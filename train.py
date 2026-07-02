from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from garl_sabre.config import EnvConfig, ModelConfig, PPOConfig, RewardConfig
from garl_sabre.dataset import generate_dataset, load_split
from garl_sabre.env import InitialLayoutEnv
from garl_sabre.model import GraphAwarePolicy
from garl_sabre.ppo import TrajectoryBuffer, ppo_update
from garl_sabre.topology import build_hardware_topology
from garl_sabre.utils import obs_to_torch, save_json, set_seed

METRIC_FIELDS = [
    "routing_score",
    "terminal_objective",
    "additional_cnot_count",
    "paper_additional_cnot_count",
    "physical_cnot_count",
    "logical_cnot_count",
    "active_logical_qubits",
    "inserted_swap_count",
    "inserted_bridge_count",
    "swap_count",
    "additional_swap_count",
    "routing_time_sec",
    "runtime",

    "input_num_qubits",
    "input_gate_count_all",
    "input_1q_count_all",
    "input_2q_count_all",
    "input_cnot_count_all",
    "input_swap_raw_count",
    "input_depth",

    "original_num_qubits",
    "original_gate_count_all",
    "original_1q_count_all",
    "original_2q_count_all",
    "original_cnot_count_all",
    "original_swap_raw_count",
    "original_cnot_equiv_count",
    "original_depth",

    "routed_gate_count_all",
    "routed_1q_count_all",
    "routed_2q_count_all",
    "routed_cnot_raw_count",
    "routed_swap_raw_count",
    "routed_cnot_equiv_count",
    "routed_depth",

    "additional_gates_total",
    "additional_1q_total",
    "additional_2q_total",
    "additional_cx_total",
    "additional_cx_total_nonnegative",
    "cnot_equiv_overhead",
    "depth_overhead",
]


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def fmt_metric(value: Optional[float], precision: int = 2, na: str = "N.A.") -> str:
    if value is None:
        return na
    return f"{float(value):.{precision}f}"


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_env(args: argparse.Namespace) -> InitialLayoutEnv:
    hardware = build_hardware_topology(
        args.phys_rows,
        args.phys_cols,
        mode=args.topology_mode,
        distance=args.topology_distance,
    )
    env_cfg = EnvConfig(
        critical_window=args.critical_window,
        lookahead_window=args.lookahead_window,
        use_physical_prior=True,
        sabre_seed=args.seed,
        optimization_level=args.optimization_level,
        topology_mode=args.topology_mode,
        topology_distance=args.topology_distance,
        action_mode=args.action_mode,
        logic_order_mode=args.logic_order_mode,
        baseline_mode=args.baseline_mode,
        evaluation_mode="paper_additional_cnot",
        metric_mode="additional_cnot_count",
        benchmark_preprocess="cnot_active_cnot_only",
        router_backend=args.router_backend,
    )
    reward_cfg = RewardConfig(terminal_scale=args.terminal_scale)
    return InitialLayoutEnv(hardware, env_cfg, reward_cfg)


def build_model_from_env(env: InitialLayoutEnv, sample, hidden_dim: int, graph_layers: int, dropout: float,
                         physical_prior_scale: float, physical_prior_clip: float, device: torch.device):
    # Populate the per-sample baseline cache before inferring observation sizes.
    if sample._cached_baseline is None:
        obs = env.reset(sample.to_circuit(), logic_graph=sample.get_logic_graph(env.env_cfg))
        sample._cached_baseline = (env.baseline_score, env.baseline_name, env.baseline_metrics)
    else:
        obs = env.reset(sample.to_circuit(), logic_graph=sample.get_logic_graph(env.env_cfg), baseline_info=sample._cached_baseline)

    logic_feat_dim = int(obs["logic_node_features"].shape[-1])
    phys_feat_dim = int(obs["physical_node_features"].shape[-1])
    edge_feat_dim = 5  # edge_freq, edge_first_layer, edge_front, edge_early, edge_log_f
    cfg = ModelConfig(
        hidden_dim=hidden_dim,
        graph_layers=graph_layers,
        dropout=dropout,
        physical_prior_scale=physical_prior_scale,
        physical_prior_clip=physical_prior_clip,
    )
    model = GraphAwarePolicy(
        cfg,
        logic_feat_dim=logic_feat_dim,
        phys_feat_dim=phys_feat_dim,
        edge_feat_dim=edge_feat_dim,
    ).to(device)
    return model, cfg


@torch.no_grad()
def run_eval_episode(model: GraphAwarePolicy, env: InitialLayoutEnv, sample, device: torch.device):
    # Reuse the cached baseline so evaluation measures only the policy layout.
    if sample._cached_baseline is None:
        obs = env.reset(sample.to_circuit(), is_training=False, logic_graph=sample.get_logic_graph(env.env_cfg))
        sample._cached_baseline = (env.baseline_score, env.baseline_name, env.baseline_metrics)
    else:
        obs = env.reset(sample.to_circuit(), is_training=False, logic_graph=sample.get_logic_graph(env.env_cfg), baseline_info=sample._cached_baseline)
    done = False
    total_reward = 0.0
    info: Dict = {}
    while not done:
        batch = obs_to_torch(obs, device)
        act_out = model.act(batch, deterministic=True)
        action = (int(act_out["action_logical"].item()), int(act_out["action_physical"].item()))
        out = env.step(action)
        total_reward += float(out.reward)
        done = bool(out.done)
        obs = out.obs
        info = out.info
    return total_reward, info


def aggregate_eval_rows(rows: List[tuple[float, Dict]]) -> Dict[str, Optional[float]]:
    if not rows:
        result: Dict[str, Optional[float]] = {"reward": 0.0}
        for field in METRIC_FIELDS:
            result[field] = None
        return result
    rewards = [r for r, _ in rows]
    infos = [i for _, i in rows]
    result: Dict[str, Optional[float]] = {"reward": float(np.mean(rewards))}
    for field in METRIC_FIELDS:
        result[field] = safe_mean([info.get(field, None) for info in infos])
    return result


def choose_eval_subset(samples: Sequence, eval_episodes: int, rng: np.random.Generator) -> List:
    if not samples:
        return []
    if eval_episodes <= 0 or eval_episodes >= len(samples):
        return list(samples)
    idx = rng.choice(len(samples), size=eval_episodes, replace=False)
    return [samples[int(i)] for i in idx]


@torch.no_grad()
def evaluate_policy(model: GraphAwarePolicy, env: InitialLayoutEnv, samples: Sequence, device: torch.device) -> Dict[str, Optional[float]]:
    model.eval()
    rows = [run_eval_episode(model, env, sample, device) for sample in samples]
    model.train()
    return aggregate_eval_rows(rows)


def stage_episode_iterator(samples: List, total_episodes: int, family_balance_mode: str, rng: np.random.Generator):
    if not samples or total_episodes <= 0:
        return
    if family_balance_mode == "none":
        yielded = 0
        while yielded < total_episodes:
            order = rng.permutation(len(samples))
            for idx in order:
                yield samples[int(idx)]
                yielded += 1
                if yielded >= total_episodes:
                    break
        return

    family_to_samples = defaultdict(list)
    for sample in samples:
        family_to_samples[str(getattr(sample, "family", "unknown"))].append(sample)
    families = sorted(family_to_samples.keys())
    counts = np.array([len(family_to_samples[f]) for f in families], dtype=np.float64)
    if family_balance_mode == "uniform":
        probs = np.ones_like(counts) / counts.size
    elif family_balance_mode == "sqrt":
        probs = np.sqrt(counts)
        probs = probs / probs.sum()
    else:
        raise ValueError(f"Unsupported family_balance_mode: {family_balance_mode}")

    yielded = 0
    while yielded < total_episodes:
        fam_idx = int(rng.choice(len(families), p=probs))
        bucket = family_to_samples[families[fam_idx]]
        item_idx = int(rng.integers(0, len(bucket)))
        yield bucket[item_idx]
        yielded += 1


def checkpoint_payload(model: GraphAwarePolicy, optimizer: torch.optim.Optimizer, model_cfg: ModelConfig,
                       env: InitialLayoutEnv, summary: Dict, args: argparse.Namespace,
                       stage_idx: int, stage_qubits: int, epoch: int) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "summary": summary,
        "model_cfg": asdict(model_cfg),
        "env_cfg": asdict(env.env_cfg),
        "reward_cfg": asdict(env.reward_cfg),
        "args": vars(args),
        "train_state": {
            "stage_idx": int(stage_idx),
            "stage_qubits": int(stage_qubits),
            "epoch": int(epoch),
        },
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def maybe_apply_stage_overrides(args: argparse.Namespace, env: InitialLayoutEnv, model: GraphAwarePolicy,
                                ppo_cfg: PPOConfig, optimizer: torch.optim.Optimizer,
                                stage_qubits: int, is_last_stage: bool,
                                num_stages: int) -> Dict[str, float | int | bool]:
    if stage_qubits < args.late_stage_start_qubits:
        return {
            "stage_physical_prior_scale": float(model.cfg.physical_prior_scale),
            "stage_entropy_coef": float(ppo_cfg.entropy_coef),
            "stage_lr": float(optimizer.param_groups[0]["lr"]),
        }
    if args.late_stage_physical_prior_scale is not None:
        model.cfg.physical_prior_scale = float(args.late_stage_physical_prior_scale)
    if args.late_stage_entropy_coef is not None:
        ppo_cfg.entropy_coef = float(args.late_stage_entropy_coef)
    if args.late_stage_lr is not None:
        set_optimizer_lr(optimizer, float(args.late_stage_lr))
    elif num_stages > 1 and is_last_stage and args.final_stage_lr_decay > 0:
        set_optimizer_lr(optimizer, args.lr * args.final_stage_lr_decay)

    return {
        "stage_physical_prior_scale": float(model.cfg.physical_prior_scale),
        "stage_entropy_coef": float(ppo_cfg.entropy_coef),
        "stage_lr": float(optimizer.param_groups[0]["lr"]),
    }


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using {device.type.upper()} device")

    if args.generate_dataset:
        generate_dataset(args.dataset_dir, args.num_circuits, args.curriculum, seed=args.seed, families=args.families)

    train_samples = load_split(args.dataset_dir, "train", args.train_manifest or None)
    valid_samples = load_split(args.dataset_dir, "valid", args.valid_manifest or None)
    if not train_samples:
        raise RuntimeError("Training split is empty.")
    if not valid_samples:
        raise RuntimeError("Validation split is empty. Provide a valid split or rebuild the dataset.")

    env = make_env(args)
    model, model_cfg = build_model_from_env(
        env,
        train_samples[0],  # 传 sample 对象过去，而不是单纯传 circuit
        args.hidden_dim,
        args.graph_layers,
        args.dropout,
        args.physical_prior_scale,
        args.physical_prior_clip,
        device,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(
        lr=args.lr,
        train_iters=args.train_iters,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        value_clip=args.value_clip,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = out_dir / "history.jsonl"
    if history_jsonl.exists():
        history_jsonl.unlink()

    save_json(
        str(out_dir / "config.json"),
        {
            "args": vars(args),
            "env_cfg": asdict(env.env_cfg),
            "reward_cfg": asdict(env.reward_cfg),
            "model_cfg": asdict(model_cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "metrics": METRIC_FIELDS,
            "mode": "paper_cnot_active_additional_cnot_initial_mapping",
        },
    )

    best_valid_additional_cnot: Optional[float] = None
    best_valid_by_stage: Dict[int, float] = {}
    curriculum = sorted(args.curriculum)

    max_hardware_qubits = int(env.hardware.num_qubits)
    for stage_idx, stage_qubits in enumerate(curriculum):
        stage_limit = min(int(stage_qubits), max_hardware_qubits)
        stage_train = [s for s in train_samples if s.num_qubits <= stage_limit]
        stage_valid = [s for s in valid_samples if s.num_qubits <= stage_limit]
        if not stage_train:
            continue
        if not stage_valid:
            raise RuntimeError(f"Validation subset is empty for stage <= {stage_limit} active logical qubits.")

        is_second_last = stage_idx == len(curriculum) - 2
        is_last = stage_idx == len(curriculum) - 1
        if len(curriculum) > 1 and stage_qubits < args.late_stage_start_qubits:
            if is_second_last:
                set_optimizer_lr(optimizer, args.lr * args.stage_lr_decay)
            elif is_last:
                set_optimizer_lr(optimizer, args.lr * args.final_stage_lr_decay)

        stage_override_info = maybe_apply_stage_overrides(
            args,
            env,
            model,
            ppo_cfg,
            optimizer,
            stage_qubits,
            is_last_stage=is_last,
            num_stages=len(curriculum),
        )
        effective_epochs = args.epochs_per_stage
        if args.late_stage_epochs_per_stage > 0 and stage_qubits >= args.late_stage_start_qubits:
            effective_epochs = args.late_stage_epochs_per_stage

        print(
            f"\n[Stage <= {stage_limit} active logical qubits] "
            f"train={len(stage_train)} valid={len(stage_valid)} "
            f"epochs={effective_epochs} episodes={args.episodes_per_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"prior={model.cfg.physical_prior_scale:.4f} "
            f"entropy={ppo_cfg.entropy_coef:.5f}"
        )

        stage_rng = np.random.default_rng(args.seed + stage_qubits * 17)
        for epoch in range(effective_epochs):
            buffer_rows = []
            episode_returns: List[float] = []
            episode_additional_cnot: List[Optional[float]] = []

            iterator = stage_episode_iterator(stage_train, args.episodes_per_epoch, args.family_balance_mode, stage_rng)
            pbar = tqdm(iterator, total=args.episodes_per_epoch, desc=f"stage={stage_qubits} epoch={epoch}")
            for sample in pbar:
                # Cache baseline routing once per sample; it dominates episode runtime.
                if sample._cached_baseline is None:
                    obs = env.reset(sample.to_circuit(), is_training=True, logic_graph=sample.get_logic_graph(env.env_cfg))
                    sample._cached_baseline = (env.baseline_score, env.baseline_name, env.baseline_metrics)
                else:
                    obs = env.reset(sample.to_circuit(), is_training=True, logic_graph=sample.get_logic_graph(env.env_cfg), baseline_info=sample._cached_baseline)
                done = False
                traj = TrajectoryBuffer()
                total_reward = 0.0
                info: Dict = {}
                while not done:
                    batch = obs_to_torch(obs, device)
                    with torch.no_grad():
                        act_out = model.act(batch)
                    action = (int(act_out["action_logical"].item()), int(act_out["action_physical"].item()))
                    out = env.step(action)
                    traj.add(
                        obs,
                        action[0],
                        action[1],
                        float(act_out["logprob"].item()),
                        float(out.reward),
                        bool(out.done),
                        float(act_out["value"].item()),
                    )
                    total_reward += float(out.reward)
                    done = bool(out.done)
                    obs = out.obs
                    info = out.info

                buffer_rows.extend(traj.compute_returns_advantages(ppo_cfg))
                episode_returns.append(total_reward)
                episode_additional_cnot.append(info.get("additional_cnot_count", info.get("routing_score", None)))
                pbar.set_postfix({
                    "reward": fmt_metric(safe_mean(episode_returns), precision=3),
                    "add_cnot": fmt_metric(safe_mean(episode_additional_cnot), precision=2),
                })

            train_metrics = ppo_update(model, optimizer, buffer_rows, ppo_cfg, device)
            eval_rng = np.random.default_rng(args.seed + stage_qubits * 1000 + epoch)
            eval_subset = choose_eval_subset(stage_valid, args.eval_episodes, rng=eval_rng)
            valid_metrics = evaluate_policy(model, env, eval_subset, device)

            summary = {
                "stage_qubits": int(stage_limit),
                "epoch": int(epoch),
                "num_eval_samples": int(len(eval_subset)),
                "train_reward": safe_mean(episode_returns),
                "train_additional_cnot": safe_mean(episode_additional_cnot),
                "valid_reward": valid_metrics["reward"],
                "valid_additional_cnot": valid_metrics.get("additional_cnot_count", valid_metrics.get("routing_score")),
                **{f"valid_{field}": valid_metrics[field] for field in METRIC_FIELDS if field != "additional_cnot_count"},
                **stage_override_info,
                **train_metrics,
            }

            append_jsonl(history_jsonl, summary)
            print(json.dumps(summary, ensure_ascii=False))

            payload = checkpoint_payload(model, optimizer, model_cfg, env, summary, args, stage_idx, stage_qubits, epoch)
            torch.save(payload, out_dir / "last_model.pt")

            stage_additional_cnot = summary["valid_additional_cnot"]
            if stage_additional_cnot is not None:
                stage_additional_cnot_val = float(stage_additional_cnot)
                prev_stage_best = best_valid_by_stage.get(stage_qubits)
                if prev_stage_best is None or stage_additional_cnot_val < prev_stage_best:
                    best_valid_by_stage[stage_qubits] = stage_additional_cnot_val
                    torch.save(payload, out_dir / f"best_model_stage_{stage_qubits}.pt")
                if best_valid_additional_cnot is None or stage_additional_cnot_val < best_valid_additional_cnot:
                    best_valid_additional_cnot = stage_additional_cnot_val
                    torch.save(payload, out_dir / "best_model.pt")

    save_json(
        str(out_dir / "final_summary.json"),
        {
            "best_valid_additional_cnot": best_valid_additional_cnot,
            "best_valid_by_stage": best_valid_by_stage,
            "history_path": str(history_jsonl),
        },
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-protocol trainer using CNOT-active circuits and additional CNOT accounting.")
    p.add_argument("--dataset_dir", type=str, default="data/demo")
    p.add_argument("--train_manifest", type=str, default="")
    p.add_argument("--valid_manifest", type=str, default="")
    p.add_argument("--save_dir", type=str, default="outputs/0607")
    p.add_argument("--generate_dataset", action="store_true")
    p.add_argument("--num_circuits", type=int, default=120)
    p.add_argument("--families", nargs="+", default=["qaoa", "hea", "qft", "grover", "adder", "random", "routing_stress"])

    p.add_argument("--phys_rows", type=int, default=0)
    p.add_argument("--phys_cols", type=int, default=0)
    p.add_argument("--topology_mode", type=str, default="ibm_q20", choices=["ibm_q20","heavy_hex"])
    p.add_argument("--topology_distance", type=int, default=5)

    p.add_argument("--curriculum", type=int, nargs="+", default=[10, 14, 16, 20])
    p.add_argument("--epochs_per_stage", type=int, default=5)
    p.add_argument("--late_stage_epochs_per_stage", type=int, default=0)
    p.add_argument("--episodes_per_epoch", type=int, default=40)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--family_balance_mode", type=str, default="sqrt", choices=["none", "uniform", "sqrt"])

    p.add_argument("--critical_window", type=int, default=8)
    p.add_argument("--lookahead_window", type=int, default=16)
    p.add_argument("--optimization_level", type=int, default=0)
    p.add_argument("--action_mode", type=str, default="fixed_order_physical", choices=["hierarchical", "fixed_order_physical"])
    
    p.add_argument("--terminal_scale", type=float, default=20.0)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--graph_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--physical_prior_scale", type=float, default=0.35)
    p.add_argument("--physical_prior_clip", type=float, default=3.5)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_iters", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--target_kl", type=float, default=0.03)
    p.add_argument("--clip_ratio", type=float, default=0.20)
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--value_clip", type=float, default=0.2)

    p.add_argument("--stage_lr_decay", type=float, default=0.75)
    p.add_argument("--final_stage_lr_decay", type=float, default=0.5)
    p.add_argument("--late_stage_start_qubits", type=int, default=32)
    p.add_argument("--late_stage_lr", type=float, default=None)
    p.add_argument("--late_stage_entropy_coef", type=float, default=None)
    p.add_argument("--late_stage_physical_prior_scale", type=float, default=None)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--logic_order_mode", type=str, default="priority_fixed", choices=["priority_fixed", "front_first", "index"])
    p.add_argument("--baseline_mode", type=str, default="dense", choices=["none", "trivial", "dense", "sabre"])
    p.add_argument("--router_backend", type=str, default="qiskit", choices=["qiskit", "tket"])
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_training(args)

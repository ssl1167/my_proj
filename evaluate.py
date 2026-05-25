from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from garl_sabre.config import EnvConfig, ModelConfig, RewardConfig
from garl_sabre.dataset import load_split
from garl_sabre.env import InitialLayoutEnv
from garl_sabre.model import GraphAwarePolicy
from garl_sabre.tabu_refine import tabu_refine_layout
from garl_sabre.topology import build_hardware_topology
from garl_sabre.utils import obs_to_torch

EVAL_COLUMNS = [
    "reward",
    "routing_score",
    "terminal_objective",
    "terminal_reward",
    "baseline_score",
    "baseline_name",
    "additional_cnot_count",
    "paper_additional_cnot_count",
    "physical_cnot_count",
    "logical_cnot_count",
    "active_logical_qubits",
    "inserted_swap_count",
    "inserted_bridge_count",
    "bridge_count_source",
    "swap_count",
    "swap_count_source",
    "additional_swap_count",
    "routing_time_sec",
    "runtime",
    "evaluating_router",
    "metric_protocol",
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

PAPER_COLUMNS = [
    "family",
    "name",
    "num_qubits",
    "initial_mapper",
    "mode",
    "logical_cnot_count",
    "active_logical_qubits",
    "inserted_swap_count",
    "inserted_bridge_count",
    "additional_cnot_count",
    "physical_cnot_count",
    "routing_time_sec",
    "evaluating_router",
    "metric_protocol",
]


@dataclass
class SearchNode:
    snapshot: Dict
    logprob: float
    total_reward: float
    done: bool
    info: Dict
    mode: str

    @property
    def rank_score(self) -> float:
        shaping_weight = float(self.info.get("beam_reward_weight", 0.20))
        return float(self.logprob + shaping_weight * self.total_reward)


def build_model_from_env(env, sample_circuit, device, state_dict):
    obs = env.reset(sample_circuit, is_training=False)
    logic_feat_dim = int(obs["logic_node_features"].shape[-1])
    phys_feat_dim = int(obs["physical_node_features"].shape[-1])
    candidate_feat_dim = int(obs["candidate_features_bank"].shape[-1])
    logical_candidate_feat_dim = int(obs["logical_candidate_features"].shape[-1])
    cfg_dict = state_dict.get("model_cfg", {}) if isinstance(state_dict, dict) else {}
    cfg = ModelConfig(**cfg_dict) if cfg_dict else ModelConfig()
    return GraphAwarePolicy(
        cfg,
        logic_feat_dim=logic_feat_dim,
        phys_feat_dim=phys_feat_dim,
        candidate_feat_dim=candidate_feat_dim,
        logical_candidate_feat_dim=logical_candidate_feat_dim,
    ).to(device)


def _metric_value(info: Dict, metric: str) -> Optional[float]:
    if metric not in {"additional_cnot_count", "swap_count", "routing_score"}:
        raise ValueError(f"Unknown eval metric: {metric}")
    val = info.get(metric, None)
    if val is None and metric == "additional_cnot_count":
        val = info.get("routing_score", None)
    return None if val is None else float(val)


def _is_better(candidate_info: Dict, best_info: Dict, metric: str) -> bool:
    cand = _metric_value(candidate_info, metric)
    best = _metric_value(best_info, metric)
    if cand is None:
        return False
    if best is None:
        return True
    return cand < best


def _as_float(info: Dict, key: str) -> Optional[float]:
    val = info.get(key, None)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _check_metric_consistency(info: Dict, name: str, atol: float = 1e-6) -> None:
    logical = _as_float(info, "logical_cnot_count")
    physical = _as_float(info, "physical_cnot_count")
    additional = _as_float(info, "additional_cnot_count")
    if logical is not None and physical is not None and additional is not None:
        expected_physical = logical + additional
        if abs(expected_physical - physical) > atol:
            raise ValueError(f"Metric inconsistency for {name}: physical_cnot_count={physical}, expected {expected_physical}.")

    paper_add = _as_float(info, "paper_additional_cnot_count")
    if paper_add is not None and additional is not None and abs(paper_add - additional) > atol:
        raise ValueError(f"Metric inconsistency for {name}: paper_additional_cnot_count={paper_add}, additional_cnot_count={additional}.")

    inserted_swap = _as_float(info, "inserted_swap_count")
    additional_swap = _as_float(info, "additional_swap_count")
    if inserted_swap is not None and additional_swap is not None and abs(inserted_swap - additional_swap) > atol:
        raise ValueError(f"Metric inconsistency for {name}: inserted_swap_count={inserted_swap}, additional_swap_count={additional_swap}.")

    additional_cx_nn = _as_float(info, "additional_cx_total_nonnegative")
    if additional is not None and inserted_swap is not None and additional_cx_nn is not None:
        expected_additional = max(additional_cx_nn, 3.0 * inserted_swap)
        if abs(additional - expected_additional) > atol:
            raise ValueError(
                f"Metric inconsistency for {name}: additional_cnot_count={additional}, "
                f"expected max(additional_cx_total_nonnegative={additional_cx_nn}, 3*inserted_swap_count={3.0 * inserted_swap}) = {expected_additional}."
            )

    routed_2q = _as_float(info, "routed_2q_count_all")
    original_2q = _as_float(info, "original_2q_count_all")
    reported_2q = _as_float(info, "additional_2q_total")
    if routed_2q is not None and original_2q is not None and reported_2q is not None and abs((routed_2q - original_2q) - reported_2q) > atol:
        raise ValueError(f"Metric inconsistency for {name}: additional_2q_total={reported_2q}, expected {routed_2q - original_2q}.")

    routed_depth = _as_float(info, "routed_depth")
    original_depth = _as_float(info, "original_depth")
    depth_overhead = _as_float(info, "depth_overhead")
    if routed_depth is not None and original_depth is not None and depth_overhead is not None and abs((routed_depth - original_depth) - depth_overhead) > atol:
        raise ValueError(f"Metric inconsistency for {name}: depth_overhead={depth_overhead}, expected {routed_depth - original_depth}.")


def run_episode(model, env, circuit, device, deterministic: bool = True, use_physical_prior: bool = True, disable_candidate_ranking_eval: bool = False):
    old_ranking = env.env_cfg.use_candidate_ranking
    if disable_candidate_ranking_eval:
        env.env_cfg.use_candidate_ranking = False
    try:
        obs = env.reset(circuit, is_training=False)
        done = False
        total_reward = 0.0
        info: Dict = {}
        with torch.no_grad():
            while not done:
                batch = obs_to_torch(obs, device)
                act_out = model.act(batch, deterministic=deterministic, use_physical_prior=use_physical_prior)
                action = (int(act_out["action_logical"].item()), int(act_out["action_physical"].item()))
                out = env.step(action)
                done = bool(out.done)
                obs = out.obs
                total_reward += float(out.reward)
                info = out.info
        return total_reward, info
    finally:
        env.env_cfg.use_candidate_ranking = old_ranking


def _topk_valid(logits: torch.Tensor, k: int) -> List[int]:
    logits = logits.detach().cpu()
    valid = torch.isfinite(logits) & (logits > -1e8)
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    vals = logits[idx]
    order = torch.argsort(vals, descending=True)
    chosen = idx[order[: min(k, idx.numel())]]
    return [int(x) for x in chosen.tolist()]


def run_beam_episode(
    model,
    env: InitialLayoutEnv,
    circuit,
    device,
    beam_width: int = 4,
    logic_branch: int = 2,
    physical_branch: int = 4,
    beam_reward_weight: float = 0.20,
    disable_candidate_ranking_eval: bool = True,
    use_physical_prior: bool = True,
    choose_metric: str = "additional_cnot_count",
):
    old_ranking = env.env_cfg.use_candidate_ranking
    if disable_candidate_ranking_eval:
        env.env_cfg.use_candidate_ranking = False
    try:
        env.reset(circuit, is_training=False)
        beams = [SearchNode(snapshot=env.snapshot(), logprob=0.0, total_reward=0.0, done=False, info={"beam_reward_weight": beam_reward_weight}, mode="beam")]
        completed: List[SearchNode] = []
        with torch.no_grad():
            for _ in range(env.logic.num_qubits if env.logic is not None else 0):
                candidates: List[SearchNode] = []
                for node in beams:
                    if node.done:
                        completed.append(node)
                        continue
                    env.restore(node.snapshot)
                    obs = env.get_obs()
                    batch = obs_to_torch(obs, device)
                    step_logits = model.get_step_logits(batch)
                    logical_logits = step_logits["logical_logits"][0]
                    logical_logprob = torch.log_softmax(logical_logits, dim=-1)
                    for logical_q in _topk_valid(logical_logits, logic_branch):
                        logical_q_t = torch.tensor([logical_q], dtype=torch.long, device=device)
                        physical_logits = model.get_physical_logits(step_logits, logical_q_t, use_physical_prior=use_physical_prior)[0]
                        physical_logprob = torch.log_softmax(physical_logits, dim=-1)
                        for phys_q in _topk_valid(physical_logits, physical_branch):
                            env.restore(node.snapshot)
                            out = env.step((logical_q, phys_q))
                            child_info = dict(out.info)
                            child_info["beam_reward_weight"] = beam_reward_weight
                            candidates.append(
                                SearchNode(
                                    snapshot=env.snapshot(),
                                    logprob=float(node.logprob + logical_logprob[logical_q].item() + physical_logprob[phys_q].item()),
                                    total_reward=float(node.total_reward + out.reward),
                                    done=bool(out.done),
                                    info=child_info,
                                    mode="beam",
                                )
                            )
                if not candidates:
                    break
                candidates.sort(key=lambda x: x.rank_score, reverse=True)
                beams = candidates[: max(1, beam_width)]
                if all(node.done for node in beams):
                    completed.extend(beams)
                    break

        final_pool = completed if completed else beams
        if not final_pool:
            raise RuntimeError("Beam search produced no candidates.")
        finished = [n for n in final_pool if _metric_value(n.info, choose_metric) is not None]
        best = min(finished, key=lambda x: float(_metric_value(x.info, choose_metric))) if finished else max(final_pool, key=lambda x: x.rank_score)
        return best.total_reward, best.info
    finally:
        env.env_cfg.use_candidate_ranking = old_ranking


def maybe_tabu_refine(best_info: Dict, env: InitialLayoutEnv, circuit, args) -> Dict:
    if args.tabu_iters <= 0:
        return best_info
    layout = best_info.get("final_layout")
    if layout is None:
        return best_info
    if env.logic is None:
        env.reset(circuit, is_training=False)
    result = tabu_refine_layout(
        circuit=circuit,
        initial_layout=layout,
        logic=env.logic,
        hardware=env.hardware,
        env_cfg=env.env_cfg,
        reward_cfg=env.reward_cfg,
        num_iters=args.tabu_iters,
        candidate_qubits=args.tabu_candidate_qubits,
        relocate_candidates=args.tabu_relocate_candidates,
        tabu_tenure=args.tabu_tenure,
        exact_eval_every=args.tabu_exact_every,
    )
    refined = dict(best_info)
    if result.routing_score + 1e-8 < float(best_info.get("routing_score", 1e18)):
        refined.update(result.metrics)
        refined["routing_score"] = float(result.routing_score)
        refined["terminal_objective"] = float(result.metrics.get("terminal_objective", result.routing_score))
        refined["final_layout"] = result.layout
        refined["tabu_improved"] = 1
    else:
        refined["tabu_improved"] = 0
    refined["tabu_iters"] = int(result.num_iters)
    refined["tabu_exact_evals"] = int(result.num_exact_evals)
    refined["tabu_surrogate_score"] = float(result.surrogate_score)
    return refined


def _default_paper_csv_path(save_csv: str) -> str:
    out = Path(save_csv)
    return str(out.with_name(f"{out.stem}_paper{out.suffix or '.csv'}"))


def _pick_best_candidate(candidates: List[Tuple[float, Dict, str]], metric: str) -> Tuple[float, Dict, str]:
    best_reward, best_info, best_mode = candidates[0]
    for cand_reward, cand_info, cand_mode in candidates[1:]:
        if _is_better(cand_info, best_info, metric):
            best_reward, best_info, best_mode = cand_reward, cand_info, cand_mode
    return best_reward, best_info, best_mode


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RL initial mapping with CNOT-only paper accounting.")
    p.add_argument("--dataset_dir", type=str, default="data/demo")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--split_manifest", type=str, default="")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--phys_rows", type=int, default=0)
    p.add_argument("--phys_cols", type=int, default=0)
    p.add_argument("--topology_mode", type=str, default="ibm_q20", choices=["grid", "bottleneck_grid", "ibm_q20", "heavy_hex"])
    p.add_argument("--topology_distance", type=int, default=5)
    p.add_argument("--router_backend", type=str, default=None, choices=["qiskit", "tket"])
    p.add_argument("--save_csv", type=str, default="outputs/eval.csv")
    p.add_argument("--paper_csv", type=str, default="")
    p.add_argument("--initial_mapper_label", type=str, default="rl+beam+tabu")
    p.add_argument("--eval_metric", type=str, default="additional_cnot_count", choices=["additional_cnot_count", "routing_score", "swap_count"])
    p.add_argument("--num_rollouts", type=int, default=1)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--beam_logic_branch", type=int, default=2)
    p.add_argument("--beam_physical_branch", type=int, default=4)
    p.add_argument("--beam_reward_weight", type=float, default=0.20)
    p.add_argument("--keep_eval_candidate_ranking", action="store_true")
    p.add_argument("--disable_eval_physical_prior", action="store_true")
    p.add_argument("--skip_greedy", action="store_true")
    p.add_argument("--tabu_iters", type=int, default=12)
    p.add_argument("--tabu_candidate_qubits", type=int, default=6)
    p.add_argument("--tabu_relocate_candidates", type=int, default=3)
    p.add_argument("--tabu_tenure", type=int, default=5)
    p.add_argument("--tabu_exact_every", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using", ("Cuda device: " + torch.cuda.get_device_name(device)) if device.type == "cuda" else "CPU device")

    state = torch.load(args.checkpoint, map_location=device)
    env_cfg = EnvConfig(**state.get("env_cfg", {})) if isinstance(state, dict) and "env_cfg" in state else EnvConfig()
    reward_cfg = RewardConfig(**state.get("reward_cfg", {})) if isinstance(state, dict) and "reward_cfg" in state else RewardConfig()
    env_cfg.sabre_seed = args.seed
    env_cfg.topology_mode = args.topology_mode
    env_cfg.topology_distance = args.topology_distance
    if args.router_backend:
        env_cfg.router_backend = args.router_backend

    hardware = build_hardware_topology(args.phys_rows, args.phys_cols, mode=args.topology_mode, distance=args.topology_distance)
    env = InitialLayoutEnv(hardware, env_cfg, reward_cfg)
    samples = load_split(args.dataset_dir, args.split, args.split_manifest or None, canonicalize_cnot_only=True)
    if not samples:
        raise RuntimeError("Evaluation split is empty.")

    model = build_model_from_env(env, samples[0].to_circuit(), device, state if isinstance(state, dict) else {})
    load_result = model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        print("Checkpoint key mismatch:")
        print("  missing:", getattr(load_result, "missing_keys", []))
        print("  unexpected:", getattr(load_result, "unexpected_keys", []))
    model.eval()

    use_physical_prior = not args.disable_eval_physical_prior
    disable_candidate_ranking_eval = not args.keep_eval_candidate_ranking

    rows = []
    for sample in tqdm(samples, desc="eval"):
        circuit = sample.to_circuit()
        candidate_runs: List[Tuple[float, Dict, str]] = []
        if args.beam_width > 1:
            beam_reward, beam_info = run_beam_episode(
                model,
                env,
                circuit,
                device,
                beam_width=args.beam_width,
                logic_branch=args.beam_logic_branch,
                physical_branch=args.beam_physical_branch,
                beam_reward_weight=args.beam_reward_weight,
                disable_candidate_ranking_eval=disable_candidate_ranking_eval,
                use_physical_prior=use_physical_prior,
                choose_metric=args.eval_metric,
            )
            candidate_runs.append((beam_reward, beam_info, f"beam{args.beam_width}"))
        else:
            if not args.skip_greedy:
                reward, info = run_episode(model, env, circuit, device, deterministic=True, use_physical_prior=use_physical_prior, disable_candidate_ranking_eval=disable_candidate_ranking_eval)
                candidate_runs.append((reward, info, "greedy"))
            for rollout_idx in range(max(0, args.num_rollouts - 1)):
                cand_reward, cand_info = run_episode(model, env, circuit, device, deterministic=False, use_physical_prior=use_physical_prior, disable_candidate_ranking_eval=disable_candidate_ranking_eval)
                candidate_runs.append((cand_reward, cand_info, f"sample_{rollout_idx + 1}"))

        if not candidate_runs:
            raise RuntimeError("No evaluation candidate was produced. Check your flags.")

        reward, info, mode = _pick_best_candidate(candidate_runs, args.eval_metric)
        info = maybe_tabu_refine(info, env, circuit, args)
        if args.tabu_iters > 0:
            mode = f"{mode}_tabu" if info.get("tabu_improved", 0) else mode

        _check_metric_consistency(info, sample.name)

        row = {"name": sample.name, "family": sample.family, "num_qubits": sample.num_qubits, "initial_mapper": args.initial_mapper_label, "mode": mode, "reward": float(reward)}
        for col in EVAL_COLUMNS:
            if col == "reward":
                continue
            elif col == "routing_time_sec":
                row[col] = info.get(col, info.get("runtime", None))
            elif col == "evaluating_router":
                row[col] = info.get(col, getattr(env_cfg, "router_backend", "qiskit"))
            else:
                row[col] = info.get(col, None)
        row["tabu_improved"] = info.get("tabu_improved", 0)
        row["tabu_exact_evals"] = info.get("tabu_exact_evals", 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.save_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    paper_path = Path(args.paper_csv or _default_paper_csv_path(args.save_csv))
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cols = [c for c in PAPER_COLUMNS if c in df.columns]
    df[existing_cols].to_csv(paper_path, index=False)

    print(df.groupby("family")[[
        "reward",
        "additional_cnot_count",
        "physical_cnot_count",
        "logical_cnot_count",
        "inserted_swap_count",
        "inserted_bridge_count",
        "baseline_score",
        "routing_time_sec",
    ]].mean(numeric_only=True))
    print(f"Saved full metrics to {out}")
    print(f"Saved paper-style metrics to {paper_path}")


if __name__ == "__main__":
    main()

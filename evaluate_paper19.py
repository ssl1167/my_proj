from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from qiskit.qasm2 import dumps

from garl_sabre.config import EnvConfig, RewardConfig
from garl_sabre.dataset import CircuitSample, save_json
from garl_sabre.env import InitialLayoutEnv
from garl_sabre.heuristics import best_of_random, dense_layout, sabre_layout, trivial_layout
from garl_sabre.model import GraphAwarePolicy
from garl_sabre.qiskit_runner import evaluate_layout_metrics, objective_from_metrics
from garl_sabre.topology import build_hardware_topology
from garl_sabre.circuit_features import build_logic_graph

from convert_revlib_real import parse_real_file
from evaluate import build_model_from_env, run_beam_episode, maybe_tabu_refine


def convert_paper19_to_json(paper19_dir: Path, output_dir: Path) -> List[CircuitSample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_files = sorted(paper19_dir.rglob("*.real"))
    if not real_files:
        raise FileNotFoundError(f"No .real files found under {paper19_dir}")
    rows = []
    for path in real_files:
        try:
            qc = parse_real_file(path)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue
        stem = path.stem
        parts = stem.rsplit("_", 1)
        family = parts[0].strip() if len(parts) == 2 and parts[1].isdigit() and parts[0].strip() else "revlib"
        rows.append({
            "name": path.stem,
            "family": family,
            "num_qubits": qc.num_qubits,
            "qasm": dumps(qc),
        })
    if not rows:
        raise RuntimeError("No .real files were successfully converted.")
    save_json(output_dir / "paper19.json", rows)
    print(f"Converted {len(rows)} circuits to {output_dir / 'paper19.json'}")
    return [CircuitSample(**row) for row in rows]


def evaluate_rl_model(samples: List[CircuitSample], checkpoint_path: str, hardware, args) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print((f"Using CUDA device: {torch.cuda.get_device_name(device)}") if device.type == "cuda" else "Using CPU device")

    state = torch.load(checkpoint_path, map_location=device)
    env_cfg = EnvConfig(**state.get("env_cfg", {})) if isinstance(state, dict) and "env_cfg" in state else EnvConfig(
        sabre_seed=args.seed,
        topology_mode=args.topology_mode,
        topology_distance=args.topology_distance,
    )
    env_cfg.sabre_seed = args.seed
    env_cfg.topology_mode = args.topology_mode
    env_cfg.topology_distance = args.topology_distance
    reward_cfg = RewardConfig(**state.get("reward_cfg", {})) if isinstance(state, dict) and "reward_cfg" in state else RewardConfig()

    env = InitialLayoutEnv(hardware, env_cfg, reward_cfg)
    model = build_model_from_env(env, samples[0].to_circuit(), device, state if isinstance(state, dict) else {})
    load_result = model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        print("Checkpoint key mismatch:")
        print("  missing:", getattr(load_result, "missing_keys", []))
        print("  unexpected:", getattr(load_result, "unexpected_keys", []))
    model.eval()

    rows: List[Dict] = []
    for sample in tqdm(samples, desc="RL evaluation"):
        circuit = sample.to_circuit()
        _, info = run_beam_episode(
            model,
            env,
            circuit,
            device,
            beam_width=args.beam_width,
            logic_branch=args.beam_logic_branch,
            physical_branch=args.beam_physical_branch,
            beam_reward_weight=args.beam_reward_weight,
            disable_candidate_ranking_eval=not args.keep_eval_candidate_ranking,
            choose_metric="swap_count",
        )
        if args.tabu_iters > 0:
            tabu_args = argparse.Namespace(
                tabu_iters=args.tabu_iters,
                tabu_candidate_qubits=args.tabu_candidate_qubits,
                tabu_relocate_candidates=args.tabu_relocate_candidates,
                tabu_tenure=args.tabu_tenure,
                tabu_exact_every=args.tabu_exact_every,
            )
            info = maybe_tabu_refine(info, env, circuit, tabu_args)

        rows.append({
            "name": sample.name,
            "family": sample.family,
            "num_qubits": sample.num_qubits,
            "method": "RL+beam+tabu",
            "swap_count": info.get("swap_count", None),
            "routing_score": info.get("routing_score", None),
            "routing_time_sec": info.get("routing_time_sec", info.get("runtime", None)),
            "original_gate_count_all": info.get("original_gate_count_all", None),
            "original_1q_count_all": info.get("original_1q_count_all", None),
            "original_2q_count_all": info.get("original_2q_count_all", None),
            "original_cnot_count_all": info.get("original_cnot_count_all", None),
            "original_depth": info.get("original_depth", None),
            "routed_gate_count_all": info.get("routed_gate_count_all", None),
            "routed_1q_count_all": info.get("routed_1q_count_all", None),
            "routed_2q_count_all": info.get("routed_2q_count_all", None),
            "routed_swap_count": info.get("routed_swap_count", info.get("swap_count", None)),
            "routed_cnot_raw_count": info.get("routed_cnot_raw_count", None),
            "routed_cnot_equiv_count": info.get("routed_cnot_equiv_count", None),
            "routed_depth": info.get("routed_depth", None),
            "additional_gates_total": info.get("additional_gates_total", None),
            "additional_1q_total": info.get("additional_1q_total", None),
            "additional_2q_total": info.get("additional_2q_total", None),
            "additional_swap_count": info.get("additional_swap_count", info.get("swap_count", None)),
            "additional_cnot_equiv_from_swap": info.get("additional_cnot_equiv_from_swap", None),
            "depth_overhead": info.get("depth_overhead", None),
            "evaluating_router": info.get("evaluating_router", "qiskit_sabre"),
        })
    return rows


def evaluate_baselines(samples: List[CircuitSample], hardware, args, topology_mode: str, topology_distance: int) -> List[Dict]:
    env_cfg = EnvConfig(sabre_seed=args.seed, topology_mode=topology_mode, topology_distance=topology_distance)
    reward_cfg = RewardConfig()
    rows: List[Dict] = []
    for sample in tqdm(samples, desc="Baseline evaluation"):
        circuit = sample.to_circuit()
        logic = build_logic_graph(circuit, critical_window=env_cfg.critical_window, lookahead_window=env_cfg.lookahead_window)
        candidates = {
            "trivial": trivial_layout(circuit.num_qubits),
            "dense": dense_layout(logic, hardware),
            "random_best": best_of_random(circuit, hardware, env_cfg, reward_cfg=reward_cfg, trials=args.random_trials, seed=args.seed),
        }
        try:
            candidates["sabre_layout"] = sabre_layout(circuit, hardware, seed=args.seed)
        except Exception as e:
            print(f"[SKIP sabre_layout for {sample.name}]: {e}")

        for method_name, layout in candidates.items():
            metrics = evaluate_layout_metrics(circuit, layout, hardware, env_cfg)
            score = objective_from_metrics(metrics, reward_cfg, env_cfg)
            rows.append({
                "name": sample.name,
                "family": sample.family,
                "num_qubits": sample.num_qubits,
                "method": method_name,
                "routing_score": float(score),
                **metrics,
            })
    return rows


def generate_comparison_table(df: pd.DataFrame, output_path: Path):
    print(f"实际评测的方法: {df['method'].unique()}")
    value_cols = [
        "swap_count",
        "original_cnot_count_all",
        "routed_cnot_equiv_count",
        "routed_cnot_raw_count",
        "additional_cnot_equiv_from_swap",
        "additional_swap_count",
        "routing_time_sec",
        "depth_overhead",
    ]
    existing_value_cols = [c for c in value_cols if c in df.columns]

    table_df = df.pivot_table(
        index=["family", "name", "num_qubits"],
        columns="method",
        values=existing_value_cols,
        aggfunc="first",
    )
    table_df = table_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    rl_method = "RL+beam+tabu"
    for method in ["trivial", "dense", "random_best", "sabre_layout"]:
        if method in df["method"].unique() and rl_method in df["method"].unique() and (method, "swap_count") in table_df.columns:
            table_df[("swap_improvement_pct", method)] = (
                (table_df[(method, "swap_count")] - table_df[(rl_method, "swap_count")])
                / table_df[(method, "swap_count")].replace(0, pd.NA)
                * 100.0
            )

    table_df.to_csv(output_path)
    print(f"对比表格已保存到 {output_path}")

    print("\n=== 按方法汇总统计 ===")
    agg_dict = {}
    for col in [
        "swap_count",
        "original_cnot_count_all",
        "routed_cnot_equiv_count",
        "routed_cnot_raw_count",
        "additional_cnot_equiv_from_swap",
        "additional_swap_count",
        "routing_score",
        "routing_time_sec",
        "depth_overhead",
    ]:
        if col in df.columns:
            agg_dict[col] = ["mean", "std"] if col != "swap_count" else ["mean", "std", "min", "max"]
    if agg_dict:
        print(df.groupby("method").agg(agg_dict).round(2))

    if rl_method in df["method"].unique():
        print("\n=== 按 family 汇总 (RL+beam+tabu) ===")
        family_cols = [c for c in ["swap_count", "original_cnot_count_all", "routed_cnot_equiv_count", "additional_cnot_equiv_from_swap", "depth_overhead"] if c in df.columns]
        if family_cols:
            print(df[df["method"] == rl_method].groupby("family")[family_cols].agg(["mean", "std"]).round(2))

    return table_df


def main():
    parser = argparse.ArgumentParser(description="评测 paper19 电路并输出统一统计口径表格")
    parser.add_argument("--paper19_dir", type=str, default="data/paper19")
    parser.add_argument("--output_dir", type=str, default="outputs/paper19_eval")
    parser.add_argument("--checkpoint", type=str, default="outputs/pure_swap_run/best_model.pt")
    parser.add_argument("--phys_rows", type=int, default=0)
    parser.add_argument("--phys_cols", type=int, default=0)
    parser.add_argument("--topology_mode", type=str, default="heavy_hex", choices=["grid", "bottleneck_grid", "ibm_q20", "heavy_hex"])
    parser.add_argument("--topology_distance", type=int, default=5)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--beam_logic_branch", type=int, default=2)
    parser.add_argument("--beam_physical_branch", type=int, default=4)
    parser.add_argument("--beam_reward_weight", type=float, default=0.20)
    parser.add_argument("--keep_eval_candidate_ranking", action="store_true")
    parser.add_argument("--tabu_iters", type=int, default=12)
    parser.add_argument("--tabu_candidate_qubits", type=int, default=6)
    parser.add_argument("--tabu_relocate_candidates", type=int, default=3)
    parser.add_argument("--tabu_tenure", type=int, default=5)
    parser.add_argument("--tabu_exact_every", type=int, default=0)
    parser.add_argument("--random_trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== 加载模型配置 ===")
    state = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    env_cfg_from_checkpoint = state.get("env_cfg", {})
    topology_mode = env_cfg_from_checkpoint.get("topology_mode", args.topology_mode)
    topology_distance = env_cfg_from_checkpoint.get("topology_distance", args.topology_distance)
    print(f"拓扑模式: {topology_mode}, 距离参数: {topology_distance}")

    print("\n=== 步骤1: 转换 paper19 电路 ===")
    samples = convert_paper19_to_json(Path(args.paper19_dir), output_path)
    if not samples:
        print("没有成功转换的电路")
        return

    print("\n=== 步骤2: 构建硬件拓扑 ===")
    hardware = build_hardware_topology(args.phys_rows, args.phys_cols, mode=topology_mode, distance=topology_distance)
    print(f"Hardware: {hardware.num_qubits} qubits")

    print("\n=== 步骤3: RL 模型评测 ===")
    rl_results = evaluate_rl_model(samples, args.checkpoint, hardware, args)

    print("\n=== 步骤4: Baseline 评测 ===")
    baseline_results = evaluate_baselines(samples, hardware, args, topology_mode, topology_distance)

    print("\n=== 步骤5: 生成对比表格 ===")
    all_results = rl_results + baseline_results
    df = pd.DataFrame(all_results)

    full_result_path = output_path / "full_results.csv"
    df.to_csv(full_result_path, index=False)
    print(f"完整结果已保存到 {full_result_path}")

    comparison_path = output_path / "comparison_table.csv"
    generate_comparison_table(df, comparison_path)

    paper_table_path = output_path / "paper_table.csv"
    df[[
        "family",
        "name",
        "method",
        "swap_count",
        "original_cnot_count_all",
        "routed_cnot_equiv_count",
        "routed_cnot_raw_count",
        "additional_cnot_equiv_from_swap",
        "additional_swap_count",
        "routing_time_sec",
    ]].to_csv(paper_table_path, index=False)
    print(f"论文表格已保存到 {paper_table_path}")

    print("\n=== 评测完成 ===")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from qiskit.qasm2 import dumps
from tqdm import tqdm

from garl_sabre.config import EnvConfig, RewardConfig
from garl_sabre.dataset import CircuitSample, save_json
from garl_sabre.env import InitialLayoutEnv
from garl_sabre.heuristics import best_of_random, dense_layout, sabre_layout, trivial_layout
from garl_sabre.model import GraphAwarePolicy
from garl_sabre.qiskit_runner import canonical_cnot_circuit, evaluate_layout_metrics, objective_from_metrics
from garl_sabre.topology import build_hardware_topology
from garl_sabre.circuit_features import build_logic_graph

from convert_revlib_real import parse_real_file
from evaluate import build_model_from_env, run_beam_episode, maybe_tabu_refine


EXPECTED_PAPER19_SANITY: Dict[str, Tuple[int, int]] = {
    "clip_206": (14, 14772),
    "rd73_252": (10, 2319),
}

PAPER_METRIC_COLUMNS = [
    "logical_cnot_count",
    "active_logical_qubits",
    "inserted_swap_count",
    "inserted_bridge_count",
    "additional_cnot_count",
    "paper_additional_cnot_count",
    "physical_cnot_count",
    "swap_count",
    "routing_score",
    "routing_time_sec",
    "evaluating_router",
    "metric_protocol",
]


def _filtered_env_config(data: Dict) -> EnvConfig:
    allowed = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in dict(data or {}).items() if k in allowed})


def _force_paper_protocol(env_cfg: EnvConfig, args: argparse.Namespace) -> EnvConfig:
    env_cfg.sabre_seed = int(args.seed)
    env_cfg.topology_mode = "ibm_q20"
    env_cfg.topology_distance = int(args.topology_distance)
    env_cfg.router_backend = str(args.router_backend)
    env_cfg.evaluation_mode = "paper_additional_cnot"
    env_cfg.metric_mode = "additional_cnot_count"
    env_cfg.benchmark_preprocess = "cnot_active_cnot_only"
    env_cfg.metric_protocol = "cnot_active_cnot_only_additional_cnot_v1"
    env_cfg.baseline_mode = str(args.baseline_mode)
    env_cfg.optimization_level = 0
    # Do not inherit old checkpoint basis settings such as swap/native 2Q gates;
    # paper19 evaluation is CNOT-only before routing, with SWAP retained only as
    # a diagnostic representation of Qiskit SABRE inserts.
    env_cfg.basis_gates = ["cx", "id", "rz", "sx", "x"]
    return env_cfg


def _row_from_info(sample: CircuitSample, method: str, info: Dict, reward: Optional[float] = None) -> Dict:
    row = {
        "name": sample.name,
        "family": sample.family,
        "num_qubits": sample.num_qubits,
        "method": method,
        "reward": reward,
    }
    for col in PAPER_METRIC_COLUMNS:
        row[col] = info.get(col, None)
    row["baseline_score"] = info.get("baseline_score", None)
    row["baseline_name"] = info.get("baseline_name", None)
    row["terminal_reward"] = info.get("terminal_reward", None)
    row["terminal_objective"] = info.get("terminal_objective", info.get("routing_score", None))
    return row


def _check_metric_consistency(row: Dict, atol: float = 1e-6) -> None:
    logical = row.get("logical_cnot_count")
    additional = row.get("additional_cnot_count")
    physical = row.get("physical_cnot_count")
    if logical is not None and additional is not None and physical is not None:
        if abs(float(logical) + float(additional) - float(physical)) > atol:
            raise ValueError(
                f"Metric inconsistency for {row.get('name')} {row.get('method')}: "
                f"physical_cnot_count={physical}, expected {float(logical) + float(additional)}."
            )
    paper_add = row.get("paper_additional_cnot_count")
    if paper_add is not None and additional is not None:
        if abs(float(paper_add) - float(additional)) > atol:
            raise ValueError(
                f"Metric inconsistency for {row.get('name')} {row.get('method')}: "
                f"paper_additional_cnot_count={paper_add}, additional_cnot_count={additional}."
            )


def convert_paper19_to_json(paper19_dir: Path, output_dir: Path, strict_sanity: bool = True) -> List[CircuitSample]:
    from qiskit import QuantumCircuit
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 优先查找 .qasm 文件（从 examples 目录）
    qasm_files = sorted((paper19_dir / "examples").rglob("*.qasm"))
    if not qasm_files:
        # 如果没有 .qasm 文件，尝试查找 .real 文件
        qasm_files = sorted(paper19_dir.rglob("*.real"))
        if not qasm_files:
            raise FileNotFoundError(f"No .qasm or .real files found under {paper19_dir}")

    rows = []
    for path in qasm_files:
        try:
            if path.suffix == ".qasm":
                # 直接加载 .qasm 文件
                raw_qc = QuantumCircuit.from_qasm_file(path)
            else:
                # 解析 .real 文件
                raw_qc = parse_real_file(path)
            qc = canonical_cnot_circuit(raw_qc)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue

        stem = path.stem
        parts = stem.rsplit("_", 1)
        family = parts[0].strip() if len(parts) == 2 and parts[1].isdigit() and parts[0].strip() else "revlib"
        logical_cnot = int(qc.count_ops().get("cx", 0))

        if strict_sanity and stem in EXPECTED_PAPER19_SANITY:
            expected_qubits, expected_cx = EXPECTED_PAPER19_SANITY[stem]
            if qc.num_qubits != expected_qubits or logical_cnot != expected_cx:
                raise ValueError(
                    f"Paper19 sanity check failed for {stem}: got {qc.num_qubits} qubits and {logical_cnot} cx; "
                    f"expected {expected_qubits} qubits and {expected_cx} cx."
                )

        rows.append({
            "name": stem,
            "family": family,
            "num_qubits": int(qc.num_qubits),
            "qasm": dumps(raw_qc),
        })

    if not rows:
        raise RuntimeError("No .real files were successfully converted.")

    if strict_sanity:
        found_names = {row["name"] for row in rows}
        missing = sorted(set(EXPECTED_PAPER19_SANITY) - found_names)
        if missing:
            raise ValueError(
                "Paper19 sanity check could not be completed because these reference circuits "
                f"were not found after conversion: {missing}.  Check --paper19_dir."
            )

    save_json(output_dir / "paper19.json", rows)
    print(f"Converted {len(rows)} canonical CNOT-only circuits to {output_dir / 'paper19.json'}")
    return [CircuitSample(**row) for row in rows]


def evaluate_rl_model(samples: List[CircuitSample], checkpoint_path: str, hardware, args) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print((f"Using CUDA device: {torch.cuda.get_device_name(device)}") if device.type == "cuda" else "Using CPU device")

    state = torch.load(checkpoint_path, map_location=device)
    env_cfg = _filtered_env_config(state.get("env_cfg", {}) if isinstance(state, dict) else {})
    env_cfg = _force_paper_protocol(env_cfg, args)
    reward_cfg = RewardConfig(**{k: v for k, v in (state.get("reward_cfg", {}) if isinstance(state, dict) else {}).items() if k in {f.name for f in fields(RewardConfig)}})

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
        reward, info = run_beam_episode(
            model,
            env,
            circuit,
            device,
            beam_width=args.beam_width,
            logic_branch=args.beam_logic_branch,
            physical_branch=args.beam_physical_branch,
            beam_reward_weight=args.beam_reward_weight,
            choose_metric="additional_cnot_count",
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

        method_name = "RL+beam+tabu" if args.tabu_iters > 0 else "RL+beam"
        row = _row_from_info(sample, method_name, info, reward=reward)
        _check_metric_consistency(row)
        rows.append(row)
    return rows


def evaluate_baselines(samples: List[CircuitSample], hardware, args) -> List[Dict]:
    env_cfg = EnvConfig(
        sabre_seed=args.seed,
        topology_mode="ibm_q20",
        topology_distance=args.topology_distance,
        router_backend=args.router_backend,
        evaluation_mode="paper_additional_cnot",
        metric_mode="additional_cnot_count",
        benchmark_preprocess="cnot_active_cnot_only",
        baseline_mode=args.baseline_mode,
        optimization_level=0,
    )
    reward_cfg = RewardConfig()
    rows: List[Dict] = []
    for sample in tqdm(samples, desc="Baseline evaluation"):
        circuit = sample.to_circuit()
        logic = build_logic_graph(circuit, critical_window=env_cfg.critical_window, lookahead_window=env_cfg.lookahead_window, decompose=True)
        candidates = {
            "trivial": trivial_layout(logic.num_qubits),
            "dense": dense_layout(logic, hardware),
            "random_best": best_of_random(circuit, hardware, env_cfg, reward_cfg=reward_cfg, trials=args.random_trials, seed=args.seed),
        }
        try:
            candidates["sabre_layout"] = sabre_layout(circuit, hardware, seed=args.seed)
        except Exception as exc:
            print(f"[SKIP sabre_layout for {sample.name}]: {exc}")

        for method_name, layout in candidates.items():
            metrics = evaluate_layout_metrics(circuit, layout, hardware, env_cfg)
            score = float(objective_from_metrics(metrics, reward_cfg, env_cfg))
            metrics = dict(metrics)
            metrics["routing_score"] = score
            metrics["terminal_objective"] = score
            row = _row_from_info(sample, method_name, metrics)
            _check_metric_consistency(row)
            rows.append(row)
    return rows


def generate_comparison_table(df: pd.DataFrame, output_path: Path):
    print(f"实际评测的方法: {df['method'].unique()}")
    value_cols = [c for c in PAPER_METRIC_COLUMNS if c in df.columns]
    table_df = df.pivot_table(
        index=["family", "name", "num_qubits"],
        columns="method",
        values=value_cols,
        aggfunc="first",
    )
    table_df = table_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    rl_method = "RL+beam+tabu"
    for method in ["trivial", "dense", "random_best", "sabre_layout"]:
        if method in df["method"].unique() and rl_method in df["method"].unique() and (method, "additional_cnot_count") in table_df.columns:
            table_df[("additional_cnot_improvement_pct", method)] = (
                (table_df[(method, "additional_cnot_count")] - table_df[(rl_method, "additional_cnot_count")])
                / table_df[(method, "additional_cnot_count")].replace(0, pd.NA)
                * 100.0
            )

    table_df.to_csv(output_path)
    print(f"对比表格已保存到 {output_path}")

    print("\n=== 按方法汇总统计 ===")
    agg_cols = [c for c in ["additional_cnot_count", "logical_cnot_count", "physical_cnot_count", "inserted_swap_count", "routing_time_sec"] if c in df.columns]
    if agg_cols:
        print(df.groupby("method")[agg_cols].agg(["mean", "std", "min", "max"]).round(2))
    return table_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper19 circuits under CNOT-active additional-CNOT protocol.")
    parser.add_argument("--paper19_dir", type=str, default="data/paper19")
    parser.add_argument("--output_dir", type=str, default="outputs/paper19_eval")
    parser.add_argument("--checkpoint", type=str, default="outputs/paper_protocol_run/best_model.pt")
    parser.add_argument("--phys_rows", type=int, default=0)
    parser.add_argument("--phys_cols", type=int, default=0)
    parser.add_argument("--topology_mode", type=str, default="ibm_q20", choices=["ibm_q20","heavy_tex"])
    parser.add_argument("--topology_distance", type=int, default=5)
    parser.add_argument("--router_backend", type=str, default="qiskit", choices=["qiskit", "tket"])
    parser.add_argument("--baseline_mode", type=str, default="dense", choices=["none", "trivial", "dense", "sabre"])
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--beam_logic_branch", type=int, default=2)
    parser.add_argument("--beam_physical_branch", type=int, default=4)
    parser.add_argument("--beam_reward_weight", type=float, default=0.20)
    parser.add_argument("--tabu_iters", type=int, default=12)
    parser.add_argument("--tabu_candidate_qubits", type=int, default=6)
    parser.add_argument("--tabu_relocate_candidates", type=int, default=3)
    parser.add_argument("--tabu_tenure", type=int, default=5)
    parser.add_argument("--tabu_exact_every", type=int, default=0)
    parser.add_argument("--random_trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_sanity_check", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: canonicalize paper19 circuits ===")
    samples = convert_paper19_to_json(Path(args.paper19_dir), output_path, strict_sanity=not args.no_sanity_check)

    print("\n=== Step 2: build IBM Q20 topology ===")
    hardware = build_hardware_topology(args.phys_rows, args.phys_cols, mode="ibm_q20", distance=args.topology_distance)
    print(f"Hardware: {hardware.num_qubits} qubits, topology={hardware.topology_name}")

    print("\n=== Step 3: RL model evaluation ===")
    rl_results = evaluate_rl_model(samples, args.checkpoint, hardware, args)

    print("\n=== Step 4: baseline evaluation ===")
    baseline_results = evaluate_baselines(samples, hardware, args)

    print("\n=== Step 5: export tables ===")
    df = pd.DataFrame(rl_results + baseline_results)
    full_result_path = output_path / "full_results.csv"
    df.to_csv(full_result_path, index=False)
    print(f"完整结果已保存到 {full_result_path}")

    comparison_path = output_path / "comparison_table.csv"
    generate_comparison_table(df, comparison_path)

    paper_table_path = output_path / "paper_table.csv"
    paper_cols = ["family", "name", "num_qubits", "method", *PAPER_METRIC_COLUMNS]
    existing = [c for c in paper_cols if c in df.columns]
    df[existing].to_csv(paper_table_path, index=False)
    print(f"论文表格已保存到 {paper_table_path}")
    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()

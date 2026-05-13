# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from qiskit.compiler import transpile
from tqdm import tqdm

from garl_sabre.config import EnvConfig
from garl_sabre.dataset import load_split
from garl_sabre.qiskit_runner import circuit_gate_profile, prepare_basis_circuit
from garl_sabre.topology import build_hardware_topology


def transpile_with_sabre(
    circuit,
    hardware,
    env_cfg: EnvConfig,
) -> Dict[str, float | str]:
    """
    Full Qiskit SABRE baseline under the unified counting protocol:
    - keep original total gate counts (all gates)
    - keep routed total gate counts (all gates)
    - keep exact additional SWAP count
    - keep CNOT-equivalent cost induced by inserted SWAPs
    """
    prepared = prepare_basis_circuit(circuit, env_cfg)
    basis_gates = list(env_cfg.basis_gates)
    if "swap" not in basis_gates:
        basis_gates.append("swap")

    start = time.perf_counter()
    routed = transpile(
        prepared,
        coupling_map=hardware.coupling_map,
        basis_gates=basis_gates,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=env_cfg.optimization_level,
        seed_transpiler=env_cfg.sabre_seed,
    )
    elapsed = time.perf_counter() - start

    original_profile = circuit_gate_profile(prepared)
    routed_profile = circuit_gate_profile(routed)

    swap_count = float(routed_profile["swap_count"])
    additional_cnot_equiv_from_swap = float(3.0 * swap_count)
    routed_cnot_equiv_count = float(original_profile["cx_count_all"] + additional_cnot_equiv_from_swap)

    additional_gates_total = float(routed_profile["gate_count_all"] - original_profile["gate_count_all"])
    additional_1q_total = float(routed_profile["oneq_count_all"] - original_profile["oneq_count_all"])
    additional_2q_total = float(routed_profile["twoq_count_all"] - original_profile["twoq_count_all"])
    depth_overhead = float(routed_profile["depth"] - original_profile["depth"])

    return {
        "swap_count": swap_count,
        "swap_count_source": "exact_swap_ops",
        "routing_score": swap_count,
        "terminal_objective": swap_count,
        "routing_time_sec": float(elapsed),
        "runtime": float(elapsed),
        "evaluating_router": "qiskit_sabre_full",
        "original_num_qubits": float(prepared.num_qubits),
        "original_gate_count_all": float(original_profile["gate_count_all"]),
        "original_1q_count_all": float(original_profile["oneq_count_all"]),
        "original_2q_count_all": float(original_profile["twoq_count_all"]),
        "original_cnot_count_all": float(original_profile["cx_count_all"]),
        "original_depth": float(original_profile["depth"]),
        "routed_gate_count_all": float(routed_profile["gate_count_all"]),
        "routed_1q_count_all": float(routed_profile["oneq_count_all"]),
        "routed_2q_count_all": float(routed_profile["twoq_count_all"]),
        "routed_swap_count": swap_count,
        "routed_cnot_raw_count": float(routed_profile["cx_count_all"]),
        "routed_cnot_equiv_count": routed_cnot_equiv_count,
        "routed_depth": float(routed_profile["depth"]),
        "additional_gates_total": additional_gates_total,
        "additional_1q_total": additional_1q_total,
        "additional_2q_total": additional_2q_total,
        "additional_swap_count": swap_count,
        "additional_cnot_equiv_from_swap": additional_cnot_equiv_from_swap,
        "depth_overhead": depth_overhead,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run a standalone SABRE baseline on a dataset split.")
    p.add_argument("--dataset_dir", type=str, default="data/demo")
    p.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--split_manifest", type=str, default="")
    p.add_argument("--phys_rows", type=int, default=0)
    p.add_argument("--phys_cols", type=int, default=0)
    p.add_argument("--topology_mode", type=str, default="heavy_hex", choices=["grid", "bottleneck_grid", "ibm_q20", "heavy_hex"])
    p.add_argument("--topology_distance", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--optimization_level", type=int, default=0)
    p.add_argument("--save_csv", type=str, default="outputs/sabre_valid.csv")
    args = p.parse_args()

    env_cfg = EnvConfig(
        sabre_seed=args.seed,
        topology_mode=args.topology_mode,
        topology_distance=args.topology_distance,
        optimization_level=args.optimization_level,
    )
    hardware = build_hardware_topology(args.phys_rows, args.phys_cols, mode=args.topology_mode, distance=args.topology_distance)
    samples = load_split(args.dataset_dir, args.split, args.split_manifest or None)

    rows: List[Dict[str, float | str]] = []
    for sample in tqdm(samples, desc="sabre_baseline"):
        circuit = sample.to_circuit()
        metrics = transpile_with_sabre(circuit, hardware, env_cfg)
        rows.append(
            {
                "name": sample.name,
                "family": sample.family,
                "num_qubits": sample.num_qubits,
                "method": "qiskit_sabre_full",
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    out = Path(args.save_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary_cols = [
        "swap_count",
        "additional_gates_total",
        "additional_swap_count",
        "routed_cnot_equiv_count",
        "routing_time_sec",
        "original_gate_count_all",
    ]
    print(df.groupby(["method", "family"])[summary_cols].mean(numeric_only=True))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from garl_sabre.circuit_features import build_logic_graph
from garl_sabre.config import EnvConfig, RewardConfig
from garl_sabre.dataset import load_split
from garl_sabre.env import InitialLayoutEnv
from garl_sabre.heuristics import (
    best_of_random,
    dense_layout,
    sabre_layout,
    trivial_layout,
)
from garl_sabre.qiskit_runner import evaluate_layout_metrics, objective_from_metrics
from garl_sabre.topology import build_hardware_topology
from run_sabre_baseline import transpile_with_sabre
from evaluate import (
    build_model_from_env,
    maybe_tabu_refine,
    run_beam_episode,
    run_episode,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, default="data/demo")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--split_manifest", type=str, default="")
    p.add_argument("--phys_rows", type=int, default=0)
    p.add_argument("--phys_cols", type=int, default=0)
    p.add_argument(
        "--topology_mode",
        type=str,
        default="heavy_hex",
        choices=["grid", "bottleneck_grid", "ibm_q20", "heavy_hex"],
    )
    p.add_argument("--topology_distance", type=int, default=5)
    p.add_argument("--random_trials", type=int, default=64)
    p.add_argument("--save_csv", type=str, default="outputs/baselines_with_rl.csv")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--include_qiskit_sabre_full",
        action="store_true",
        help="Add a full transpile-based Qiskit SABRE baseline to the same CSV.",
    )
    p.add_argument("--rl_checkpoint", type=str, default="")
    p.add_argument("--rl_initial_mapper_label", type=str, default="rl+beam+tabu")
    p.add_argument("--rl_num_rollouts", type=int, default=1)
    p.add_argument("--rl_beam_width", type=int, default=4)
    p.add_argument("--rl_beam_logic_branch", type=int, default=2)
    p.add_argument("--rl_beam_physical_branch", type=int, default=4)
    p.add_argument("--rl_beam_reward_weight", type=float, default=0.20)
    p.add_argument("--rl_tabu_iters", type=int, default=12)
    p.add_argument("--rl_tabu_candidate_qubits", type=int, default=6)
    p.add_argument("--rl_tabu_relocate_candidates", type=int, default=3)
    p.add_argument("--rl_tabu_tenure", type=int, default=5)
    p.add_argument("--rl_tabu_exact_every", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


def make_common_env_cfg(args) -> EnvConfig:
    return EnvConfig(
        sabre_seed=args.seed,
        topology_mode=args.topology_mode,
        topology_distance=args.topology_distance,
    )


def build_rl_components(args, samples, hardware):
    if not args.rl_checkpoint:
        return None
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using Cuda device:", torch.cuda.get_device_name(device))
    else:
        print("Using CPU device")
        device = torch.device("cpu")

    state = torch.load(args.rl_checkpoint, map_location=device)

    env_cfg = (
        EnvConfig(**state.get("env_cfg", {}))
        if isinstance(state, dict) and "env_cfg" in state
        else EnvConfig()
    )
    env_cfg.sabre_seed = args.seed
    env_cfg.topology_mode = args.topology_mode
    env_cfg.topology_distance = args.topology_distance

    reward_cfg = (
        RewardConfig(**state.get("reward_cfg", {}))
        if isinstance(state, dict) and "reward_cfg" in state
        else RewardConfig()
    )
    env = InitialLayoutEnv(hardware, env_cfg, reward_cfg)
    model = build_model_from_env(env, samples[0].to_circuit(), device, state if isinstance(state, dict) else {})
    load_result = model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        print("RL checkpoint key mismatch:")
        print("  missing:", getattr(load_result, "missing_keys", []))
        print("  unexpected:", getattr(load_result, "unexpected_keys", []))
    model.eval()

    tabu_args = SimpleNamespace(
        tabu_iters=args.rl_tabu_iters,
        tabu_candidate_qubits=args.rl_tabu_candidate_qubits,
        tabu_relocate_candidates=args.rl_tabu_relocate_candidates,
        tabu_tenure=args.rl_tabu_tenure,
        tabu_exact_every=args.rl_tabu_exact_every,
    )

    return {
        "device": device,
        "state": state,
        "env": env,
        "model": model,
        "tabu_args": tabu_args,
    }


def evaluate_rl_on_sample(args, rl_pack, sample):
    circuit = sample.to_circuit()
    model = rl_pack["model"]
    env = rl_pack["env"]
    device = rl_pack["device"]

    if args.rl_beam_width > 1:
        reward, info = run_beam_episode(
            model,
            env,
            circuit,
            device,
            beam_width=args.rl_beam_width,
            logic_branch=args.rl_beam_logic_branch,
            physical_branch=args.rl_beam_physical_branch,
            beam_reward_weight=args.rl_beam_reward_weight,
        )
        mode = f"beam{args.rl_beam_width}"
    else:
        reward, info = run_episode(model, env, circuit, device, deterministic=True)
        mode = "greedy"
        for _ in range(max(0, args.rl_num_rollouts - 1)):
            cand_reward, cand_info = run_episode(model, env, circuit, device, deterministic=False)
            if info.get("routing_score") is None or (
                cand_info.get("routing_score") is not None
                and float(cand_info["routing_score"]) < float(info["routing_score"])
            ):
                reward, info = cand_reward, cand_info
                mode = "sample_best"

    info = maybe_tabu_refine(info, env, circuit, rl_pack["tabu_args"])
    if args.rl_tabu_iters > 0:
        mode = f"{mode}_tabu" if info.get("tabu_improved", 0) else mode

    row = {
        "name": sample.name,
        "family": sample.family,
        "num_qubits": sample.num_qubits,
        "method": args.rl_initial_mapper_label,
        "swap_count": info.get("swap_count", None),
        "routing_score": info.get("routing_score", None),
        "routing_time_sec": info.get("routing_time_sec", info.get("runtime", None)),
        "original_num_qubits": info.get("original_num_qubits", sample.num_qubits),
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
        "mode": mode,
    }
    return row


if __name__ == "__main__":
    args = build_parser().parse_args()

    hardware = build_hardware_topology(
        args.phys_rows,
        args.phys_cols,
        mode=args.topology_mode,
        distance=args.topology_distance,
    )
    samples = load_split(args.dataset_dir, args.split, args.split_manifest or None)

    env_cfg = make_common_env_cfg(args)
    reward_cfg = RewardConfig()
    rows = []

    rl_pack = build_rl_components(args, samples, hardware)

    for sample in tqdm(samples, desc="compare"):
        circuit = sample.to_circuit()
        logic = build_logic_graph(
            circuit,
            critical_window=env_cfg.critical_window,
            lookahead_window=env_cfg.lookahead_window,
        )

        candidates = {
            "trivial": trivial_layout(circuit.num_qubits),
            "dense": dense_layout(logic, hardware),
            "random_best": best_of_random(
                circuit,
                hardware,
                env_cfg,
                reward_cfg=reward_cfg,
                trials=args.random_trials,
                seed=args.seed,
            ),
        }

        try:
            candidates["sabre_layout"] = sabre_layout(circuit, hardware, seed=args.seed)
        except Exception:
            pass

        for method_name, layout in candidates.items():
            metrics = evaluate_layout_metrics(circuit, layout, hardware, env_cfg)
            score = objective_from_metrics(metrics, reward_cfg, env_cfg)
            rows.append(
                {
                    "name": sample.name,
                    "family": sample.family,
                    "num_qubits": sample.num_qubits,
                    "method": method_name,
                    "routing_score": float(score),
                    **metrics,
                }
            )

        if args.include_qiskit_sabre_full:
            sabre_metrics = transpile_with_sabre(circuit, hardware, env_cfg)
            sabre_score = objective_from_metrics(sabre_metrics, reward_cfg, env_cfg)
            rows.append(
                {
                    "name": sample.name,
                    "family": sample.family,
                    "num_qubits": sample.num_qubits,
                    "method": "qiskit_sabre_full",
                    "routing_score": float(sabre_score),
                    **sabre_metrics,
                }
            )

        if rl_pack is not None:
            rows.append(evaluate_rl_on_sample(args, rl_pack, sample))

    df = pd.DataFrame(rows)
    out = Path(args.save_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    preferred_cols = [
        "swap_count",
        "additional_gates_total",
        "additional_swap_count",
        "routed_cnot_equiv_count",
        "routing_time_sec",
        "original_gate_count_all",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]
    if existing_cols:
        print(df.groupby(["method", "family"])[existing_cols].mean(numeric_only=True))
    else:
        print("No preferred columns found to summarize.")

    print(f"Saved to {out}")

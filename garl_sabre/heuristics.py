from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from .circuit_features import LogicGraphData, build_logic_graph
from .config import EnvConfig, RewardConfig
from .qiskit_runner import evaluate_layout_metrics, objective_from_metrics, prepare_basis_circuit
from .topology import HardwareTopology


def trivial_layout(num_logical: int) -> List[int]:
    return list(range(num_logical))


def dense_layout(logic: LogicGraphData, hardware: HardwareTopology) -> List[int]:
    logical_order = logic.placement_order
    centrality = (
        hardware.node_features[:, 0]
        + hardware.node_features[:, 1]
        + hardware.node_features[:, 2]
        + hardware.node_features[:, 3]
    )
    physical_order = list(np.argsort(-centrality))

    layout = [-1] * logic.num_qubits
    for q, p in zip(logical_order, physical_order):
        layout[q] = int(p)

    free = [p for p in range(hardware.num_qubits) if p not in layout]
    for i in range(len(layout)):
        if layout[i] < 0:
            layout[i] = free.pop(0)

    return layout


def random_layout(num_logical: int, num_physical: int, rng: random.Random) -> List[int]:
    picks = list(range(num_physical))
    rng.shuffle(picks)
    return picks[:num_logical]


def sabre_layout(
    circuit: QuantumCircuit,
    hardware: HardwareTopology,
    seed: int = 7,
    optimization_level: int = 1,
) -> List[int]:
    """
    Extract an initial SABRE layout robustly across circuits / Qiskit versions.

    Note:
    - We keep this as a lightweight heuristic baseline.
    - Full-strength SABRE comparisons should use the standalone/full baseline script.
    """
    prepared = prepare_basis_circuit(circuit)
    transpiled = transpile(
        prepared,
        coupling_map=hardware.coupling_map,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=optimization_level,
        seed_transpiler=seed,
        basis_gates=["cx", "id", "rz", "sx", "x"],
    )

    init_layout = transpiled.layout.initial_layout
    mapping = [-1] * prepared.num_qubits

    for virt, phys in init_layout.get_virtual_bits().items():
        try:
            logical_idx = prepared.find_bit(virt).index
        except Exception:
            # Ancilla / extra virtual bits may not belong to the original circuit.
            continue

        if 0 <= logical_idx < prepared.num_qubits:
            mapping[logical_idx] = int(phys)

    free = [p for p in range(hardware.num_qubits) if p not in mapping]
    for i, v in enumerate(mapping):
        if v < 0:
            mapping[i] = free.pop(0)

    return mapping


def best_of_random(
    circuit: QuantumCircuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig | None = None,
    trials: int = 32,
    seed: int = 7,
) -> List[int]:
    """
    Random-best baseline.

    This baseline is intentionally bounded to a moderate number of trials because
    each trial invokes a full downstream evaluation. For main-paper comparisons,
    32 or 64 trials are usually sufficient; larger values should be reserved for
    supplementary fairness checks.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    reward_cfg = reward_cfg or RewardConfig()
    rng = random.Random(seed)
    prepared = prepare_basis_circuit(circuit, env_cfg)

    best_layout = None
    best_metric = float("inf")

    for _ in range(int(trials)):
        layout = random_layout(prepared.num_qubits, hardware.num_qubits, rng)
        metrics = evaluate_layout_metrics(circuit, layout, hardware, env_cfg)
        metric = objective_from_metrics(metrics, reward_cfg, env_cfg)
        if metric < best_metric:
            best_metric = metric
            best_layout = layout

    assert best_layout is not None
    return best_layout


def teacher_layouts(
    circuit: QuantumCircuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    seed: int = 7,
    random_trials: int = 32,
    logic: Optional[LogicGraphData] = None,
) -> Dict[str, List[int]]:
    """
    Teacher layouts for imitation / warmup.

    The caller may pass a precomputed logic graph to avoid redundant graph
    construction when this function is called repeatedly.
    """
    if logic is None:
        logic = build_logic_graph(
            circuit,
            critical_window=env_cfg.critical_window,
            lookahead_window=env_cfg.lookahead_window,
        )

    result = {
        "trivial": trivial_layout(logic.num_qubits),
        "dense": dense_layout(logic, hardware),
        "random_best": best_of_random(
            circuit,
            hardware,
            env_cfg,
            trials=random_trials,
            seed=seed,
        ),
    }

    try:
        result["sabre"] = sabre_layout(circuit, hardware, seed=seed)
    except Exception:
        pass

    return result

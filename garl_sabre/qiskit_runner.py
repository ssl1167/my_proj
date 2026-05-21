from __future__ import annotations

import time
from typing import Dict, List, Sequence

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from .config import EnvConfig, RewardConfig
from .topology import HardwareTopology

_PREPARED_FLAG = "_garl_basis_prepared"
_PREPARED_BASIS = "_garl_basis_gates"
_PREPARED_OPT = "_garl_basis_opt_level"


def _routing_basis_gates(env_cfg: EnvConfig) -> List[str]:
    """
    Keep SWAP explicit in the routed circuit so that we can report both:
    - exact additional SWAP count
    - total routed gate count
    - CNOT-equivalent cost induced by inserted SWAPs
    """
    basis = list(env_cfg.basis_gates)
    if "swap" not in basis:
        basis.append("swap")
    return basis


def prepare_basis_circuit(circuit: QuantumCircuit, env_cfg: EnvConfig | None = None) -> QuantumCircuit:
    env_cfg = env_cfg or EnvConfig()
    metadata = dict(circuit.metadata or {})
    basis_tag = tuple(env_cfg.basis_gates)

    if (
        metadata.get(_PREPARED_FLAG, False)
        and tuple(metadata.get(_PREPARED_BASIS, ())) == basis_tag
        and int(metadata.get(_PREPARED_OPT, -1)) == 0
    ):
        return circuit

    prepared = transpile(
        circuit,
        basis_gates=env_cfg.basis_gates,
        optimization_level=env_cfg.optimization_level,
        seed_transpiler=env_cfg.sabre_seed,
    )
    new_metadata = dict(prepared.metadata or {})
    new_metadata[_PREPARED_FLAG] = True
    new_metadata[_PREPARED_BASIS] = list(basis_tag)
    new_metadata[_PREPARED_OPT] = env_cfg.optimization_level
    prepared.metadata = new_metadata
    return prepared


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    total = 0
    for item in circuit.data:
        qubits = getattr(item, "qubits", item[1])
        if len(qubits) == 2:
            total += 1
    return total


def count_swaps(circuit: QuantumCircuit) -> int:
    total = 0
    for item in circuit.data:
        op = getattr(item, "operation", item[0])
        if op.name == "swap":
            total += 1
    return total


def count_named_gate(circuit: QuantumCircuit, gate_name: str) -> int:
    try:
        counts = circuit.count_ops()
        return int(counts.get(gate_name, 0))
    except Exception:
        total = 0
        for item in circuit.data:
            op = getattr(item, "operation", item[0])
            if op.name == gate_name:
                total += 1
        return total


def circuit_gate_profile(circuit: QuantumCircuit) -> Dict[str, float]:
    physical_ops = [
        item for item in circuit.data
        if getattr(item, "operation", item[0]).name not in ["barrier", "measure", "rz", "delay"]
    ]

    total_physical_gates = float(len(physical_ops))
    twoq_gates = float(count_two_qubit_gates(circuit))
    oneq_gates = max(0.0, total_physical_gates - twoq_gates)
    swap_count = float(count_swaps(circuit))
    cx_count = float(count_named_gate(circuit, "cx"))
    try:
        depth = float(circuit.depth())
    except Exception:
        depth = float("nan")
    return {
        "num_qubits": float(circuit.num_qubits),
        "gate_count_all": total_physical_gates,
        "oneq_count_all": float(oneq_gates),
        "twoq_count_all": float(twoq_gates),
        "cx_count_all": cx_count,
        "swap_count": swap_count,
        "depth": depth,
    }


def _build_metrics(prepared: QuantumCircuit, routed: QuantumCircuit, elapsed: float, evaluating_router: str) -> Dict[str, float | str]:
    original_profile = circuit_gate_profile(prepared)
    routed_profile = circuit_gate_profile(routed)

    additional_swap_count = float(routed_profile["swap_count"] - original_profile["swap_count"])
    additional_cnot_equiv_from_swap = float(3.0 * additional_swap_count)
    routed_cnot_equiv_count = float(routed_profile["cx_count_all"] + 3.0 * routed_profile["swap_count"])

    additional_gates_total = float(routed_profile["gate_count_all"] - original_profile["gate_count_all"])
    additional_1q_total = float(routed_profile["oneq_count_all"] - original_profile["oneq_count_all"])
    additional_2q_total = float(routed_profile["twoq_count_all"] - original_profile["twoq_count_all"])
    depth_overhead = float(routed_profile["depth"] - original_profile["depth"])

    return {
        # optimization target
        "swap_count": additional_swap_count,
        "swap_count_source": "exact_swap_ops",
        "routing_score": additional_swap_count,
        "terminal_objective": additional_swap_count,
        # runtime
        "routing_time_sec": float(elapsed),
        "runtime": float(elapsed),
        "evaluating_router": evaluating_router,
        # original circuit statistics (all gates retained)
        "original_num_qubits": float(prepared.num_qubits),
        "original_gate_count_all": float(original_profile["gate_count_all"]),
        "original_1q_count_all": float(original_profile["oneq_count_all"]),
        "original_2q_count_all": float(original_profile["twoq_count_all"]),
        "original_cnot_count_all": float(original_profile["cx_count_all"]),
        "original_depth": float(original_profile["depth"]),
        # routed circuit statistics (all gates retained)
        "routed_gate_count_all": float(routed_profile["gate_count_all"]),
        "routed_1q_count_all": float(routed_profile["oneq_count_all"]),
        "routed_2q_count_all": float(routed_profile["twoq_count_all"]),
        "routed_cnot_raw_count": float(routed_profile["cx_count_all"]),
        "routed_cnot_equiv_count": routed_cnot_equiv_count,
        "routed_swap_count": additional_swap_count,
        "routed_depth": float(routed_profile["depth"]),
        # deltas / added cost
        "additional_gates_total": additional_gates_total,
        "additional_1q_total": additional_1q_total,
        "additional_2q_total": additional_2q_total,
        "additional_swap_count": additional_swap_count,
        "additional_cnot_equiv_from_swap": additional_cnot_equiv_from_swap,
        "depth_overhead": depth_overhead,
    }


def routing_score_from_metrics(
    metrics: Dict[str, float | None],
    alpha_swap: float = 1.0,
    beta_depth: float = 0.0,
    gamma_twoq: float = 0.0,
    eta_added_twoq: float = 0.0,
    theta_depth_overhead: float = 0.0,
) -> float:
    del alpha_swap, beta_depth, gamma_twoq, eta_added_twoq, theta_depth_overhead
    return float(metrics.get("swap_count", 0.0) or 0.0)


def transpile_with_layout(
    circuit: QuantumCircuit,
    layout: List[int],
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    prepared = prepare_basis_circuit(circuit, env_cfg)

    start = time.perf_counter()
    routed = transpile(
        prepared,
        coupling_map=hardware.coupling_map,
        basis_gates=_routing_basis_gates(env_cfg),
        routing_method="sabre",
        initial_layout=layout,
        optimization_level=env_cfg.optimization_level,
        seed_transpiler=env_cfg.sabre_seed,
    )
    elapsed = time.perf_counter() - start
    return _build_metrics(prepared, routed, elapsed, evaluating_router="qiskit_sabre")


def evaluate_initial_mapping_with_router(
    circuit: QuantumCircuit,
    layout: Sequence[int],
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


def evaluate_layout_metrics(
    circuit: QuantumCircuit,
    layout: Sequence[int],
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


def objective_from_metrics(metrics: Dict[str, float | str | None], reward_cfg: RewardConfig, env_cfg: EnvConfig) -> float:
    del reward_cfg, env_cfg
    return float(metrics.get("swap_count", 0.0) or 0.0)


from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, transpile

from .config import EnvConfig, RewardConfig
from .topology import HardwareTopology

try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.architecture import Architecture
    from pytket.passes import RoutingPass
    from pytket.circuit import Node as TKNode
    HAS_TKET = True
except ImportError:
    HAS_TKET = False

CNOT_CANONICAL_BASIS = ["rz", "sx", "x", "cx"]
_PREPARED_FLAG = "_garl_cnot_only_prepared"
_PREPARED_PROTOCOL = "_garl_protocol"


def _instruction_parts(inst):
    if hasattr(inst, "operation"):
        return inst.operation, inst.qubits, inst.clbits
    op = inst[0]
    qargs = inst[1]
    cargs = inst[2] if len(inst) > 2 else []
    return op, qargs, cargs


def _basis_with_swap(env_cfg: EnvConfig) -> List[str]:
    basis = list(getattr(env_cfg, "basis_gates", []) or CNOT_CANONICAL_BASIS)
    if "swap" not in basis:
        basis.append("swap")
    if "cx" not in basis:
        basis.append("cx")
    return list(dict.fromkeys(basis))


def decompose_to_cx_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    不进行门分解，直接返回原电路。
    仅在后续处理中保留原本就是 cx 的门。
    """
    return circuit


def canonical_cnot_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    C++-style benchmark canonicalization:
    不进行门分解，仅保留原本就是 cx 的门操作。
    其他门（单比特门、cz、swap等）直接忽略，不做任何处理。
    """
    cx_ops: List[Tuple[int, int]] = []
    active: set[int] = set()

    for inst in circuit.data:
        op, qargs, _ = _instruction_parts(inst)
        # 仅保留名称为 "cx" 的门，其他门直接跳过
        if getattr(op, "name", "") != "cx" or len(qargs) != 2:
            continue
        c = circuit.find_bit(qargs[0]).index
        t = circuit.find_bit(qargs[1]).index
        cx_ops.append((c, t))
        active.add(c)
        active.add(t)

    if not cx_ops:
        return QuantumCircuit(max(1, min(circuit.num_qubits, 1)), name=circuit.name)

    active_sorted = sorted(active)
    remap = {old: new for new, old in enumerate(active_sorted)}
    out = QuantumCircuit(len(active_sorted), name=circuit.name)
    for c, t in cx_ops:
        out.cx(remap[c], remap[t])
    out.metadata = dict(circuit.metadata or {})
    out.metadata[_PREPARED_FLAG] = True
    out.metadata[_PREPARED_PROTOCOL] = "cnot_active_cnot_only_v1"
    return out


def prepare_basis_circuit(circuit: QuantumCircuit, env_cfg: EnvConfig | None = None) -> QuantumCircuit:
    del env_cfg
    metadata = dict(circuit.metadata or {})
    if metadata.get(_PREPARED_FLAG, False) and metadata.get(_PREPARED_PROTOCOL) == "cnot_active_cnot_only_v1":
        return circuit
    return canonical_cnot_circuit(circuit)


def count_named_gate(circuit: QuantumCircuit, gate_name: str) -> int:
    try:
        return int(circuit.count_ops().get(gate_name, 0))
    except Exception:
        total = 0
        for item in circuit.data:
            op, _, _ = _instruction_parts(item)
            if getattr(op, "name", "") == gate_name:
                total += 1
        return total


def count_swaps(circuit: QuantumCircuit) -> int:
    return count_named_gate(circuit, "swap")


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    total = 0
    for item in circuit.data:
        _, qargs, _ = _instruction_parts(item)
        if len(qargs) == 2:
            total += 1
    return total


def circuit_gate_profile(circuit: QuantumCircuit, active_qubits_only: bool = False) -> Dict[str, float]:
    ignored_ops = {"barrier", "measure", "delay", "reset"}
    physical_ops = []
    active_qubits: set[int] = set()
    for item in circuit.data:
        op, qargs, _ = _instruction_parts(item)
        if getattr(op, "name", "") not in ignored_ops:
            physical_ops.append(item)
            for q in qargs:
                active_qubits.add(int(circuit.find_bit(q).index))

    twoq_gates = float(count_two_qubit_gates(circuit))
    swap_count = float(count_swaps(circuit))
    cx_count = float(count_named_gate(circuit, "cx"))
    num_qubits = len(active_qubits) if active_qubits_only and active_qubits else circuit.num_qubits
    try:
        depth = float(circuit.depth())
    except Exception:
        depth = float("nan")

    return {
        "num_qubits": float(num_qubits),
        "gate_count_all": float(len(physical_ops)),
        "oneq_count_all": float(max(0.0, float(len(physical_ops)) - twoq_gates)),
        "twoq_count_all": float(twoq_gates),
        "cx_count_all": float(cx_count),
        "swap_count": float(swap_count),
        "depth": depth,
    }


def _additional_cnot_from_profiles(original_profile: Dict[str, float], routed_profile: Dict[str, float]) -> tuple[float, float, float]:
    inserted_swap_count = max(0.0, float(routed_profile["swap_count"] - original_profile["swap_count"]))
    additional_cx_total = max(0.0, float(routed_profile["cx_count_all"] - original_profile["cx_count_all"]))
    swap_decomposition_overhead = 3.0 * inserted_swap_count

    # If the backend preserves SWAP, routed CX does not increase, so use 3*SWAP.
    # If the backend decomposes swaps/bridges to CX, routed CX - original CX captures that overhead.
    additional_cnot_count = max(additional_cx_total, swap_decomposition_overhead)
    return inserted_swap_count, additional_cx_total, additional_cnot_count


def _build_metrics(raw_circuit: QuantumCircuit, prepared: QuantumCircuit, routed: QuantumCircuit, elapsed: float, evaluating_router: str) -> Dict[str, float | str]:
    source_profile = circuit_gate_profile(raw_circuit, active_qubits_only=True)
    original_profile = circuit_gate_profile(prepared, active_qubits_only=True)
    routed_profile = circuit_gate_profile(routed)

    logical_cnot_count = float(source_profile["cx_count_all"])
    active_logical_qubits = float(original_profile["num_qubits"])

    inserted_swap_count, additional_cx_total_nonnegative, additional_cnot_count = _additional_cnot_from_profiles(original_profile, routed_profile)
    inserted_bridge_count = 0.0  # Qiskit SABRE and tket RoutingPass do not expose bridge choices through this API.
    physical_cnot_count = logical_cnot_count + additional_cnot_count

    routed_swap_raw = float(routed_profile["swap_count"])
    original_swap_raw = float(original_profile["swap_count"])
    additional_swap_count = inserted_swap_count

    original_2q = float(original_profile["twoq_count_all"])
    routed_2q = float(routed_profile["twoq_count_all"])
    additional_2q_total = float(routed_2q - original_2q)
    additional_cx_total = float(routed_profile["cx_count_all"] - original_profile["cx_count_all"])

    additional_gates_total = float(routed_profile["gate_count_all"] - original_profile["gate_count_all"])
    additional_1q_total = float(routed_profile["oneq_count_all"] - original_profile["oneq_count_all"])
    depth_overhead = float(routed_profile["depth"] - original_profile["depth"])

    routed_cnot_equiv_count = float(routed_2q + 2.0 * routed_swap_raw)
    original_cnot_equiv_count = float(original_2q + 2.0 * original_swap_raw)
    cnot_equiv_overhead = max(0.0, routed_cnot_equiv_count - original_cnot_equiv_count)

    return {
        # Paper-compatible optimization target.
        "routing_score": additional_cnot_count,
        "terminal_objective": additional_cnot_count,
        "additional_cnot_count": additional_cnot_count,
        "paper_additional_cnot_count": additional_cnot_count,
        "physical_cnot_count": physical_cnot_count,
        "logical_cnot_count": logical_cnot_count,
        "active_logical_qubits": active_logical_qubits,
        "inserted_swap_count": inserted_swap_count,
        "inserted_bridge_count": inserted_bridge_count,
        "bridge_count_source": "not_exposed_by_backend",

        # Diagnostic swap fields retained for debugging only.
        "swap_count": additional_swap_count,
        "swap_count_source": "raw_swap_gate_delta_diagnostic_not_paper_metric",
        "additional_swap_count": additional_swap_count,

        "routing_time_sec": float(elapsed),
        "runtime": float(elapsed),
        "evaluating_router": evaluating_router,

        "input_num_qubits": float(source_profile["num_qubits"]),
        "input_gate_count_all": float(source_profile["gate_count_all"]),
        "input_1q_count_all": float(source_profile["oneq_count_all"]),
        "input_2q_count_all": float(source_profile["twoq_count_all"]),
        "input_cnot_count_all": float(source_profile["cx_count_all"]),
        "input_swap_raw_count": float(source_profile["swap_count"]),
        "input_depth": float(source_profile["depth"]),

        "original_num_qubits": float(source_profile["num_qubits"]),
        "original_gate_count_all": float(source_profile["gate_count_all"]),
        "original_1q_count_all": float(source_profile["oneq_count_all"]),
        "original_2q_count_all": float(source_profile["twoq_count_all"]),
        "original_cnot_count_all": float(source_profile["cx_count_all"]),
        "original_swap_raw_count": float(source_profile["swap_count"]),
        "original_cnot_equiv_count": float(source_profile["twoq_count_all"] + 2.0 * source_profile["swap_count"]),
        "original_depth": float(source_profile["depth"]),
        "routing_input_num_qubits": float(original_profile["num_qubits"]),
        "routing_input_gate_count_all": float(original_profile["gate_count_all"]),
        "routing_input_cnot_count_all": float(original_profile["cx_count_all"]),

        "routed_gate_count_all": float(routed_profile["gate_count_all"]),
        "routed_1q_count_all": float(routed_profile["oneq_count_all"]),
        "routed_2q_count_all": routed_2q,
        "routed_cnot_raw_count": float(routed_profile["cx_count_all"]),
        "routed_swap_raw_count": routed_swap_raw,
        "routed_cnot_equiv_count": routed_cnot_equiv_count,
        "routed_depth": float(routed_profile["depth"]),

        "additional_gates_total": additional_gates_total,
        "additional_1q_total": additional_1q_total,
        "additional_2q_total": additional_2q_total,
        "additional_cx_total": additional_cx_total,
        "additional_cx_total_nonnegative": additional_cx_total_nonnegative,
        "cnot_equiv_overhead": cnot_equiv_overhead,
        "depth_overhead": depth_overhead,
        "metric_protocol": "cnot_active_cnot_only_additional_cnot_v1",
    }


def routing_score_from_metrics(metrics: Dict[str, float | None], alpha_swap: float = 1.0, beta_depth: float = 0.0, gamma_twoq: float = 0.0, eta_added_twoq: float = 0.0, theta_depth_overhead: float = 0.0) -> float:
    del alpha_swap, beta_depth, gamma_twoq, eta_added_twoq, theta_depth_overhead
    return float(metrics.get("additional_cnot_count", metrics.get("routing_score", 0.0)) or 0.0)


def transpile_with_layout(circuit: QuantumCircuit, layout: List[int], hardware: HardwareTopology, env_cfg: EnvConfig | None = None) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    prepared = prepare_basis_circuit(circuit, env_cfg)
    start = time.perf_counter()
    routed = transpile(
        prepared,
        coupling_map=hardware.coupling_map,
        basis_gates=_basis_with_swap(env_cfg),
        initial_layout=layout,
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=0,
        seed_transpiler=env_cfg.sabre_seed,
    )
    elapsed = time.perf_counter() - start
    return _build_metrics(circuit, prepared, routed, elapsed, evaluating_router="qiskit_sabre")


def evaluate_initial_mapping_with_router(circuit: QuantumCircuit, layout: Sequence[int], hardware: HardwareTopology, env_cfg: EnvConfig | None = None) -> Dict[str, float | str]:
    return evaluate_layout_metrics(circuit, layout, hardware, env_cfg or EnvConfig())


def evaluate_full_sabre_metrics(
    circuit: QuantumCircuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
    seeds: Sequence[int] | None = None,
    optimization_level: int = 0,
) -> Dict[str, float | str]:
    """Evaluate Qiskit SABRE when layout and routing are optimized together.

    This is a routing baseline rather than an initial-layout-only score.  It is
    useful for paper-table comparisons because it keeps the original CNOT-only
    gate sequence intact at optimization_level=0 while allowing SABRE to choose
    a layout and route in one pass.
    """
    env_cfg = env_cfg or EnvConfig()
    prepared = prepare_basis_circuit(circuit, env_cfg)
    seed_list = list(seeds) if seeds is not None else [int(getattr(env_cfg, "sabre_seed", 7))]
    if not seed_list:
        seed_list = [int(getattr(env_cfg, "sabre_seed", 7))]

    best_metrics: Dict[str, float | str] | None = None
    best_score: float | None = None
    total_elapsed = 0.0
    best_seed = seed_list[0]

    for seed in seed_list:
        start = time.perf_counter()
        routed = transpile(
            prepared,
            coupling_map=hardware.coupling_map,
            basis_gates=_basis_with_swap(env_cfg),
            layout_method="sabre",
            routing_method="sabre",
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed),
        )
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        metrics = _build_metrics(circuit, prepared, routed, elapsed, evaluating_router="qiskit_full_sabre")
        score = float(objective_from_metrics(metrics, RewardConfig(), env_cfg))
        if best_score is None or score < best_score:
            best_score = score
            best_metrics = dict(metrics)
            best_seed = int(seed)

    assert best_metrics is not None
    best_metrics["routing_time_sec"] = float(total_elapsed)
    best_metrics["runtime"] = float(total_elapsed)
    best_metrics["evaluating_router"] = "qiskit_full_sabre_seed_sweep"
    best_metrics["sabre_best_seed"] = float(best_seed)
    best_metrics["sabre_num_seeds"] = float(len(seed_list))
    best_metrics["sabre_optimization_level"] = float(optimization_level)
    return best_metrics


def evaluate_layout_metrics(circuit: QuantumCircuit, layout: Sequence[int], hardware: HardwareTopology, env_cfg: EnvConfig | None = None) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    if getattr(env_cfg, "router_backend", "qiskit") == "tket":
        if not HAS_TKET:
            raise ImportError("pytket is required for router_backend='tket'.")
        prepared = prepare_basis_circuit(circuit, env_cfg)
        tk_circ = qiskit_to_tk(prepared)
        edges = [(TKNode("q", int(u)), TKNode("q", int(v))) for u, v in hardware.coupling_map.get_edges()]
        tk_architecture = Architecture(edges)
        placement_map = {}
        for logical_idx, phys_idx in enumerate(layout):
            if logical_idx < len(tk_circ.qubits):
                placement_map[tk_circ.qubits[logical_idx]] = TKNode("q", int(phys_idx))
        from pytket.placement import Placement
        Placement(tk_architecture).place_with_map(tk_circ, placement_map)
        start = time.perf_counter()
        RoutingPass(tk_architecture).apply(tk_circ)
        elapsed = time.perf_counter() - start
        return _build_metrics(circuit, prepared, tk_to_qiskit(tk_circ), elapsed, evaluating_router="tket")
    return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


def objective_from_metrics(metrics: Dict[str, float | str | None], reward_cfg: RewardConfig, env_cfg: EnvConfig) -> float:
    del reward_cfg, env_cfg
    return float(metrics.get("additional_cnot_count", metrics.get("routing_score", 0.0)) or 0.0)

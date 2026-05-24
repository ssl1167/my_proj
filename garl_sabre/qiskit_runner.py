from __future__ import annotations

import time
from typing import Dict, List, Sequence

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from .config import EnvConfig, RewardConfig
from .topology import HardwareTopology

# --- 1. 在文件顶部引入相关的库 ---
try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.architecture import Architecture
    from pytket.passes import RoutingPass
    from pytket.circuit import Node as TKNode
    HAS_TKET = True
except ImportError:
    HAS_TKET = False

_PREPARED_FLAG = "_garl_basis_prepared"
_PREPARED_BASIS = "_garl_basis_gates"
_PREPARED_OPT = "_garl_basis_opt_level"


def _basis_with_swap(env_cfg: EnvConfig) -> List[str]:
    """Return basis gates with SWAP preserved for raw swap counting."""
    basis = list(getattr(env_cfg, "basis_gates", []) or [])
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
        # 修复点 1: 完美对齐当前配置的优化等级，激活毫秒级缓存机制
        and int(metadata.get(_PREPARED_OPT, -1)) == env_cfg.optimization_level
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


def _build_metrics(raw_circuit: QuantumCircuit, prepared: QuantumCircuit, routed: QuantumCircuit, elapsed: float, evaluating_router: str) -> Dict[str, float | str]:
    raw_profile = circuit_gate_profile(raw_circuit)     # 最原始的 QASM 电路 (用于写进论文)
    prepared_profile = circuit_gate_profile(prepared)   # 预编译电路 (用于计算真实的路由物理开销)
    routed_profile = circuit_gate_profile(routed)

    # === 物理开销计算 (必须基于 prepared_profile) ===
    prepared_cnot_equiv = prepared_profile["twoq_count_all"] + 2.0 * prepared_profile["swap_count"]
    routed_cnot_equiv = routed_profile["twoq_count_all"] + 2.0 * routed_profile["swap_count"]

    added_cnot_equiv = max(0.0, float(routed_cnot_equiv - prepared_cnot_equiv))
    additional_swap_count = added_cnot_equiv / 3.0

    additional_gates_total = float(routed_profile["gate_count_all"] - prepared_profile["gate_count_all"])
    additional_1q_total = float(routed_profile["oneq_count_all"] - prepared_profile["oneq_count_all"])
    additional_2q_total = float(routed_profile["twoq_count_all"] - prepared_profile["twoq_count_all"])
    depth_overhead = float(routed_profile["depth"] - prepared_profile["depth"])

    return {
        "swap_count": additional_swap_count,
        "swap_count_source": "cnot_equiv_derived",
        "routing_score": additional_swap_count,
        "terminal_objective": additional_swap_count,
        "routing_time_sec": float(elapsed),
        "runtime": float(elapsed),
        "evaluating_router": evaluating_router,

        # === 核心修正：打印给论文看的“原始统计”，使用 raw_profile ===
        "original_num_qubits": float(raw_profile["num_qubits"]),
        "original_gate_count_all": float(raw_profile["gate_count_all"]),
        "original_1q_count_all": float(raw_profile["oneq_count_all"]),
        # 这就是完美对齐论文 #gates 列的终极指标！
        "original_2q_count_all": float(raw_profile["twoq_count_all"]),
        "original_cnot_count_all": float(raw_profile["cx_count_all"]),
        "original_depth": float(raw_profile["depth"]),

        # === 路由后的统计 ===
        "routed_cnot_raw_count": float(routed_profile["cx_count_all"]),
        "routed_cnot_equiv_count": routed_cnot_equiv,
        "routed_swap_count": additional_swap_count,
        "routed_depth": float(routed_profile["depth"]),

        # deltas / added cost
        "additional_gates_total": additional_gates_total,
        "additional_1q_total": additional_1q_total,
        "additional_2q_total": additional_2q_total,
        "additional_swap_count": additional_swap_count,
        "additional_cnot_equiv_from_swap": added_cnot_equiv,
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
        basis_gates=_basis_with_swap(env_cfg),
        initial_layout=layout,
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=0,
        seed_transpiler=env_cfg.sabre_seed,
    )
    elapsed = time.perf_counter() - start
    return _build_metrics(circuit, prepared, routed, elapsed, evaluating_router="qiskit_sabre")


def evaluate_initial_mapping_with_router(
    circuit: QuantumCircuit,
    layout: Sequence[int],
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


# --- 2. 重写 evaluate_layout_metrics 函数 ---
def evaluate_layout_metrics(
    circuit: QuantumCircuit,
    layout: Sequence[int],
    hardware: HardwareTopology,
    env_cfg: EnvConfig | None = None,
) -> Dict[str, float | str]:
    env_cfg = env_cfg or EnvConfig()
    
    # 统一管控双后端，修复 Tabu Search 等处被绕过的漏洞
    if getattr(env_cfg, "router_backend", "qiskit") == "tket":
        if not HAS_TKET:
            raise ImportError("pytket is required for router_backend='tket'.")
        
        prepared = prepare_basis_circuit(circuit, env_cfg)
        tk_circ = qiskit_to_tk(prepared)
        
        # 强制指定寄存器名称为 "q"，完美兼容任何版本的 qiskit_to_tk
        edges = [(TKNode("q", int(u)), TKNode("q", int(v))) for u, v in hardware.coupling_map.get_edges()]
        tk_architecture = Architecture(edges)
        
        placement_map = {}
        for logical_idx, phys_idx in enumerate(layout):
            if logical_idx < len(tk_circ.qubits):
                # 两边强制完全同构：TKNode("q", xxx)
                placement_map[tk_circ.qubits[logical_idx]] = TKNode("q", int(phys_idx))
                
        from pytket.placement import Placement
        Placement(tk_architecture).place_with_map(tk_circ, placement_map)
        
        start = time.perf_counter()
        routing_pass = RoutingPass(tk_architecture)
        routing_pass.apply(tk_circ)
        elapsed = time.perf_counter() - start
        
        routed_circ_qiskit = tk_to_qiskit(tk_circ)
        return _build_metrics(circuit, prepared, routed_circ_qiskit, elapsed, evaluating_router="tket")
        
    else:
        # 默认回退至 Qiskit Sabre
        return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


def objective_from_metrics(metrics: Dict[str, float | str | None], reward_cfg: RewardConfig, env_cfg: EnvConfig) -> float:
    del reward_cfg, env_cfg
    return float(metrics.get("swap_count", 0.0) or 0.0)

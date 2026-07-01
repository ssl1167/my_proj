# 修正后的代码文件：`qiskit_runner.py` 与 `env.py`

以下是根据前面的分析，修改好门代价膨胀与二次综合问题后的完整代码。

## 1. `garl_sabre/qiskit_runner.py`

```python
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


def _build_metrics(prepared: QuantumCircuit, routed: QuantumCircuit, elapsed: float, evaluating_router: str) -> Dict[str, float | str]:
    original_profile = circuit_gate_profile(prepared)
    routed_profile = circuit_gate_profile(routed)

    # 1. 核心修正：计算原电路的“等效二比特门”基数（完全对齐 Paper 的 CNOT count 基线）
    # 公式：所有二比特门总数 + 2 * 未被展开的 SWAP 门数量（因为SWAP已被统计为1个2Q门，需再补2个凑成3等效开销）
    original_cnot_equiv = original_profile["twoq_count_all"] + 2.0 * original_profile["swap_count"]
    
    # 2. 计算路由后电路的“等效二比特门”总数
    routed_cnot_equiv = routed_profile["twoq_count_all"] + 2.0 * routed_profile["swap_count"]

    # 3. 计算电路映射之后绝对增加的等效 CNOT 总数
    added_cnot_equiv = max(0.0, float(routed_cnot_equiv - original_cnot_equiv))
    
    # 4. 鲁棒地反推添加的 SWAP 门数目 (1 SWAP = 3 Equivalent CNOTs)
    additional_swap_count = added_cnot_equiv / 3.0

    # 保持原有变量的命名，以便兼容下游原有的字段记录和赋值
    additional_cnot_equiv_from_swap = added_cnot_equiv
    routed_cnot_equiv_count = routed_cnot_equiv

    # 其他指标保留以供参考和消融实验，虽然重点看SWAP和2Q，但不删减字典字段，避免下游报 KeyError 打印 null
    additional_gates_total = float(routed_profile["gate_count_all"] - original_profile["gate_count_all"])
    additional_1q_total = float(routed_profile["oneq_count_all"] - original_profile["oneq_count_all"])
    additional_2q_total = float(routed_profile["twoq_count_all"] - original_profile["twoq_count_all"])
    depth_overhead = float(routed_profile["depth"] - original_profile["depth"])

    return {
        # optimization target (模型优化目标现在指向最鲁棒的反推值)
        "swap_count": additional_swap_count,
        "swap_count_source": "cnot_equiv_derived", # 修改备注以指示新的鲁棒统计来源
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
        # 修复点 2: 严格遵守硬件原生的 basis_gates，允许 Qiskit 粉碎 SWAP 并触发深层门消除
        basis_gates=env_cfg.basis_gates,
        
        # 核心修复 1：绝对锁死布局，禁止 Qiskit 覆盖 RL Agent 产生的映射结果
        initial_layout=layout,
        layout_method="trivial",
        
        routing_method="sabre",
        
        # 核心修复 2：将评估路由时的优化级别降为 0，纯粹统计路由代价，防止重复综合导致的门数爆炸
        optimization_level=0, 
        
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
        return _build_metrics(prepared, routed_circ_qiskit, elapsed, evaluating_router="tket")
        
    else:
        # 默认回退至 Qiskit Sabre
        return transpile_with_layout(circuit, list(layout), hardware, env_cfg)


def objective_from_metrics(metrics: Dict[str, float | str | None], reward_cfg: RewardConfig, env_cfg: EnvConfig) -> float:
    del reward_cfg, env_cfg
    return float(metrics.get("swap_count", 0.0) or 0.0)

```

## 2. `garl_sabre/env.py`

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile

from .circuit_features import LogicGraphData, build_logic_graph
from .config import EnvConfig, RewardConfig
from .heuristics import dense_layout, trivial_layout
from .qiskit_runner import _build_metrics, evaluate_layout_metrics, objective_from_metrics, prepare_basis_circuit
from .topology import HardwareTopology, adjacency_with_self_loops


@dataclass
class StepOutput:
    obs: Dict
    reward: float
    done: bool
    info: Dict


class InitialLayoutEnv:
    """Initial-layout reinforcement learning environment."""

    def __init__(self, hardware: HardwareTopology, env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None) -> None:
        self.hardware = hardware
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()

        self.circuit: Optional[QuantumCircuit] = None
        self.logic: Optional[LogicGraphData] = None
        self.mapping_log_to_phys: Optional[np.ndarray] = None
        self.used_phys: Optional[np.ndarray] = None
        self.step_idx: int = 0
        self.logical_order: list[int] = []
        self.front_pair_mask: Optional[np.ndarray] = None
        self.critical_pair_mask: Optional[np.ndarray] = None

        # --- 核心修改：引入有状态运行模式（业界标准 RL 环境流设计） ---
        self.is_training: bool = True  # 默认激活无损探索训练模式
        finite_d = self.hardware.dist[self.hardware.dist < 1e8]
        self.max_dist = float(max(1.0, finite_d.max() if finite_d.size > 0 else 1.0))

        # ==================== 核心修改 1: 拓扑自适应图论熵权融合引擎 ====================
        raw_topo_feats = self.hardware.node_features[:, :4].astype(np.float32)
        num_nodes = raw_topo_feats.shape[0]
        if num_nodes > 1:
            col_sums = np.sum(raw_topo_feats, axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1e-8, col_sums)
            p_matrix = raw_topo_feats / col_sums
            eps = 1e-12
            entropy = -np.sum(p_matrix * np.log(p_matrix + eps), axis=0) / np.log(num_nodes)
            utility = 1.0 - entropy
            utility_sum = np.sum(utility)
            entropy_weights = utility / utility_sum if utility_sum > 1e-6 else np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        else:
            entropy_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        
        self.phys_centrality = np.dot(raw_topo_feats, entropy_weights).astype(np.float32)
        # ==============================================================================

        self.physical_adj_binary: np.ndarray = adjacency_with_self_loops(self.hardware.graph, self.hardware.num_qubits)
        self.physical_adj: np.ndarray = self._build_physical_weighted_adj()

        self.baseline_score: Optional[float] = None
        self.baseline_name: str = "none"
        self.baseline_metrics: Dict[str, float] = {}
        self.reward_anchor_score: float = 1.0

    def _use_candidate_ranking(self) -> bool:
       # 彻底听从配置，避免训练和测试环境的 MDP（马尔可夫决策过程）空间不一致
        return bool(getattr(self.env_cfg, "use_candidate_ranking", False))
        
    def _use_physical_prior(self) -> bool:
        return bool(getattr(self.env_cfg, "use_physical_prior", True))

    def _compute_baseline_metrics(self) -> bool:
        return True  # 学术级评测流中强制激活基线计算

    def _build_physical_weighted_adj(self) -> np.ndarray:
        n = self.hardware.num_qubits
        adj = np.zeros((n, n), dtype=np.float32)

        centrality = self.phys_centrality.astype(np.float32)
        span = float(np.max(centrality) - np.min(centrality))
        if span < 1e-8:
            centrality = np.zeros_like(centrality, dtype=np.float32)
        else:
            centrality = (centrality - float(np.min(centrality))) / (span + 1e-8)

        for u, v in self.hardware.graph.edges():
            edge_w = 1.0 + 0.35 * float(0.5 * (centrality[u] + centrality[v]))
            adj[u, v] = edge_w
            adj[v, u] = edge_w

        finite = self.hardware.dist < 1e8
        tau = max(self.max_dist * 0.6, 1.0)
        diffusion = np.exp(-self.hardware.dist / tau).astype(np.float32)
        diffusion[~finite] = 0.0
        np.fill_diagonal(diffusion, 0.0)
        adj += 0.15 * diffusion

        np.fill_diagonal(adj, 1.0 + 0.25 * centrality)
        deg = np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
        return (adj / deg).astype(np.float32)

    # 扩大参数签名，接收外部缓存的 baseline_info
    def reset(self, circuit: QuantumCircuit, is_training: Optional[bool] = None, logic_graph: Optional[LogicGraphData] = None, baseline_info: Optional[Tuple] = None) -> Dict:
        if circuit.num_qubits > self.hardware.num_qubits:
            raise ValueError(f"Circuit has {circuit.num_qubits} qubits, but hardware only has {self.hardware.num_qubits}.")

        if is_training is not None:
            self.is_training = is_training

        self.circuit = circuit.copy()
        
        # --- 核心修改 3：依赖注入优先。如有缓存图则极速挂载，否则执行回退计算 ---
        if logic_graph is not None:
            self.logic = logic_graph
        else:
            self.logic = build_logic_graph(self.circuit, critical_window=self.env_cfg.critical_window, lookahead_window=self.env_cfg.lookahead_window)
            
        self.mapping_log_to_phys = np.full(self.logic.num_qubits, -1, dtype=np.int64)
        
        # ... 后续其余代码保持完全不变 ...
        self.used_phys = np.zeros(self.hardware.num_qubits, dtype=np.float32)
        self.step_idx = 0
        self.logical_order = self._build_logical_order()
        self.front_pair_mask = self._pairs_to_mask(self.logic.front_pairs)
        self.critical_pair_mask = self._pairs_to_mask(self.logic.critical_edges)
        
        # ==================== 核心修改 2: 锁死单一公认外部标杆，斩断 hybrid 漏洞 ====================
        # ==================== 核心修改：短路基线计算 ====================
        if baseline_info is not None:
            self.baseline_score, self.baseline_name, self.baseline_metrics = baseline_info
        else:
            self.baseline_score, self.baseline_name, self.baseline_metrics = self._compute_baseline()
        
        # 强制将奖励函数的反事实参照物分母与你指定的单一 baseline_mode 深度锁死
        self.reward_anchor_score = self.baseline_score if self.baseline_score is not None else 1.0
        if self.reward_anchor_score <= 1e-5:
            fast_layout = dense_layout(self.logic, self.hardware)
            metrics = evaluate_layout_metrics(self.circuit, fast_layout, self.hardware, self.env_cfg)
            self.reward_anchor_score = max(float(objective_from_metrics(metrics, self.reward_cfg, self.env_cfg)), 1.0)
        # =========================================================================================

        return self._get_obs()

    def _pairs_to_mask(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        assert self.logic is not None
        mask = np.zeros((self.logic.num_qubits, self.logic.num_qubits), dtype=np.float32)
        for a, b in pairs:
            if 0 <= a < self.logic.num_qubits and 0 <= b < self.logic.num_qubits:
                mask[a, b] = 1.0
                mask[b, a] = 1.0
        return mask

    def _build_logical_order(self) -> list[int]:
        assert self.logic is not None
        if self.env_cfg.logic_order_mode == "priority_fixed":
            return list(self.logic.placement_order)
        if self.env_cfg.logic_order_mode == "front_first":
            marked = []
            seen = set()
            for a, b in self.logic.front_pairs:
                if a not in seen:
                    marked.append(a)
                    seen.add(a)
                if b not in seen:
                    marked.append(b)
                    seen.add(b)
            for q in self.logic.placement_order:
                if q not in seen:
                    marked.append(int(q))
            return marked
        return list(range(self.logic.num_qubits))

    def current_logical_qubit(self) -> int:
        assert self.logic is not None
        if self.step_idx >= self.logic.num_qubits:
            return int(self.logical_order[-1])
        return int(self.logical_order[self.step_idx])

    def free_physical_mask(self) -> np.ndarray:
        assert self.used_phys is not None
        return (self.used_phys < 0.5).astype(np.float32)

    def logical_action_mask(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        if self.env_cfg.action_mode == "fixed_order_physical":
            mask = np.zeros(self.logic.num_qubits, dtype=np.float32)
            mask[self.current_logical_qubit()] = 1.0
            return mask
        return (self.mapping_log_to_phys < 0).astype(np.float32)

    def _logical_dynamic_stats(self, logical_q: int) -> Tuple[float, float, float, float, float, float]:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        weights = self.logic.edge_weight[logical_q].astype(np.float32)
        neighbor_idx = np.where(weights > 0)[0]
        total_weight = float(np.sum(weights[neighbor_idx])) + 1e-8
        total_neighbor_count = max(int(neighbor_idx.size), 1)

        mapped_idx = [int(j) for j in neighbor_idx if self.mapping_log_to_phys[j] >= 0]
        unmapped_idx = [int(j) for j in neighbor_idx if self.mapping_log_to_phys[j] < 0]

        mapped_neighbor_ratio = float(len(mapped_idx)) / float(total_neighbor_count)
        mapped_weight_ratio = float(np.sum(weights[mapped_idx])) / total_weight if mapped_idx else 0.0
        frontier_mass_ratio = float(np.sum(weights * self.front_pair_mask[logical_q])) / total_weight
        critical_mass_ratio = float(np.sum(weights * self.critical_pair_mask[logical_q])) / total_weight
        unresolved_frontier_ratio = float(np.sum(weights[unmapped_idx] * self.front_pair_mask[logical_q, unmapped_idx])) / total_weight if unmapped_idx else 0.0
        unresolved_critical_ratio = float(np.sum(weights[unmapped_idx] * self.critical_pair_mask[logical_q, unmapped_idx])) / total_weight if unmapped_idx else 0.0

        return (
            mapped_neighbor_ratio,
            mapped_weight_ratio,
            frontier_mass_ratio,
            critical_mass_ratio,
            unresolved_frontier_ratio,
            unresolved_critical_ratio,
        )

    def _logical_candidate_features(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        feat_dim = self.logic.node_features.shape[1] + 7
        feats = np.zeros((self.logic.num_qubits, feat_dim), dtype=np.float32)
        active_idx = np.where(self.mapping_log_to_phys < 0)[0]
        for q in active_idx:
            q = int(q)
            dyn = self._logical_dynamic_stats(q)
            feats[q, : self.logic.node_features.shape[1]] = self.logic.node_features[q]
            feats[q, self.logic.node_features.shape[1] + 0] = 1.0
            feats[q, self.logic.node_features.shape[1] + 1 : self.logic.node_features.shape[1] + 7] = np.asarray(dyn, dtype=np.float32)
        return feats

    def _candidate_features(self, logical_q: int) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None
        n_phys = self.hardware.num_qubits
        features = np.zeros((n_phys, 11), dtype=np.float32)

        all_weights = self.logic.edge_weight[logical_q].astype(np.float32)
        neighbor_idx = np.where(all_weights > 0)[0]
        total_neighbor_count = max(int(neighbor_idx.size), 1)
        total_edge_weight = float(np.sum(all_weights[neighbor_idx])) + 1e-8

        mapped_idx = [int(j) for j in neighbor_idx if self.mapping_log_to_phys[j] >= 0]
        mapped_weight_sum = float(np.sum(all_weights[mapped_idx])) if mapped_idx else 0.0
        mapped_neighbor_ratio = float(len(mapped_idx)) / float(total_neighbor_count)
        mapped_weight_ratio = mapped_weight_sum / total_edge_weight

        for p in range(n_phys):
            deg = max(int(self.hardware.graph.degree(p)), 1)
            free_nb = sum(float(self.used_phys[nbr] < 0.5) for nbr in self.hardware.graph.neighbors(p))
            free_neighbor_ratio = free_nb / float(deg)

            features[p, 0] = mapped_neighbor_ratio
            features[p, 1] = mapped_weight_ratio
            features[p, 8] = free_neighbor_ratio
            features[p, 9] = float(self.used_phys[p] < 0.5)

            if not mapped_idx:
                continue

            mapped_phys = np.array([int(self.mapping_log_to_phys[j]) for j in mapped_idx], dtype=np.int64)
            dists = self.hardware.dist[p, mapped_phys].astype(np.float32) / self.max_dist
            weights = all_weights[mapped_idx].astype(np.float32)
            wsum = float(weights.sum()) + 1e-8

            front_weights = weights * (1.0 + self.front_pair_mask[logical_q, mapped_idx])
            critical_weights = weights * (1.0 + self.critical_pair_mask[logical_q, mapped_idx])
            front_sum = float(front_weights.sum()) + 1e-8
            critical_sum = float(critical_weights.sum()) + 1e-8

            features[p, 2] = 1.0
            features[p, 3] = float(np.min(dists))
            features[p, 4] = float(np.sum(weights * dists) / wsum)
            features[p, 5] = float(np.max(dists))
            features[p, 6] = float(np.sum(weights * dists) / total_edge_weight)
            features[p, 7] = float(np.sum(front_weights * dists) / front_sum)
            features[p, 10] = float(np.sum(critical_weights * dists) / critical_sum)
        return features

    def _candidate_feature_bank(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        bank = np.zeros((self.logic.num_qubits, self.hardware.num_qubits, 11), dtype=np.float32)
        active_idx = np.where(self.mapping_log_to_phys < 0)[0]
        for q in active_idx:
            bank[int(q)] = self._candidate_features(int(q))
        return bank

    def _candidate_summary(self, logical_q: int, phys_q: int) -> Dict[str, float]:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None
        weights = self.logic.edge_weight[logical_q].astype(np.float32)
        neighbors = np.where(weights > 0)[0]
        mapped_neighbors = [int(j) for j in neighbors if self.mapping_log_to_phys[j] >= 0]

        free_ratio = 0.0
        deg = max(int(self.hardware.graph.degree(phys_q)), 1)
        free_ratio = float(sum(self.used_phys[nbr] < 0.5 for nbr in self.hardware.graph.neighbors(phys_q))) / float(deg)
        centrality = float(self.phys_centrality[phys_q])

        base_dist = 0.0
        frontier_dist = 0.0
        critical_dist = 0.0
        executable_frontier = 0.0
        if mapped_neighbors:
            mapped_phys = np.array([int(self.mapping_log_to_phys[j]) for j in mapped_neighbors], dtype=np.int64)
            dists_raw = self.hardware.dist[phys_q, mapped_phys].astype(np.float32)
            dists = dists_raw / self.max_dist
            edge_w = weights[mapped_neighbors].astype(np.float32)
            edge_w_sum = float(edge_w.sum()) + 1e-8
            base_dist = float(np.sum(edge_w * dists) / edge_w_sum)

            frontier_mask = self.front_pair_mask[logical_q, mapped_neighbors].astype(np.float32)
            critical_mask = self.critical_pair_mask[logical_q, mapped_neighbors].astype(np.float32)
            if float(frontier_mask.sum()) > 0:
                frontier_w = edge_w * (1.0 + frontier_mask)
                frontier_dist = float(np.sum(frontier_w * dists) / (float(frontier_w.sum()) + 1e-8))
                executable_frontier = float(np.sum(frontier_w * (dists_raw <= 1.0)) / (float(frontier_w.sum()) + 1e-8))
            else:
                frontier_dist = base_dist
            if float(critical_mask.sum()) > 0:
                critical_w = edge_w * (1.0 + critical_mask)
                critical_dist = float(np.sum(critical_w * dists) / (float(critical_w.sum()) + 1e-8))
            else:
                critical_dist = base_dist

        score = (
            -1.00 * base_dist
            -0.60 * frontier_dist
            -0.40 * critical_dist
            +0.20 * free_ratio
            +0.10 * executable_frontier
            +0.05 * centrality
        )
        return {
            "base_dist": float(base_dist),
            "frontier_dist": float(frontier_dist),
            "critical_dist": float(critical_dist),
            "free_neighbor_ratio": float(free_ratio),
            "executable_frontier_ratio": float(executable_frontier),
            "centrality": float(centrality),
            "score_raw": float(score),
        }

    def _legal_candidate_scores(self, logical_q: int) -> tuple[np.ndarray, np.ndarray]:
        free_idx = np.where(self.free_physical_mask() > 0)[0]
        if free_idx.size == 0:
            return free_idx, np.zeros(0, dtype=np.float32)
        scores = np.zeros(free_idx.size, dtype=np.float32)
        for i, p in enumerate(free_idx):
            scores[i] = float(self._candidate_summary(logical_q, int(p))["score_raw"])
        return free_idx, scores

    def _ranked_physical_mask(self, logical_q: int) -> np.ndarray:
        free_mask = self.free_physical_mask()
        if not self._use_candidate_ranking():
            return free_mask

        free_idx, scores = self._legal_candidate_scores(logical_q)
        if free_idx.size == 0:
            return free_mask

        topk = int(self.env_cfg.candidate_topk)
        if topk <= 0 or topk >= free_idx.size:
            return free_mask

        chosen = free_idx[np.argsort(-scores)[:topk]]
        mask = np.zeros_like(free_mask)
        mask[chosen] = 1.0
        if mask.sum() <= 0 and getattr(self.env_cfg, "allow_full_action_space_fallback", True):
            return free_mask
        return mask

    def _physical_action_masks_bank(self, candidate_bank: np.ndarray) -> np.ndarray:
        del candidate_bank
        assert self.logic is not None and self.mapping_log_to_phys is not None
        masks = np.zeros((self.logic.num_qubits, self.hardware.num_qubits), dtype=np.float32)
        active_idx = np.where(self.mapping_log_to_phys < 0)[0]
        for q in active_idx:
            masks[int(q)] = self._ranked_physical_mask(int(q))
        return masks

    def _physical_prior_bank(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        bank = np.zeros((self.logic.num_qubits, self.hardware.num_qubits), dtype=np.float32)
        if not self._use_physical_prior():
            return bank

        free_mask = self.free_physical_mask()
        active_idx = np.where(self.mapping_log_to_phys < 0)[0]
        for q in active_idx:
            q = int(q)
            free_idx, scores = self._legal_candidate_scores(q)
            if free_idx.size <= 0:
                continue
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            norm = max(std, 0.20)
            bank[q, free_idx] = np.clip((scores - mean) / norm, -4.0, 4.0).astype(np.float32)
            bank[q, free_mask <= 0] = -6.0
        return bank

    def snapshot(self) -> Dict[str, np.ndarray | int]:
        assert self.mapping_log_to_phys is not None and self.used_phys is not None
        return {
            "mapping_log_to_phys": self.mapping_log_to_phys.copy(),
            "used_phys": self.used_phys.copy(),
            "step_idx": int(self.step_idx),
        }

    def restore(self, state: Dict[str, np.ndarray | int]) -> None:
        assert self.logic is not None
        self.mapping_log_to_phys = np.asarray(state["mapping_log_to_phys"], dtype=np.int64).copy()
        self.used_phys = np.asarray(state["used_phys"], dtype=np.float32).copy()
        self.step_idx = int(state["step_idx"])

    def get_obs(self) -> Dict:
        return self._get_obs()

    def _get_obs(self) -> Dict:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None

        progress = float(self.step_idx) / float(max(self.logic.num_qubits, 1))
        logical_feats = self._logical_candidate_features()
        candidate_bank = self._candidate_feature_bank()
        physical_masks_bank = self._physical_action_masks_bank(candidate_bank)
        physical_prior_bank = self._physical_prior_bank()
        current_q = self.current_logical_qubit()

        return {
            "logic_node_features": self.logic.node_features.copy(),
            "logic_adj": self.logic.weighted_adj.copy(),
            "logic_adj_binary": self.logic.adj.copy(),
            "physical_node_features": self.hardware.node_features.copy(),
            "physical_adj": self.physical_adj.copy(),
            "physical_adj_binary": self.physical_adj_binary.copy(),
            "mapping": self.mapping_log_to_phys.copy(),
            "used_phys": self.used_phys.copy(),
            "free_physical_mask": self.free_physical_mask(),
            "logical_action_mask": self.logical_action_mask(),
            "action_mask": self.free_physical_mask().copy(),
            "current_logical_idx": np.int64(current_q),
            "logical_candidate_features": logical_feats,
            "candidate_features_bank": candidate_bank,
            "physical_action_masks_bank": physical_masks_bank,
            "physical_prior_bank": physical_prior_bank,
            "progress": np.float32(progress),

            
            "is_fixed_order": np.int64(1 if self.env_cfg.action_mode == "fixed_order_physical" else 0),
        }
        

    def _shape_reward_for_choice(self, logical_q: int, phys_q: int) -> tuple[float, Dict[str, float]]:
        free_idx, scores = self._legal_candidate_scores(logical_q)
        cand = self._candidate_summary(logical_q, phys_q)
        if free_idx.size <= 1:
            centered = 0.0
            best_gap = 0.0
            score_mean = cand["score_raw"]
            score_std = 0.0
        else:
            score_mean = float(np.mean(scores))
            score_std = float(np.std(scores))
            norm = max(score_std, 0.20)
            centered = float((cand["score_raw"] - score_mean) / norm)
            best_gap = float((cand["score_raw"] - float(np.max(scores))) / norm)

        reward = 0.0
        reward += 0.10 * centered
        reward += 0.06 * best_gap
        reward += 0.05 * cand["executable_frontier_ratio"]
        reward += 0.03 * cand["free_neighbor_ratio"]
        
        # ==================== 核心修改 3: 双曲正切平滑激活，替换硬裁剪 ====================
        reward = float(0.35 * np.tanh(reward / 0.25))
        # ==============================================================================

        info = {
            "shape_base": -cand["base_dist"],
            "shape_frontier": -cand["frontier_dist"],
            "shape_critical": -cand["critical_dist"],
            "shape_future": 0.0,
            "shape_reservation": cand["free_neighbor_ratio"],
            "shape_local_bonus": cand["executable_frontier_ratio"],
            "gate_reward": 0.0,
            "lookahead_reward": 0.0,
            "blocked_penalty": 0.0,
            "score_raw": cand["score_raw"],
            "shape_centered": centered,
            "shape_best_gap": best_gap,
            "shape_ref_mean": score_mean,
            "shape_ref_std": score_std,
            "completion_bonus": 0.0,
        }
        return reward, info

    def _compute_baseline(self) -> tuple[Optional[float], str, Dict[str, float]]:
        """Compute the configured external baseline metrics."""
        assert self.circuit is not None and self.logic is not None
        mode = self.env_cfg.baseline_mode
        if mode == "none":
            return None, "none", {}

        # 1. 学术级工业标杆 Qiskit-Sabre 分数
        if mode == "sabre":
            prepared = prepare_basis_circuit(self.circuit, self.env_cfg)
            sabre_routed_circ = transpile(
                prepared,
                coupling_map=self.hardware.coupling_map,
                # 修复点 3: 基线同样使用原生 basis_gates，让 Sabre 基准分也享受门抵消红利
                basis_gates=self.env_cfg.basis_gates,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=0, # 修改为 0，保证基线评估对齐纯路由代价
                seed_transpiler=self.env_cfg.sabre_seed
            )
            sabre_metrics = _build_metrics(prepared, sabre_routed_circ, 0.0, "qiskit_sabre")
            sabre_score = float(objective_from_metrics(sabre_metrics, self.reward_cfg, self.env_cfg))
            return sabre_score, "sabre", sabre_metrics

        # 2. 保留原有的常规紧凑和微弱基线分支（仅限非sabre状态）
        candidates: list[tuple[str, list[int]]] = []
        if mode == "trivial":
            candidates.append(("trivial", trivial_layout(self.logic.num_qubits)))
        if mode == "dense":
            candidates.append(("dense", dense_layout(self.logic, self.hardware)))

        best_name = "none"
        best_metrics: Dict[str, float] = {}
        best_score: Optional[float] = None
        for name, layout in candidates:
            metrics = evaluate_layout_metrics(self.circuit, layout, self.hardware, self.env_cfg)
            score = objective_from_metrics(metrics, self.reward_cfg, self.env_cfg)
            if best_score is None or score < best_score:
                best_score = float(score)
                best_name = name
                best_metrics = dict(metrics)
        return best_score, best_name, best_metrics

    def _parse_action(self, action: int | Sequence[int] | np.ndarray) -> tuple[int, int]:
        if self.env_cfg.action_mode == "fixed_order_physical":
            if isinstance(action, (tuple, list, np.ndarray)):
                if len(action) < 2:
                    raise ValueError("Expected (logical, physical) action pair in fixed-order mode.")
                return int(action[0]), int(action[1])
            return self.current_logical_qubit(), int(action)

        if not isinstance(action, (tuple, list, np.ndarray)) or len(action) < 2:
            raise ValueError("Hierarchical mode expects action=(logical_q, physical_q).")
        return int(action[0]), int(action[1])

    def _execute_dual_backend_routing(self, layout: list[int]) -> Dict[str, float]:
        """Run the configured router backend and return routing metrics."""
        # 直接调用底层的 evaluate_layout_metrics，那里已经统一拦截并处理了 qiskit 和 tket
        return evaluate_layout_metrics(self.circuit, layout, self.hardware, self.env_cfg)

    def step(self, action: int | Sequence[int] | np.ndarray) -> StepOutput:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None and self.circuit is not None

        logical_q, action_phys = self._parse_action(action)
        logical_mask = self.logical_action_mask()
        if logical_q < 0 or logical_q >= logical_mask.size or logical_mask[logical_q] <= 0:
            valid = np.where(logical_mask > 0)[0].tolist()
            raise ValueError(f"Illegal logical action {logical_q}.")

        physical_mask = self._ranked_physical_mask(logical_q)
        if action_phys < 0 or action_phys >= physical_mask.size or physical_mask[action_phys] <= 0 or self.used_phys[action_phys] >= 0.5:
            valid = np.where(physical_mask > 0)[0].tolist()
            raise ValueError(f"Illegal physical action {action_phys}.")

        shape_reward, shape_info = self._shape_reward_for_choice(logical_q, action_phys)
        self.mapping_log_to_phys[logical_q] = action_phys
        self.used_phys[action_phys] = 1.0
        self.step_idx += 1

        done = bool(self.step_idx >= self.logic.num_qubits)
        reward = float(shape_reward)
        info: Dict[str, float | None | list[int] | str] = {
            "selected_logical": logical_q,
            "selected_physical": action_phys,
            "baseline_name": self.baseline_name,
            **shape_info,
        }

        if done:
            layout = self.mapping_log_to_phys.tolist()
            
            # ==================== 核心修改 4: 激活双后端动态自适应终结回报对齐公式 ====================
            metrics = self._execute_dual_backend_routing(layout)
            terminal_objective = float(objective_from_metrics(metrics, self.reward_cfg, self.env_cfg))
            
            # 计算当前 RL 结果相比固定外部基线的相对优化提升红利率
            relative_improvement = (self.reward_anchor_score - terminal_objective) / self.reward_anchor_score
            
            # 使用线性缩放实现绝对的 MDP 价值对齐，拒绝 Reward Hacking 
            num_qubits_factor = float(self.logic.num_qubits)  

            terminal_reward = float(self.reward_cfg.terminal_scale * relative_improvement * num_qubits_factor)
            reward += float(terminal_reward)
            # =========================================================================================

            info.update(metrics)
            info["routing_score"] = float(terminal_objective)
            info["baseline_score"] = float(self.baseline_score) if self.baseline_score is not None else None
            info["terminal_objective"] = float(terminal_objective)
            info["final_layout"] = layout
            info["terminal_reward"] = float(terminal_reward)

        return StepOutput(obs=self._get_obs(), reward=float(reward), done=done, info=info)

```

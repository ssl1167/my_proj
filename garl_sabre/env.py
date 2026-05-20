from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile

from .circuit_features import LogicGraphData, build_logic_graph
from .config import EnvConfig, RewardConfig
from .heuristics import dense_layout, trivial_layout
from .qiskit_runner import evaluate_layout_metrics, objective_from_metrics
from .topology import HardwareTopology, adjacency_with_self_loops

# 学术级级联惰性导入：动态探测环境中是否安有 Quantinuum t|ket> 编译生态
try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.architecture import Architecture
    from pytket.passes import RoutingPass
    from pytket.circuit import Node
    HAS_TKET = True
except ImportError:
    HAS_TKET = False


@dataclass
class StepOutput:
    obs: Dict
    reward: float
    done: bool
    info: Dict


class InitialLayoutEnv:
    """高级量子初始映射强化学习环境：融合自适应图熵权、双路由后端评测与可控外部标杆锚定。"""

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
        # --- 学术级修正：拒绝双重偏置，训练期强制放开全空间 ---
        if self.is_training:
            return False  # 训练期间拒绝任何硬掩码拦截，允许模型全域探索未知最优解
        # 仅在非训练模式（评估、测试或工业推断部署）下，才允许遵从用户配置项
        return bool(getattr(self.env_cfg, "use_candidate_ranking", True))

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

    # 修改 reset 方法签名，追加 logic_graph 参数
    def reset(self, circuit: QuantumCircuit, is_training: Optional[bool] = None, logic_graph: Optional[LogicGraphData] = None) -> Dict:
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

            # --- 必须补齐的核心广播锚点 ---
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
        """最高学术水准的基线评测黑盒：支持标准的 Sabre 初始映射。"""
        assert self.circuit is not None and self.logic is not None
        mode = self.env_cfg.baseline_mode
        if mode == "none":
            return None, "none", {}

        # 1. 锁死工业黄金基线 Qiskit-Sabre 分数
        if mode == "sabre":
            # 激活 Qiskit 官方一揽子 Sabre 映射编译，使用环境指定的随机种子
            sabre_routed_circ = transpile(
                self.circuit,
                coupling_map=self.hardware.coupling_map,
                layout_method="sabre",
                routing_method="sabre",
                optimization_level=self.env_cfg.optimization_level,
                seed_transpiler=self.env_cfg.sabre_seed
            )
            sabre_metrics = {
                "cnot_count": float(sabre_routed_circ.count_ops().get("cx", 0)),
                "depth": float(sabre_routed_circ.depth())
            }
            sabre_score = float(sabre_metrics["cnot_count"])
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
        """核心路由模块：无缝解耦支持 Qiskit-Sabre 或 Quantinuum-tket 路由流。"""
        backend = self.env_cfg.router_backend
        
        if backend == "qiskit":
            # 锁死由强化学习环境层演进出的布局映射，禁止后续路由算法擅自更改 Initial Layout
            initial_layout_dict = {self.circuit.qubits[i]: layout[i] for i in range(len(layout))}
            routed_circ = transpile(
                self.circuit,
                coupling_map=self.hardware.coupling_map,
                initial_layout=initial_layout_dict,
                layout_method=None,  # 强行锁死，阻止 Qiskit 乱动初始位置
                routing_method="sabre",
                optimization_level=self.env_cfg.optimization_level,
                seed_transpiler=self.env_cfg.sabre_seed
            )
            return {
                "cnot_count": float(routed_circ.count_ops().get("cx", 0)),
                "depth": float(routed_circ.depth())
            }
            
        elif backend == "tket":
            if not HAS_TKET:
                raise ImportError("未探测到 pytket 生态，请执行 `pip install pytket pytket-qiskit`。")
            
            # 将 Qiskit 线路骨架平滑序列化至 pytket 核心图表征
            tk_circ = qiskit_to_tk(self.circuit)
            edges = [(int(u), int(v)) for u, v in self.hardware.coupling_map.get_edges()]
            tk_architecture = Architecture(edges)
            
            # 将强化学习网络生成的布局锁死到 tket 节点双射映射字典上
            placement_map = {}
            for logical_idx, phys_idx in enumerate(layout):
                if logical_idx < len(tk_circ.qubits):
                    placement_map[tk_circ.qubits[logical_idx]] = Node(phys_idx)
            
            from pytket.placement import Placement
            Placement().place_with_map(tk_circ, placement_map)  # 安全、可控地锁定初始布局
            
            # 调用 tket 核心正统图结构并发交换路由 Pass
            routing_pass = RoutingPass(tk_architecture)
            routing_pass.apply(tk_circ)
            
            # 重新安全回写为 Qiskit 线路格式以保持全局统计口径绝对对齐
            routed_circ_qiskit = tk_to_qiskit(tk_circ)
            return {
                "cnot_count": float(routed_circ_qiskit.count_ops().get("cx", 0)),
                "depth": float(routed_circ_qiskit.depth())
            }
        else:
            raise ValueError(f"不受支持的路由后端: {backend}")

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
            terminal_objective = float(metrics["cnot_count"])
            
            # 计算当前 RL 结果相比固定外部基线的相对优化提升红利率
            relative_improvement = (self.reward_anchor_score - terminal_objective) / self.reward_anchor_score
            
            # 使用对数缩放因子抵消不同规模量子线路的步长总回报累积方差
            num_qubits_factor = np.log(max(float(self.logic.num_qubits), 2.0))
            
            # 最终的具有清晰物理含义的大奖励
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

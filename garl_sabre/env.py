from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .circuit_features import LogicGraphData, build_logic_graph
from .config import EnvConfig, RewardConfig
from .heuristics import dense_layout, sabre_layout, trivial_layout
from .qiskit_runner import evaluate_layout_metrics, objective_from_metrics, prepare_basis_circuit
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
        self.raw_circuit: Optional[QuantumCircuit] = None
        self.logic: Optional[LogicGraphData] = None
        self.mapping_log_to_phys: Optional[np.ndarray] = None
        self.used_phys: Optional[np.ndarray] = None
        self.step_idx: int = 0
        self.logical_order: list[int] = []
        self.front_pair_mask: Optional[np.ndarray] = None
        self.critical_pair_mask: Optional[np.ndarray] = None

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


        
    def _use_physical_prior(self) -> bool:
        return bool(getattr(self.env_cfg, "use_physical_prior", True))

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
        # is_training 参数保留用于向后兼容，但不再实际使用
        # Paper protocol guardrail: the environment itself canonicalizes the input
        # circuit before any qubit-count check, graph construction, baseline
        # computation, or final routing evaluation.  This prevents a raw .real/qasm
        # declared width from leaking back into the RL state space.
        prepared_circuit = prepare_basis_circuit(circuit, self.env_cfg)
        if prepared_circuit.num_qubits > self.hardware.num_qubits:
            raise ValueError(
                f"Canonical circuit has {prepared_circuit.num_qubits} active logical qubits, "
                f"but hardware only has {self.hardware.num_qubits}."
            )

        self.raw_circuit = circuit.copy()
        self.circuit = prepared_circuit.copy()
        
        # --- 核心修改 3：依赖注入优先。如有缓存图则极速挂载，否则执行回退计算 ---
        if logic_graph is not None:
            if int(logic_graph.num_qubits) != int(self.circuit.num_qubits):
                raise ValueError(
                    f"Cached logic graph qubit count ({logic_graph.num_qubits}) does not match "
                    f"canonical circuit qubit count ({self.circuit.num_qubits}). "
                    "Clear dataset caches and rebuild the canonical CNOT-active dataset."
                )
            self.logic = logic_graph
        else:
            self.logic = build_logic_graph(
                self.circuit,
                critical_window=self.env_cfg.critical_window,
                lookahead_window=self.env_cfg.lookahead_window,
                decompose=True,
                feature_norm_mode=getattr(self.env_cfg, "feature_norm_mode", "adaptive"),
                feature_norm_scale=float(getattr(self.env_cfg, "feature_norm_scale", 1.2)),
                feat_norm_neighbors=float(getattr(self.env_cfg, "feat_norm_neighbors", 15.0)),
                feat_norm_twoq=float(getattr(self.env_cfg, "feat_norm_twoq", 100.0)),
                feat_norm_front=float(getattr(self.env_cfg, "feat_norm_front", 10.0)),
                feat_norm_early=float(getattr(self.env_cfg, "feat_norm_early", 20.0)),
                feat_norm_weighted_degree=float(getattr(self.env_cfg, "feat_norm_weighted_degree", 150.0)),
                feat_norm_pagerank=float(getattr(self.env_cfg, "feat_norm_pagerank", 1.0)),
                feat_norm_critical=float(getattr(self.env_cfg, "feat_norm_critical", 30.0)),
            )
            
        self.mapping_log_to_phys = np.full(self.logic.num_qubits, -1, dtype=np.int64)
        
        self.used_phys = np.zeros(self.hardware.num_qubits, dtype=np.float32)
        self.step_idx = 0
        self.logical_order = self._build_logical_order()
        self.front_pair_mask = self._pairs_to_mask(self.logic.front_pairs)
        self.critical_pair_mask = self._pairs_to_mask(self.logic.critical_edges)
        
        # Baseline cache injection.
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

    def _ranked_physical_mask(self, logical_q: int) -> np.ndarray:
        # 不再使用候选排名，返回所有可用物理比特的掩码
        return self.free_physical_mask()

    def _physical_action_masks_bank(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None
        masks = np.zeros((self.logic.num_qubits, self.hardware.num_qubits), dtype=np.float32)
        active_idx = np.where(self.mapping_log_to_phys < 0)[0]
        for q in active_idx:
            masks[int(q)] = self._ranked_physical_mask(int(q))
        return masks

    def _physical_prior_bank(self) -> np.ndarray:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None
        n_logic = self.logic.num_qubits
        n_phys = self.hardware.num_qubits
        free_mask = self.free_physical_mask()
        prior = np.zeros((n_logic, n_phys), dtype=np.float32)
        centrality = self.phys_centrality.astype(np.float32)
        if centrality.size > 0:
            c_min = float(np.min(centrality))
            c_span = float(np.max(centrality) - c_min)
            centrality = (centrality - c_min) / (c_span + 1e-8) if c_span > 1e-8 else np.zeros_like(centrality)

        for logical_q in range(n_logic):
            if self.mapping_log_to_phys[logical_q] >= 0:
                continue
            weights = self.logic.edge_weight[logical_q].astype(np.float32)
            mapped_neighbors = [int(j) for j in np.where(weights > 0)[0] if self.mapping_log_to_phys[int(j)] >= 0]
            if mapped_neighbors:
                mapped_phys = self.mapping_log_to_phys[mapped_neighbors].astype(np.int64)
                edge_w = weights[mapped_neighbors].astype(np.float32)
                edge_w_sum = float(edge_w.sum()) + 1e-8
                for phys_q in range(n_phys):
                    dists = self.hardware.dist[phys_q, mapped_phys].astype(np.float32)
                    avg_dist = float(np.sum(edge_w * dists) / edge_w_sum)
                    prior[logical_q, phys_q] = 1.0 - (avg_dist / max(self.max_dist, 1.0))
            else:
                interaction_strength = float(np.sum(weights))
                strength = interaction_strength / (interaction_strength + 1.0)
                prior[logical_q] = 0.25 * strength * centrality

            for phys_q in range(n_phys):
                deg = max(int(self.hardware.graph.degree(phys_q)), 1)
                free_neighbors = sum(float(self.used_phys[nbr] < 0.5) for nbr in self.hardware.graph.neighbors(phys_q))
                prior[logical_q, phys_q] += 0.10 * (free_neighbors / float(deg)) + 0.05 * centrality[phys_q]
            prior[logical_q] *= free_mask

        return np.clip(prior, -1.0, 1.0).astype(np.float32)



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
        physical_masks_bank = self._physical_action_masks_bank()
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
            "physical_action_masks_bank": physical_masks_bank,
            "progress": np.float32(progress),
            "is_fixed_order": np.int64(1 if self.env_cfg.action_mode == "fixed_order_physical" else 0),
            "physical_centrality": self.phys_centrality.copy(),
            "physical_prior_bank": self._physical_prior_bank(),
            # 原始边统计量，让 GNN 学习最优组合
            "edge_freq": self.logic.edge_freq.copy(),
            "edge_first_layer": self.logic.edge_first_layer.copy(),
            "edge_front": self.logic.edge_front.copy(),
            "edge_early": self.logic.edge_early.copy(),
            "edge_log_f": self.logic.edge_log_f.copy(),
        }
        

    def _shape_reward_for_choice(self, logical_q: int, phys_q: int) -> tuple[float, Dict[str, float]]:
        assert self.logic is not None and self.mapping_log_to_phys is not None and self.used_phys is not None
        weights = self.logic.edge_weight[logical_q].astype(np.float32)
        neighbors = np.where(weights > 0)[0]
        mapped_neighbors = [int(j) for j in neighbors if self.mapping_log_to_phys[int(j)] >= 0]

        reward = 0.0
        avg_dist = 0.0
        mapped_weight_ratio = 0.0
        if mapped_neighbors:
            mapped_phys = self.mapping_log_to_phys[mapped_neighbors].astype(np.int64)
            dists = self.hardware.dist[phys_q, mapped_phys].astype(np.float32)
            edge_w = weights[mapped_neighbors].astype(np.float32)
            edge_w_sum = float(edge_w.sum()) + 1e-8
            avg_dist = float(np.sum(edge_w * dists) / edge_w_sum)
            total_w = float(np.sum(weights)) + 1e-8
            mapped_weight_ratio = float(edge_w.sum() / total_w)
            reward += 0.35 * mapped_weight_ratio * (1.0 - avg_dist / max(self.max_dist, 1.0))

        front_bonus = 0.0
        critical_bonus = 0.0
        for other in mapped_neighbors:
            other_phys = int(self.mapping_log_to_phys[other])
            dist = float(self.hardware.dist[phys_q, other_phys])
            close_bonus = max(0.0, 1.0 - dist / max(self.max_dist, 1.0))
            if self.front_pair_mask is not None:
                front_bonus += float(self.front_pair_mask[logical_q, other]) * close_bonus
            if self.critical_pair_mask is not None:
                critical_bonus += float(self.critical_pair_mask[logical_q, other]) * close_bonus
        reward += 0.12 * front_bonus + 0.08 * critical_bonus

        deg = max(int(self.hardware.graph.degree(phys_q)), 1)
        free_neighbors = sum(float(self.used_phys[nbr] < 0.5) for nbr in self.hardware.graph.neighbors(phys_q))
        free_ratio = free_neighbors / float(deg)
        reward += 0.05 * free_ratio

        centrality = float(self.phys_centrality[phys_q])
        reward += 0.03 * centrality
        reward = float(0.10 * np.tanh(reward / 0.50))

        info = {
            "shape_base": -avg_dist if mapped_neighbors else 0.0,
            "shape_frontier": float(front_bonus),
            "shape_critical": float(critical_bonus),
            "shape_future": 0.0,
            "shape_reservation": float(free_ratio),
            "shape_local_bonus": float(centrality),
            "gate_reward": 0.0,
            "lookahead_reward": 0.0,
            "blocked_penalty": 0.0,
            "score_raw": 0.0,
            "shape_centered": float(reward),
            "shape_best_gap": float(mapped_weight_ratio),
            "shape_ref_mean": 0.0,
            "shape_ref_std": 0.0,
            "completion_bonus": 0.0,
        }
        return reward, info

    def _compute_baseline(self) -> tuple[Optional[float], str, Dict[str, float]]:
        """Compute the configured external baseline under the paper protocol.

        The returned score is always obtained through objective_from_metrics(),
        which uses additional_cnot_count in the paper/CNOT-active protocol.
        Raw swap_count is retained only inside metrics as a diagnostic field.
        """
        assert self.circuit is not None and self.logic is not None
        mode = str(getattr(self.env_cfg, "baseline_mode", "dense"))
        if mode == "none":
            return None, "none", {}

        candidates: list[tuple[str, list[int]]] = []
        if mode == "trivial":
            candidates.append(("trivial", trivial_layout(self.logic.num_qubits)))
        elif mode == "dense":
            candidates.append(("dense", dense_layout(self.logic, self.hardware)))
        elif mode == "sabre":
            try:
                candidates.append(("sabre", sabre_layout(self.circuit, self.hardware, seed=self.env_cfg.sabre_seed)))
            except Exception:
                candidates.append(("sabre_fallback_dense", dense_layout(self.logic, self.hardware)))
        else:
            raise ValueError(f"Unsupported baseline_mode: {mode}")

        best_name = "none"
        best_metrics: Dict[str, float] = {}
        best_score: Optional[float] = None
        for name, layout in candidates:
            metrics = evaluate_layout_metrics(self.circuit, layout, self.hardware, self.env_cfg)
            score = float(objective_from_metrics(metrics, self.reward_cfg, self.env_cfg))
            if best_score is None or score < best_score:
                best_score = score
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
        return evaluate_layout_metrics(self.raw_circuit or self.circuit, layout, self.hardware, self.env_cfg)

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
            
            # 【修改点】：废弃与 Sabre 基线的相对提升率，直接将代价（如 SWAP 数量）作为绝对惩罚
            # 乘以 0.1 的目的是缩放方差，假设产生 50 个 SWAP，那么最后一步的惩罚是 -5.0，
            # 这与前面的单步奖励量级是匹配的，能稳定 Critic 网络的预估。
            anchor = float(self.reward_anchor_score if self.reward_anchor_score is not None else 0.0)
            logical_cnot = float(metrics.get("logical_cnot_count", 0.0) or 0.0)
            active_qubits = float(metrics.get("active_logical_qubits", self.logic.num_qubits) or self.logic.num_qubits)
            normalizer = max(anchor, logical_cnot, active_qubits, 1.0)
            terminal_reward = float((anchor - terminal_objective) / normalizer)
            clip = float(getattr(self.reward_cfg, "terminal_clip", 5.0) or 5.0)
            terminal_reward = float(np.clip(terminal_reward, -clip, clip) * self.reward_cfg.terminal_scale)

            reward += float(terminal_reward)
            # =========================================================================================

            info.update(metrics)
            info["routing_score"] = float(terminal_objective)
            info["baseline_score"] = float(self.baseline_score) if self.baseline_score is not None else None
            info["terminal_objective"] = float(terminal_objective)
            info["final_layout"] = layout
            info["terminal_reward"] = float(terminal_reward)
            info["terminal_reward_anchor"] = float(anchor)
            info["terminal_reward_normalizer"] = float(normalizer)

        return StepOutput(obs=self._get_obs(), reward=float(reward), done=done, info=info)

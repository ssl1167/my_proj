from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile

from .qiskit_runner import canonical_cnot_circuit
from qiskit.converters import circuit_to_dag

from .topology import adjacency_with_self_loops

CNOT_CANONICAL_BASIS = ["rz", "sx", "x", "cx"]
IGNORED_OPS = {"barrier", "measure", "delay", "reset"}


@dataclass
class LogicGraphData:
    num_qubits: int
    graph: nx.Graph
    adj: np.ndarray
    weighted_adj: np.ndarray
    node_features: np.ndarray
    edge_weight: np.ndarray
    # 原始边统计量，让 GNN 学习最优组合而非人工压缩
    edge_freq: np.ndarray       # 交互频率
    edge_first_layer: np.ndarray # 首次出现层
    edge_front: np.ndarray      # 前端计数
    edge_early: np.ndarray       # 前瞻窗口计数
    edge_log_f: np.ndarray      # 对数频率
    placement_order: List[int]
    critical_edges: List[Tuple[int, int]]
    front_pairs: List[Tuple[int, int]]
    critical_score: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


def _instruction_parts(inst):
    if hasattr(inst, "operation"):
        return inst.operation, inst.qubits, inst.clbits
    op = inst[0]
    qargs = inst[1]
    cargs = inst[2] if len(inst) > 2 else []
    return op, qargs, cargs


def _make_empty_like(circuit: QuantumCircuit) -> QuantumCircuit:
    return QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=f"{circuit.name}_twoq")


def _layerize_cx_ops(circuit: QuantumCircuit) -> List[List[Tuple[int, int, str]]]:
    cx_circ = _make_empty_like(circuit)
    for inst in circuit.data:
        op, qargs, cargs = _instruction_parts(inst)
        op_name = getattr(op, "name", "")
        if op_name in IGNORED_OPS or op_name != "cx" or len(qargs) != 2:
            continue
        q_indices = [circuit.find_bit(q).index for q in qargs]
        c_indices = [circuit.find_bit(c).index for c in cargs] if cargs else []
        op_to_append = op.copy() if hasattr(op, "copy") else op
        cx_circ.append(op_to_append, qargs=q_indices, cargs=c_indices)

    if not cx_circ.data:
        return []

    dag = circuit_to_dag(cx_circ)
    layers: List[List[Tuple[int, int, str]]] = []
    for layer_dict in dag.layers():
        layer_dag = layer_dict["graph"]
        layer_ops: List[Tuple[int, int, str]] = []
        for node in layer_dag.op_nodes():
            op_name = getattr(node.op, "name", node.name)
            if op_name != "cx" or len(node.qargs) != 2:
                continue
            q0 = cx_circ.find_bit(node.qargs[0]).index
            q1 = cx_circ.find_bit(node.qargs[1]).index
            if q0 != q1:
                layer_ops.append((int(q0), int(q1), "cx"))
        if layer_ops:
            layers.append(layer_ops)
    return layers


def _normalize_dense_adj(adj: np.ndarray) -> np.ndarray:
    deg = np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
    return (adj / deg).astype(np.float32)


def weighted_adjacency_with_self_loops(graph: nx.Graph, n: int, weight_attr: str = "weight") -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    weights: List[float] = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight_attr, 1.0))
        adj[int(u), int(v)] = w
        adj[int(v), int(u)] = w
        weights.append(w)
    self_loop = float(np.mean(weights)) if weights else 1.0
    np.fill_diagonal(adj, max(self_loop, 1.0))
    return _normalize_dense_adj(adj)


def _normalize_feature(x: np.ndarray, mode: str, max_expected: float, scale: float = 1.2) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    
    if mode == "adaptive":
        max_val = float(np.max(x))
        if max_val < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip(x / (max_val * scale), 0.0, 1.0).astype(np.float32)
    elif mode == "absolute":
        return np.clip(x / max_expected, 0.0, 1.0).astype(np.float32)
    elif mode == "standard":
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mean) / (std * scale)).astype(np.float32)
    else:
        return np.clip(x / max_expected, 0.0, 1.0).astype(np.float32)


def _select_critical_edges(critical_score_map: Dict[Tuple[int, int], float], first_layer: Dict[Tuple[int, int], int], freq: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    if not critical_score_map:
        return []
    ranked_edges = sorted(critical_score_map.items(), key=lambda kv: (-kv[1], first_layer[kv[0]], -freq[kv[0]]))
    scores = np.array([score for _, score in ranked_edges], dtype=np.float32)
    adaptive_thr = float(np.mean(scores)) + 0.25 * float(np.std(scores))
    selected = [edge for edge, score in ranked_edges if score >= adaptive_thr]
    if selected:
        return selected
    keep = max(1, min(len(ranked_edges), ceil(np.sqrt(len(ranked_edges)))))
    return [edge for edge, _ in ranked_edges[:keep]]


def _empty_logic_graph(n: int) -> LogicGraphData:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    empty_adj = adjacency_with_self_loops(g, n)
    return LogicGraphData(
        num_qubits=n,
        graph=g,
        adj=empty_adj,
        weighted_adj=empty_adj.copy(),
        node_features=np.zeros((n, 8), dtype=np.float32),
        edge_weight=np.zeros((n, n), dtype=np.float32),
        edge_freq=np.zeros((n, n), dtype=np.float32),
        edge_first_layer=np.zeros((n, n), dtype=np.float32),
        edge_front=np.zeros((n, n), dtype=np.float32),
        edge_early=np.zeros((n, n), dtype=np.float32),
        edge_log_f=np.zeros((n, n), dtype=np.float32),
        placement_order=list(range(n)),
        critical_edges=[],
        front_pairs=[],
        critical_score=np.zeros((n, n), dtype=np.float32),
    )


def build_logic_graph(
    circuit: QuantumCircuit,
    critical_window: int = 8,
    lookahead_window: int = 16,
    basis_gates: Sequence[str] | None = None,
    decompose: bool = True,
    cnot_only: bool = True,
    feature_norm_mode: str = "adaptive",
    feature_norm_scale: float = 1.2,
    feat_norm_neighbors: float = 15.0,
    feat_norm_twoq: float = 100.0,
    feat_norm_front: float = 10.0,
    feat_norm_early: float = 20.0,
    feat_norm_weighted_degree: float = 150.0,
    feat_norm_pagerank: float = 1.0,
    feat_norm_critical: float = 30.0,
) -> LogicGraphData:
    """
    Build a CNOT-interaction graph for initial-layout learning.

    In paper-compatible mode, the graph is built from a compact CNOT-only circuit:
    single-qubit gates are ignored, multi-qubit gates are first decomposed to CX
    basis, and qubits that never participate in CX are removed.
    """
    if cnot_only:
        work_circuit = canonical_cnot_circuit(circuit)
    else:
        work_circuit = circuit

    n = work_circuit.num_qubits
    if n <= 0:
        return _empty_logic_graph(0)

    layers = _layerize_cx_ops(work_circuit)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if not layers:
        return _empty_logic_graph(n)

    freq: Dict[Tuple[int, int], float] = defaultdict(float)
    first_layer: Dict[Tuple[int, int], int] = {}
    front_count: Dict[Tuple[int, int], float] = defaultdict(float)
    early_count: Dict[Tuple[int, int], float] = defaultdict(float)
    layer_positions: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    front_layer_count = min(max(int(critical_window), 1), len(layers))
    lookahead_layer_count = min(max(int(lookahead_window), front_layer_count), len(layers))

    for layer_idx, ops in enumerate(layers):
        for q0, q1, _ in ops:
            a, b = sorted((int(q0), int(q1)))
            if a == b:
                continue
            edge = (a, b)
            freq[edge] += 1.0
            layer_positions[edge].append(layer_idx)
            first_layer.setdefault(edge, layer_idx)
            if layer_idx < front_layer_count:
                front_count[edge] += 1.0
            if layer_idx < lookahead_layer_count:
                early_count[edge] += 1.0

    edge_weight = np.zeros((n, n), dtype=np.float32)
    critical_score_mat = np.zeros((n, n), dtype=np.float32)
    critical_incident = np.zeros(n, dtype=np.float32)
    distinct_neighbors = np.zeros(n, dtype=np.float32)
    total_twoq = np.zeros(n, dtype=np.float32)
    front_incident = np.zeros(n, dtype=np.float32)
    early_density = np.zeros(n, dtype=np.float32)
    critical_score_map: Dict[Tuple[int, int], float] = {}
    total_layers = max(len(layers), 1)
    # 新增：节点平均执行深度（越早执行完的节点紧迫度越高）
    mean_layer = np.zeros(n, dtype=np.float32)
    # 原始边特征数组（让 GNN 学习最优组合）
    edge_freq_arr = np.zeros((n, n), dtype=np.float32)
    edge_first_layer_arr = np.full((n, n), float(total_layers), dtype=np.float32)
    edge_front_arr = np.zeros((n, n), dtype=np.float32)
    edge_early_arr = np.zeros((n, n), dtype=np.float32)
    edge_log_f_arr = np.zeros((n, n), dtype=np.float32)

    for (a, b), f in freq.items():
        fl = first_layer[(a, b)]
        front = front_count[(a, b)]
        early = early_count[(a, b)]
        positions = layer_positions[(a, b)]
        if len(positions) > 1:
            gaps = np.diff(positions).astype(np.float32)
            mean_gap = float(np.mean(gaps))
        else:
            mean_gap = float(total_layers)

        log_f = np.log1p(f)
        # [修改点]：保留原始边特征供 GNN 学习，人工压缩的 w 仅用于邻接矩阵
        w = 1.00 * log_f + 1.50 * (1.0 / (1.0 + fl)) + 0.80 * np.log1p(front) + 0.50 * (1.0 / (1.0 + mean_gap))
        critical_score = 0.75 * early + 0.60 * front + 0.55 * log_f + 0.50 * (1.0 / (1.0 + fl))
        critical_score_map[(a, b)] = float(critical_score)
        g.add_edge(a, b, weight=float(w), freq=float(f), first_layer=int(fl), front=float(front), early=float(early), critical_score=float(critical_score))
        edge_weight[a, b] = edge_weight[b, a] = float(w)
        # 保存原始边特征供 GNN 学习
        edge_freq_arr[a, b] = edge_freq_arr[b, a] = float(f)
        edge_first_layer_arr[a, b] = edge_first_layer_arr[b, a] = 1.0 / (1.0 + float(fl))
        edge_front_arr[a, b] = edge_front_arr[b, a] = float(front)
        edge_early_arr[a, b] = edge_early_arr[b, a] = float(early)
        edge_log_f_arr[a, b] = edge_log_f_arr[b, a] = float(log_f)
        critical_score_mat[a, b] = critical_score_mat[b, a] = float(critical_score)
        total_twoq[a] += f
        total_twoq[b] += f
        distinct_neighbors[a] += 1.0
        distinct_neighbors[b] += 1.0
        front_incident[a] += front
        front_incident[b] += front
        early_density[a] += early
        early_density[b] += early
        critical_incident[a] += critical_score
        critical_incident[b] += critical_score

    # [修改点]：对边权重进行全局归一化，稳定 GNN 训练
    edge_weight_flat = edge_weight[edge_weight > 0]
    if edge_weight_flat.size > 0:
        edge_weight_max = float(edge_weight_flat.max())
        edge_weight = np.clip(edge_weight / edge_weight_max, 0.0, 1.0).astype(np.float32)

    # [修改点]：计算每个节点的平均执行深度（替代冗余的 future_heat）
    layer_sum = np.zeros(n, dtype=np.float32)
    layer_cnt = np.zeros(n, dtype=np.float32)
    for (a, b) in first_layer.keys():
        fl = float(first_layer[(a, b)])
        layer_sum[a] += fl
        layer_sum[b] += fl
        layer_cnt[a] += 1.0
        layer_cnt[b] += 1.0
    mean_layer = np.where(layer_cnt > 0, layer_sum / layer_cnt, float(total_layers))

    weighted_degree = np.array([sum(float(g[u][v]["weight"]) for v in g.neighbors(u)) for u in range(n)], dtype=np.float32)
    if g.number_of_edges() > 0:
        pr = nx.pagerank(g, weight="weight")
        pagerank = np.array([float(pr.get(i, 0.0)) for i in range(n)], dtype=np.float32)
    else:
        pagerank = np.ones(n, dtype=np.float32) / max(n, 1)

    node_features = np.stack([
        _normalize_feature(distinct_neighbors, feature_norm_mode, feat_norm_neighbors, feature_norm_scale),      # 逻辑邻居数
        _normalize_feature(total_twoq, feature_norm_mode, feat_norm_twoq, feature_norm_scale),             # 整体交互频率
        _normalize_feature(front_incident, feature_norm_mode, feat_norm_front, feature_norm_scale),         # 前端交互频率
        _normalize_feature(early_density, feature_norm_mode, feat_norm_early, feature_norm_scale),          # 前瞻窗口频率
        _normalize_feature(weighted_degree, feature_norm_mode, feat_norm_weighted_degree, feature_norm_scale),       # 权重度
        _normalize_feature(pagerank, feature_norm_mode, feat_norm_pagerank, feature_norm_scale),                 # PageRank
        _normalize_feature(critical_incident, feature_norm_mode, feat_norm_critical, feature_norm_scale),       # 关键得分
        _normalize_feature(total_layers - mean_layer, "absolute", float(total_layers), feature_norm_scale),  # 节点紧迫度
    ], axis=1).astype(np.float32)

    priority = 1.0 * _normalize_feature(distinct_neighbors, feature_norm_mode, feat_norm_neighbors, feature_norm_scale) + 1.2 * _normalize_feature(total_twoq, feature_norm_mode, feat_norm_twoq, feature_norm_scale) + 1.2 * _normalize_feature(front_incident, feature_norm_mode, feat_norm_front, feature_norm_scale) + 0.9 * _normalize_feature(early_density, feature_norm_mode, feat_norm_early, feature_norm_scale) + 0.7 * _normalize_feature(critical_incident, feature_norm_mode, feat_norm_critical, feature_norm_scale)
    placement_order = [int(x) for x in np.argsort(-priority)]
    critical_edges = _select_critical_edges(critical_score_map, first_layer, freq)

    front_pairs: List[Tuple[int, int]] = []
    seen = set()
    for layer_ops in layers[:front_layer_count]:
        for q0, q1, _ in layer_ops:
            edge = tuple(sorted((int(q0), int(q1))))
            if edge not in seen:
                front_pairs.append(edge)
                seen.add(edge)

    return LogicGraphData(
        num_qubits=n,
        graph=g,
        adj=adjacency_with_self_loops(g, n),
        weighted_adj=weighted_adjacency_with_self_loops(g, n, weight_attr="weight"),
        node_features=node_features,
        edge_weight=edge_weight,
        edge_freq=edge_freq_arr,
        edge_first_layer=edge_first_layer_arr,
        edge_front=edge_front_arr,
        edge_early=edge_early_arr,
        edge_log_f=edge_log_f_arr,
        placement_order=placement_order,
        critical_edges=critical_edges,
        front_pairs=front_pairs,
        critical_score=critical_score_mat.astype(np.float32),
    )
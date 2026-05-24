from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile

from .topology import adjacency_with_self_loops
from qiskit.converters import circuit_to_dag


@dataclass
class LogicGraphData:
    num_qubits: int
    graph: nx.Graph
    adj: np.ndarray
    weighted_adj: np.ndarray
    node_features: np.ndarray
    edge_weight: np.ndarray
    placement_order: List[int]
    critical_edges: List[Tuple[int, int]]
    front_pairs: List[Tuple[int, int]]
    critical_score: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


def _layerize_two_qubit_ops(circuit: QuantumCircuit) -> List[List[Tuple[int, int, str]]]:
    two_q_circ = circuit.copy()
    two_q_circ.data.clear()  
    
    for inst in circuit.data:
        if hasattr(inst, "operation"):
            op = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits
        else:
            op = inst[0]
            qargs = inst[1]
            cargs = inst[2] if len(inst) > 2 else []

        if len(qargs) == 2:
            # 核心修复 3：统一用整数 index 桥接跨 circuit 的对象绑定，杜绝版本兼容性 Bug
            q_mapped = [two_q_circ.qubits[circuit.find_bit(q).index] for q in qargs]
            c_mapped = [two_q_circ.clbits[circuit.find_bit(c).index] for c in cargs] if cargs else []
            op_to_append = op.copy() if hasattr(op, "copy") else op
            two_q_circ.append(op_to_append, qargs=q_mapped, cargs=c_mapped)
            
    dag = circuit_to_dag(two_q_circ)
    layers: List[List[Tuple[int, int, str]]] = []
    
    for layer_dict in dag.layers():
        layer_dag = layer_dict["graph"]
        layer_ops: List[Tuple[int, int, str]] = []
        
        for node in layer_dag.op_nodes():
            q0 = circuit.find_bit(node.qargs[0]).index
            q1 = circuit.find_bit(node.qargs[1]).index
            op_name = getattr(node.op, "name", node.name)
            layer_ops.append((q0, q1, op_name))
            
        if layer_ops:
            layers.append(layer_ops)
            
    return layers


def _normalize_dense_adj(adj: np.ndarray) -> np.ndarray:
    deg = np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
    return (adj / deg).astype(np.float32)


def weighted_adjacency_with_self_loops(graph: nx.Graph, n: int, weight_attr: str = "weight") -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    weights = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight_attr, 1.0))
        adj[u, v] = w
        adj[v, u] = w
        weights.append(w)

    self_loop = float(np.mean(weights)) if weights else 1.0
    np.fill_diagonal(adj, max(self_loop, 1.0))
    return _normalize_dense_adj(adj)


def _safe_norm(x: np.ndarray) -> np.ndarray:
    span = np.max(x) - np.min(x)
    if span < 1e-8:
        return np.zeros_like(x)
    return (x - np.min(x)) / (span + 1e-8)


def _select_critical_edges(
    critical_score_map: Dict[Tuple[int, int], float],
    first_layer: Dict[Tuple[int, int], int],
    freq: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int]]:
    if not critical_score_map:
        return []

    ranked_edges = sorted(
        critical_score_map.items(),
        key=lambda kv: (-kv[1], first_layer[kv[0]], -freq[kv[0]]),
    )
    scores = np.array([score for _, score in ranked_edges], dtype=np.float32)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    adaptive_thr = mean + 0.25 * std

    selected = [edge for edge, score in ranked_edges if score >= adaptive_thr]
    if selected:
        return selected

    keep = max(1, min(len(ranked_edges), ceil(np.sqrt(len(ranked_edges)))))
    return [edge for edge, _ in ranked_edges[:keep]]


def build_logic_graph(circuit: QuantumCircuit, critical_window: int = 8, lookahead_window: int = 16) -> LogicGraphData:
    # 核心修复 2：在特征建图之前强制解构出标准通用基，让 CCX 等多控门的内在依赖完全显露
    circuit = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
    )
    
    n = circuit.num_qubits
    layers = _layerize_two_qubit_ops(circuit)

    g = nx.Graph()
    g.add_nodes_from(range(n))

    if not layers:
        empty_adj = adjacency_with_self_loops(g, n)
        return LogicGraphData(
            num_qubits=n,
            graph=g,
            adj=empty_adj,
            weighted_adj=empty_adj.copy(),
            node_features=np.zeros((n, 8), dtype=np.float32),
            edge_weight=np.zeros((n, n), dtype=np.float32),
            placement_order=list(range(n)),
            critical_edges=[],
            front_pairs=[],
            critical_score=np.zeros((n, n), dtype=np.float32),
        )

    freq = defaultdict(float)
    first_layer: Dict[Tuple[int, int], int] = {}
    front_count = defaultdict(float)
    early_count = defaultdict(float)
    layer_positions: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    front_layer_count = min(max(int(critical_window), 1), len(layers))
    lookahead_layer_count = min(max(int(lookahead_window), front_layer_count), len(layers))

    for layer_idx, ops in enumerate(layers):
        for q0, q1, _ in ops:
            a, b = sorted((q0, q1))
            freq[(a, b)] += 1.0
            layer_positions[(a, b)].append(layer_idx)
            if (a, b) not in first_layer:
                first_layer[(a, b)] = layer_idx
            if layer_idx < front_layer_count:
                front_count[(a, b)] += 1.0
            if layer_idx < lookahead_layer_count:
                early_count[(a, b)] += 1.0

    edge_weight = np.zeros((n, n), dtype=np.float32)
    critical_score_mat = np.zeros((n, n), dtype=np.float32)
    future_heat = np.zeros(n, dtype=np.float32)
    critical_incident = np.zeros(n, dtype=np.float32)
    distinct_neighbors = np.zeros(n, dtype=np.float32)
    total_twoq = np.zeros(n, dtype=np.float32)
    front_incident = np.zeros(n, dtype=np.float32)
    early_density = np.zeros(n, dtype=np.float32)
    critical_score_map: Dict[Tuple[int, int], float] = {}

    total_layers = max(len(layers), 1)
    for (a, b), f in freq.items():
        fl = first_layer[(a, b)]
        front = front_count[(a, b)]
        early = early_count[(a, b)]
        positions = layer_positions[(a, b)]

        if len(positions) > 1:
            gaps = np.diff(positions).astype(np.float32)
            mean_gap = float(np.mean(gaps))
            iqr = float(np.percentile(gaps, 75) - np.percentile(gaps, 25)) if len(gaps) > 2 else mean_gap
        else:
            mean_gap = float(total_layers)
            iqr = mean_gap

        w = (
            1.00 * f
            + 1.20 * (1.0 / (1.0 + fl))
            + 0.90 * front
            + 0.55 * (1.0 / (1.0 + mean_gap))
            + 0.25 * (1.0 / (1.0 + iqr))
        )

        critical_score = (
            0.75 * early
            + 0.60 * front
            + 0.55 * f
            + 0.50 * (1.0 / (1.0 + fl))
        )
        critical_score_map[(a, b)] = float(critical_score)

        g.add_edge(a, b, weight=w, freq=f, first_layer=fl, front=front, early=early, critical_score=critical_score)
        edge_weight[a, b] = edge_weight[b, a] = w
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
        future_heat[a] += early
        future_heat[b] += early

    weighted_degree = np.array([sum(g[u][v]["weight"] for v in g.neighbors(u)) for u in range(n)], dtype=np.float32)

    pagerank = (
        np.array(list(nx.pagerank(g, weight="weight").values()), dtype=np.float32)
        if g.number_of_edges() > 0
        else np.ones(n, dtype=np.float32) / max(n, 1)
    )

    node_features = np.stack(
        [
            _safe_norm(distinct_neighbors),
            _safe_norm(total_twoq),
            _safe_norm(front_incident),
            _safe_norm(early_density),
            _safe_norm(weighted_degree),
            _safe_norm(pagerank),
            _safe_norm(critical_incident),
            _safe_norm(future_heat),
        ],
        axis=1,
    ).astype(np.float32)

    priority = (
        1.0 * _safe_norm(distinct_neighbors)
        + 1.2 * _safe_norm(total_twoq)
        + 1.2 * _safe_norm(front_incident)
        + 0.9 * _safe_norm(early_density)
        + 0.7 * _safe_norm(critical_incident)
    )
    placement_order = list(np.argsort(-priority))

    critical_edges = _select_critical_edges(critical_score_map, first_layer, freq)

    front_pairs: List[Tuple[int, int]] = []
    seen = set()
    for layer_ops in layers[:front_layer_count]:
        for q0, q1, _ in layer_ops:
            edge = tuple(sorted((q0, q1)))
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
        placement_order=placement_order,
        critical_edges=critical_edges,
        front_pairs=front_pairs,
        critical_score=critical_score_mat.astype(np.float32),
    )

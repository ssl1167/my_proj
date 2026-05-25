from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
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


def _decompose_to_cx_basis(circuit: QuantumCircuit, basis_gates: Sequence[str] | None = None) -> QuantumCircuit:
    basis = list(basis_gates or CNOT_CANONICAL_BASIS)
    if "cx" not in basis:
        basis = list(CNOT_CANONICAL_BASIS)
    return transpile(circuit, basis_gates=basis, optimization_level=0)


def _decompose_to_strict_cx_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Always decompose to the canonical 1Q+CX basis used by the paper protocol.

    This function intentionally ignores EnvConfig.basis_gates, because allowing
    "swap" or other two-qubit basis gates here would make CNOT-only graph
    construction silently drop those gates instead of decomposing them to CX.
    """
    return transpile(circuit, basis_gates=CNOT_CANONICAL_BASIS, optimization_level=0)


def _canonical_cnot_circuit(circuit: QuantumCircuit, basis_gates: Sequence[str] | None = None) -> QuantumCircuit:
    del basis_gates
    work = _decompose_to_strict_cx_basis(circuit)
    cx_ops: List[Tuple[int, int]] = []
    active: set[int] = set()

    for inst in work.data:
        op, qargs, _ = _instruction_parts(inst)
        if getattr(op, "name", "") != "cx" or len(qargs) != 2:
            continue
        c = work.find_bit(qargs[0]).index
        t = work.find_bit(qargs[1]).index
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
    return out


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


def _safe_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    span = float(np.max(x) - np.min(x))
    if span < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - np.min(x)) / (span + 1e-8)).astype(np.float32)


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
) -> LogicGraphData:
    """
    Build a CNOT-interaction graph for initial-layout learning.

    In paper-compatible mode, the graph is built from a compact CNOT-only circuit:
    single-qubit gates are ignored, multi-qubit gates are first decomposed to CX
    basis, and qubits that never participate in CX are removed.
    """
    if cnot_only:
        work_circuit = _canonical_cnot_circuit(circuit, basis_gates=basis_gates)
    elif decompose:
        work_circuit = _decompose_to_cx_basis(circuit, basis_gates=basis_gates)
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

        w = 1.00 * f + 1.20 * (1.0 / (1.0 + fl)) + 0.90 * front + 0.55 * (1.0 / (1.0 + mean_gap)) + 0.25 * (1.0 / (1.0 + iqr))
        critical_score = 0.75 * early + 0.60 * front + 0.55 * f + 0.50 * (1.0 / (1.0 + fl))
        critical_score_map[(a, b)] = float(critical_score)
        g.add_edge(a, b, weight=float(w), freq=float(f), first_layer=int(fl), front=float(front), early=float(early), critical_score=float(critical_score))
        edge_weight[a, b] = edge_weight[b, a] = float(w)
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

    weighted_degree = np.array([sum(float(g[u][v]["weight"]) for v in g.neighbors(u)) for u in range(n)], dtype=np.float32)
    if g.number_of_edges() > 0:
        pr = nx.pagerank(g, weight="weight")
        pagerank = np.array([float(pr.get(i, 0.0)) for i in range(n)], dtype=np.float32)
    else:
        pagerank = np.ones(n, dtype=np.float32) / max(n, 1)

    node_features = np.stack([
        _safe_norm(distinct_neighbors),
        _safe_norm(total_twoq),
        _safe_norm(front_incident),
        _safe_norm(early_density),
        _safe_norm(weighted_degree),
        _safe_norm(pagerank),
        _safe_norm(critical_incident),
        _safe_norm(future_heat),
    ], axis=1).astype(np.float32)

    priority = 1.0 * _safe_norm(distinct_neighbors) + 1.2 * _safe_norm(total_twoq) + 1.2 * _safe_norm(front_incident) + 0.9 * _safe_norm(early_density) + 0.7 * _safe_norm(critical_incident)
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
        placement_order=placement_order,
        critical_edges=critical_edges,
        front_pairs=front_pairs,
        critical_score=critical_score_mat.astype(np.float32),
    )

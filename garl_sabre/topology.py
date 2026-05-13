from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap


@dataclass
class HardwareTopology:
    rows: int
    cols: int
    graph: nx.Graph
    coupling_map: CouplingMap
    dist: np.ndarray
    node_features: np.ndarray
    topology_name: str = ""

    @property
    def num_qubits(self) -> int:
        return self.graph.number_of_nodes()


def grid_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def _build_grid_graph(rows: int, cols: int, mode: str) -> nx.Graph:
    g = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            u = grid_index(r, c, cols)
            g.add_node(u, row=r, col=c)
    for r in range(rows):
        for c in range(cols):
            u = grid_index(r, c, cols)
            if r + 1 < rows:
                v = grid_index(r + 1, c, cols)
                g.add_edge(u, v)
            if c + 1 < cols:
                v = grid_index(r, c + 1, cols)
                if mode == "bottleneck_grid":
                    if r % 2 == 0 or c in (0, cols - 2):
                        g.add_edge(u, v)
                else:
                    g.add_edge(u, v)
    return g


def _build_ibm_q20_graph() -> tuple[nx.Graph, int, int]:
    # Manual Tokyo-style 20-qubit lattice used in older mapping papers.
    rows, cols = 4, 5
    g = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            u = grid_index(r, c, cols)
            g.add_node(u, row=r, col=c)

    horizontal = [(grid_index(r, c, cols), grid_index(r, c + 1, cols)) for r in range(rows) for c in range(cols - 1)]
    vertical = [(grid_index(r, c, cols), grid_index(r + 1, c, cols)) for r in range(rows - 1) for c in range(cols)]
    diagonals = [
        (1, 5), (1, 7), (3, 7), (3, 9),
        (5, 11), (7, 11), (7, 13), (9, 13),
        (11, 15), (11, 17), (13, 17), (13, 19),
    ]
    g.add_edges_from(horizontal + vertical + diagonals)
    return g, rows, cols


def _build_heavy_hex_graph(distance: int) -> tuple[nx.Graph, int, int, CouplingMap]:
    if distance <= 0 or distance % 2 == 0:
        raise ValueError("heavy_hex distance must be a positive odd integer")

    coupling_map = CouplingMap.from_heavy_hex(distance, bidirectional=True)
    edges = [(int(u), int(v)) for u, v in coupling_map.get_edges()]
    n = max(max(u, v) for u, v in edges) + 1 if edges else 0

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g, distance, distance, coupling_map


def _make_bidirectional_coupling_map(g: nx.Graph) -> CouplingMap:
    directed_edges: List[Tuple[int, int]] = []
    for u, v in g.edges():
        directed_edges.append((int(u), int(v)))
        directed_edges.append((int(v), int(u)))
    return CouplingMap(directed_edges)


def _safe_norm(x: np.ndarray) -> np.ndarray:
    span = np.max(x) - np.min(x)
    if span < 1e-8:
        return np.zeros_like(x)
    return (x - np.min(x)) / (span + 1e-8)


def _spectral_position_features(g: nx.Graph, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable pseudo-spatial features for topologies without explicit row/col.

    This avoids feeding arbitrary node-index order (np.arange(n)) to the model.
    """
    if n <= 1:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    try:
        pos = nx.spectral_layout(g, dim=2)
        xs = np.array([float(pos[i][0]) for i in range(n)], dtype=np.float32)
        ys = np.array([float(pos[i][1]) for i in range(n)], dtype=np.float32)
        return _safe_norm(xs), _safe_norm(ys)
    except Exception:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)


def _compute_node_features(g: nx.Graph, rows: int, cols: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    n = g.number_of_nodes()
    degree = np.array([g.degree(i) for i in range(n)], dtype=np.float32)
    betweenness = np.array(list(nx.betweenness_centrality(g).values()), dtype=np.float32)
    closeness = np.array(list(nx.closeness_centrality(g).values()), dtype=np.float32)

    path_lengths = dict(nx.all_pairs_shortest_path_length(g))
    dist = np.full((n, n), 1e9, dtype=np.float32)
    for i in range(n):
        dist[i, i] = 0.0
        for j, d in path_lengths[i].items():
            dist[i, j] = float(d)
    avg_dist = np.array([dist[i][dist[i] < 1e8].mean() for i in range(n)], dtype=np.float32)

    if all(("row" in g.nodes[i] and "col" in g.nodes[i]) for i in range(n)):
        row_feat = np.array([g.nodes[i]["row"] / max(rows - 1, 1) for i in range(n)], dtype=np.float32)
        col_feat = np.array([g.nodes[i]["col"] / max(cols - 1, 1) for i in range(n)], dtype=np.float32)
    else:
        # For heavy-hex and other topologies without explicit coordinates, use a
        # spectral embedding rather than arbitrary node indices.
        row_feat, col_feat = _spectral_position_features(g, n)

    bottleneck_flag = (betweenness > np.median(betweenness)).astype(np.float32)
    junction_threshold = 4 if mode in {"grid", "bottleneck_grid", "ibm_q20"} else 3
    junction_threshold = min(junction_threshold, int(np.max(degree)) if n > 0 else junction_threshold)
    core_flag = np.array([1.0 if g.degree(i) >= junction_threshold else 0.0 for i in range(n)], dtype=np.float32)

    node_features = np.stack(
        [
            _safe_norm(degree),
            _safe_norm(betweenness),
            _safe_norm(closeness),
            1.0 - _safe_norm(avg_dist),
            row_feat,
            col_feat,
            core_flag,
            bottleneck_flag,
        ],
        axis=1,
    ).astype(np.float32)
    return dist, node_features


def build_hardware_topology(rows: int, cols: int, mode: str = "grid", distance: int = 5) -> HardwareTopology:
    if mode not in {"grid", "bottleneck_grid", "ibm_q20", "heavy_hex"}:
        raise ValueError(f"Unsupported topology mode: {mode}")

    if mode == "ibm_q20":
        g, rows, cols = _build_ibm_q20_graph()
        coupling_map = _make_bidirectional_coupling_map(g)
    elif mode == "heavy_hex":
        g, rows, cols, coupling_map = _build_heavy_hex_graph(distance)
    else:
        g = _build_grid_graph(rows, cols, mode)
        coupling_map = _make_bidirectional_coupling_map(g)

    dist, node_features = _compute_node_features(g, rows, cols, mode)

    return HardwareTopology(
        rows=rows,
        cols=cols,
        graph=g,
        coupling_map=coupling_map,
        dist=dist,
        node_features=node_features,
        topology_name=mode,
    )


def build_grid_topology(rows: int, cols: int) -> HardwareTopology:
    return build_hardware_topology(rows, cols, mode="grid")


def adjacency_with_self_loops(graph: nx.Graph, n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in graph.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    return adj / deg

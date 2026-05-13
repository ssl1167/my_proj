from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .circuit_features import LogicGraphData
from .config import EnvConfig, RewardConfig
from .qiskit_runner import evaluate_layout_metrics, objective_from_metrics
from .topology import HardwareTopology


@dataclass
class TabuResult:
    layout: List[int]
    metrics: Dict[str, float]
    routing_score: float
    surrogate_score: float
    num_exact_evals: int
    num_iters: int


def _incident_priority(logic: LogicGraphData) -> np.ndarray:
    return logic.edge_weight.sum(axis=1).astype(np.float32)


def _build_pair_masks(logic: LogicGraphData) -> Tuple[np.ndarray, np.ndarray]:
    front_mask = np.zeros_like(logic.edge_weight, dtype=np.float32)
    critical_mask = np.zeros_like(logic.edge_weight, dtype=np.float32)
    for a, b in logic.front_pairs:
        front_mask[a, b] = front_mask[b, a] = 1.0
    for a, b in logic.critical_edges:
        critical_mask[a, b] = critical_mask[b, a] = 1.0
    return front_mask, critical_mask


def _build_edge_list(logic: LogicGraphData) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    n = logic.num_qubits
    for a in range(n):
        for b in range(a + 1, n):
            w = float(logic.edge_weight[a, b])
            if w > 0.0:
                edges.append((a, b, w))
    return edges


def surrogate_layout_score(
    layout: Sequence[int],
    logic: LogicGraphData,
    hardware: HardwareTopology,
    edge_list: List[Tuple[int, int, float]] | None = None,
    front_mask: np.ndarray | None = None,
    critical_mask: np.ndarray | None = None,
    centrality: np.ndarray | None = None,
    priority: np.ndarray | None = None,
) -> float:
    layout_arr = np.asarray(layout, dtype=np.int64)
    if edge_list is None:
        edge_list = _build_edge_list(logic)
    if front_mask is None or critical_mask is None:
        front_mask, critical_mask = _build_pair_masks(logic)
    if centrality is None:
        centrality = hardware.node_features[:, 0] + hardware.node_features[:, 1] + hardware.node_features[:, 2] + hardware.node_features[:, 3]
    if priority is None:
        priority = _incident_priority(logic)

    total = 0.0
    for a, b, w in edge_list:
        d = float(hardware.dist[layout_arr[a], layout_arr[b]])
        emphasis = 1.0 + 0.85 * float(front_mask[a, b]) + 0.55 * float(critical_mask[a, b])
        total += emphasis * w * d

    if float(np.sum(priority)) > 1e-8:
        total -= 0.10 * float(np.sum(priority * centrality[layout_arr]) / (np.sum(priority) + 1e-8))
    return float(total)


def _evaluate_exact(
    layout: Sequence[int],
    circuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
) -> Tuple[Dict[str, float], float]:
    metrics = evaluate_layout_metrics(circuit, list(layout), hardware, env_cfg)
    score = objective_from_metrics(metrics, reward_cfg, env_cfg)
    return metrics, float(score)


def _local_position_candidates(
    q: int,
    layout: np.ndarray,
    logic: LogicGraphData,
    hardware: HardwareTopology,
    max_candidates: int,
) -> List[int]:
    if max_candidates <= 0:
        return []

    weights = logic.edge_weight[q].astype(np.float32)
    neighbors = np.where(weights > 0)[0]
    if neighbors.size <= 0:
        centrality = hardware.node_features[:, 0] + hardware.node_features[:, 1]
        order = np.argsort(-centrality)
        return [int(x) for x in order[:max_candidates] if int(x) != int(layout[q])]

    scores = []
    denom = float(np.sum(weights[neighbors])) + 1e-8
    for p in range(hardware.num_qubits):
        if int(p) == int(layout[q]):
            continue
        d = hardware.dist[int(p), layout[neighbors]].astype(np.float32)
        score = float(np.sum(weights[neighbors] * d) / denom)
        scores.append((score, int(p)))
    scores.sort(key=lambda x: x[0])
    return [p for _, p in scores[:max_candidates]]


def _tabu_aspiration_allowed(
    move: Tuple,
    it: int,
    score: float,
    best_surrogate: float,
    current_layout: np.ndarray,
    trial_layout: np.ndarray,
    circuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    exact_eval_every: int,
    best_exact: float,
    exact_evals: int,
) -> Tuple[bool, float, int]:
    """
    Allow a tabu move if:
    1) it is clearly better by surrogate, or
    2) on periodic exact checks it improves the true objective.
    """
    if score < best_surrogate - 1e-8:
        return True, best_exact, exact_evals

    if exact_eval_every > 0 and (it % exact_eval_every == 0):
        _, trial_exact = _evaluate_exact(trial_layout, circuit, hardware, env_cfg, reward_cfg)
        exact_evals += 1
        if trial_exact < best_exact - 1e-8:
            return True, min(best_exact, trial_exact), exact_evals

    return False, best_exact, exact_evals


def tabu_refine_layout(
    circuit,
    initial_layout: Sequence[int],
    logic: LogicGraphData,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    num_iters: int = 12,
    candidate_qubits: int = 6,
    relocate_candidates: int = 3,
    tabu_tenure: int = 5,
    exact_eval_every: int = 0,
) -> TabuResult:
    current = np.asarray(initial_layout, dtype=np.int64).copy()
    front_mask, critical_mask = _build_pair_masks(logic)
    edge_list = _build_edge_list(logic)
    centrality = hardware.node_features[:, 0] + hardware.node_features[:, 1] + hardware.node_features[:, 2] + hardware.node_features[:, 3]
    priority = _incident_priority(logic)

    best_layout = current.copy()
    best_surrogate = surrogate_layout_score(
        best_layout, logic, hardware,
        edge_list=edge_list,
        front_mask=front_mask,
        critical_mask=critical_mask,
        centrality=centrality,
        priority=priority,
    )
    best_metrics, best_exact = _evaluate_exact(best_layout, circuit, hardware, env_cfg, reward_cfg)
    exact_evals = 1

    tabu_until: Dict[Tuple, int] = {}

    for it in range(1, max(0, num_iters) + 1):
        incident_cost = np.zeros(logic.num_qubits, dtype=np.float32)
        for q in range(logic.num_qubits):
            neighbors = np.where(logic.edge_weight[q] > 0)[0]
            if neighbors.size <= 0:
                continue
            d = hardware.dist[current[q], current[neighbors]].astype(np.float32)
            incident_cost[q] = float(np.sum(logic.edge_weight[q, neighbors] * d))
        rank = np.argsort(-(incident_cost + 0.15 * priority))
        hot_qubits = [int(x) for x in rank[: max(2, min(candidate_qubits, logic.num_qubits))]]

        phys_owner = np.full(hardware.num_qubits, -1, dtype=np.int64)
        for q, p in enumerate(current.tolist()):
            if 0 <= p < hardware.num_qubits:
                phys_owner[int(p)] = int(q)

        best_move = None
        best_move_surrogate = None
        best_move_layout = None

        # 1) direct swaps among hot qubits
        for i in range(len(hot_qubits)):
            for j in range(i + 1, len(hot_qubits)):
                qa, qb = hot_qubits[i], hot_qubits[j]
                move = ("swap_hot", min(qa, qb), max(qa, qb))
                trial = current.copy()
                trial[qa], trial[qb] = trial[qb], trial[qa]
                score = surrogate_layout_score(
                    trial, logic, hardware,
                    edge_list=edge_list,
                    front_mask=front_mask,
                    critical_mask=critical_mask,
                    centrality=centrality,
                    priority=priority,
                )
                if tabu_until.get(move, -1) >= it:
                    aspiration, best_exact, exact_evals = _tabu_aspiration_allowed(
                        move, it, score, best_surrogate, current, trial,
                        circuit, hardware, env_cfg, reward_cfg,
                        exact_eval_every, best_exact, exact_evals,
                    )
                    if not aspiration:
                        continue
                if best_move_surrogate is None or score < best_move_surrogate:
                    best_move = move
                    best_move_surrogate = score
                    best_move_layout = trial

        # 2) hotspot -> candidate physical positions (free relocation OR swap with current occupant)
        for q in hot_qubits:
            candidate_positions = _local_position_candidates(
                q=q,
                layout=current,
                logic=logic,
                hardware=hardware,
                max_candidates=relocate_candidates,
            )
            for p in candidate_positions:
                owner = int(phys_owner[p])
                if owner < 0:
                    move = ("move", int(q), int(p))
                    trial = current.copy()
                    trial[q] = int(p)
                else:
                    if owner == q:
                        continue
                    move = ("swap_pos", min(int(q), owner), max(int(q), owner), int(p))
                    trial = current.copy()
                    trial[q], trial[owner] = trial[owner], trial[q]

                score = surrogate_layout_score(
                    trial, logic, hardware,
                    edge_list=edge_list,
                    front_mask=front_mask,
                    critical_mask=critical_mask,
                    centrality=centrality,
                    priority=priority,
                )
                if tabu_until.get(move, -1) >= it:
                    aspiration, best_exact, exact_evals = _tabu_aspiration_allowed(
                        move, it, score, best_surrogate, current, trial,
                        circuit, hardware, env_cfg, reward_cfg,
                        exact_eval_every, best_exact, exact_evals,
                    )
                    if not aspiration:
                        continue
                if best_move_surrogate is None or score < best_move_surrogate:
                    best_move = move
                    best_move_surrogate = score
                    best_move_layout = trial

        if best_move_layout is None:
            break

        current = best_move_layout
        tabu_until[best_move] = it + max(1, tabu_tenure)

        if best_move_surrogate is not None and best_move_surrogate < best_surrogate:
            best_surrogate = float(best_move_surrogate)
            best_layout = current.copy()

        if exact_eval_every > 0 and it % exact_eval_every == 0:
            metrics, score = _evaluate_exact(current, circuit, hardware, env_cfg, reward_cfg)
            exact_evals += 1
            if score < best_exact:
                best_exact = score
                best_metrics = metrics
                best_layout = current.copy()
                best_surrogate = surrogate_layout_score(
                    best_layout, logic, hardware,
                    edge_list=edge_list,
                    front_mask=front_mask,
                    critical_mask=critical_mask,
                    centrality=centrality,
                    priority=priority,
                )

    final_metrics, final_score = _evaluate_exact(best_layout, circuit, hardware, env_cfg, reward_cfg)
    exact_evals += 1
    if final_score < best_exact:
        best_exact = final_score
        best_metrics = final_metrics

    return TabuResult(
        layout=[int(x) for x in best_layout.tolist()],
        metrics=best_metrics,
        routing_score=float(best_exact),
        surrogate_score=float(best_surrogate),
        num_exact_evals=int(exact_evals),
        num_iters=int(num_iters),
    )

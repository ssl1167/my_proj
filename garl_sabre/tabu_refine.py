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
        if 0 <= a < logic.num_qubits and 0 <= b < logic.num_qubits:
            front_mask[a, b] = front_mask[b, a] = 1.0

    for a, b in logic.critical_edges:
        if 0 <= a < logic.num_qubits and 0 <= b < logic.num_qubits:
            critical_mask[a, b] = critical_mask[b, a] = 1.0

    return front_mask, critical_mask


def _build_edge_list(logic: LogicGraphData) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    n = int(logic.num_qubits)

    for a in range(n):
        for b in range(a + 1, n):
            w = float(logic.edge_weight[a, b])
            if w > 0.0:
                edges.append((a, b, w))

    return edges


def _centrality_score(hardware: HardwareTopology) -> np.ndarray:
    feats = np.asarray(hardware.node_features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] != hardware.num_qubits:
        return np.zeros(hardware.num_qubits, dtype=np.float32)
    use_dim = min(4, feats.shape[1])
    return feats[:, :use_dim].sum(axis=1).astype(np.float32)


def _validate_layout(layout: Sequence[int], logic: LogicGraphData, hardware: HardwareTopology) -> np.ndarray:
    layout_arr = np.asarray(layout, dtype=np.int64).copy()
    n_logic = int(logic.num_qubits)
    n_phys = int(hardware.num_qubits)

    if layout_arr.ndim != 1:
        raise ValueError(f"Layout must be one-dimensional, got shape={layout_arr.shape}.")
    if layout_arr.size != n_logic:
        raise ValueError(
            f"Layout length mismatch: layout has {layout_arr.size} entries, "
            f"but logic graph has {n_logic} logical qubits."
        )
    if n_logic > n_phys:
        raise ValueError(
            f"Logic graph has {n_logic} logical qubits, but hardware has only {n_phys} physical qubits."
        )
    if np.any(layout_arr < 0) or np.any(layout_arr >= n_phys):
        bad = layout_arr[(layout_arr < 0) | (layout_arr >= n_phys)]
        raise ValueError(f"Layout contains invalid physical qubits {bad.tolist()} for hardware size {n_phys}.")
    if len(set(int(x) for x in layout_arr.tolist())) != layout_arr.size:
        raise ValueError(f"Layout must be injective; duplicate physical assignments found in {layout_arr.tolist()}.")

    return layout_arr


def _assert_paper_objective_metrics(metrics: Dict[str, float]) -> None:
    """
    Fail fast if the exact evaluator is not using the paper protocol fields.

    In the revised protocol, tabu refinement must optimize additional CNOTs.
    swap_count is allowed to exist only as a diagnostic field.
    """
    if "additional_cnot_count" not in metrics:
        raise KeyError(
            "evaluate_layout_metrics() did not return 'additional_cnot_count'. "
            "tabu_refine_layout() must be used with the paper-protocol qiskit_runner.py."
        )
    if "paper_additional_cnot_count" in metrics:
        a = float(metrics["additional_cnot_count"])
        b = float(metrics["paper_additional_cnot_count"])
        if abs(a - b) > 1e-6:
            raise ValueError(
                "Metric inconsistency: paper_additional_cnot_count and additional_cnot_count differ "
                f"({b} vs {a})."
            )


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
    """
    Cheap layout-only surrogate used to propose local moves.

    This is not the paper metric and is not reported as a routing result.  The
    exact objective is always obtained from evaluate_layout_metrics() and
    objective_from_metrics(), which must use additional_cnot_count under the
    paper protocol.
    """
    layout_arr = _validate_layout(layout, logic, hardware)

    if edge_list is None:
        edge_list = _build_edge_list(logic)
    if front_mask is None or critical_mask is None:
        front_mask, critical_mask = _build_pair_masks(logic)
    if centrality is None:
        centrality = _centrality_score(hardware)
    if priority is None:
        priority = _incident_priority(logic)

    total = 0.0
    for a, b, w in edge_list:
        d = float(hardware.dist[layout_arr[a], layout_arr[b]])
        emphasis = 1.0 + 0.85 * float(front_mask[a, b]) + 0.55 * float(critical_mask[a, b])
        total += emphasis * w * d

    priority_sum = float(np.sum(priority))
    if priority_sum > 1e-8:
        total -= 0.10 * float(np.sum(priority * centrality[layout_arr]) / (priority_sum + 1e-8))

    return float(total)


def _evaluate_exact(
    layout: Sequence[int],
    circuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    logic: LogicGraphData,
) -> Tuple[Dict[str, float], float]:
    layout_arr = _validate_layout(layout, logic, hardware)
    metrics = evaluate_layout_metrics(circuit, [int(x) for x in layout_arr.tolist()], hardware, env_cfg)
    _assert_paper_objective_metrics(metrics)
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
        centrality = _centrality_score(hardware)
        order = np.argsort(-centrality)
        return [int(x) for x in order[:max_candidates] if int(x) != int(layout[q])]

    scores: List[Tuple[float, int]] = []
    denom = float(np.sum(weights[neighbors])) + 1e-8

    for p in range(hardware.num_qubits):
        if int(p) == int(layout[q]):
            continue
        d = hardware.dist[int(p), layout[neighbors]].astype(np.float32)
        score = float(np.sum(weights[neighbors] * d) / denom)
        scores.append((score, int(p)))

    scores.sort(key=lambda x: x[0])
    return [p for _, p in scores[:max_candidates]]


def _physical_owner(layout: np.ndarray, hardware: HardwareTopology) -> np.ndarray:
    owner = np.full(hardware.num_qubits, -1, dtype=np.int64)
    for q, p in enumerate(layout.tolist()):
        owner[int(p)] = int(q)
    return owner


def _candidate_is_tabu(
    move: Tuple,
    iteration: int,
    candidate_surrogate: float,
    best_surrogate: float,
    tabu_until: Dict[Tuple, int],
) -> bool:
    """
    Surrogate-only aspiration rule.

    The previous implementation updated best_exact inside tabu aspiration without
    updating the matching layout/metrics, which could return mismatched results.
    Here a tabu move is allowed only if it improves the best surrogate; exact
    objective state is updated only where the corresponding layout and metrics
    are also available.
    """
    if tabu_until.get(move, -1) < iteration:
        return False
    return not (candidate_surrogate < best_surrogate - 1e-8)


def _maybe_update_best_exact(
    layout: np.ndarray,
    circuit,
    hardware: HardwareTopology,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    logic: LogicGraphData,
    best_layout: np.ndarray,
    best_metrics: Dict[str, float],
    best_score: float,
) -> Tuple[np.ndarray, Dict[str, float], float, int]:
    metrics, score = _evaluate_exact(layout, circuit, hardware, env_cfg, reward_cfg, logic)
    if score < best_score - 1e-8:
        return layout.copy(), metrics, float(score), 1
    return best_layout, best_metrics, float(best_score), 1


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
    """
    Refine an initial logical-to-physical layout by local search.

    This module does not implement an alternative routing algorithm and does not
    count inserted SWAPs by itself.  Every exact score is delegated to
    evaluate_layout_metrics() and objective_from_metrics(), which, under the
    paper protocol, optimize additional_cnot_count.  The surrogate is used only
    to rank local layout moves before exact evaluation.
    """
    current = _validate_layout(initial_layout, logic, hardware)

    front_mask, critical_mask = _build_pair_masks(logic)
    edge_list = _build_edge_list(logic)
    centrality = _centrality_score(hardware)
    priority = _incident_priority(logic)

    best_surrogate_layout = current.copy()
    best_surrogate = surrogate_layout_score(
        best_surrogate_layout,
        logic,
        hardware,
        edge_list=edge_list,
        front_mask=front_mask,
        critical_mask=critical_mask,
        centrality=centrality,
        priority=priority,
    )

    best_exact_metrics, best_exact_score = _evaluate_exact(
        current,
        circuit,
        hardware,
        env_cfg,
        reward_cfg,
        logic,
    )
    best_exact_layout = current.copy()
    exact_evals = 1

    tabu_until: Dict[Tuple, int] = {}
    completed_iters = 0

    for it in range(1, max(0, int(num_iters)) + 1):
        completed_iters = it

        incident_cost = np.zeros(logic.num_qubits, dtype=np.float32)
        for q in range(logic.num_qubits):
            neighbors = np.where(logic.edge_weight[q] > 0)[0]
            if neighbors.size <= 0:
                continue
            d = hardware.dist[current[q], current[neighbors]].astype(np.float32)
            incident_cost[q] = float(np.sum(logic.edge_weight[q, neighbors] * d))

        rank = np.argsort(-(incident_cost + 0.15 * priority))
        hot_count = max(1, min(int(candidate_qubits), logic.num_qubits))
        hot_qubits = [int(x) for x in rank[:hot_count]]
        phys_owner = _physical_owner(current, hardware)

        best_move: Tuple | None = None
        best_move_surrogate: float | None = None
        best_move_layout: np.ndarray | None = None

        def consider_move(move: Tuple, trial: np.ndarray) -> None:
            nonlocal best_move, best_move_surrogate, best_move_layout

            score = surrogate_layout_score(
                trial,
                logic,
                hardware,
                edge_list=edge_list,
                front_mask=front_mask,
                critical_mask=critical_mask,
                centrality=centrality,
                priority=priority,
            )
            if _candidate_is_tabu(move, it, score, best_surrogate, tabu_until):
                return
            if best_move_surrogate is None or score < best_move_surrogate:
                best_move = move
                best_move_surrogate = float(score)
                best_move_layout = trial

        # 1) Swap two logical qubits' physical positions.  This is a local-search
        # layout move, not an inserted routing SWAP and must not affect gate counts.
        for i in range(len(hot_qubits)):
            for j in range(i + 1, len(hot_qubits)):
                qa, qb = hot_qubits[i], hot_qubits[j]
                trial = current.copy()
                trial[qa], trial[qb] = trial[qb], trial[qa]
                move = ("swap_layout", min(qa, qb), max(qa, qb))
                consider_move(move, trial)

        # 2) Move a hot logical qubit to a promising free physical position, or
        # swap it with the current occupant if the physical position is occupied.
        for q in hot_qubits:
            for p in _local_position_candidates(
                q=q,
                layout=current,
                logic=logic,
                hardware=hardware,
                max_candidates=int(relocate_candidates),
            ):
                owner = int(phys_owner[p])
                if owner < 0:
                    trial = current.copy()
                    trial[q] = int(p)
                    move = ("relocate_layout", int(q), int(p))
                else:
                    if owner == q:
                        continue
                    trial = current.copy()
                    trial[q], trial[owner] = trial[owner], trial[q]
                    move = ("swap_position", min(int(q), owner), max(int(q), owner), int(p))
                consider_move(move, trial)

        if best_move_layout is None or best_move is None:
            break

        current = _validate_layout(best_move_layout, logic, hardware)
        tabu_until[best_move] = it + max(1, int(tabu_tenure))

        if best_move_surrogate is not None and best_move_surrogate < best_surrogate - 1e-8:
            best_surrogate = float(best_move_surrogate)
            best_surrogate_layout = current.copy()

        if exact_eval_every > 0 and it % int(exact_eval_every) == 0:
            updated_layout, updated_metrics, updated_score, inc = _maybe_update_best_exact(
                current,
                circuit,
                hardware,
                env_cfg,
                reward_cfg,
                logic,
                best_exact_layout,
                best_exact_metrics,
                best_exact_score,
            )
            exact_evals += inc
            best_exact_layout = updated_layout
            best_exact_metrics = updated_metrics
            best_exact_score = updated_score

    # Final exact checks.  Evaluate both the current endpoint and the best
    # surrogate layout, because surrogate improvement is not guaranteed to improve
    # the true additional-CNOT objective.
    final_candidates = [current, best_surrogate_layout]
    seen: set[Tuple[int, ...]] = set()
    for candidate in final_candidates:
        key = tuple(int(x) for x in candidate.tolist())
        if key in seen:
            continue
        seen.add(key)
        updated_layout, updated_metrics, updated_score, inc = _maybe_update_best_exact(
            candidate,
            circuit,
            hardware,
            env_cfg,
            reward_cfg,
            logic,
            best_exact_layout,
            best_exact_metrics,
            best_exact_score,
        )
        exact_evals += inc
        best_exact_layout = updated_layout
        best_exact_metrics = updated_metrics
        best_exact_score = updated_score

    best_exact_surrogate = surrogate_layout_score(
        best_exact_layout,
        logic,
        hardware,
        edge_list=edge_list,
        front_mask=front_mask,
        critical_mask=critical_mask,
        centrality=centrality,
        priority=priority,
    )

    return TabuResult(
        layout=[int(x) for x in best_exact_layout.tolist()],
        metrics=best_exact_metrics,
        routing_score=float(best_exact_score),
        surrogate_score=float(best_exact_surrogate),
        num_exact_evals=int(exact_evals),
        num_iters=int(completed_iters),
    )

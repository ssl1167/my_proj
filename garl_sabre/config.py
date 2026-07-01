from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RewardConfig:
    """Reward configuration for the paper/CNOT-active protocol.

    The terminal objective is additional_cnot_count.  Legacy compatibility coefficients
    are retained only so older checkpoints and scripts can still be loaded; they
    are ignored by the current objective_from_metrics implementation.
    """

    alpha_swap: float = 1.0
    beta_depth: float = 0.0
    gamma_twoq: float = 0.0
    eta_added_twoq: float = 0.0
    theta_depth_overhead: float = 0.0

    terminal_scale: float = 20.0
    reward_scale: float = 1.0

    shape_scale: float = 0.0
    frontier_scale: float = 0.0
    critical_scale: float = 0.0
    future_cost_scale: float = 0.0
    local_gate_bonus: float = 0.0
    executable_gate_reward: float = 0.0
    executable_lookahead_reward: float = 0.0
    placement_penalty: float = 0.0
    blocked_gate_penalty: float = 0.0
    blocked_lookahead_penalty: float = 0.0
    completion_bonus: float = 0.0

    use_relative_terminal: bool = False
    depth_norm_weight: float = 0.0
    terminal_clip: float = 5.0

    @classmethod
    def strong(cls) -> "RewardConfig":
        return cls()

    @classmethod
    def light(cls) -> "RewardConfig":
        return cls()


@dataclass
class EnvConfig:
    """Environment and evaluation protocol configuration.

    Defaults are set for the paper-compatible protocol: CNOT-active logical
    circuits, IBM Q20 coupling graph, and additional CNOT as the optimization
    objective.  raw swap_count remains available only as a diagnostic metric.
    """

    critical_window: int = 8
    lookahead_window: int = 16
    use_physical_prior: bool = True

    sabre_seed: int = 7
    optimization_level: int = 0
    basis_gates: List[str] = field(default_factory=lambda: ["cx", "id", "rz", "sx", "x"])
    topology_mode: str = "ibm_q20"
    topology_distance: int = 5

    action_mode: str = "fixed_order_physical"
    logic_order_mode: str = "priority_fixed"

    baseline_mode: str = "dense"
    router_backend: str = "qiskit"

    evaluation_mode: str = "paper_additional_cnot"
    metric_mode: str = "additional_cnot_count"
    benchmark_preprocess: str = "cnot_active_cnot_only"
    metric_protocol: str = "cnot_active_cnot_only_additional_cnot_v1"

    feature_norm_mode: str = "adaptive"
    feature_norm_scale: float = 1.2

    feat_norm_neighbors: float = 15.0
    feat_norm_twoq: float = 100.0
    feat_norm_front: float = 10.0
    feat_norm_early: float = 20.0
    feat_norm_weighted_degree: float = 150.0
    feat_norm_pagerank: float = 1.0
    feat_norm_critical: float = 30.0

    # Legacy compatibility knobs.  They should not control the paper metric.
    paper_two_qubit_only: bool = True

    @classmethod
    def strong(cls) -> "EnvConfig":
        return cls()

    @classmethod
    def light(cls) -> "EnvConfig":
        return cls()




@dataclass
class ModelConfig:
    hidden_dim: int = 128
    graph_layers: int = 3
    dropout: float = 0.1
    placement_dim: int = 64
    physical_prior_scale: float = 0.35
    physical_prior_clip: float = 3.5

    @classmethod
    def strong(cls) -> "ModelConfig":
        return cls(
            hidden_dim=128,
            graph_layers=3,
            dropout=0.1,
            placement_dim=64,
            physical_prior_scale=0.45,
            physical_prior_clip=3.5,
        )

    @classmethod
    def light(cls) -> "ModelConfig":
        return cls(
            hidden_dim=128,
            graph_layers=3,
            dropout=0.1,
            placement_dim=64,
            physical_prior_scale=0.35,
            physical_prior_clip=3.5,
        )


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    train_iters: int = 10
    minibatch_size: int = 64
    target_kl: float = 0.03
    value_clip: float = 0.2

from dataclasses import dataclass, field
from typing import List


@dataclass
class RewardConfig:
    """Swap-centric reward configuration.

    Only terminal swap_count is part of the main optimization target. The extra
    fields are retained strictly for backward checkpoint/script compatibility.
    """

    alpha_swap: float = 1.0
    beta_depth: float = 0.0
    gamma_twoq: float = 0.0
    eta_added_twoq: float = 0.0
    theta_depth_overhead: float = 0.0

    terminal_scale: float = 1.0
    reward_scale: float = 1.0

    # Legacy shaping knobs (kept only for compatibility with older checkpoints).
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
    terminal_clip: float = 0.0

    @classmethod
    def strong(cls) -> "RewardConfig":
        return cls()

    @classmethod
    def light(cls) -> "RewardConfig":
        return cls()


@dataclass
class EnvConfig:
    """Environment / evaluation protocol configuration."""

    critical_window: int = 8
    lookahead_window: int = 16
    candidate_topk: int = 16

    use_candidate_ranking: bool = True
    allow_full_action_space_fallback: bool = True
    use_physical_prior: bool = True

    sabre_seed: int = 7
    optimization_level: int = 0
    basis_gates: List[str] = field(default_factory=lambda: ["cx", "id", "rz", "sx", "x"])
    topology_mode: str = "heavy_hex"
    topology_distance: int = 5

    action_mode: str = "hierarchical"
    logic_order_mode: str = "priority_fixed"
    baseline_mode: str = "dense"

    evaluation_mode: str = "legacy"
    router_backend: str = "qiskit"

    # Legacy field retained for compatibility with older scripts/checkpoints.
    paper_two_qubit_only: bool = True

    routing_backend: str = "sabre"  # 可选控制流: "sabre" 或 "tket"
    baseline_mode: str = "hybrid"   # 建议扩充基线模式

    @classmethod
    def strong(cls) -> "EnvConfig":
        return cls(candidate_topk=20)

    @classmethod
    def light(cls) -> "EnvConfig":
        return cls(candidate_topk=16)

    @classmethod
    def no_ranking(cls) -> "EnvConfig":
        cfg = cls.light()
        cfg.use_candidate_ranking = False
        return cfg


@dataclass
class ModelConfig:
    """Policy/value network configuration."""

    hidden_dim: int = 128
    graph_layers: int = 3
    dropout: float = 0.1
    placement_dim: int = 64
    candidate_hidden_dim: int = 64
    logical_candidate_hidden_dim: int = 64
    physical_prior_scale: float = 0.15
    physical_prior_clip: float = 3.5

    @classmethod
    def strong(cls) -> "ModelConfig":
        # Keep a moderate prior in the 'strong' preset; 0.85 is too aggressive and
        # tends to drag the policy back toward heuristic behavior.
        return cls(
            hidden_dim=128,
            graph_layers=3,
            dropout=0.1,
            placement_dim=64,
            candidate_hidden_dim=64,
            logical_candidate_hidden_dim=64,
            physical_prior_scale=0.30,
            physical_prior_clip=3.5,
        )

    @classmethod
    def light(cls) -> "ModelConfig":
        return cls(
            hidden_dim=128,
            graph_layers=3,
            dropout=0.1,
            placement_dim=64,
            candidate_hidden_dim=64,
            logical_candidate_hidden_dim=64,
            physical_prior_scale=0.15,
            physical_prior_clip=3.5,
        )


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    train_iters: int = 6
    minibatch_size: int = 16
    value_clip: float = 0.2
    target_kl: float = 0.08
    

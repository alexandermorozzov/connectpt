"""Scalar records; logger derives all tags from phases and field names."""
from dataclasses import dataclass, field
from typing import ClassVar
from ..objectives.const import METRIC_TAG_ALIASES


@dataclass
class DataCounts:
    PHASE: ClassVar[str] = "time"
    training_graphs: int | None = None
    validation_graphs: int | None = None


@dataclass
class ValidationRecord:
    PHASE: ClassVar[str] = "eval"
    cost: float
    objective: dict
    penalty: dict
    monitor: dict
    feasible_fraction: float


@dataclass
class ValueRecord:
    PHASE: ClassVar[str] = "rollout"
    value_mean: float
    value: dict


@dataclass
class DiagnosticRecord:
    PHASE: ClassVar[str] = "rollout"
    initial_ratio_error: float
    graph_instances_loaded: int
    graph_batches_loaded: int
    completed_networks: int
    return_target_mean: float
    step_reward: float
    cost: float | None = None
    ATT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['ATT']})
    RTT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['RTT']})
    feasible_fraction: float | None = None
    unserved_fraction: float | None = None
    completed_alpha_mean: float | None = None
    completed_alpha_0_fraction: float | None = None
    completed_alpha_1_fraction: float | None = None
    completed_alpha_intermediate_fraction: float | None = None


@dataclass
class ParetoRecord:
    """Preference-grid evaluation; the caller passes the prefix
    (``pareto/<mode>`` or ``eval/test/<checkpoint>/<mode>``)."""

    PHASE: ClassVar[str] = "pareto"
    ATT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['ATT']})
    RTT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['RTT']})
    feasible_fraction: float | None = None
    mean_unserved_fraction: float | None = None
    hv_penalized: float | None = None
    hv: float | None = None
    hv_reference_coverage: float | None = None
    eu_conditioned: float | None = None
    eu_archive: float | None = None
    eu_archive_feasible: float | None = None
    pareto_points_penalized: float | None = None
    pareto_points_feasible: float | None = None
    spacing_feasible: float | None = None
    distinct_networks: float | None = None
    distinct_feasible_networks: float | None = None
    distinct_pareto_networks: float | None = None
    distinct_networks_total: float | None = None
    distinct_feasible_networks_total: float | None = None
    graph_count: float | None = None
    outcome_count: float | None = None
    reference_0: float | None = None
    reference_1: float | None = None
    evaluation_seconds: float | None = None


@dataclass
class TrainRecord:
    PHASE: ClassVar[str] = "train"
    policy_gradient_loss: float
    value_loss: float
    entropy_loss: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float
    diversity_loss: float
    learning_rate: float
    clip_range: float
    n_updates: int
    loss: float
    diversity_kl: float
    diversity_target: float


@dataclass
class GoalTrainRecord:
    PHASE: ClassVar[str] = "train"
    value_loss: dict
    explained_variance: dict
    advantage_std: dict


@dataclass
class RolloutRecord:
    PHASE: ClassVar[str] = "rollout"
    ep_rew_mean: float
    ep_len_mean: float
    objective: dict
    penalty: dict


@dataclass
class TimeRecord:
    PHASE: ClassVar[str] = "time"
    fps: float
    iterations: int
    time_elapsed: float
    total_timesteps: int

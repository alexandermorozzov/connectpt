"""Training metric records (one dataclass per logging point).

Every current metric keeps its old tag in ``metadata["tag"]`` until Э9, so old
and new runs plot side by side; new metrics use the ``<phase>/<name>`` scheme.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


def _tag(name, sep=None):
    """Legacy tag; ``sep`` joins a dict field's tag and key (``/`` by default)."""
    return field(default=None, metadata={"tag": name, **({"sep": sep} if sep else {})})


@dataclass
class DataCounts:
    """Graph counts of a construction run, logged once at step 0."""
    PHASE: ClassVar[str] = "data"
    training_graphs: int | None = _tag("training graph count")
    validation_graphs: int | None = _tag("validation graph count")


@dataclass
class ValidationRecord:
    """Construction validation pass (mean over the validation graphs)."""
    PHASE: ClassVar[str] = "eval"
    cost: float | None = _tag("val cost")
    att: float | None = _tag("val ATT")
    rtt: float | None = _tag("val RTT")
    d_un: float | None = _tag("val $d_{un}$")
    disconnected_pairs: float | None = _tag("val # disconnected node pairs")
    wmc: float | None = _tag("val median_connectivity_weighted")


@dataclass
class ParetoRecord:
    """Preference-grid evaluation; the caller passes the prefix
    (``pareto/<mode>`` or ``test/<checkpoint>/<mode>``)."""
    PHASE: ClassVar[str] = "pareto"
    att_minutes: float | None = None
    rtt_minutes: float | None = None
    feasible_fraction: float | None = None
    mean_unserved_fraction: float | None = None
    hv_penalized: float | None = None
    hv_feasible: float | None = None
    hv_reference_coverage: float | None = None
    eu_conditioned_penalized: float | None = None
    eu_archive_penalized: float | None = None
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
    reference_att: float | None = None
    reference_rtt: float | None = None
    evaluation_seconds: float | None = None


@dataclass
class ConstructionRolloutRecord:
    """Returns and critic values of one construction rollout."""
    PHASE: ClassVar[str] = "rollout"
    avg_return: float | None = _tag("avg return")
    baseline: float | None = _tag("baseline")
    # D3PO: per objective, keyed ATT / RTT / connectivity
    avg_return_by_objective: dict | None = _tag("avg return", sep=" ")
    baseline_by_objective: dict | None = _tag("baseline", sep=" ")


@dataclass
class ConstructionTrainRecord:
    """One construction training iteration (PPO or D3PO update)."""
    PHASE: ClassVar[str] = "train"
    route_obj: float | None = _tag("route obj")
    objective: float | None = _tag("objective")
    entropy: float | None = _tag("entropy")
    value_loss: float | None = _tag("value loss")
    clip_fraction: float | None = _tag("clip fraction")
    initial_ratio_error: float | None = _tag("initial ratio error")
    diversity_loss: float | None = _tag("diversity loss")
    explained_variance: float | None = _tag("critic/explained_variance")
    # D3PO: per objective, keyed att / rtt / connectivity
    explained_variance_by_objective: dict | None = _tag("critic/explained_variance", sep="_")
    train_seconds: float | None = _tag("train seconds")
    elapsed_seconds: float | None = _tag("elapsed seconds")
    env_transitions: int | None = _tag("env transitions")
    total_timesteps: int | None = field(default=None, metadata={"tag": "time/total_timesteps"})
    graph_instances_loaded: int | None = _tag("graph instances loaded")
    graph_batches_loaded: int | None = _tag("graph batches loaded")
    completed_networks: int | None = _tag("completed networks")
    mean_episode_reward: float | None = _tag("mean episode reward")
    mean_episode_length: float | None = _tag("mean episode length")
    return_target_mean: float | None = _tag("return target mean")
    step_reward: float | None = _tag("train/step_reward")
    cost: float | None = _tag("train/cost")
    att_minutes: float | None = _tag("train/att_minutes")
    rtt_minutes: float | None = _tag("train/rtt_minutes")
    feasible_fraction: float | None = _tag("train/feasible_fraction")
    unserved_fraction: float | None = _tag("train/unserved_fraction")
    completed_alpha_mean: float | None = _tag("train/completed_alpha_mean")
    completed_alpha_0_fraction: float | None = _tag("train/completed_alpha_0_fraction")
    completed_alpha_1_fraction: float | None = _tag("train/completed_alpha_1_fraction")
    completed_alpha_intermediate_fraction: float | None = _tag(
        "train/completed_alpha_intermediate_fraction")


@dataclass
class EditTrainRecord:
    """One edit training iteration (PPO or D3PO)."""
    PHASE: ClassVar[str] = "train"
    seed_cost: float | None = _tag("train_seed_cost")
    final_cost: float | None = _tag("train_final_cost")
    delta: float | None = _tag("train_delta")
    component_demand_delta: float | None = _tag("train_component_demand_delta")
    component_route_delta: float | None = _tag("train_component_route_delta")
    component_connectivity_delta: float | None = _tag("train_component_connectivity_delta")
    reward_mean: float | None = _tag("train_reward_mean")
    reward_sum: float | None = _tag("train_reward_sum")
    reward_per_episode: float | None = _tag("train_reward_per_episode")
    scaled_delta_sum: float | None = _tag("train_scaled_delta_sum")
    reward_delta_residual: float | None = _tag("train_reward_delta_residual")
    episode_count: int | None = _tag("train_episode_count")
    return_mean: float | None = _tag("train_return_mean")
    advantage_mean: float | None = _tag("train_advantage_mean")
    active_sample_count: int | None = _tag("train_active_sample_count")
    ppo_ratio_mean: float | None = _tag("train_ppo_ratio_mean")
    ppo_clip_fraction: float | None = _tag("train_ppo_clip_fraction")
    ppo_objective: float | None = _tag("train_ppo_objective")
    d3po_ratio_mean: float | None = _tag("train_d3po_ratio_mean")
    d3po_clip_fraction: float | None = _tag("train_d3po_clip_fraction")
    d3po_objective: float | None = _tag("train_d3po_objective")
    d3po_diversity_loss: float | None = _tag("train_d3po_diversity_loss")
    critic_mse_mean: float | None = _tag("train_critic_mse_mean")
    critic_explained_variance: float | None = _tag("train_critic_explained_variance")
    # D3PO: per objective, keyed demand / route / connectivity
    return_mean_by_objective: dict | None = _tag("train_return_mean", sep="_")
    advantage_mean_by_objective: dict | None = _tag("train_advantage_mean", sep="_")
    critic_mse_by_objective: dict | None = _tag("train_critic_mse", sep="_")
    critic_explained_variance_by_objective: dict | None = _tag(
        "train_critic_explained_variance", sep="_")
    changed_route_rate: float | None = _tag("train_changed_route_rate")
    val_gap: float | None = _tag("train_val_gap")
    is_eval_iteration: bool | None = _tag("is_eval_iteration")
    # action counts and rates, keyed by action statistic
    action: dict | None = _tag("train_action", sep="_")


@dataclass
class EditValRecord:
    """Edit validation: the last evaluation, carried until the next one."""
    PHASE: ClassVar[str] = "eval"
    seed_cost: float | None = _tag("val_seed_cost")
    final_cost: float | None = _tag("val_final_cost")
    delta: float | None = _tag("val_delta")
    win_rate: float | None = _tag("val_win_rate")
    changed_route_rate: float | None = _tag("val_changed_route_rate")
    changed_graph_rate: float | None = _tag("val_changed_graph_rate")
    component_delta_demand: float | None = _tag("val_component_delta_demand")
    component_delta_route: float | None = _tag("val_component_delta_route")
    component_delta_connectivity: float | None = _tag("val_component_delta_connectivity")

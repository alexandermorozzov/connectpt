"""Typed dataset and training parameters, independent of configuration libraries."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EditDatasetParams:
    dataset_dirname: Path
    target_n_routes: int
    min_route_len: int
    max_route_len: int
    train_fraction: float
    split_seed: int
    n_val_per_tier: int


@dataclass(frozen=True)
class DatasetBuildParams:
    n_graphs: int
    raw_n_nodes: int
    raw_graph_type: str
    raw_graph_seed: int
    lc_n_samples: int
    lc_combos: list
    tiers: list
    force_regen: bool
    batch_size_cpu: int
    batch_size_cuda: int
    corruption_seed: int
    tier_cfg: dict


@dataclass(frozen=True)
class CurriculumParams:
    enabled: bool
    schedule: list


@dataclass(frozen=True)
class OptimizerParams:
    name: str
    learning_rate: float
    decay: float


@dataclass(frozen=True)
class LoopParams:
    n_iterations: int
    horizon: int
    val_period: int
    validate_before_update: bool
    max_seconds: float | None
    batch_size: int
    batch_size_cuda: int
    cpu_threads: int | None
    warmup_batches: int
    max_rollout_samples: int | None
    calibrate_before_validation: bool
    action_seed: int | None
    force_nonhalt_until: int | None
    resample_preferences_on_terminal: bool

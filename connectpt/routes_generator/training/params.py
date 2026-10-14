"""Typed dataset and training parameters, independent of configuration libraries."""
from dataclasses import dataclass
from pathlib import Path
from .const import DatasetKind, DataSourceKind, PreferenceDistribution
from .dataset import DatasetLimits
from .split import SplitParams
from .pipeline import GraphPipeline


@dataclass(frozen=True)
class DatasetInputParams:
    kind: DatasetKind
    source: DataSourceKind
    path: Path | None
    limits: DatasetLimits
    legacy_split: SplitParams
    validation_path: Path | None
    test_path: Path | None


@dataclass(frozen=True)
class PreparedDataInput:
    params: DatasetInputParams
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline
    online_source: object | None
    benchmark: object | None


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
    preference_distribution: PreferenceDistribution

    def __post_init__(self):
        object.__setattr__(self, 'preference_distribution', PreferenceDistribution(self.preference_distribution))

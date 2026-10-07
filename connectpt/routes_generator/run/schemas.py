"""Typed parameters of the config groups (B6).

``Run.setup`` turns a composed group into its dataclass with :func:`params`;
below ``Run.setup`` only these objects travel, never a ``DictConfig``. The
dataclasses have no default values: a missing, extra or renamed YAML key raises
``TypeError`` (a dry run catches it). Groups that differ between run types have
one class per shape. The remaining groups get their classes in their stage:
``cost`` Э4, ``features`` Э5, ``model``/``model_profile`` Э6, ``algorithm``/``env`` Э7,
``method`` Э8.
"""
from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf


def params(cls, node):
    """``cls(**node)`` for a composed config node."""
    return cls(**OmegaConf.to_container(node, resolve=True))


@dataclass(frozen=True)
class RunParams:
    type: str
    name: str
    seed: int


@dataclass(frozen=True)
class TrainingRunParams(RunParams):
    cpu: bool


@dataclass(frozen=True)
class SearchRunParams(RunParams):
    cpu: bool
    silent: bool
    label: str


@dataclass(frozen=True)
class SearchPaths:
    weights_dir: str
    runs_dir: str
    output_dir: str


@dataclass(frozen=True)
class TrainingPaths:
    weights_dir: str
    output_dir: str
    checkpoint_path: str
    init_checkpoint_path: str | None


@dataclass(frozen=True)
class OutputPaths:
    """Runs with one output folder (dataset, comparison)."""
    output_dir: str | None


@dataclass(frozen=True)
class InstanceParams:
    name: str
    path: str
    n_routes: int
    min_route_len: int
    max_route_len: int
    crs: str | None


@dataclass(frozen=True)
class SweepParams:
    """One search point; ``None`` keeps the adjustment value of the cost."""
    alpha: float
    n_iterations: int
    adj_target: float | None
    adj_weight: float | None


@dataclass(frozen=True)
class AdjustmentParams:
    """The ``adjustment/*.yaml`` group; it lands in ``cost.kwargs``."""
    adjustment_degree_weight: float
    adjustment_degree_target: float
    adjustment_degree_objective: str
    adjustment_degree_gap: float
    adjustment_degree_mode: str

    @classmethod
    def from_cost(cls, cost_kwargs):
        return cls(**{k: v for k, v in OmegaConf.to_container(cost_kwargs, resolve=True).items()
                      if k.startswith("adjustment_degree_")})


@dataclass(frozen=True)
class EditDatasetParams:
    dataset_dirname: str
    target_n_routes: int
    min_route_len: int
    max_route_len: int
    train_fraction: float
    split_seed: int


@dataclass(frozen=True)
class ConstructionDatasetParams:
    type: str
    kwargs: dict
    test_path: str


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


@dataclass(frozen=True)
class ComparisonParams:
    """Runs under ``<paths.output_dir>/runs/<name>``, compared on one test grid."""
    names: list
    checkpoint: str
    mode: str
    budget_mode: str


@dataclass(frozen=True)
class CurriculumParams:
    tiers: list
    n_val_per_tier: int
    schedule: list


def logger_names(node) -> list[str]:
    """The ``logger_list`` key: names from ``run.logger.LOGGERS``."""
    from .logger import LOGGERS

    names = list(node)
    unknown = [n for n in names if n not in LOGGERS]
    if unknown:
        raise TypeError(f"unknown loggers {unknown}; known: {sorted(LOGGERS)}")
    return names

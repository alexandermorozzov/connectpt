"""Typed configuration groups owned by run. params resolves them at setup."""
from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf
from .logger import LOGGERS


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
    """Dataset preparation output folder."""
    output_dir: str | None
    weights_dir: str


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

    @classmethod
    def from_cost(cls, cost_kwargs):
        values = OmegaConf.to_container(cost_kwargs, resolve=True)
        return cls(adjustment_degree_weight=values["penalties"]["adjustment"],
                   **{k: v for k, v in values.items() if k.startswith("adjustment_degree_")})


@dataclass(frozen=True)
class InitialParams:
    """``cfg/initial``: heuristic initial networks (John et al.)."""
    alpha: float
    prioritize_direct_connections: bool


def logger_names(node) -> list[str]:
    """The ``logger_list`` key: names from ``run.logger.LOGGERS``."""

    names = list(node)
    unknown = [n for n in names if n not in LOGGERS]
    if unknown:
        raise TypeError(f"unknown loggers {unknown}; known: {sorted(LOGGERS)}")
    return names

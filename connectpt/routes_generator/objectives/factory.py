"""Build cost modules from the cost config.

``cfg/cost/<name>.yaml`` is the single source of truth for
the cost: ``type`` selects the module class, ``kwargs`` are its constructor
arguments (weights, connectivity mode, disabled components, adjustment penalty).
"""
from __future__ import annotations

from functools import lru_cache

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..transit_time_estimator import get_cost_module_from_cfg

CFG_DIR = Path(__file__).resolve().parents[1] / "cfg"
COST_DIR = CFG_DIR / "cost"
ADJUSTMENT_DIR = CFG_DIR / "adjustment"

DEFAULT_COST = "rtt_wmc"
DEFAULT_ADJUSTMENT = "cap"


class CostFactory:
    """Construct cost modules from cost configs."""

    @staticmethod
    def load_cost_config(name: str = DEFAULT_COST,
                         adjustment: str = DEFAULT_ADJUSTMENT) -> DictConfig:
        """Load a cost and merge one adjustment preset into ``kwargs``."""
        path = COST_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
        if not path.exists():
            raise FileNotFoundError(f"cost config not found: {path}")
        cfg = OmegaConf.load(path)
        adj_path = ADJUSTMENT_DIR / (
            adjustment if adjustment.endswith(".yaml") else f"{adjustment}.yaml")
        if not adj_path.exists():
            raise FileNotFoundError(f"adjustment config not found: {adj_path}")
        return OmegaConf.merge(cfg, {"kwargs": OmegaConf.load(adj_path)})

    @staticmethod
    def build(cost_cfg, low_memory_mode: bool = False, symmetric_routes: bool = True):
        """Build a cost module from an ``experiment.cost_function`` node."""
        return get_cost_module_from_cfg(cost_cfg, low_memory_mode, symmetric_routes)

    @staticmethod
    def build_cost(name: str = DEFAULT_COST, *, adjustment: str = DEFAULT_ADJUSTMENT,
                   symmetric_routes: bool = True, low_memory_mode: bool = False):
        """Load the cost config by name and build the module from it."""
        return CostFactory.build(CostFactory.load_cost_config(name, adjustment),
                                 low_memory_mode, symmetric_routes)


@lru_cache(maxsize=None)
def cost_kwargs(name: str = DEFAULT_COST, adjustment: str = DEFAULT_ADJUSTMENT):
    """The cost config's ``kwargs`` node -- the single source for the weights,
    the connectivity mode and the adjustment-degree penalty. Read-only: it is
    cached, so callers copy before mutating."""
    return CostFactory.load_cost_config(name, adjustment).kwargs


def adjustment_kwargs(name: str = DEFAULT_COST,
                      adjustment: str = DEFAULT_ADJUSTMENT) -> dict:
    """The ``adjustment_degree_*`` block, as kwargs for the BCO loop."""
    return {k: v for k, v in cost_kwargs(name, adjustment).items()
            if k.startswith("adjustment_degree_")}

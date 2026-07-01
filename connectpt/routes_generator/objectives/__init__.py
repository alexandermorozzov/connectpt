"""Objective / cost construction for route-generation experiments.

The unified paper objective lives in ``cfg/objective/*.yaml`` (the single source
of truth). :class:`CostFactory` loads those YAMLs and builds / configures the
existing :class:`MyCostModule` so training, search and evaluation all optimize
the same cost.
"""

from .factory import CostFactory
from .unified import (UnifiedObjective, load_bco_algo_config,
                      load_unified_objective)


__all__ = [
    "CostFactory",
    "UnifiedObjective",
    "load_unified_objective",
    "load_bco_algo_config",
]

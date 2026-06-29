"""Objective / cost construction for route-generation experiments.

The unified paper objective lives in ``cfg/objective/*.yaml`` (the single source
of truth). :class:`CostFactory` loads those YAMLs and builds / configures the
existing :class:`MyCostModule` so training, search and evaluation all optimize
the same cost.
"""

from .factory import CostFactory


__all__ = ["CostFactory"]

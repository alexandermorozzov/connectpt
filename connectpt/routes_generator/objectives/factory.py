"""Build and configure cost modules from the objective YAMLs.

The objective YAMLs (``cfg/objective/*.yaml``) are the single source of truth
for *what* the cost is (weights, connectivity mode, adjustment penalty). This
factory:

* delegates raw cost-module construction to the existing
  :func:`get_cost_module_from_cfg` (no behaviour change -- the v1 path);
* applies an objective YAML onto an already-built cost module, centralising the
  weight / connectivity / adjustment / disabled-component assignment that the
  notebook used to do by hand.

It deliberately does NOT own any objective constants -- it reads them from YAML.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..transit_time_estimator import get_cost_module_from_cfg

CFG_DIR = Path(__file__).resolve().parents[1] / "cfg"
OBJECTIVE_DIR = CFG_DIR / "objective"


class CostFactory:
    """Construct / configure cost modules from objective configs."""

    @staticmethod
    def load_objective(name: str) -> DictConfig:
        """Load ``cfg/objective/<name>.yaml`` as an OmegaConf node."""
        path = OBJECTIVE_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
        if not path.exists():
            raise FileNotFoundError(f"objective config not found: {path}")
        return OmegaConf.load(path)

    @staticmethod
    def build(cost_cfg, low_memory_mode: bool = False, symmetric_routes: bool = True):
        """Build a cost module from an ``experiment.cost_function`` node.

        Thin delegate over the existing construction path; kept so callers depend
        on ``CostFactory`` rather than reaching into ``transit_time_estimator``.
        """
        return get_cost_module_from_cfg(cost_cfg, low_memory_mode, symmetric_routes)

    @staticmethod
    def apply_objective(cost_obj, objective, *, for_training: bool = False):
        """Configure ``cost_obj`` in place from an objective config.

        ``objective`` is either an objective name (str) or a loaded objective
        node. Sets the cost weights, connectivity mode, disabled components and
        the adjustment-degree penalty. ``for_training`` selects the one-sided
        ``cap`` adjustment objective used for PPO reward shaping; otherwise the
        two-sided ``target`` objective used for search/eval acceptance is used.
        """
        if isinstance(objective, str):
            objective = CostFactory.load_objective(objective)

        weights = objective.weights
        cost_obj.demand_time_weight = float(weights.demand_time_weight)
        cost_obj.route_time_weight = float(weights.route_time_weight)
        cost_obj.median_connectivity_weight = float(weights.median_connectivity_weight)
        cost_obj.connectivity_mode = str(objective.connectivity_mode)

        disabled = list(objective.get("disabled_components", []) or [])
        cost_obj.set_enabled_components(disabled_components=disabled)

        adj = objective.adjustment
        cost_obj.adjustment_degree_weight = float(adj.weight)
        cost_obj.adjustment_degree_target = float(adj.target)
        cost_obj.adjustment_degree_objective = str(
            adj.train_objective if for_training else adj.objective
        )
        cost_obj.adjustment_degree_gap = float(adj.gap)
        cost_obj.adjustment_degree_mode = str(adj.mode)
        return cost_obj

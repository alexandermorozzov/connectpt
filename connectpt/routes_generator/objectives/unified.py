"""Read the unified paper objective as plain Python values.

The objective YAML (``cfg/objective/rtt_wmc_no_demand.yaml``) is the single
source of truth for the cost objective (see :class:`CostFactory`). This module
exposes those values as a small frozen dataclass so notebook / eval_lib code can
read the weights, connectivity mode and the adjustment-penalty kwargs WITHOUT a
pile of module-level ``CONST = _cfg.field`` constants (the old
``eval_lib.params``). It reads the YAML through :class:`CostFactory`; it owns no
objective values itself.

The BCO *algorithm* config (bee counts, worse-accept schedule, ...) is a
separate concern and lives in ``cfg/bco_mumford.yaml``; :func:`load_bco_algo_config`
reads it on demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from omegaconf import OmegaConf

from ..core.paths import CFG_DIR
from .factory import CostFactory


@dataclass(frozen=True)
class UnifiedObjective:
    """The unified paper objective, read from ``cfg/objective/<name>.yaml``."""

    name: str
    connectivity_mode: str
    disabled_components: tuple[str, ...]
    demand_time_weight: float
    route_time_weight: float
    median_connectivity_weight: float
    adj_weight: float
    adj_target: float
    adj_objective: str        # eval / BCO search acceptance (two-sided target)
    adj_train_objective: str  # PPO reward shaping (one-sided cap)
    adj_gap: float
    adj_mode: str

    @property
    def weights(self) -> dict:
        """Cost weights as an ``experiment.cost_function.kwargs`` sub-dict."""
        return {
            "demand_time_weight": self.demand_time_weight,
            "route_time_weight": self.route_time_weight,
            "median_connectivity_weight": self.median_connectivity_weight,
        }

    @property
    def adj_kwargs(self) -> dict:
        """Two-sided adjustment kwargs every unified-objective search/eval run
        passes to the BCO loop (formerly ``eval_lib.paper.UNIFIED_ADJ``)."""
        return {
            "adjustment_degree_weight": self.adj_weight,
            "adjustment_degree_target": self.adj_target,
            "adjustment_degree_objective": self.adj_objective,
            "adjustment_degree_gap": self.adj_gap,
            "adjustment_degree_mode": self.adj_mode,
        }


@lru_cache(maxsize=None)
def load_unified_objective(name: str = "rtt_wmc_no_demand") -> UnifiedObjective:
    """Load the unified objective values from ``cfg/objective/<name>.yaml``."""
    obj = CostFactory.load_objective(name)
    weights = obj.weights
    adj = obj.adjustment
    return UnifiedObjective(
        name=name,
        connectivity_mode=str(obj.connectivity_mode),
        disabled_components=tuple(obj.get("disabled_components", []) or []),
        demand_time_weight=float(weights.demand_time_weight),
        route_time_weight=float(weights.route_time_weight),
        median_connectivity_weight=float(weights.median_connectivity_weight),
        adj_weight=float(adj.weight),
        adj_target=float(adj.target),
        adj_objective=str(adj.objective),
        adj_train_objective=str(adj.train_objective),
        adj_gap=float(adj.gap),
        adj_mode=str(adj.mode),
    )


@lru_cache(maxsize=1)
def load_bco_algo_config():
    """The BCO algorithm config (``cfg/bco_mumford.yaml``), loaded once.

    Consumers read fields off the returned OmegaConf node
    (``load_bco_algo_config().n_bees``) -- the search knobs the old
    ``build_bco_cfg`` reader resolved (bee counts, worse-accept schedule,
    ``ignore_type*_max_route_len``, ...).
    """
    return OmegaConf.load(CFG_DIR / "bco_mumford.yaml")

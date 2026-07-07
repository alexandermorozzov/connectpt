"""Legacy ``n_type1..n_type7`` taxonomy compatibility surface.

Isolated here so the engine, the invocation layer and the declarative bee-set
path carry no per-type taxonomy of their own. New code should not import from
this package -- bee sets are declarative specs
(:meth:`~connectpt.routes_generator.search.executable_plan.ExecutablePlan.from_specs`).
"""
from .bco_config import (
    apply_disabled_components,
    compose_bco_cfg,
    safe_run_name,
)
from .plans import bee_colony, plan_from_counts, plan_from_flat_cfg

__all__ = [
    "apply_disabled_components",
    "bee_colony",
    "compose_bco_cfg",
    "plan_from_counts",
    "plan_from_flat_cfg",
    "safe_run_name",
]

"""Cost construction for route-generation experiments.

The cost config (``cfg/cost/*.yaml`` plus ``cfg/adjustment/*.yaml``) is the source of
truth; :class:`CostFactory` builds :class:`MyCostModule` from it, so training,
search and evaluation optimize the same cost. :func:`cost_kwargs` exposes the
same values to callers that need the numbers without building a module.
"""

from .factory import CostFactory, adjustment_kwargs, cost_kwargs


__all__ = [
    "CostFactory",
    "cost_kwargs",
    "adjustment_kwargs",
]

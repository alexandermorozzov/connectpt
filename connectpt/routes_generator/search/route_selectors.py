"""Route selectors -- which route a bee mutates.

Named strategy objects matching the search config vocabulary
(``route_selection``). They are intentionally light: the BCO runner uses the
*name* to drive the existing bee_colony route-picking, and these objects make
the vocabulary explicit and validated.
"""
from __future__ import annotations


class RouteSelector:
    name = "base"

    def select(self, state, context):  # pragma: no cover - overridden
        raise NotImplementedError


class UniformRouteSelector(RouteSelector):
    name = "uniform"


class LowDemandRouteSelector(RouteSelector):
    """Prefer routes carrying little demand (good trim candidates)."""

    name = "low_demand"


class SameRouteSelector(RouteSelector):
    """Reuse the route the previous compound step touched."""

    name = "same_route"


_SELECTORS = {c.name: c for c in (UniformRouteSelector, LowDemandRouteSelector,
                                  SameRouteSelector)}


def get_route_selector(name: str) -> RouteSelector:
    try:
        return _SELECTORS[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown route_selection {name!r}; known: {sorted(_SELECTORS)}"
        ) from exc

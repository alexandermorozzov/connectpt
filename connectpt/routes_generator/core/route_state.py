"""The shared batch state -- a stable import surface.

``RouteGenBatchState`` is the central state object used identically by training
and search (it holds the graph batch, current/finished routes and the cost
weights). It is defined in ``transit_time_estimator``; re-export it here so the
training / search / evaluation layers import it from ``core`` rather than from
the large estimator module.
"""
from __future__ import annotations

from ..transit_time_estimator import RouteGenBatchState

__all__ = ["RouteGenBatchState"]

"""The one alpha x adj_target sweep driver (grid iteration + results table).

An experiment sweep is: for each method, for each ``alpha`` (RTT/WMC trade-off)
and each ``adj_target`` (adjustment-degree target), run ONE seeded search from
the shared initial network and score the resulting routes. This module owns that
grid loop + the results-table assembly once; the per-point *execution* (which
engine runs the search) is injected via ``run_point`` so both the library run
(``BeeColonySearchRun``) and the notebook-side runner share this driver instead
of each re-implementing the double loop.

``run_point(label, method, alpha, adj_target) -> (routes, engine_metrics)`` and
``score(engine_metrics, routes) -> dict`` are the two seams. ``methods`` is an
iterable of ``(label, method_obj)``; a single-method sweep passes one pair.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd


def run_sweep_table(
    *,
    methods: Iterable[tuple[str, Any]],
    alpha_grid: Iterable,
    adj_targets: Iterable,
    init_routes: Any,
    run_point: Callable[[str, Any, Any, Any], tuple[Any, Any]],
    score: Callable[[Any, Any], dict],
    extra: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the (method x alpha x adj_target) grid into a table + routes map.

    Returns ``(DataFrame, routes)`` where ``routes`` maps ``"Initial"`` and one
    ``"<label> a=<alpha> t=<adj_target>"`` key per point to its route set, and
    each table row is ``score(...) + {method, alpha, adj_target, **extra}``.
    """
    extra = dict(extra or {})
    rows, routes = [], {"Initial": init_routes}
    for label, method in methods:
        for alpha in alpha_grid:
            for adj_target in adj_targets:
                out_routes, m = run_point(label, method, alpha, adj_target)
                row = dict(score(m, out_routes))
                row.update(method=label, alpha=alpha, adj_target=adj_target, **extra)
                rows.append(row)
                routes[f"{label} a={alpha} t={adj_target}"] = out_routes
    return pd.DataFrame(rows), routes

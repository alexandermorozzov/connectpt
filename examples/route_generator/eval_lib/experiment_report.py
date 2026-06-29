"""Unified experiment report: one renderer for every experiment's output.

Given an :class:`ExperimentResult`, build the results table + the figure the
spec asks for (``report.routes_plot``: ``pareto`` for an RTT x WMC front, or
``network`` / ``gis`` for route-set panels). All plotting goes through the
atomic functions in :mod:`eval_lib.viz` so every experiment renders in one
style; this module only selects which atomic call to make.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import viz


@dataclass
class ReportArtifact:
    table: pd.DataFrame
    figures: dict = field(default_factory=dict)


def render_report(result, *, max_route_panels: int = 4) -> ReportArtifact:
    """Build the table + the spec-selected figure for an experiment result."""
    report_cfg = {}
    if result.spec is not None:
        report_cfg = dict(result.spec.get("report", {}) or {})
    kind = report_cfg.get("routes_plot", "network")

    name = getattr(result, "name", "experiment")
    figures = {}
    if kind == "pareto":
        figures["pareto"] = viz.plot_pareto(
            result.table, title=f"{name}: RTT x WMC front")
    else:  # "network" / "gis" -- route panels (coords + street_adj)
        inst = result.instance
        figures["routes"] = viz.plot_routes_grid(
            result.routes, inst.coords, inst.street_adj,
            title=inst.label, max_panels=max_route_panels)
    return ReportArtifact(table=viz.style_table(result.table), figures=figures)

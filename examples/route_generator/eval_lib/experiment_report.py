"""Unified experiment report: one renderer for every experiment's output.

Given an :class:`ExperimentResult`, build the results table + the figure the
spec asks for (``report.routes_plot``: ``pareto`` for an RTT x WMC front, or
``network`` / ``gis`` for route-set panels). The same renderer serves EKB,
MACSA and benchmark because they all carry coords + street_adj on the loaded
instance -- so the notebook stops hand-rolling per-experiment plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .plots import plot_plain_route_set


@dataclass
class ReportArtifact:
    table: pd.DataFrame
    figures: dict = field(default_factory=dict)


def _pareto_fig(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    if {"RTT", "WMC"} <= set(table.columns):
        ax.plot(table["RTT"], table["WMC"], "o-")
        for _, r in table.iterrows():
            ax.annotate(f"a={r.get('alpha')}", (r["RTT"], r["WMC"]),
                        fontsize="small")
    ax.set_xlabel("RTT")
    ax.set_ylabel("WMC")
    ax.set_title("Pareto front: RTT x WMC")
    return fig


def _routes_fig(result, max_panels: int):
    import matplotlib.pyplot as plt
    inst = result.instance
    items = list(result.routes.items())[:max_panels]
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
    if len(items) == 1:
        axes = [axes]
    for ax, (label, routes) in zip(axes, items):
        plot_plain_route_set(ax, routes, inst.coords, inst.street_adj)
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(inst.label)
    return fig


def render_report(result, *, max_route_panels: int = 4) -> ReportArtifact:
    """Build the table + the spec-selected figure for an experiment result."""
    report_cfg = {}
    if result.spec is not None:
        report_cfg = dict(result.spec.get("report", {}) or {})
    kind = report_cfg.get("routes_plot", "network")

    figures: dict[str, Any] = {}
    if kind == "pareto":
        figures["pareto"] = _pareto_fig(result.table)
    else:  # "network" / "gis" -- route panels (coords + street_adj)
        figures["routes"] = _routes_fig(result, max_route_panels)
    return ReportArtifact(table=result.table, figures=figures)

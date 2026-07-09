"""Atomic, uniform visualization for every experiment.

Each experiment renders the same kinds of artifacts -- route-set panels, an
RTT x WMC front, a styled metrics table -- differing only in labels/titles.
These atomic functions render them in ONE consistent style so neither the
notebook nor render_report hand-rolls plotting. New experiments call these; they
never copy a plotting block.
"""
from __future__ import annotations

import math
from typing import Mapping

from .route_plots import plot_demand_graph, plot_plain_route_set, plot_route_diff

# One shared style for every figure/table.
STYLE = dict(
    figsize_per_panel=5.0,
    node_size=8,
    title_fontsize=10,
    suptitle_fontsize=13,
    annotate_fontsize="x-small",
    line="o-",
    palette="tab20",
)


def plot_routes_grid(route_sets: Mapping, coords, street_adj, *, ncols: int = 4,
                     title: str | None = None, diff_against: str | None = None,
                     diff_ref_routes=None, demand=None,
                     max_panels: int | None = None,
                     subtitles: Mapping | None = None,
                     node_label_offset: int = 0, **panel_kwargs):
    """Grid of route-set panels (the one route plotter for EKB/MACSA/benchmark).

    ``route_sets`` maps a label -> route tensor; ``diff_against`` (a label) draws
    every other panel as a diff vs that reference instead of a plain set;
    ``diff_ref_routes`` (a route tensor) does the same against an external
    reference that is not itself a panel.
    ``demand`` (an OD matrix) prepends an OD-demand panel; ``subtitles`` maps a
    panel label -> subtitle string (e.g. its metric row); ``node_label_offset``
    shifts the drawn node ids (MACSA figures are 1-indexed).
    ``panel_kwargs`` (e.g. ``node_size`` / ``palette`` / ``with_overlap_curves``
    / ``show_node_labels``) are forwarded to each panel plotter -- the geo/GIS
    render path uses them to style panels over a street underlay.
    """
    import matplotlib.pyplot as plt

    items = list(route_sets.items())
    if max_panels is not None:
        items = items[:max_panels]
    subtitles = subtitles or {}
    n = len(items) + (1 if demand is not None else 0)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fp = STYLE["figsize_per_panel"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fp * ncols, fp * nrows),
                             squeeze=False)
    flat = list(axes.flat)

    def _relabel(ax):
        if not node_label_offset:
            return
        for txt in ax.texts:
            value = txt.get_text()
            if value.isdigit():
                txt.set_text(str(int(value) + node_label_offset))

    if demand is not None:
        plot_demand_graph(flat[0], demand, coords, street_adj, title="OD demand",
                          subtitle="edge color/width = demand")
        _relabel(flat[0])
        flat = flat[1:]
    for ax, (label, routes) in zip(flat, items):
        ref = (route_sets.get(diff_against) if diff_against is not None
               else diff_ref_routes)
        if ref is not None and label != diff_against:
            plot_route_diff(ax, routes, ref, coords, street_adj,
                            **panel_kwargs)
        else:
            plot_plain_route_set(ax, routes, coords, street_adj, **panel_kwargs)
        subtitle = subtitles.get(label)
        ax.set_title(f"{label}\n{subtitle}" if subtitle else str(label),
                     fontsize=STYLE["title_fontsize"])
        _relabel(ax)
        ax.set_axis_off()
    for ax in flat[len(items):]:
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=STYLE["suptitle_fontsize"], fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pareto(df, *, x: str = "RTT", y: str = "WMC", hue: str = "method",
                title: str | None = None, annotate: str | None = "alpha"):
    """RTT x WMC (or any x/y) front, one line per ``hue`` value."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = df.groupby(hue) if hue in df.columns else [(None, df)]
    for key, grp in groups:
        grp = grp.sort_values(x) if x in grp.columns else grp
        if {x, y} <= set(grp.columns):
            ax.plot(grp[x], grp[y], STYLE["line"], label=str(key))
            if annotate and annotate in grp.columns:
                for _, r in grp.iterrows():
                    ax.annotate(f"{r[annotate]:g}", (r[x], r[y]),
                                fontsize=STYLE["annotate_fontsize"])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if hue in df.columns:
        ax.legend(fontsize="small")
    ax.set_title(title or f"{x} x {y} front")
    return fig


def style_table(df, *, ndigits: int = 3):
    """Round + return a table for uniform display() in the notebook."""
    return df.round(ndigits)

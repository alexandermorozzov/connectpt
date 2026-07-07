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

from .route_plots import plot_plain_route_set, plot_route_diff

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
                     demand=None, max_panels: int | None = None):
    """Grid of route-set panels (the one route plotter for EKB/MACSA/benchmark).

    ``route_sets`` maps a label -> route tensor; ``diff_against`` (a label) draws
    every other panel as a diff vs that reference instead of a plain set.
    """
    import matplotlib.pyplot as plt

    items = list(route_sets.items())
    if max_panels is not None:
        items = items[:max_panels]
    n = len(items)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fp = STYLE["figsize_per_panel"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fp * ncols, fp * nrows),
                             squeeze=False)
    flat = axes.flat
    for ax, (label, routes) in zip(flat, items):
        if diff_against is not None and label != diff_against and diff_against in route_sets:
            plot_route_diff(ax, routes, route_sets[diff_against], coords, street_adj)
        else:
            plot_plain_route_set(ax, routes, coords, street_adj)
        ax.set_title(str(label), fontsize=STYLE["title_fontsize"])
        ax.set_axis_off()
    for ax in list(flat)[n:]:
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

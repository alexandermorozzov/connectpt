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

from .route_plots import (GEO_STREET_STYLE, get_first_route_set, large_palette,
                          plot_changed_route_slots, plot_demand_graph,
                          plot_plain_route_set, plot_route_adj_gradient,
                          plot_route_diff, route_colors_for,
                          summarize_route_changes)

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

# Geo (map-underlay) panels need their own scale: a real city has hundreds of
# stops, so nodes shrink and fade, labels go away, and the street graph darkens
# to stay visible over the basemap.
GEO_STYLE = dict(
    # stops are context, not the message: small and quiet, or 700 of them turn
    # the city centre into a field of dots competing with the routes.
    node_size=6,
    node_alpha=0.45,
    with_overlap_curves=True,
    street_style=GEO_STREET_STYLE,
)
GEO_PANEL_SIZE = 15.0     # inches per panel -- these are poster-sized figures
GEO_TITLE_FONTSIZE = 15
GEO_LEGEND_ROWS = 6       # route legend shape: columns follow from the route count
GEO_LEGEND_MAX_COLS = 14


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


def metric_line(row, *, include_adj: bool = True) -> str:
    """Two-line metric caption for a route panel (objective row, then demand split).

    ``row`` is one results-table record (a dict or a pandas Series). Missing or
    non-numeric fields are simply skipped, so the same helper captions an Initial
    network (no adjustment yet) and a search result.
    """
    row = dict(row) if row is not None else {}

    def value(*keys):
        for key in keys:
            if key not in row:
                continue
            raw = row[key]
            if raw is None or raw == "":
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if number != number:      # NaN
                continue
            return number
        return None

    head = []
    wmc = value("WMC", "median_connectivity_weighted", "WMC_median")
    rtt = value("RTT")
    adj = value("adj_vs_seed", "adj")
    if wmc is not None:
        head.append(f"WMC={wmc:.2f}")
    if rtt is not None:
        head.append(f"RTT={rtt:.0f}")
    if include_adj and adj is not None:
        head.append(f"adj={adj:.2f}")

    tail = []
    for label, keys in (("d0", ("d0", "$d_0$")), ("d1", ("d1", "$d_1$")),
                        ("d2", ("d2", "$d_2$")), ("dun", ("d_un", "$d_{un}$"))):
        number = value(*keys)
        if number is not None:
            tail.append(f"{label}={number:.2f}")

    return "\n".join(part for part in ("  ".join(head), "  ".join(tail)) if part)


def plot_geo_map_panels(routes, coords, street_adj, *, reference_routes=None,
                        basemap=None, extent=None, title=None, label="candidate",
                        metrics=None, reference_metrics=None,
                        reference_title="Current / initial routes"):
    """The geo case-study figure: route panels over a map basemap.

    With a ``reference_routes`` network this is the paper's 2x2 read of one
    result -- initial | changed | which route slots changed | how much each route
    changed (adjustment gradient with a colorbar). Without one it degrades to a
    single panel of ``routes``, which is how a seed network alone is shown.

    ``coords`` must already be in the basemap's projection (EPSG:3857 via
    ``geo.project_coords_3857``); ``basemap``/``extent`` come from
    ``reports.basemap`` and may be ``None`` -- the panels then render over a
    plain background at their own data limits.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from .basemap import add_map_underlay, apply_panel_limits
    from .geo import route_adjustments

    n_routes = int(get_first_route_set(routes).shape[0])
    palette = large_palette(max(n_routes, 1))
    colored = dict(GEO_STYLE, palette=palette)
    plain = dict(colored, show_node_labels=False)
    single = reference_routes is None

    nrows, ncols = (1, 1) if single else (2, 2)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                             figsize=(GEO_PANEL_SIZE * ncols, GEO_PANEL_SIZE * nrows))
    flat = list(axes.ravel())
    for ax in flat:
        add_map_underlay(ax, basemap)

    if single:
        plot_plain_route_set(flat[0], routes, coords, street_adj,
                             title=reference_title,
                             subtitle=metric_line(metrics, include_adj=False),
                             **plain)
    else:
        plot_plain_route_set(
            flat[0], reference_routes, coords, street_adj, title=reference_title,
            subtitle=metric_line(reference_metrics, include_adj=False), **plain)
        plot_plain_route_set(
            flat[1], routes, coords, street_adj, title=f"Changed routes: {label}",
            subtitle=metric_line(metrics), **plain)

        changes = summarize_route_changes(routes, reference_routes)
        plot_changed_route_slots(
            flat[2], routes, reference_routes, coords, street_adj,
            title=f"Changed route slots highlighted: {label}",
            subtitle=(f"changed_routes={changes['changed_routes']}  "
                      f"+edges={changes['added_edges']}  "
                      f"-edges={changes['removed_edges']}  "
                      f"+stops={changes['added_stops']}  "
                      f"-stops={changes['removed_stops']}"),
            **colored)

        # adjustment is computed once so the subtitle can quote the mean the
        # gradient is drawn from -- no second, possibly divergent, calculation.
        adj_values = route_adjustments(routes, reference_routes)
        scalar, _ = plot_route_adj_gradient(
            flat[3], routes, reference_routes, coords, street_adj,
            title=f"Route-wise adj gradient: {label}",
            subtitle=("gray=0, yellow->purple=increasing adj, "
                      f"mean adj={float(np.mean(adj_values)):.3f}"),
            adj_values=adj_values, **GEO_STYLE)
        # outside the axes (constrained_layout reserves the strip) so the bar
        # never sits on top of the routes it describes.
        cbar = fig.colorbar(scalar, ax=flat[3], location="left", fraction=0.035,
                            pad=0.02, shrink=0.85)
        cbar.set_label("route adj", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    for ax in flat:
        if extent is not None:
            apply_panel_limits(ax, extent)
        else:
            ax.set_axis_off()

    add_route_legend(fig, routes, palette)
    if title:
        fig.suptitle(title, fontsize=GEO_TITLE_FONTSIZE, fontweight="bold")
    return fig


def add_route_legend(fig, routes, palette, *, label_offset: int = 1):
    """Colour key for the route panels, laid out under the whole figure.

    Panels 1-3 colour edges by route identity, and with dozens of routes that
    mapping is unreadable without a key. The legend is placed outside the axes
    (constrained_layout reserves the band), so it costs no map area.
    ``label_offset`` shifts the printed ids -- 1 by default, since a figure reads
    as "route 1..N" rather than from zero.
    """
    from matplotlib.lines import Line2D

    colors = route_colors_for(routes, palette=palette)
    if not len(colors):
        return
    handles = [Line2D([0], [0], color=color, linewidth=4,
                      label=str(idx + label_offset))
               for idx, color in enumerate(colors)]
    n_routes = len(handles)
    ncol = min(GEO_LEGEND_MAX_COLS,
               max(4, math.ceil(n_routes / GEO_LEGEND_ROWS)))
    nrow = math.ceil(n_routes / ncol)
    # matplotlib fills a legend column by column and leaves the last column
    # short, so the block comes out ragged. Padding up to a full nrow x ncol
    # grid with invisible entries makes every row hold exactly ``ncol`` routes.
    handles += [Line2D([0], [0], color="none", label=" ")
                for _ in range(nrow * ncol - n_routes)]
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol,
               frameon=True, framealpha=1.0, edgecolor="#c8ccd0",
               fontsize=10, handlelength=1.8, columnspacing=1.4,
               labelspacing=0.5, borderpad=0.9,
               title=f"Routes {label_offset}-{n_routes + label_offset - 1}"
                     " (panel colour = route id)",
               title_fontsize=12)


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

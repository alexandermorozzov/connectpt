"""Unified route-comparison figure builder shared by both notebooks.

`render_route_comparison_figure` draws one grid figure: cell 0 is a reference
:class:`RunResult` (plain route set), the remaining cells are case
:class:`RunResult` objects drawn as route diffs vs the reference. Each cell has
a subtitle with raw metric values (ATT / RTT / d_un / disconnected pairs) and --
depending on the run kind -- RL action counts or BCO mutation counts, plus an
optional node-sequence table underneath (``show_tables`` /
``SHOW_ROUTE_SEQUENCE_TABLES``).

It is the extracted rendering half of the training notebook's
``render_lc_improvement_case``. The training notebook feeds it weight-scenario
runs (labels = cost coefficients); ``evaluation_seeded`` feeds it BCO accept
modes / objective-weight combos. Only the ``cases`` list and their labels
differ -- the grid, subtitles and legend are identical.
"""
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .params import SHOW_ROUTE_SEQUENCE_TABLES
from .helpers import (metric_value, as_route_tensor, aggregate_mutation_stats,
                      plot_mutation_histogram)
from . import plots as _plots


# matplotlib line samples explaining the route-diff markings; reused verbatim
# from the training notebook's render_lc_improvement_case legend.
_DIFF_LEGEND_HANDLES = [
    Line2D([0], [0], color="dimgray", lw=2.2, linestyle="--",
           label="initial-only segment / shortened away"),
    Line2D([0], [0], color="tab:blue", lw=2.0, alpha=0.35,
           label="shared segment"),
    Line2D([0], [0], color="tab:blue", lw=4.0,
           label="new / extended segment"),
    Line2D([0], [0], marker="o", color="black", markerfacecolor="tab:blue",
           markersize=7, lw=0, label="added stop"),
    Line2D([0], [0], marker="x", color="crimson", markersize=7, lw=0,
           label="removed stop"),
]


def _count_nonempty_routes(routes) -> int:
    """Routes with at least one edge (>1 real stop) in the first route set."""
    routes = _plots.get_first_route_set(as_route_tensor(routes))
    return int(((routes > -1).sum(dim=-1) > 1).sum().item())


def _fmt(value, decimals: int = 2) -> str:
    """Fixed-point format with NaN-safe handling. No ``e`` notation."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def _raw_line(metrics: dict) -> str:
    """Raw (un-normalized) metric values read from a RunResult.metrics dict."""
    metrics = metrics or {}
    return (
        f"ATT={_fmt(metric_value(metrics, 'ATT'))} "
        f"RTT={_fmt(metric_value(metrics, 'RTT'))} "
        f"d_un={_fmt(metric_value(metrics, '$d_{un}$'))} "
        f"disconn={_fmt(metric_value(metrics, '# disconnected node pairs'))}"
    )


def _raw_delta_line(metrics: dict, ref_metrics: dict) -> str:
    """Raw metric values as reference -> case transitions."""
    metrics, ref_metrics = metrics or {}, ref_metrics or {}
    parts = []
    for label, key in (("ATT", "ATT"), ("RTT", "RTT"),
                        ("d_un", "$d_{un}$"),
                        ("disconn", "# disconnected node pairs")):
        parts.append(f"{label} {_fmt(metric_value(ref_metrics, key))}"
                     f"->{_fmt(metric_value(metrics, key))}")
    return ", ".join(parts)


def _action_line(action_stats: dict):
    """RL ext/trim/halt counters line, or None when there are no action stats."""
    if not action_stats:
        return None
    total = int(action_stats.get("total_count", 0))
    if total == 0:
        return None
    return (
        f"actions: ext={int(action_stats.get('extend_count', 0))}, "
        f"trim_s={int(action_stats.get('trim_start_count', 0))}, "
        f"trim_e={int(action_stats.get('trim_end_count', 0))}, "
        f"halt={int(action_stats.get('halt_count', 0))}"
    )


def _mutation_line(mutation_stats: dict):
    """BCO attempted/accepted/worse-accepted counters line, or None."""
    if not mutation_stats:
        return None
    attempted = sum(int(v) for v in (mutation_stats.get("attempted") or {}).values())
    accepted = sum(int(v) for v in (mutation_stats.get("accepted") or {}).values())
    worse = sum(int(v) for v in (mutation_stats.get("worse_accepted") or {}).values())
    if attempted == 0 and accepted == 0:
        return None
    return (f"mutations: attempted={attempted}, accepted={accepted}, "
            f"worse_accepted={worse}")


def default_reference_subtitle(result) -> str:
    """Subtitle for the reference cell: raw metric values + route count."""
    return (
        f"{_raw_line(result.metrics)}\n"
        f"routes={_count_nonempty_routes(result.routes)}"
    )


def default_case_subtitle(result, reference) -> str:
    """Subtitle for a case cell: raw metric values (reference -> case) plus the
    RL action / BCO mutation counters."""
    lines = [
        f"{_raw_delta_line(result.metrics, reference.metrics)}, "
        f"routes={_count_nonempty_routes(result.routes)}",
    ]
    extra = (_action_line(getattr(result, "action_stats", None))
             or _mutation_line(getattr(result, "mutation_stats", None)))
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def default_plain_subtitle(result) -> str:
    """Subtitle for the plain (no-diff) figure: cost first, then raw metrics
    and route count, plus the optional RL action / BCO mutation counter line.

    Used by :func:`render_route_set_figure`; cost is the top line so it acts
    as the per-panel headline.
    """
    metrics = result.metrics or {}
    lines = [
        f"cost={_fmt(metric_value(metrics, 'cost'))}",
        f"{_raw_line(metrics)}, "
        f"routes={_count_nonempty_routes(result.routes)}",
    ]
    extra = (_action_line(getattr(result, "action_stats", None))
             or _mutation_line(getattr(result, "mutation_stats", None)))
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def render_route_comparison_figure(reference, cases, graph, street_adj=None, *,
                                   title="", ncols=3, palette="tab20",
                                   with_overlap_curves=True, show_tables=None,
                                   reference_subtitle_fn=default_reference_subtitle,
                                   case_subtitle_fn=default_case_subtitle,
                                   figsize=None, save_as=None):
    """Grid figure comparing a reference RunResult against case RunResults.

    Cell 0 draws ``reference`` as a plain route set; cells 1.. draw each entry
    of ``cases`` as a route diff vs the reference. A node-sequence table sits
    under every cell (``show_tables``). ``graph`` is either a PyG ``HeteroData``
    graph or an explicit coords array (with ``street_adj`` then supplied) -- both
    conventions are forwarded straight to :func:`plot_plain_route_set` /
    :func:`plot_route_diff`.

    Returns the matplotlib ``Figure`` (the caller is responsible for showing /
    saving it). When ``save_as`` is given, the underlying route data (every
    RunResult + the graph) is persisted via ``save_route_results`` so the
    figure can be regenerated later -- the *data* is saved, not the image.
    """
    if show_tables is None:
        show_tables = SHOW_ROUTE_SEQUENCE_TABLES
    items = [reference] + list(cases)
    n_cells = len(items)
    ncols = max(1, min(int(ncols), n_cells))
    n_cell_rows = math.ceil(n_cells / ncols)

    if figsize is None:
        figsize = (8.0 * ncols, (11.0 if show_tables else 8.0) * n_cell_rows)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if show_tables:
        grid = fig.add_gridspec(2 * n_cell_rows, ncols,
                                height_ratios=[4.2, 1.15] * n_cell_rows)
    else:
        grid = fig.add_gridspec(n_cell_rows, ncols)

    for cell_idx in range(n_cell_rows * ncols):
        cell_row, col = divmod(cell_idx, ncols)
        plot_row = 2 * cell_row if show_tables else cell_row
        plot_ax = fig.add_subplot(grid[plot_row, col])
        table_ax = (fig.add_subplot(grid[plot_row + 1, col])
                    if show_tables else None)

        if cell_idx >= n_cells:
            plot_ax.axis("off")
            if table_ax is not None:
                table_ax.axis("off")
            continue

        result = items[cell_idx]
        result_label = getattr(result, "label", "") or f"run {cell_idx}"
        if cell_idx == 0:
            _plots.plot_plain_route_set(
                plot_ax, result.routes, graph, street_adj,
                title=result_label,
                subtitle=reference_subtitle_fn(result),
                palette=palette, with_overlap_curves=with_overlap_curves)
        else:
            _plots.plot_route_diff(
                plot_ax, result.routes, reference.routes, graph, street_adj,
                title=result_label,
                subtitle=case_subtitle_fn(result, reference),
                palette=palette, with_overlap_curves=with_overlap_curves)
        if table_ax is not None:
            _plots.draw_route_sequence_table(
                table_ax, result.routes, f"{result_label} route nodes",
                palette=palette)

    if cases:
        fig.legend(handles=_DIFF_LEGEND_HANDLES, loc="lower center",
                   bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=True)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    if save_as:
        # Persist the route data (not the image) so the figure can be redrawn.
        from .results_io import save_route_results
        save_route_results(items, save_as, coords=graph, street_adj=street_adj)
    return fig


def render_route_set_figure(results, graph, street_adj=None, *,
                            title="", ncols=3, palette="tab20",
                            with_overlap_curves=True, show_tables=None,
                            subtitle_fn=default_plain_subtitle,
                            figsize=None, save_as=None):
    """Grid figure drawing each :class:`RunResult` as a *plain* route set.

    Companion to :func:`render_route_comparison_figure`. Every cell is drawn
    via :func:`plot_plain_route_set` (no diff vs reference, no diff legend).
    Per-panel subtitle is produced by ``subtitle_fn`` -- default
    :func:`default_plain_subtitle` puts ``cost=...`` on the first line so the
    cost shows up as the per-panel headline.

    Parameters mirror :func:`render_route_comparison_figure` except there is
    no separate reference cell. ``show_tables`` defaults to
    ``SHOW_ROUTE_SEQUENCE_TABLES``; ``save_as`` persists the route data so the
    figure can be regenerated later via
    :func:`results_io.redraw_route_set`.
    """
    if show_tables is None:
        show_tables = SHOW_ROUTE_SEQUENCE_TABLES
    items = list(results)
    n_cells = len(items)
    if n_cells == 0:
        raise ValueError("render_route_set_figure: results is empty")
    ncols = max(1, min(int(ncols), n_cells))
    n_cell_rows = math.ceil(n_cells / ncols)

    if figsize is None:
        figsize = (8.0 * ncols, (11.0 if show_tables else 8.0) * n_cell_rows)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if show_tables:
        grid = fig.add_gridspec(2 * n_cell_rows, ncols,
                                height_ratios=[4.2, 1.15] * n_cell_rows)
    else:
        grid = fig.add_gridspec(n_cell_rows, ncols)

    for cell_idx in range(n_cell_rows * ncols):
        cell_row, col = divmod(cell_idx, ncols)
        plot_row = 2 * cell_row if show_tables else cell_row
        plot_ax = fig.add_subplot(grid[plot_row, col])
        table_ax = (fig.add_subplot(grid[plot_row + 1, col])
                    if show_tables else None)

        if cell_idx >= n_cells:
            plot_ax.axis("off")
            if table_ax is not None:
                table_ax.axis("off")
            continue

        result = items[cell_idx]
        result_label = getattr(result, "label", "") or f"run {cell_idx}"
        _plots.plot_plain_route_set(
            plot_ax, result.routes, graph, street_adj,
            title=result_label,
            subtitle=subtitle_fn(result),
            palette=palette, with_overlap_curves=with_overlap_curves)
        if table_ax is not None:
            _plots.draw_route_sequence_table(
                table_ax, result.routes, f"{result_label} route nodes",
                palette=palette)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    if save_as:
        from .results_io import save_route_results
        save_route_results(items, save_as, coords=graph, street_adj=street_adj)
    return fig


def plot_alpha_pareto_grid(rows_df, *, cities=None, methods=None,
                           x_metric="ATT", y_metric="RTT",
                           figsize_per_city=(5.5, 4.6),
                           title_prefix="Pareto trade-off across alpha"):
    """Figure-5-style Pareto trade-off grid: ``(x_metric, y_metric)`` per city.

    ``rows_df`` is the per-run table from :func:`run_alpha_pareto_sweep` (must
    contain ``dataset``, ``method``, ``alpha`` and both metric columns). Each
    subplot is one ``dataset``; each curve is one ``method``, with one point
    per ``alpha`` averaged across seeds. Points are connected in increasing
    α order so the trade-off curve sweeps down-and-leftward from α=0
    (operator perspective, lowest C_o) to α=1 (passenger perspective, lowest
    C_p) -- matching the paper's Figure 3/4/5 reading direction. Error bars
    show seed std (zero-width with one seed).

    Defaults to ATT/RTT (the paper's C_p/C_o); pass ``x_metric``/``y_metric``
    if a different pair is wanted. Returns the matplotlib ``Figure``.
    """
    if cities is None:
        cities = list(dict.fromkeys(rows_df["dataset"].tolist()))
    if methods is None:
        methods = list(dict.fromkeys(rows_df["method"].tolist()))

    n_cities = len(cities)
    n_cols = min(n_cities, 3)
    n_rows = math.ceil(n_cities / n_cols) if n_cities else 1
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(figsize_per_city[0] * n_cols,
                 figsize_per_city[1] * n_rows))

    cmap = plt.get_cmap("tab10")
    method_colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for idx, city in enumerate(cities):
        ax = axes[idx // n_cols, idx % n_cols]
        city_df = rows_df[rows_df["dataset"] == city]
        for method in methods:
            mdf = city_df[city_df["method"] == method]
            if mdf.empty:
                continue
            agg = (mdf.groupby("alpha")[[x_metric, y_metric]]
                       .agg(["mean", "std"])
                       .reset_index()
                       .sort_values("alpha"))
            x_means = agg[(x_metric, "mean")]
            y_means = agg[(y_metric, "mean")]
            x_stds = agg[(x_metric, "std")].fillna(0.0)
            y_stds = agg[(y_metric, "std")].fillna(0.0)
            ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds,
                        marker="o", linestyle="-", linewidth=1.6,
                        markersize=5, label=method,
                        color=method_colors[method], capsize=2, alpha=0.9)
        ax.set_xlabel(f"{x_metric}  (lower = better)")
        ax.set_ylabel(f"{y_metric}  (lower = better)")
        ax.set_title(city, fontweight="bold")
        ax.grid(alpha=0.25)

    for j in range(n_cities, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    # One shared legend below the grid.
    seen_labels, handles = set(), []
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                handles.append(h)
    if handles:
        fig.legend(handles, [h.get_label() for h in handles],
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(4, len(handles)), frameon=True)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def plot_seed_sweep_mutation_grid(sweep, title_prefix=""):
    """Mutation-histogram grid for a :func:`run_seed_sweep` result.

    Rows = accept modes, columns = BCO methods; each subplot is the
    seed-averaged :func:`plot_mutation_histogram`. ``sweep`` is either the
    ``run_seed_sweep`` result dict or its ``results`` ``RunResult`` list. The
    unified replacement for ``plot_sweep_mutation_grid`` on the new sweep shape.
    """
    results = sweep["results"] if isinstance(sweep, dict) else list(sweep)
    bco = [r for r in results if getattr(r, "kind", None) == "bco"]
    if not bco:
        return None
    accept_modes = list(dict.fromkeys(r.accept_mode for r in bco))
    methods = list(dict.fromkeys(r.label for r in bco))
    fig, axes = plt.subplots(
        len(accept_modes), len(methods),
        figsize=(4.7 * len(methods), 4.4 * len(accept_modes)),
        squeeze=False)
    for row_idx, accept_mode in enumerate(accept_modes):
        for col_idx, method in enumerate(methods):
            runs = [r for r in bco
                    if r.accept_mode == accept_mode and r.label == method]
            agg = aggregate_mutation_stats([r.mutation_stats for r in runs])
            plot_mutation_histogram(
                agg, f"{method}\n{accept_mode}", ax=axes[row_idx, col_idx])
    fig.suptitle(f"{title_prefix}: mutation attempts vs accepts",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig

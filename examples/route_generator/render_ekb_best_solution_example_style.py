# -*- coding: utf-8 -*-
"""Example-style one-off EKB best-solution renderer.

Produces a separate 2x2 figure matching the reference layout:
  current/initial routes | changed routes
  changed route slots highlighted | route-wise adjustment gradient
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from connectpt.routes_generator.bee_colony import get_adjustment_degrees  # noqa: E402
from eval_lib import plots as route_plots  # noqa: E402
from eval_lib.ekb import load_ekb_tensors, make_ekb_street_underlay_adj  # noqa: E402
from eval_lib.helpers import as_route_tensor  # noqa: E402
from eval_lib.paper import PAPER_DIR  # noqa: E402
from eval_lib.params import ADJ_GAP, ADJ_MODE  # noqa: E402
from render_ekb_best_solution_figure import (  # noqa: E402
    DEFAULT_INITIAL_METRICS_PATH,
    DEFAULT_ROUTES_PATH,
    DEFAULT_TILE_CACHE,
    apply_panel_limits,
    build_basemap,
    default_final_metrics_path,
    load_first_matching_row,
    metric_value,
    padded_extent,
    transform_coords_to_3857,
)


DEFAULT_OUT_PATH = PAPER_DIR / "final_ekb_best_solution_example_style.png"
SHOW_NODE_LABELS = False
NODE_SIZE = 15
NODE_ALPHA = 0.30
WITH_OVERLAP_CURVES = True
ROAD_COLOR = "#6f7378"
ROAD_LINEWIDTH = 1.2
ROAD_ALPHA = 0.50
ADJ_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "route_adj_yellow_red_purple",
    ["#ffd84d", "#f05a28", "#7b1fa2"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes-path", type=Path, default=DEFAULT_ROUTES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--tile-cache", type=Path, default=DEFAULT_TILE_CACHE)
    parser.add_argument("--initial-metrics", type=Path, default=DEFAULT_INITIAL_METRICS_PATH)
    parser.add_argument("--final-metrics", type=Path, default=None)
    parser.add_argument("--zoom", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pad-m", type=float, default=2500.0)
    parser.add_argument("--allow-no-basemap", action="store_true")
    return parser.parse_args()


def first_route_set(routes) -> torch.Tensor:
    routes = as_route_tensor(routes).long()
    return routes[0] if routes.ndim == 3 else routes


def pad_route_width(routes: torch.Tensor, width: int) -> torch.Tensor:
    routes = first_route_set(routes)
    if routes.shape[-1] >= width:
        return routes[..., :width]
    pad = torch.full((routes.shape[0], width - routes.shape[-1]), -1, dtype=routes.dtype)
    return torch.cat([routes, pad], dim=-1)


def route_adjustments(routes: torch.Tensor, reference_routes: torch.Tensor) -> np.ndarray:
    width = max(int(routes.shape[-1]), int(reference_routes.shape[-1]))
    routes_padded = pad_route_width(routes, width)[None]
    reference_padded = pad_route_width(reference_routes, width)[None]
    adj = get_adjustment_degrees(
        routes_padded,
        reference_padded,
        True,
        gap=ADJ_GAP,
        mode=ADJ_MODE,
    )
    return adj[0].detach().cpu().numpy()


def register_example_palette(name: str = "example_glasbey67", n: int = 67) -> str:
    colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            colors.extend(cmap.colors)
    idx = 0
    while len(colors) < n:
        colors.append(plt.cm.hsv((idx * 0.61803398875) % 1.0))
        idx += 1
    route_plots._PALETTE_CACHE[name] = np.asarray([mcolors.to_rgba(color) for color in colors[:n]])
    return name


def metric_line(row: dict[str, str], include_adj: bool) -> str:
    rtt = metric_value(row, "RTT")
    wmc = metric_value(row, "WMC", "median_connectivity_weighted", "WMC_median")
    adj = metric_value(row, "adj_vs_seed", "adj")
    d0 = metric_value(row, "$d_0$", "d0")
    d1 = metric_value(row, "$d_1$", "d1")
    d2 = metric_value(row, "$d_2$", "d2")
    dun = metric_value(row, "$d_{un}$", "d_un")
    parts = []
    if wmc is not None:
        parts.append(f"WMC={wmc:.2f}")
    if rtt is not None:
        parts.append(f"RTT={rtt:.0f}")
    if include_adj and adj is not None:
        parts.append(f"adj={adj:.2f}")
    d_parts = []
    if d0 is not None:
        d_parts.append(f"d0={d0:.2f}")
    if d1 is not None:
        d_parts.append(f"d1={d1:.2f}")
    if d2 is not None:
        d_parts.append(f"d2={d2:.2f}")
    if dun is not None:
        d_parts.append(f"dun={dun:.2f}")
    lines = []
    if parts:
        lines.append("  ".join(parts))
    if d_parts:
        lines.append("  ".join(d_parts))
    return "\n".join(lines)


def draw_street_graph_stronger(ax, coords, street_adj) -> None:
    coords = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    street_adj = street_adj.detach().cpu().numpy() if isinstance(street_adj, torch.Tensor) else np.asarray(street_adj)
    n_nodes = coords.shape[0]
    for start in range(n_nodes):
        for end in range(start + 1, n_nodes):
            if np.isfinite(street_adj[start, end]) or np.isfinite(street_adj[end, start]):
                ax.plot(
                    [coords[start, 0], coords[end, 0]],
                    [coords[start, 1], coords[end, 1]],
                    color=ROAD_COLOR,
                    linewidth=ROAD_LINEWIDTH,
                    alpha=ROAD_ALPHA,
                    solid_capstyle="round",
                    zorder=1.2,
                )


def add_map_underlay(ax, basemap) -> None:
    if basemap is None:
        ax.set_facecolor("#f6f2ea")
        return
    image, extent = basemap
    ax.imshow(image, extent=extent, origin="upper", zorder=0, alpha=0.92)


def quiet_plain_nodes(ax) -> None:
    for collection in ax.collections:
        collection.set_alpha(NODE_ALPHA)


def plot_changed_route_slots(
    ax,
    routes,
    reference_routes,
    coords,
    street_adj,
    *,
    title: str,
    subtitle: str,
    palette: str,
) -> None:
    routes = route_plots.get_first_route_set(routes)
    reference_routes = route_plots.get_first_route_set(reference_routes)
    coords, street_adj_arr = route_plots.extract_coords_street_adj(coords, street_adj)
    route_plots.draw_street_graph(ax, coords, street_adj_arr)

    colors = route_plots.route_colors_for(routes, palette=palette)
    overlap_map = route_plots.build_edge_overlap_map(routes) if WITH_OVERLAP_CURVES else None
    for route_idx, route_tensor in enumerate(routes):
        route = route_plots.route_to_list(route_tensor)
        if len(route) < 2:
            continue
        reference_route = (
            route_plots.route_to_list(reference_routes[route_idx])
            if route_idx < reference_routes.shape[0]
            else []
        )
        changed = route != reference_route
        route_plots.plot_edges(
            ax,
            coords,
            route_plots.route_edge_list(route),
            color=colors[route_idx % len(colors)],
            linewidth=3.8 if changed else 1.2,
            alpha=0.96 if changed else 0.16,
            route_idx=route_idx,
            overlap_map=overlap_map,
            zorder=4 if changed else 3,
        )

    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=NODE_SIZE, alpha=NODE_ALPHA, zorder=5)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_route_adj_gradient(
    ax,
    routes,
    reference_routes,
    coords,
    street_adj,
    *,
    title: str,
    subtitle: str,
):
    routes = route_plots.get_first_route_set(routes)
    reference_routes = route_plots.get_first_route_set(reference_routes)
    coords, street_adj_arr = route_plots.extract_coords_street_adj(coords, street_adj)
    route_plots.draw_street_graph(ax, coords, street_adj_arr)

    adj_values = route_adjustments(routes, reference_routes)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    overlap_map = route_plots.build_edge_overlap_map(routes) if WITH_OVERLAP_CURVES else None

    route_items = []
    for route_idx, route_tensor in enumerate(routes):
        route = route_plots.route_to_list(route_tensor)
        if len(route) < 2:
            continue
        value = float(adj_values[route_idx]) if route_idx < len(adj_values) else 0.0
        route_items.append((value, route_idx, route))

    # Draw unchanged routes first as a quiet context layer, then changed routes
    # on top so high-adj routes do not disappear into the dense network.
    for value, route_idx, route in sorted(route_items, key=lambda item: item[0]):
        if value <= 1e-6:
            route_plots.plot_edges(
                ax,
                coords,
                route_plots.route_edge_list(route),
                color="#9aa1a8",
                linewidth=0.9,
                alpha=0.16,
                route_idx=route_idx,
                overlap_map=overlap_map,
                zorder=2.6,
            )
            continue

        route_plots.plot_edges(
            ax,
            coords,
            route_plots.route_edge_list(route),
            color=ADJ_CMAP(norm(value)),
            linewidth=2.0 + 3.5 * value,
            alpha=0.98,
            route_idx=route_idx,
            overlap_map=overlap_map,
            zorder=4 + value,
        )

    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=NODE_SIZE, alpha=NODE_ALPHA, zorder=5.5)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=ADJ_CMAP)
    scalar.set_array(adj_values)
    return scalar, float(np.mean(adj_values))

def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    palette = register_example_palette(n=67)
    original_draw_street_graph = route_plots.draw_street_graph
    route_plots.draw_street_graph = draw_street_graph_stronger

    try:
        dump = torch.load(args.routes_path, map_location="cpu", weights_only=False)
        routes_obj = dump["routes"] if isinstance(dump, dict) and "routes" in dump else dump
        if not isinstance(routes_obj, dict):
            raise ValueError("Expected a .pt file with a route mapping, e.g. dump['routes']")

        keys = list(routes_obj)
        initial_key = next((key for key in keys if "Initial" in str(key)), keys[0])
        candidate_keys = [key for key in keys if key != initial_key]
        if not candidate_keys:
            raise ValueError("The route dump contains only initial routes; nothing to compare.")
        candidate_key = candidate_keys[0]

        init_routes = first_route_set(routes_obj[initial_key])
        cand_routes = first_route_set(routes_obj[candidate_key])
        tensors = load_ekb_tensors()
        coords_3857 = transform_coords_to_3857(tensors["node_locs"])
        plot_adj = make_ekb_street_underlay_adj(tensors["street_adj"])
        extent = padded_extent(coords_3857, float(args.pad_m))

        try:
            basemap = build_basemap(extent, int(args.zoom), args.tile_cache)
            print(f"[EKB example viz] loaded map tiles at z={args.zoom}")
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            if not args.allow_no_basemap:
                raise RuntimeError("Could not load map tiles; pass --allow-no-basemap to render without them.") from exc
            basemap = None
            print(f"[EKB example viz] basemap unavailable, using fallback: {exc}")

        final_metrics_path = args.final_metrics or default_final_metrics_path(args.routes_path)
        init_row = load_first_matching_row(args.initial_metrics, str(initial_key))
        cand_row = load_first_matching_row(final_metrics_path, str(candidate_key))
        changes = route_plots.summarize_route_changes(cand_routes, init_routes)
        init_metrics = metric_line(init_row, include_adj=False)
        cand_metrics = metric_line(cand_row, include_adj=True)
        alpha_label = "best"

        fig, axes = plt.subplots(2, 2, figsize=(30, 24), squeeze=False, constrained_layout=True)
        flat_axes = axes.ravel()
        for ax in flat_axes:
            add_map_underlay(ax, basemap)

        route_plots.plot_plain_route_set(
            flat_axes[0],
            init_routes,
            coords_3857,
            plot_adj,
            title=f"Current / initial routes\n{init_metrics}" if init_metrics else "Current / initial routes",
            palette=palette,
            with_overlap_curves=WITH_OVERLAP_CURVES,
            show_node_labels=SHOW_NODE_LABELS,
            node_size=NODE_SIZE,
        )
        quiet_plain_nodes(flat_axes[0])

        route_plots.plot_plain_route_set(
            flat_axes[1],
            cand_routes,
            coords_3857,
            plot_adj,
            title=f"Changed routes: {alpha_label}\n{cand_metrics}" if cand_metrics else f"Changed routes: {alpha_label}",
            palette=palette,
            with_overlap_curves=WITH_OVERLAP_CURVES,
            show_node_labels=SHOW_NODE_LABELS,
            node_size=NODE_SIZE,
        )
        quiet_plain_nodes(flat_axes[1])

        change_subtitle = (
            f"changed_routes={changes['changed_routes']}  "
            f"+edges={changes['added_edges']}  -edges={changes['removed_edges']}  "
            f"+stops={changes['added_stops']}  -stops={changes['removed_stops']}"
        )
        plot_changed_route_slots(
            flat_axes[2],
            cand_routes,
            init_routes,
            coords_3857,
            plot_adj,
            title=f"Changed route slots highlighted: {alpha_label}",
            subtitle=change_subtitle,
            palette=palette,
        )

        mean_adj_title = float(np.mean(route_adjustments(cand_routes, init_routes)))
        adj_scalar, mean_adj = plot_route_adj_gradient(
            flat_axes[3],
            cand_routes,
            init_routes,
            coords_3857,
            plot_adj,
            title=f"Route-wise adj gradient: {alpha_label}",
            subtitle=f"gray=0, yellow->purple=increasing adj, mean adj={mean_adj_title:.3f}",
        )
        cax = flat_axes[3].inset_axes([0.94, 0.08, 0.03, 0.80])
        cbar = fig.colorbar(adj_scalar, cax=cax)
        cbar.set_label("route adj", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        for ax in flat_axes:
            apply_panel_limits(ax, extent)
            ax.set_axis_off()

        fig.suptitle(f"{args.routes_path.name}: {alpha_label}", fontsize=15, fontweight="bold")
        fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[EKB example viz] candidate: {candidate_key}")
        print(f"[EKB example viz] saved -> {args.out}")
    finally:
        route_plots.draw_street_graph = original_draw_street_graph


if __name__ == "__main__":
    main()
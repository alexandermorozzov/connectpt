# -*- coding: utf-8 -*-
"""One-off renderer for the EKB best-solution checkpoint.

Creates a single 1x3 PNG:
  Initial solution | Diff solution | Plain
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PathCollection
from PIL import Image
from pyproj import Transformer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lib import plots as route_plots  # noqa: E402
from eval_lib.ekb import (  # noqa: E402
    EKB_COORD_CRS,
    load_ekb_tensors,
    make_ekb_street_underlay_adj,
)
from eval_lib.helpers import as_route_tensor  # noqa: E402
from eval_lib.paper import PAPER_DIR  # noqa: E402


DEFAULT_ROUTES_PATH = PAPER_DIR / "final_ekb_best_solution_routes.pt"
DEFAULT_OUT_PATH = PAPER_DIR / "final_ekb_best_solution_three_panel_map.png"
DEFAULT_TILE_CACHE = PAPER_DIR / "_tile_cache"
DEFAULT_INITIAL_METRICS_PATH = PAPER_DIR / "final_ekb_nbco_gnn_trimextend_iter50_seqbees.csv"
TILE_URL = "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
WEB_MERCATOR_HALF_WORLD = 20037508.342789244


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes-path", type=Path, default=DEFAULT_ROUTES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--tile-cache", type=Path, default=DEFAULT_TILE_CACHE)
    parser.add_argument("--initial-metrics", type=Path, default=DEFAULT_INITIAL_METRICS_PATH)
    parser.add_argument("--final-metrics", type=Path, default=None)
    parser.add_argument("--zoom", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--pad-m", type=float, default=2500.0)
    parser.add_argument("--allow-no-basemap", action="store_true")
    return parser.parse_args()


def first_2d(routes) -> torch.Tensor:
    routes = as_route_tensor(routes).long()
    if routes.ndim == 3:
        return routes[0]
    return routes


def load_route_pair(path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    dump = torch.load(path, map_location="cpu", weights_only=False)
    routes = dump.get("routes", {})
    if "Initial EKB routes" not in routes:
        raise KeyError(f"{path} has no 'Initial EKB routes' route set")
    final_key = next((key for key in routes if key != "Initial EKB routes"), None)
    if final_key is None:
        raise KeyError(f"{path} has no final route set")
    return first_2d(routes["Initial EKB routes"]), first_2d(routes[final_key]), final_key


def default_final_metrics_path(routes_path: Path) -> Path:
    name = routes_path.name
    if name.endswith("_routes.pt"):
        return routes_path.with_name(name[: -len("_routes.pt")] + ".csv")
    return routes_path.with_suffix(".csv")


def load_first_matching_row(path: Path | None, method: str | None = None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    if method is None:
        return rows[0]
    return next((row for row in rows if row.get("method") == method), rows[0])


def metric_value(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def metric_label(title: str, row: dict[str, str]) -> str:
    cost = metric_value(row, "cost", "objective_cost")
    rtt = metric_value(row, "RTT")
    wmc = metric_value(row, "WMC", "median_connectivity_weighted", "WMC_median")
    adj = metric_value(row, "adj_vs_seed")
    parts = []
    if cost is not None:
        parts.append(f"cost={cost:.3f}")
    if rtt is not None:
        parts.append(f"RTT={rtt:.0f}")
    if wmc is not None:
        parts.append(f"WMC={wmc:.2f}")
    if adj is not None:
        parts.append(f"adj={adj:.3f}")
    return f"{title}\n{'  '.join(parts)}" if parts else title

def transform_coords_to_3857(coords, source_crs: str = EKB_COORD_CRS) -> np.ndarray:
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)
    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    x_coord, y_coord = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack((x_coord, y_coord))


def padded_extent(coords_3857: np.ndarray, pad_m: float) -> tuple[float, float, float, float]:
    min_x = float(coords_3857[:, 0].min()) - pad_m
    max_x = float(coords_3857[:, 0].max()) + pad_m
    min_y = float(coords_3857[:, 1].min()) - pad_m
    max_y = float(coords_3857[:, 1].max()) + pad_m
    return min_x, max_x, min_y, max_y


def tile_span_m(zoom: int) -> float:
    return 2.0 * WEB_MERCATOR_HALF_WORLD / (2**zoom)


def meters_to_tile(x_coord: float, y_coord: float, zoom: int) -> tuple[int, int]:
    span = tile_span_m(zoom)
    limit = 2**zoom - 1
    tile_x = int(math.floor((x_coord + WEB_MERCATOR_HALF_WORLD) / span))
    tile_y = int(math.floor((WEB_MERCATOR_HALF_WORLD - y_coord) / span))
    return max(0, min(limit, tile_x)), max(0, min(limit, tile_y))


def tile_bounds(tile_x: int, tile_y: int, zoom: int) -> tuple[float, float, float, float]:
    span = tile_span_m(zoom)
    min_x = tile_x * span - WEB_MERCATOR_HALF_WORLD
    max_x = (tile_x + 1) * span - WEB_MERCATOR_HALF_WORLD
    max_y = WEB_MERCATOR_HALF_WORLD - tile_y * span
    min_y = WEB_MERCATOR_HALF_WORLD - (tile_y + 1) * span
    return min_x, max_x, min_y, max_y


def download_tile(tile_x: int, tile_y: int, zoom: int, cache_dir: Path) -> Image.Image:
    cache_path = cache_dir / str(zoom) / str(tile_x) / f"{tile_y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = TILE_URL.format(z=zoom, x=tile_x, y=tile_y)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "connectpt-ekb-figure/1.0"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read()
    cache_path.write_bytes(data)
    return Image.open(cache_path).convert("RGB")


def build_basemap(
    extent: tuple[float, float, float, float],
    zoom: int,
    cache_dir: Path,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    min_x, max_x, min_y, max_y = extent
    left, bottom = meters_to_tile(min_x, min_y, zoom)
    right, top = meters_to_tile(max_x, max_y, zoom)
    x_tiles = range(left, right + 1)
    y_tiles = range(top, bottom + 1)

    tile_size = 256
    canvas = Image.new("RGB", (len(x_tiles) * tile_size, len(y_tiles) * tile_size))
    for row, tile_y in enumerate(y_tiles):
        for col, tile_x in enumerate(x_tiles):
            tile = download_tile(tile_x, tile_y, zoom, cache_dir)
            canvas.paste(tile.resize((tile_size, tile_size)), (col * tile_size, row * tile_size))
            time.sleep(0.02)

    map_min_x, _, _, map_max_y = tile_bounds(left, top, zoom)
    _, map_max_x, map_min_y, _ = tile_bounds(right, bottom, zoom)
    return np.asarray(canvas), (map_min_x, map_max_x, map_min_y, map_max_y)


def add_map_underlay(ax, basemap: tuple[np.ndarray, tuple[float, float, float, float]] | None) -> None:
    if basemap is None:
        ax.set_facecolor("#f6f2ea")
        return
    image, extent = basemap
    ax.imshow(image, extent=extent, origin="upper", zorder=0, alpha=0.92)


def apply_panel_limits(ax, extent: tuple[float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y = extent
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.axis("off")


def soften_node_markers(
    ax,
    black_alpha: float = 0.10,
    green_alpha: float = 0.62,
    size: float = 4.0,
    green_size: float = 14.0,
) -> None:
    """Make node markers quiet while preserving added stops as green."""
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        facecolors = collection.get_facecolors()
        if facecolors.size == 0:
            continue

        rgb = facecolors[:, :3]
        is_black = np.all(rgb < 0.08, axis=1)
        is_green = (rgb[:, 1] > 0.75) & (rgb[:, 0] < 0.25) & (rgb[:, 2] < 0.25)
        if bool(np.all(is_black)):
            facecolors[:, 3] = float(black_alpha)
            collection.set_facecolors(facecolors)
            edgecolors = collection.get_edgecolors()
            if edgecolors.size:
                edgecolors[:, 3] = float(black_alpha)
                collection.set_edgecolors(edgecolors)
            collection.set_alpha(float(black_alpha))
            collection.set_sizes(np.full_like(collection.get_sizes(), float(size)))
            continue

        if bool(np.all(is_green)):
            facecolors[:, :3] = np.array([0.0, 0.85, 0.12])
            facecolors[:, 3] = float(green_alpha)
            collection.set_facecolors(facecolors)
            green_edges = np.tile(facecolors[0], (max(len(facecolors), 1), 1))
            green_edges[:, 3] = float(green_alpha)
            collection.set_edgecolors(green_edges)
            collection.set_alpha(float(green_alpha))
            collection.set_sizes(np.full_like(collection.get_sizes(), float(green_size)))
            collection.set_zorder(max(collection.get_zorder(), 8))

def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    initial_routes, final_routes, final_key = load_route_pair(args.routes_path)
    final_metrics_path = args.final_metrics or default_final_metrics_path(args.routes_path)
    initial_metrics = load_first_matching_row(args.initial_metrics, "Initial EKB routes")
    final_metrics = load_first_matching_row(final_metrics_path, final_key)
    tensors = load_ekb_tensors()
    coords_3857 = transform_coords_to_3857(tensors["node_locs"])
    street_adj = make_ekb_street_underlay_adj(tensors["street_adj"])
    extent = padded_extent(coords_3857, float(args.pad_m))

    try:
        basemap = build_basemap(extent, int(args.zoom), args.tile_cache)
        print(f"[EKB viz] loaded map tiles at z={args.zoom}")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if not args.allow_no_basemap:
            raise RuntimeError(
                "Could not load map tiles. Re-run with network access or pass "
                "--allow-no-basemap for a street-graph-only fallback."
            ) from exc
        basemap = None
        print(f"[EKB viz] basemap unavailable, using fallback background: {exc}")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 7.2),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]
    for ax in axes:
        add_map_underlay(ax, basemap)

    common_kwargs = dict(
        graph_or_coords=coords_3857,
        street_adj=street_adj,
        palette="tab20",
        with_overlap_curves=True,
        show_node_labels=False,
        node_size=7,
    )
    route_plots.plot_plain_route_set(
        axes[0],
        initial_routes,
        title=metric_label("Initial solution", initial_metrics),
        **common_kwargs,
    )
    route_plots.plot_route_diff(
        axes[1],
        final_routes,
        initial_routes,
        title=metric_label("Diff solution", final_metrics),
        **common_kwargs,
    )
    route_plots.plot_plain_route_set(
        axes[2],
        final_routes,
        title=metric_label("Plain", final_metrics),
        **common_kwargs,
    )
    for ax in axes:
        soften_node_markers(ax)
        apply_panel_limits(ax, extent)

    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[EKB viz] final route set: {final_key}")
    print(f"[EKB viz] saved -> {args.out}")


if __name__ == "__main__":
    main()

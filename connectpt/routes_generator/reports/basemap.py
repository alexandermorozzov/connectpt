"""Slippy-map basemap underlay for geo route figures -- general, not city-specific.

A geo instance (real-world coordinates with a CRS) reads much better over an
actual street map than over the bare stop graph. This module is the whole
underlay story: project an extent, fetch/cache the web-mercator tiles covering
it, and paste them under a matplotlib axis.

Offline first: a tile is served from the on-disk cache when present and only
downloaded when it is missing, so a machine with a warm cache (or no network at
all) still renders. When tiles cannot be fetched, :func:`load_basemap` returns
``None`` and :func:`add_map_underlay` falls back to a flat paper-coloured
background -- a figure is never lost to a network error.

Nothing here knows about EKB; ``reports.render`` supplies the coordinates.
"""
from __future__ import annotations

import logging
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..core.paths import ARTIFACTS_DIR

log = logging.getLogger(__name__)

# CARTO "light_all" -- a quiet grey basemap that does not fight route colors.
TILE_URL = "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
TILE_SIZE = 256
WEB_MERCATOR_HALF_WORLD = 20037508.342789244

# Canonical write target, plus the tile set already shipped in the repo (kept
# read-only so a warm offline render works without duplicating the cache).
TILE_CACHE_DIR = ARTIFACTS_DIR / "_tile_cache"
FALLBACK_TILE_CACHE_DIR = ARTIFACTS_DIR / "archive" / "paper_results" / "_tile_cache"

DEFAULT_ZOOM = 11
DEFAULT_PAD_M = 1000.0   # just enough context; wider margins shrink the network
NO_BASEMAP_FACECOLOR = "#f6f2ea"


def padded_extent(coords_3857, pad_m: float = DEFAULT_PAD_M):
    """Bounding box ``(min_x, max_x, min_y, max_y)`` of the coords + a margin."""
    coords = np.asarray(coords_3857, dtype=float)
    return (float(coords[:, 0].min()) - pad_m, float(coords[:, 0].max()) + pad_m,
            float(coords[:, 1].min()) - pad_m, float(coords[:, 1].max()) + pad_m)


def _tile_span_m(zoom: int) -> float:
    return 2.0 * WEB_MERCATOR_HALF_WORLD / (2 ** zoom)


def _meters_to_tile(x_coord: float, y_coord: float, zoom: int) -> tuple[int, int]:
    span = _tile_span_m(zoom)
    limit = 2 ** zoom - 1
    tile_x = int(math.floor((x_coord + WEB_MERCATOR_HALF_WORLD) / span))
    tile_y = int(math.floor((WEB_MERCATOR_HALF_WORLD - y_coord) / span))
    return max(0, min(limit, tile_x)), max(0, min(limit, tile_y))


def _tile_bounds(tile_x: int, tile_y: int, zoom: int):
    span = _tile_span_m(zoom)
    return (tile_x * span - WEB_MERCATOR_HALF_WORLD,
            (tile_x + 1) * span - WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD - (tile_y + 1) * span,
            WEB_MERCATOR_HALF_WORLD - tile_y * span)


def _cached_tile_path(tile_x: int, tile_y: int, zoom: int, cache_dir: Path):
    """First existing cached copy of a tile (canonical dir, then the shipped one)."""
    rel = Path(str(zoom)) / str(tile_x) / f"{tile_y}.png"
    for base in (cache_dir, FALLBACK_TILE_CACHE_DIR):
        candidate = Path(base) / rel
        if candidate.exists():
            return candidate
    return None


def _load_tile(tile_x: int, tile_y: int, zoom: int, cache_dir: Path, *,
               download: bool, throttle_s: float):
    from PIL import Image

    cached = _cached_tile_path(tile_x, tile_y, zoom, cache_dir)
    if cached is not None:
        return Image.open(cached).convert("RGB")
    if not download:
        raise FileNotFoundError(f"tile z{zoom}/{tile_x}/{tile_y} not in cache")

    request = urllib.request.Request(
        TILE_URL.format(z=zoom, x=tile_x, y=tile_y),
        headers={"User-Agent": "connectpt-route-figure/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read()
    target = Path(cache_dir) / str(zoom) / str(tile_x) / f"{tile_y}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if throttle_s:
        time.sleep(throttle_s)  # be polite to the tile server
    return Image.open(target).convert("RGB")


def load_basemap(extent, zoom: int = DEFAULT_ZOOM, cache_dir=None, *,
                 download: bool = True, throttle_s: float = 0.02):
    """Mosaic the tiles covering ``extent`` -> ``(image, tile_extent)`` or ``None``.

    ``extent`` is ``(min_x, max_x, min_y, max_y)`` in EPSG:3857 metres. The
    returned tile extent is the mosaic's own bounds (tile-aligned, so slightly
    larger than ``extent``) -- pass it straight to :func:`add_map_underlay`.
    Returns ``None`` when the tiles are neither cached nor reachable, which the
    callers treat as "draw without a basemap" rather than as a failure.
    """
    from PIL import Image

    cache_dir = Path(cache_dir) if cache_dir is not None else TILE_CACHE_DIR
    min_x, max_x, min_y, max_y = extent
    left, bottom = _meters_to_tile(min_x, min_y, zoom)
    right, top = _meters_to_tile(max_x, max_y, zoom)
    x_tiles, y_tiles = range(left, right + 1), range(top, bottom + 1)

    canvas = Image.new("RGB", (len(x_tiles) * TILE_SIZE, len(y_tiles) * TILE_SIZE))
    try:
        for row, tile_y in enumerate(y_tiles):
            for col, tile_x in enumerate(x_tiles):
                tile = _load_tile(tile_x, tile_y, zoom, cache_dir,
                                  download=download, throttle_s=throttle_s)
                canvas.paste(tile.resize((TILE_SIZE, TILE_SIZE)),
                             (col * TILE_SIZE, row * TILE_SIZE))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        log.warning("basemap unavailable (z=%s), rendering without tiles: %s",
                    zoom, exc)
        return None

    map_min_x, _, _, map_max_y = _tile_bounds(left, top, zoom)
    _, map_max_x, map_min_y, _ = _tile_bounds(right, bottom, zoom)
    log.info("basemap: %dx%d tiles at z=%d", len(x_tiles), len(y_tiles), zoom)
    return np.asarray(canvas), (map_min_x, map_max_x, map_min_y, map_max_y)


def add_map_underlay(ax, basemap) -> None:
    """Paste a loaded basemap under ``ax`` (or paint the no-tiles fallback)."""
    if basemap is None:
        ax.set_facecolor(NO_BASEMAP_FACECOLOR)
        return
    image, extent = basemap
    ax.imshow(image, extent=extent, origin="upper", zorder=0, alpha=0.92)


def apply_panel_limits(ax, extent) -> None:
    """Clip a panel to the data extent (tiles are drawn wider than the network)."""
    min_x, max_x, min_y, max_y = extent
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_axis_off()


__all__ = [
    "TILE_CACHE_DIR",
    "DEFAULT_ZOOM",
    "DEFAULT_PAD_M",
    "padded_extent",
    "load_basemap",
    "add_map_underlay",
    "apply_panel_limits",
]

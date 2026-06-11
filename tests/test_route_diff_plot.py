"""Regression test: plot_route_diff must show reduced edge *coverage*, not just
fully-removed edges. A duplicated edge trimmed from N routes to M<N is a real
change (the dedup signal) and must appear as dashed grey copies."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

from eval_lib.plots import plot_route_diff  # noqa: E402


def _count_removed_artists(ax):
    """Count dashed dimgray artists (the 'removed coverage' markers)."""
    n = 0
    for ln in ax.get_lines():
        if ln.get_linestyle() in ("--", "dashed") and \
                matplotlib.colors.to_hex(ln.get_color()) == \
                matplotlib.colors.to_hex("dimgray"):
            n += 1
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch) and p.get_linestyle() in (
                "--", "dashed") and matplotlib.colors.to_hex(
                p.get_edgecolor()) == matplotlib.colors.to_hex("dimgray"):
            n += 1
    return n


def _grid_coords(side=4):
    return np.array([[c, r] for r in range(side) for c in range(side)],
                    dtype=float)


def _full_street_adj(n):
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0.0)
    adj[adj == np.inf] = 1.0  # fully connected so every route leg is valid
    np.fill_diagonal(adj, 0.0)
    return adj


def test_deduplication_is_drawn():
    coords = _grid_coords(4)            # 16 nodes
    adj = _full_street_adj(16)
    # seed: edge (5,6) duplicated across 3 routes
    seed = torch.tensor([[5, 6, 7, -1], [5, 6, 8, -1],
                         [5, 6, 9, -1], [1, 2, 3, -1]])
    # candidate: dedup -> only one route keeps (5,6); the (6,8) and (6,9) legs go
    cand = torch.tensor([[5, 6, 7, -1], [8, -1, -1, -1],
                         [9, -1, -1, -1], [1, 2, 3, -1]])

    fig, ax = plt.subplots()
    plot_route_diff(ax, cand, seed, coords, adj, title="t",
                    with_overlap_curves=True)
    removed = _count_removed_artists(ax)
    plt.close(fig)
    # 2 trimmed copies of (5,6) + the (6,8) and (6,9) legs = 4 removed markers.
    assert removed >= 4, f"expected >=4 removed markers, got {removed}"


def test_pure_reorder_shows_no_removed():
    coords = _grid_coords(4)
    adj = _full_street_adj(16)
    seed = torch.tensor([[1, 2, 3, -1], [5, 6, 7, -1], [9, 10, 11, -1]])
    # same network, routes reordered -> nothing removed
    cand = torch.tensor([[9, 10, 11, -1], [1, 2, 3, -1], [5, 6, 7, -1]])
    fig, ax = plt.subplots()
    plot_route_diff(ax, cand, seed, coords, adj, title="t")
    removed = _count_removed_artists(ax)
    plt.close(fig)
    assert removed == 0, f"reorder should show 0 removed, got {removed}"

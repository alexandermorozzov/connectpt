"""Shared route visualization helpers for the LC-improvement notebooks.

Both ``lc_improvement_training.ipynb`` and
``evaluation_seeded_lc_bco_mumford0.ipynb`` previously carried near-duplicate
copies of these helpers. The lc_improvement_training version is the richer
one (curved edges for overlapping routes, tab20 palette, accepts PyG
``HeteroData``); the evaluation_seeded version is simpler (straight edges,
tab10 palette, raw ``coords``/``street_adj`` arrays).

The functions here unify both. Adapter ``extract_coords_street_adj`` accepts
either a PyG graph or raw coords+street_adj. ``plot_plain_route_set`` /
``plot_route_diff`` expose ``palette`` and ``with_overlap_curves`` kwargs so
each notebook can pick its visual style.

Canonical lineage: V2 of cell ``80161e50`` in
``lc_improvement_training.ipynb``. Do not re-introduce the earlier V1 of
``compute_cost_breakdown`` etc. — they are dead code.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch

from connectpt.routes_generator.citygraph_dataset import STOP_KEY
from connectpt.routes_generator.torch_utils import get_batch_tensor_from_routes


# Palette cache — built lazily because matplotlib colormaps return RGBA tuples
# that we want as numpy arrays for fast indexing.
_PALETTE_CACHE: dict[str, np.ndarray] = {}


def _palette_colors(palette: str) -> np.ndarray:
    """Return cached RGBA colors for a named matplotlib colormap palette."""
    if palette not in _PALETTE_CACHE:
        _PALETTE_CACHE[palette] = np.asarray(plt.get_cmap(palette).colors)
    return _PALETTE_CACHE[palette]


def route_colors_for_n(n_routes: int, palette: str = "tab20") -> np.ndarray:
    """Return ``n_routes`` colors cycling through the named palette."""
    cycle = _palette_colors(palette)
    if n_routes <= 0:
        return cycle[:0]
    color_indices = np.arange(n_routes) % len(cycle)
    return cycle[color_indices]


def route_colors_for(routes, palette: str = "tab20") -> np.ndarray:
    """Return one color per route in ``routes`` (first batch element)."""
    routes = get_first_route_set(routes)
    return route_colors_for_n(routes.shape[0], palette=palette)


def route_to_list(route_tensor):
    """Convert a route tensor (with ``-1`` padding) to a python list of ints."""
    return [int(node) for node in route_tensor.tolist() if int(node) >= 0]


def route_edge_list(route):
    """Yield consecutive (u, v) edges along ``route``."""
    return [(route[idx], route[idx + 1]) for idx in range(len(route) - 1)]


def undirected_edge_key(edge):
    """Canonical edge key (sorted endpoints) for undirected comparisons."""
    return tuple(sorted(edge))


def get_first_route_set(routes):
    """Coerce routes into a 2-D ``[n_routes, max_route_len]`` tensor (CPU).

    Accepts: ``torch.Tensor`` of shape ``[n_routes, L]`` or ``[1, n_routes, L]``
    or ``[B, n_routes, L]`` (returns batch index 0); a nested list of routes.
    """
    if isinstance(routes, torch.Tensor):
        routes = routes.detach().cpu()
    else:
        routes = get_batch_tensor_from_routes(routes).detach().cpu()
    if routes.ndim == 3:
        return routes[0]
    return routes


def extract_coords_street_adj(graph_or_coords, street_adj=None):
    """Adapter: accept a PyG ``HeteroData`` graph or raw ``(coords, street_adj)``.

    Returns ``(coords_np, street_adj_np)`` as numpy arrays.
    """
    if street_adj is None:
        graph = graph_or_coords
        coords = graph[STOP_KEY].pos.detach().cpu().numpy()
        street_adj = graph.street_adj.detach().cpu().numpy()
        return coords, street_adj

    coords = graph_or_coords
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.asarray(coords)
    if isinstance(street_adj, torch.Tensor):
        street_adj = street_adj.detach().cpu().numpy()
    else:
        street_adj = np.asarray(street_adj)
    return coords, street_adj


# Street-underlay styles. The default is the faint grey used on benchmark
# figures; the geo one is darker and thicker so the road network stays
# readable on top of a map basemap. A style is a plain kwargs dict passed to
# ``draw_street_graph`` -- panels take it as ``street_style=``, which is what
# the paper figures used to achieve by monkey-patching this module.
DEFAULT_STREET_STYLE = dict(color="lightgray", linewidth=1.0, alpha=0.35, zorder=1)
GEO_STREET_STYLE = dict(color="#6f7378", linewidth=1.2, alpha=0.50, zorder=1.2)


def draw_street_graph(ax, coords, street_adj, *, color="lightgray",
                      linewidth=1.0, alpha=0.35, zorder=1):
    """Draw the underlying street graph as faint lines (caller picks the style)."""
    n_nodes = coords.shape[0]
    for start in range(n_nodes):
        for end in range(start + 1, n_nodes):
            if np.isfinite(street_adj[start, end]) or \
                    np.isfinite(street_adj[end, start]):
                ax.plot(
                    [coords[start, 0], coords[end, 0]],
                    [coords[start, 1], coords[end, 1]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                    zorder=zorder,
                )


# Palette generation bounds. Colors are picked for a LIGHT map basemap: nothing
# so pale it washes out, nothing so dark it reads as the street graph, and enough
# chroma that hue -- not lightness -- is what distinguishes two routes.
PALETTE_LIGHTNESS = (24.0, 74.0)     # CIE L*
PALETTE_MIN_CHROMA = 22.0            # sqrt(a*^2 + b*^2)
PALETTE_GRID_STEPS = 14              # sRGB cube sampling per channel


def _srgb_to_lab(rgb):
    """Convert sRGB in ``[0, 1]`` to CIE L*a*b* (D65) -- vectorized, no deps.

    Route colors have to be spaced by how DIFFERENT THEY LOOK, and RGB distance
    is a poor proxy for that; L*a*b* is close enough to perceptual that greedy
    farthest-point selection in it produces a genuinely distinguishable set.
    """
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear @ to_xyz.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def _spaced_colors(n_colors: int):
    """``n_colors`` maximally distinguishable sRGB colors, deterministically.

    Samples the sRGB cube, drops anything too light / too dark / too grey for a
    map figure, then greedily takes the candidate furthest (in L*a*b*) from
    everything already taken. The greedy order does not depend on ``n_colors``,
    so a 20-route figure uses the first 20 colors of the 67-route palette.
    """
    axis = np.linspace(0.0, 1.0, PALETTE_GRID_STEPS)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    candidates = grid.reshape(-1, 3)
    lab = _srgb_to_lab(candidates)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = ((lab[:, 0] >= PALETTE_LIGHTNESS[0]) & (lab[:, 0] <= PALETTE_LIGHTNESS[1])
            & (chroma >= PALETTE_MIN_CHROMA))
    candidates, lab, chroma = candidates[keep], lab[keep], chroma[keep]

    picked = [int(np.argmax(chroma))]     # start from the most saturated color
    min_dist = np.linalg.norm(lab - lab[picked[0]], axis=1)
    while len(picked) < min(n_colors, len(candidates)):
        nxt = int(np.argmax(min_dist))
        picked.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(lab - lab[nxt], axis=1))
    return candidates[picked]


def large_palette(n_routes: int, name: str = "connectpt_large") -> str:
    """Register (once) and return a palette with ``n_routes`` distinct colors.

    Named matplotlib qualitative colormaps top out around 20 entries, and simply
    chaining them (tab20 + Set3 + ...) yields near-duplicate pastels that vanish
    over a light basemap. This builds the palette by perceptual spacing instead
    and caches it under ``name`` so :func:`route_colors_for` treats it like any
    built-in palette.
    """
    n_routes = max(int(n_routes), 1)
    cached = _PALETTE_CACHE.get(name)
    if cached is not None and len(cached) >= n_routes:
        return name
    rgb = _spaced_colors(n_routes)
    _PALETTE_CACHE[name] = np.concatenate(
        [rgb, np.ones((len(rgb), 1))], axis=1)
    return name


def drawn_node_mask(street_adj, *route_sets):
    """Boolean mask of the nodes worth drawing on a route figure.

    A stop with no street edge at all is an artefact of the source data, not a
    place a route can reach -- drawing it leaves a dot floating in empty space.
    A node is kept when the street graph connects it, or when one of the drawn
    route sets visits it anyway.
    """
    adj = np.asarray(street_adj)
    finite = np.isfinite(adj)
    mask = finite.any(axis=1) | finite.any(axis=0)
    for route_set in route_sets:
        if route_set is None:
            continue
        routes = get_first_route_set(route_set)
        nodes = routes[routes >= 0]
        if nodes.numel():
            mask[np.unique(nodes.numpy())] = True
    return mask


def scatter_nodes(ax, coords, mask, *, size, alpha=None, color="black", zorder=5):
    """Draw the node markers for the kept nodes only."""
    if size is None or size <= 0:
        return
    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=size, alpha=alpha,
               zorder=zorder)


def label_nodes(ax, coords, mask, *, zorder=6):
    """Write node ids on the kept nodes only."""
    for node_idx in np.flatnonzero(mask):
        x_coord, y_coord = coords[node_idx]
        ax.text(x_coord, y_coord, str(int(node_idx)), fontsize=7, color="white",
                ha="center", va="center", zorder=zorder)

def build_edge_overlap_map(*route_sets):
    """Build a map ``(route_idx, edge_key) -> (pos, count)`` for curved-edge
    rendering of overlapping routes."""
    edge_to_routes: dict[tuple[int, int], set[int]] = {}
    for route_set in route_sets:
        if route_set is None:
            continue
        route_set = get_first_route_set(route_set)
        for route_idx, route_tensor in enumerate(route_set):
            route = route_to_list(route_tensor)
            for edge in route_edge_list(route):
                edge_to_routes.setdefault(
                    undirected_edge_key(edge), set()).add(route_idx)

    overlap_map = {}
    for edge_key, route_idxs in edge_to_routes.items():
        ordered_route_idxs = sorted(route_idxs)
        count = len(ordered_route_idxs)
        for pos, route_idx in enumerate(ordered_route_idxs):
            overlap_map[(route_idx, edge_key)] = (pos, count)
    return overlap_map


def overlapping_edge_rad(route_idx, edge, overlap_map,
                         base_step=0.16, max_rad=0.28):
    """Return the matplotlib ``arc3,rad=...`` value for an edge so multiple
    routes sharing the same edge fan out instead of overlapping."""
    if overlap_map is None:
        return 0.0
    edge_key = undirected_edge_key(edge)
    pos, count = overlap_map.get((route_idx, edge_key), (0, 1))
    if count <= 1:
        return 0.0

    centered_pos = pos - (count - 1) / 2.0
    max_centered_pos = max((count - 1) / 2.0, 1.0)
    step = min(base_step, max_rad / max_centered_pos)
    direction_sign = 1.0 if tuple(edge) == edge_key else -1.0
    return centered_pos * step * direction_sign


def plot_edge(ax, coords, edge, *, color, linewidth, alpha, route_idx=0,
              overlap_map=None, linestyle="-", zorder=3):
    """Draw a single route edge.

    When ``overlap_map`` is ``None`` or no overlap is recorded for this edge,
    the edge is drawn as a straight line (matches the evaluation_seeded
    notebook's prior style). Otherwise it is drawn as an arc that fans out
    relative to other routes sharing the same edge.
    """
    start, end = edge
    start_xy = coords[start]
    end_xy = coords[end]
    rad = overlapping_edge_rad(route_idx, edge, overlap_map)

    if abs(rad) < 1e-9:
        ax.plot(
            [start_xy[0], end_xy[0]],
            [start_xy[1], end_xy[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
        return

    patch = FancyArrowPatch(
        posA=start_xy,
        posB=end_xy,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad:.4f}",
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=1,
        zorder=zorder,
    )
    ax.add_patch(patch)


def plot_edges(ax, coords, edges, *, color, linewidth, alpha, route_idx=0,
               overlap_map=None, linestyle="-", zorder=3):
    """Draw a sequence of edges using ``plot_edge``."""
    for edge in edges:
        plot_edge(
            ax,
            coords,
            edge,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            route_idx=route_idx,
            overlap_map=overlap_map,
            linestyle=linestyle,
            zorder=zorder,
        )


def summarize_route_changes(routes, reference_routes):
    """Count changed routes, added/removed edges and stops between two route
    sets (first batch element each)."""
    routes = get_first_route_set(routes)
    reference_routes = get_first_route_set(reference_routes)
    n_routes = max(routes.shape[0], reference_routes.shape[0])
    summary = {
        "changed_routes": 0,
        "added_edges": 0,
        "removed_edges": 0,
        "added_stops": 0,
        "removed_stops": 0,
    }

    for route_idx in range(n_routes):
        route = route_to_list(routes[route_idx]) \
            if route_idx < routes.shape[0] else []
        reference = route_to_list(reference_routes[route_idx]) \
            if route_idx < reference_routes.shape[0] else []

        route_edges = {undirected_edge_key(edge)
                       for edge in route_edge_list(route)}
        reference_edges = {undirected_edge_key(edge)
                           for edge in route_edge_list(reference)}
        route_nodes = set(route)
        reference_nodes = set(reference)

        added_edges = route_edges - reference_edges
        removed_edges = reference_edges - route_edges
        added_stops = route_nodes - reference_nodes
        removed_stops = reference_nodes - route_nodes

        if added_edges or removed_edges or added_stops or removed_stops:
            summary["changed_routes"] += 1
        summary["added_edges"] += len(added_edges)
        summary["removed_edges"] += len(removed_edges)
        summary["added_stops"] += len(added_stops)
        summary["removed_stops"] += len(removed_stops)

    return summary


def plot_plain_route_set(ax, routes, graph_or_coords, street_adj=None,
                         title=None, subtitle=None, *,
                         palette="tab20", with_overlap_curves=True,
                         show_node_labels=True, node_size=55, node_alpha=None,
                         street_style=None):
    """Draw a single route set on top of the underlying street graph.

    Two positional conventions are supported:

    * PyG graph:  ``plot_plain_route_set(ax, routes, graph, title, ...)`` —
      ``graph_or_coords`` is a ``HeteroData`` with ``[STOP_KEY].pos`` and
      ``.street_adj``; the 4th positional arg is the title.
    * Raw arrays: ``plot_plain_route_set(ax, routes, coords, street_adj,
      title, ...)`` — explicit coords + street_adj arrays.

    A string in the ``street_adj`` slot is interpreted as the title (the
    PyG-graph convention).
    """
    if isinstance(street_adj, str):
        street_adj, title = None, street_adj
    if title is None:
        title = ""
    routes = get_first_route_set(routes)
    coords, street_adj_arr = extract_coords_street_adj(
        graph_or_coords, street_adj)
    draw_street_graph(ax, coords, street_adj_arr,
                      **(street_style or DEFAULT_STREET_STYLE))

    colors = route_colors_for(routes, palette=palette)
    overlap_map = build_edge_overlap_map(routes) \
        if with_overlap_curves else None
    for route_idx, route_tensor in enumerate(routes):
        route = route_to_list(route_tensor)
        if len(route) < 2:
            continue
        color = colors[route_idx % len(colors)]
        plot_edges(
            ax,
            coords,
            route_edge_list(route),
            color=color,
            linewidth=3.2,
            alpha=0.95,
            route_idx=route_idx,
            overlap_map=overlap_map,
            zorder=3,
        )

    shown = drawn_node_mask(street_adj_arr, routes)
    scatter_nodes(ax, coords, shown, size=node_size, alpha=node_alpha)
    if show_node_labels:
        label_nodes(ax, coords, shown)

    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_demand_graph(ax, demand, graph_or_coords, street_adj=None,
                      title=None, subtitle=None, *, cmap="plasma", top_frac=None):
    """Draw OD demand as edges colored / weighted by demand volume.

    Each node pair ``(i, j)`` with positive total demand is drawn as a straight
    segment over the faint street graph; both its color and width scale with the
    symmetric demand ``demand[i, j] + demand[j, i]``. ``top_frac`` (0..1)
    optionally keeps only the busiest fraction of pairs to de-clutter dense
    matrices. Same positional conventions as ``plot_plain_route_set`` (a string
    in the ``street_adj`` slot is treated as the title).
    """
    from matplotlib.collections import LineCollection
    if isinstance(street_adj, str):
        street_adj, title = None, street_adj
    coords, street_adj_arr = extract_coords_street_adj(graph_or_coords, street_adj)
    if isinstance(demand, torch.Tensor):
        demand = demand.detach().cpu().numpy()
    demand = np.asarray(demand, dtype=float)
    draw_street_graph(ax, coords, street_adj_arr)

    n_nodes = coords.shape[0]
    segments, values = [], []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            total = float(demand[i, j]) + float(demand[j, i])
            if total <= 0:
                continue
            segments.append([(coords[i, 0], coords[i, 1]),
                             (coords[j, 0], coords[j, 1])])
            values.append(total)
    if segments:
        values = np.asarray(values)
        if top_frac is not None and 0.0 < top_frac < 1.0:
            keep = values >= np.quantile(values, 1.0 - top_frac)
            segments = [s for s, k in zip(segments, keep) if k]
            values = values[keep]
        vmax = float(values.max()) or 1.0
        line_coll = LineCollection(
            segments, cmap=cmap, norm=plt.Normalize(0.0, vmax),
            linewidths=0.5 + 3.5 * (values / vmax), alpha=0.85, zorder=2)
        line_coll.set_array(values)
        ax.add_collection(line_coll)
        ax.figure.colorbar(line_coll, ax=ax, fraction=0.046, pad=0.04, label="demand")

    shown = drawn_node_mask(street_adj_arr)
    scatter_nodes(ax, coords, shown, size=55)
    label_nodes(ax, coords, shown)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else (title or ""),
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_route_diff(ax, routes, reference_routes, graph_or_coords,
                    street_adj=None, title=None, subtitle=None, *,
                    palette="tab20", with_overlap_curves=True,
                    show_node_labels=True, node_size=45, street_style=None):
    """Draw ``routes`` overlaid with diff markings vs ``reference_routes``.

    Edge coverage that the candidate dropped relative to the seed is drawn as
    dashed grey lines -- this is multiplicity-aware, so trimming a duplicated
    edge from several routes down to fewer (or zero) shows one dashed copy per
    lost coverage, not just full removals. Shared edges are dimmed in the route
    color; edges new to the network are emphasized in the route color. Nodes
    added/removed are marked distinctly.

    Like :func:`plot_plain_route_set`, accepts either the PyG-graph
    convention ``(..., graph, title, ...)`` or the raw-arrays convention
    ``(..., coords, street_adj, title, ...)``; a string in the
    ``street_adj`` slot is taken as the title.
    """
    if isinstance(street_adj, str):
        street_adj, title = None, street_adj
    if title is None:
        title = ""
    routes = get_first_route_set(routes)
    reference_routes = get_first_route_set(reference_routes)
    coords, street_adj_arr = extract_coords_street_adj(
        graph_or_coords, street_adj)
    draw_street_graph(ax, coords, street_adj_arr,
                      **(street_style or DEFAULT_STREET_STYLE))

    colors = route_colors_for(routes, palette=palette)
    # Curve overlapping edges using the CANDIDATE routes (added/shared belong to
    # them); removed edges are not in the candidate so they need no curving.
    overlap_map = build_edge_overlap_map(routes) if with_overlap_curves else None

    # NETWORK-level edge MULTISETS (undirected key -> coverage count + a
    # representative edge). The diff is over the whole network WITH multiplicity,
    # NOT per route slot: reordering routes between the seed and the candidate is
    # not flagged, but reducing how many routes cover an edge IS. So a duplicated
    # edge trimmed from 3 routes down to 1 shows up as two removed copies -- the
    # core dedup signal, which a plain set difference would hide.
    def _net_counts(route_set):
        counts, rep = Counter(), {}
        for route_tensor in route_set:
            for edge in route_edge_list(route_to_list(route_tensor)):
                key = undirected_edge_key(edge)
                counts[key] += 1
                rep.setdefault(key, edge)
        return counts, rep

    ref_counts, ref_rep = _net_counts(reference_routes)
    cand_counts, _ = _net_counts(routes)
    ref_edge_map = ref_rep  # presence test for the per-route classification below

    def _removed_rad(slot):
        # straight for the first copy; fan further copies out (alternating sides)
        # so a trimmed duplicate stays visible next to the kept candidate edge.
        if slot == 0:
            return 0.0
        sign = 1.0 if slot % 2 == 1 else -1.0
        lane = (slot + 1) // 2
        return sign * min(lane * 0.16, 0.28)

    # removed COPIES: any edge whose candidate coverage dropped below the seed
    # coverage (full removal, or a duplicate trimmed away) -> dashed grey arcs.
    for key, ref_n in ref_counts.items():
        cand_n = cand_counts.get(key, 0)
        edge = ref_rep[key]
        start_xy, end_xy = coords[edge[0]], coords[edge[1]]
        for j in range(ref_n - cand_n):
            rad = _removed_rad(cand_n + j)
            if abs(rad) < 1e-9:
                ax.plot([start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]],
                        color="dimgray", linewidth=2.2, alpha=0.85,
                        linestyle="--", solid_capstyle="round", zorder=2)
            else:
                ax.add_patch(FancyArrowPatch(
                    posA=start_xy, posB=end_xy, arrowstyle="-",
                    connectionstyle=f"arc3,rad={rad:.4f}", color="dimgray",
                    linewidth=2.2, alpha=0.85, linestyle="--",
                    shrinkA=0, shrinkB=0, mutation_scale=1, zorder=2))

    # candidate edges, per route, classified added (new to the network) vs
    # shared (also in the seed network), coloured by route.
    for route_idx, route_tensor in enumerate(routes):
        route = route_to_list(route_tensor)
        color = colors[route_idx % len(colors)]
        shared_edges = [e for e in route_edge_list(route)
                        if undirected_edge_key(e) in ref_edge_map]
        added_edges = [e for e in route_edge_list(route)
                       if undirected_edge_key(e) not in ref_edge_map]
        if shared_edges:
            plot_edges(ax, coords, shared_edges, color=color, linewidth=2.0,
                       alpha=0.35, route_idx=route_idx, overlap_map=overlap_map,
                       zorder=3)
        if added_edges:
            plot_edges(ax, coords, added_edges, color=color, linewidth=4.0,
                       alpha=0.95, route_idx=route_idx, overlap_map=overlap_map,
                       zorder=4)

    # NETWORK-level node diff (added = new to the network, removed = gone).
    ref_nodes = {n for r in reference_routes for n in route_to_list(r)}
    cand_nodes = {n for r in routes for n in route_to_list(r)}
    added_nodes = sorted(cand_nodes - ref_nodes)
    removed_nodes = sorted(ref_nodes - cand_nodes)
    if added_nodes:
        ax.scatter(coords[added_nodes, 0], coords[added_nodes, 1],
                   s=45, facecolors="lime", edgecolors="black", linewidths=0.6,
                   zorder=6)
    if removed_nodes:
        ax.scatter(coords[removed_nodes, 0], coords[removed_nodes, 1],
                   s=55, c="crimson", marker="x", linewidths=1.6, zorder=6)

    shown = drawn_node_mask(street_adj_arr, routes, reference_routes)
    scatter_nodes(ax, coords, shown, size=node_size, alpha=0.65, zorder=4)
    if show_node_labels:
        label_nodes(ax, coords, shown, zorder=7)

    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Change-oriented panels (candidate vs reference network)
# ---------------------------------------------------------------------------
# ``plot_route_diff`` answers "which edges moved"; these two answer "which
# ROUTES moved, and by how much" -- the view the paper's case-study figure
# needs, where a 67-route network makes a per-edge diff unreadable.

# yellow -> red -> purple: low adjustment stays warm and thin, a heavily
# rewritten route reads as a thick purple line.
ADJ_CMAP_COLORS = ("#ffd84d", "#f05a28", "#7b1fa2")
UNCHANGED_ROUTE_COLOR = "#9aa1a8"


def adj_colormap(name: str = "route_adj_yellow_red_purple"):
    """The route-adjustment colormap (yellow -> red -> purple)."""
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(name, list(ADJ_CMAP_COLORS))


def plot_changed_route_slots(ax, routes, reference_routes, graph_or_coords,
                             street_adj=None, title=None, subtitle=None, *,
                             palette="tab20", with_overlap_curves=True,
                             node_size=15, node_alpha=0.30, street_style=None):
    """Highlight the route slots that changed; ghost the ones that did not.

    Comparison is per SLOT (route ``i`` against reference route ``i``), which is
    what an edit policy actually rewrites -- unlike the network-level multiset
    diff in :func:`plot_route_diff`. Changed routes are drawn thick and opaque
    in their route color, unchanged ones stay as a faint context layer.
    """
    routes = get_first_route_set(routes)
    reference_routes = get_first_route_set(reference_routes)
    coords, street_adj_arr = extract_coords_street_adj(graph_or_coords, street_adj)
    draw_street_graph(ax, coords, street_adj_arr,
                      **(street_style or DEFAULT_STREET_STYLE))

    colors = route_colors_for(routes, palette=palette)
    overlap_map = build_edge_overlap_map(routes) if with_overlap_curves else None
    for route_idx, route_tensor in enumerate(routes):
        route = route_to_list(route_tensor)
        if len(route) < 2:
            continue
        reference = (route_to_list(reference_routes[route_idx])
                     if route_idx < reference_routes.shape[0] else [])
        changed = route != reference
        plot_edges(
            ax, coords, route_edge_list(route),
            color=colors[route_idx % len(colors)],
            linewidth=3.8 if changed else 1.2,
            alpha=0.96 if changed else 0.16,
            route_idx=route_idx, overlap_map=overlap_map,
            zorder=4 if changed else 3,
        )

    scatter_nodes(ax, coords, drawn_node_mask(street_adj_arr, routes,
                                              reference_routes),
                  size=node_size, alpha=node_alpha)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else (title or ""),
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_route_adj_gradient(ax, routes, reference_routes, graph_or_coords,
                            street_adj=None, title=None, subtitle=None, *,
                            adj_values=None, with_overlap_curves=True,
                            node_size=15, node_alpha=0.30, street_style=None):
    """Color each route by its adjustment degree vs the reference network.

    ``adj_values`` is the per-route adjustment vector; when omitted it is
    computed via ``reports.geo.route_adjustments`` (objective-configured gap and
    mode). Untouched routes (adj ~ 0) are drawn grey and thin underneath so the
    rewritten ones stand out of a dense network. Returns
    ``(ScalarMappable, mean_adj)`` -- the mappable is what a colorbar needs.
    """
    import matplotlib.colors as mcolors

    routes = get_first_route_set(routes)
    reference_routes = get_first_route_set(reference_routes)
    coords, street_adj_arr = extract_coords_street_adj(graph_or_coords, street_adj)
    draw_street_graph(ax, coords, street_adj_arr,
                      **(street_style or DEFAULT_STREET_STYLE))

    if adj_values is None:
        from .geo import route_adjustments
        adj_values = route_adjustments(routes, reference_routes)
    adj_values = np.asarray(adj_values, dtype=float)
    cmap = adj_colormap()
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    overlap_map = build_edge_overlap_map(routes) if with_overlap_curves else None

    items = []
    for route_idx, route_tensor in enumerate(routes):
        route = route_to_list(route_tensor)
        if len(route) < 2:
            continue
        value = float(adj_values[route_idx]) if route_idx < len(adj_values) else 0.0
        items.append((value, route_idx, route))

    # Quiet context layer first, then rewritten routes on top (ascending adj) so
    # the most-changed route is never buried under an untouched one.
    for value, route_idx, route in sorted(items, key=lambda item: item[0]):
        if value <= 1e-6:
            plot_edges(ax, coords, route_edge_list(route),
                       color=UNCHANGED_ROUTE_COLOR, linewidth=0.9, alpha=0.16,
                       route_idx=route_idx, overlap_map=overlap_map, zorder=2.6)
            continue
        plot_edges(ax, coords, route_edge_list(route), color=cmap(norm(value)),
                   linewidth=2.0 + 3.5 * value, alpha=0.98,
                   route_idx=route_idx, overlap_map=overlap_map,
                   zorder=4 + value)

    scatter_nodes(ax, coords, drawn_node_mask(street_adj_arr, routes,
                                              reference_routes),
                  size=node_size, alpha=node_alpha, zorder=5.5)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else (title or ""),
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(adj_values)
    mean_adj = float(np.mean(adj_values)) if adj_values.size else 0.0
    return scalar, mean_adj

# ---------------------------------------------------------------------------
# Cost-component enable / disable helpers
# ---------------------------------------------------------------------------
# Both notebooks can switch individual cost components off (demand / route /
# connectivity). When a component is disabled the library drops it from the
# weighted cost; these helpers let the notebook tables / plots drop the
# matching columns and panels so a disabled component is never displayed.

COST_COMPONENT_NAMES = ("demand", "route", "connectivity")

# Human-readable axis / panel labels per component.

# Base (undecorated) dataframe columns owned by each cost component, across
# both the evaluation_seeded and lc_improvement_training notebooks. The
# filter below also recognizes these wrapped in ``mean_`` / ``std_``
# prefixes and ``_with_worse`` / ``_without_worse`` suffixes (worse-accept
# summary tables).


def resolve_enabled_components(enabled):
    """Normalize an ``enabled`` argument to a tuple of component names.

    Accepts ``None`` (all enabled), a cost module exposing
    ``enabled_component_names``, a result/history dict carrying an
    ``"enabled_components"`` key, or an explicit iterable of names.
    """
    if enabled is None:
        return COST_COMPONENT_NAMES
    if hasattr(enabled, "enabled_component_names"):
        return tuple(enabled.enabled_component_names)
    if isinstance(enabled, dict):
        names = enabled.get("enabled_components", COST_COMPONENT_NAMES)
        return tuple(names)
    return tuple(enabled)


def disabled_components(enabled):
    """Return the disabled component names (canonical order)."""
    active = set(resolve_enabled_components(enabled))
    return tuple(c for c in COST_COMPONENT_NAMES if c not in active)


__all__ = [
    "route_to_list",
    "route_edge_list",
    "undirected_edge_key",
    "get_first_route_set",
    "route_colors_for",
    "route_colors_for_n",
    "extract_coords_street_adj",
    "draw_street_graph",
    "drawn_node_mask",
    "DEFAULT_STREET_STYLE",
    "GEO_STREET_STYLE",
    "large_palette",
    "adj_colormap",
    "build_edge_overlap_map",
    "overlapping_edge_rad",
    "plot_edge",
    "plot_edges",
    "summarize_route_changes",
    "plot_plain_route_set",
    "plot_route_diff",
    "plot_changed_route_slots",
    "plot_route_adj_gradient",
    "COST_COMPONENT_NAMES",
    "resolve_enabled_components",
    "disabled_components",
]

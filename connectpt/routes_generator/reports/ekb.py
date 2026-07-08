"""EKB (Ekaterinburg) case-study helpers: GIS projection, connectivity stats,
route-figure street underlays. Library home (was eval_lib.ekb)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer

from ..core.paths import ARTIFACTS_DIR, DATASETS_DIR
from ..data.routes import as_route_tensor


EKB_DATA_DIR = DATASETS_DIR / "EKB"
EKB_COORD_CRS = "EPSG:32641"  # UTM zone 41N: Ekaterinburg -> WGS84.
EKB_CITY_NAME = "EKB"
EKB_STATIC_MAP_PATH = ARTIFACTS_DIR / "paper_results" / "ekb_seed_routes_static.png"


# EKB data loading moved to the library -- single implementation.
from connectpt.routes_generator.data.loaders import (  # noqa: F401
    load_ekb_tensors, load_ekb_routes, ekb_spec)


def make_ekb_crop_case(
    tensors,
    routes,
    target_nodes=300,
    min_route_len=None,
    max_route_len=None,
    keep_largest_component=True,
):
    """Build a safely reindexed geometric EKB crop for experiments.

    The crop keeps the nearest ``target_nodes`` to the median coordinate, slices
    all node-pair tensors, splits seed routes into contiguous in-crop fragments,
    and remaps route node ids to the cropped graph.
    """
    node_locs = tensors["node_locs"]
    xy = node_locs.detach().cpu().numpy() if isinstance(node_locs, torch.Tensor) else np.asarray(node_locs)
    n_total = int(xy.shape[0])
    target_nodes = int(target_nodes)
    if target_nodes <= 0 or target_nodes > n_total:
        raise ValueError(f"target_nodes must be in [1, {n_total}], got {target_nodes}")

    center = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - center[None, :], axis=1)
    selected = np.argsort(dist, kind="mergesort")[:target_nodes]
    mask = np.zeros(n_total, dtype=bool)
    mask[selected] = True
    radius = float(dist[selected].max())

    component_sizes_before = None
    if keep_largest_component:
        components = _components_from_adj(tensors["street_adj"], mask)
        component_sizes_before = [len(comp) for comp in components]
        if len(components) > 1:
            mask[:] = False
            mask[np.asarray(components[0], dtype=int)] = True

    crop_ids = np.flatnonzero(mask)
    old_to_new = np.full(n_total, -1, dtype=np.int64)
    old_to_new[crop_ids] = np.arange(len(crop_ids), dtype=np.int64)

    cropped_tensors = {}
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and value.shape[0] == n_total and value.shape[1] == n_total:
                cropped_tensors[key] = value[crop_ids][:, crop_ids].clone()
            elif value.ndim >= 1 and value.shape[0] == n_total:
                cropped_tensors[key] = value[crop_ids].clone()
            else:
                cropped_tensors[key] = value.clone()
        else:
            arr = np.asarray(value)
            if arr.ndim >= 2 and arr.shape[0] == n_total and arr.shape[1] == n_total:
                cropped_tensors[key] = arr[np.ix_(crop_ids, crop_ids)].copy()
            elif arr.ndim >= 1 and arr.shape[0] == n_total:
                cropped_tensors[key] = arr[crop_ids].copy()
            else:
                cropped_tensors[key] = arr.copy()

    rr_in = as_route_tensor(routes).long()
    batched = rr_in.ndim == 3
    rr = rr_in[0] if batched else rr_in
    route_lens = (rr > -1).sum(dim=-1)
    if min_route_len is None:
        nonempty = route_lens[route_lens > 0]
        min_route_len = int(nonempty.min().item()) if nonempty.numel() else 1
    min_route_len = int(min_route_len)
    if max_route_len is None:
        max_route_len = int(rr.shape[-1])
    max_route_len = int(max_route_len)

    route_segments = []
    source_route_ids = []
    for route_idx, route_tensor in enumerate(rr):
        current = []
        for raw_node in route_tensor.tolist():
            node = int(raw_node)
            if node >= 0 and mask[node]:
                current.append(int(old_to_new[node]))
            else:
                _append_crop_segment(
                    route_segments, source_route_ids, current, route_idx,
                    min_route_len, max_route_len)
                current = []
        _append_crop_segment(
            route_segments, source_route_ids, current, route_idx,
            min_route_len, max_route_len)

    if not route_segments:
        raise ValueError(
            "EKB crop produced no valid route fragments; lower min_route_len "
            "or increase target_nodes.")

    cropped_routes = torch.full(
        (len(route_segments), max_route_len), -1, dtype=torch.long)
    for idx, segment in enumerate(route_segments):
        cropped_routes[idx, :len(segment)] = torch.tensor(segment, dtype=torch.long)

    valid_nodes = cropped_routes[cropped_routes >= 0]
    if valid_nodes.numel() and int(valid_nodes.max().item()) >= len(crop_ids):
        raise AssertionError("cropped routes contain node ids outside cropped tensors")
    cropped_lens = (cropped_routes > -1).sum(dim=-1)
    if bool((cropped_lens < min_route_len).any()):
        raise AssertionError("cropped routes shorter than min_route_len were kept")
    if batched:
        cropped_routes = cropped_routes[None]

    meta = {
        "crop_enabled": True,
        "target_nodes": target_nodes,
        "n_nodes_full": n_total,
        "n_nodes": int(len(crop_ids)),
        "n_routes_full": int(rr.shape[0]),
        "n_routes": int(len(route_segments)),
        "min_route_len": min_route_len,
        "max_route_len": max_route_len,
        "center_xy": [float(center[0]), float(center[1])],
        "radius_m": radius,
        "kept_old_node_ids": crop_ids.tolist(),
        "source_route_ids": [int(v) for v in source_route_ids],
        "keep_largest_component": bool(keep_largest_component),
        "component_sizes_before_filter": component_sizes_before,
    }
    return cropped_tensors, cropped_routes, meta


def project_ekb_coords(coords, source_crs=EKB_COORD_CRS):
    """Project EKB metric coordinates to WGS84 ``(lat, lon)`` pairs."""
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack((lat, lon))


def ekb_route_stats(routes):
    """Compact route-set statistics for map titles and notebook tables."""
    rr = as_route_tensor(routes)
    if rr.ndim == 3:
        rr = rr[0]
    route_lens = (rr > -1).sum(dim=-1)
    nonempty = route_lens[route_lens > 0]
    stops = {int(node) for route in rr for node in route.tolist() if int(node) >= 0}
    return {
        "n_routes": int(rr.shape[0]),
        "min_len": int(nonempty.min().item()) if nonempty.numel() else 0,
        "max_len": int(nonempty.max().item()) if nonempty.numel() else 0,
        "mean_len": float(nonempty.float().mean().item()) if nonempty.numel() else 0.0,
        "unique_stops": len(stops),
    }


def ekb_connectivity_stats(tensors, routes=None):
    """Connectivity diagnostics for the EKB street graph."""
    street_adj = tensors["street_adj"]
    demand = tensors["demand"]
    if isinstance(street_adj, torch.Tensor):
        street_adj = street_adj.detach().cpu()
    else:
        street_adj = torch.as_tensor(street_adj)
    if isinstance(demand, torch.Tensor):
        demand = demand.detach().cpu()
    else:
        demand = torch.as_tensor(demand)

    n_nodes = int(street_adj.shape[0])
    directed = (street_adj > 0) & torch.isfinite(street_adj)
    undirected = directed | directed.T
    seen = torch.zeros(n_nodes, dtype=torch.bool)
    components = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neighbors = torch.where(undirected[node] & ~seen)[0].tolist()
            for nxt in neighbors:
                seen[nxt] = True
                stack.append(int(nxt))
        components.append(comp)
    components = sorted(components, key=len, reverse=True)

    comp_id = torch.empty(n_nodes, dtype=torch.long)
    for idx, comp in enumerate(components):
        comp_id[torch.tensor(comp)] = idx
    cross_component = comp_id[:, None] != comp_id[None, :]
    total_demand = demand.sum()
    cross_demand = demand[cross_component].sum()

    route_nodes = set()
    route_self_loops = []
    routes_with_repeats = []
    if routes is not None:
        rr = as_route_tensor(routes)
        if rr.ndim == 3:
            rr = rr[0]
        giant = set(components[0]) if components else set()
        for route_idx, route_tensor in enumerate(rr):
            nodes = _route_nodes(route_tensor)
            route_nodes.update(nodes)
            if len(nodes) != len(set(nodes)):
                routes_with_repeats.append(route_idx)
            for start, end in zip(nodes[:-1], nodes[1:]):
                if start == end:
                    route_self_loops.append((route_idx, start))
        route_nodes_outside_giant = sorted(route_nodes - giant)
    else:
        route_nodes_outside_giant = []

    isolated_nodes = [comp[0] for comp in components if len(comp) == 1]
    return {
        "n_nodes": n_nodes,
        "directed_arcs": int(directed.sum().item()),
        "undirected_edges": int(torch.triu(undirected, diagonal=1).sum().item()),
        "symmetric": bool(torch.equal(directed, directed.T)),
        "n_components": len(components),
        "component_sizes": [len(comp) for comp in components],
        "isolated_nodes": isolated_nodes,
        "cross_component_demand": float(cross_demand.item()),
        "cross_component_demand_pct": (
            float(100.0 * cross_demand.item() / total_demand.item())
            if float(total_demand.item()) > 0 else 0.0),
        "route_covered_nodes": len(route_nodes) if routes is not None else None,
        "route_nodes_outside_giant": route_nodes_outside_giant,
        "routes_with_repeats": routes_with_repeats,
        "route_self_loops": route_self_loops,
    }


def make_ekb_plot_street_adj(*route_sets, n_nodes=None, source_adj=None):
    """Sparse route-edge adjacency for readable matplotlib EKB route figures.

    ``EkbTravelTimes.txt`` is a dense travel-time matrix. Passing it to the
    generic route plotting helper would draw every node pair as a base edge,
    so this helper builds a lightweight underlay from just the route edges
    present in the supplied seed/final route sets.
    """
    src = None
    if source_adj is not None:
        src = source_adj.detach().cpu().numpy() if isinstance(source_adj, torch.Tensor) else source_adj
        src = np.asarray(src, dtype=float)
        if n_nodes is None:
            n_nodes = int(src.shape[0])

    route_sets_2d = []
    max_node = -1
    for route_set in route_sets:
        if route_set is None:
            continue
        rr = as_route_tensor(route_set)
        if rr.ndim == 3:
            rr = rr[0]
        route_sets_2d.append(rr)
        valid = rr[rr >= 0]
        if valid.numel():
            max_node = max(max_node, int(valid.max().item()))

    if n_nodes is None:
        n_nodes = max_node + 1
    adj = np.full((int(n_nodes), int(n_nodes)), np.inf, dtype=np.float32)

    for rr in route_sets_2d:
        for route_tensor in rr:
            nodes = _route_nodes(route_tensor)
            for start, end in zip(nodes[:-1], nodes[1:]):
                if start < 0 or end < 0 or start >= n_nodes or end >= n_nodes:
                    continue
                weight = 1.0
                if src is not None:
                    forward = src[start, end]
                    backward = src[end, start]
                    candidates = [v for v in (forward, backward) if np.isfinite(v)]
                    if candidates:
                        weight = float(min(candidates))
                adj[start, end] = min(adj[start, end], weight)
                adj[end, start] = min(adj[end, start], weight)
    return adj


def make_ekb_street_underlay_adj(source_adj):
    """Static EKB street-graph underlay for matplotlib route figures.

    Missing EKB street edges are encoded as zeros, while the generic plotting
    helper treats every finite value as drawable. This converts the travel-time
    matrix to the plotting convention: finite positive arcs remain visible and
    all missing/self edges become ``inf``.
    """
    src = source_adj.detach().cpu().numpy() if isinstance(source_adj, torch.Tensor) else source_adj
    src = np.asarray(src, dtype=float)
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"EKB street adjacency must be square, got {src.shape}")

    positive = np.isfinite(src) & (src > 0)
    forward = np.where(positive, src, np.inf)
    backward = np.where(positive.T, src.T, np.inf)
    underlay = np.minimum(forward, backward).astype(np.float32)
    underlay[~(positive | positive.T)] = np.inf
    np.fill_diagonal(underlay, np.inf)
    return underlay


def _route_nodes(route_tensor):
    return [int(node) for node in route_tensor.tolist() if int(node) >= 0]


def _append_crop_segment(
    route_segments,
    source_route_ids,
    current,
    route_idx,
    min_route_len,
    max_route_len,
):
    if len(current) < min_route_len:
        return
    if len(current) > max_route_len:
        current = current[:max_route_len]
    route_segments.append(list(current))
    source_route_ids.append(int(route_idx))


def _components_from_adj(street_adj, mask):
    src = street_adj.detach().cpu().numpy() if isinstance(street_adj, torch.Tensor) else np.asarray(street_adj)
    active = np.flatnonzero(mask)
    active_set = set(int(v) for v in active.tolist())
    positive = np.isfinite(src) & (src > 0)
    undirected = positive | positive.T
    seen = set()
    components = []
    for start in active:
        start = int(start)
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neighbors = np.flatnonzero(undirected[node] & mask)
            for nxt in neighbors:
                nxt = int(nxt)
                if nxt in active_set and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return sorted(components, key=len, reverse=True)

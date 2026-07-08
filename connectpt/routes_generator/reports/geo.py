"""Geo / GIS rendering helpers -- general, not city-specific.

An instance whose ``node_locs`` are real-world coordinates (a ``crs`` in its
meta) can be drawn on a street underlay and projected to lat/lon. These helpers
were generalized out of the EKB case study so any geo city (not just EKB) can use
the optional ``kind="gis"`` render path -- ``reports.ekb`` just supplies the EKB
constants + crop and re-exports these.
"""
from __future__ import annotations

import numpy as np
import torch
from pyproj import Transformer

from ..data.routes import as_route_tensor


def _route_nodes(route_tensor):
    return [int(node) for node in route_tensor.tolist() if int(node) >= 0]


def project_coords(coords, source_crs):
    """Project metric coordinates in ``source_crs`` to WGS84 ``(lat, lon)`` pairs."""
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=float)
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack((lat, lon))


def route_stats(routes):
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


def network_connectivity_stats(tensors, routes=None):
    """Connectivity diagnostics for a street graph (+ optional route coverage)."""
    street_adj = tensors["street_adj"]
    demand = tensors["demand"]
    street_adj = (street_adj.detach().cpu() if isinstance(street_adj, torch.Tensor)
                  else torch.as_tensor(street_adj))
    demand = (demand.detach().cpu() if isinstance(demand, torch.Tensor)
              else torch.as_tensor(demand))

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


def route_underlay_adj(*route_sets, n_nodes=None, source_adj=None):
    """Sparse route-edge adjacency for readable matplotlib route figures.

    A dense travel-time matrix would draw every node pair as a base edge; this
    builds a lightweight underlay from only the edges present in the supplied
    route sets (optionally weighted by ``source_adj``)."""
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
                    forward, backward = src[start, end], src[end, start]
                    candidates = [v for v in (forward, backward) if np.isfinite(v)]
                    if candidates:
                        weight = float(min(candidates))
                adj[start, end] = min(adj[start, end], weight)
                adj[end, start] = min(adj[end, start], weight)
    return adj


def street_underlay_adj(source_adj):
    """Static street-graph underlay for matplotlib route figures.

    Converts a travel-time matrix (missing edges as 0/inf) to the plotting
    convention: finite positive arcs stay visible, missing/self edges -> ``inf``.
    """
    src = source_adj.detach().cpu().numpy() if isinstance(source_adj, torch.Tensor) else source_adj
    src = np.asarray(src, dtype=float)
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError(f"street adjacency must be square, got {src.shape}")
    positive = np.isfinite(src) & (src > 0)
    forward = np.where(positive, src, np.inf)
    backward = np.where(positive.T, src.T, np.inf)
    underlay = np.minimum(forward, backward).astype(np.float32)
    underlay[~(positive | positive.T)] = np.inf
    np.fill_diagonal(underlay, np.inf)
    return underlay

"""EKB case-study helpers for paper_combined.ipynb."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer

from .context import ARTIFACTS_DIR, DATASETS_DIR
from .helpers import as_route_tensor


EKB_DATA_DIR = DATASETS_DIR / "EKB"
EKB_COORD_CRS = "EPSG:32641"  # UTM zone 41N: Ekaterinburg -> WGS84.
EKB_CITY_NAME = "EKB"
EKB_STATIC_MAP_PATH = ARTIFACTS_DIR / "paper_results" / "ekb_seed_routes_static.png"


def load_ekb_tensors(data_dir=EKB_DATA_DIR, travel_time_scale=60.0):
    """Load EKB coords, travel times, and demand in tensor-dataset format.

    The travel-time files follow the Mumford convention used elsewhere in the
    repo, so values are interpreted as minutes and converted to seconds.
    """
    data_dir = Path(data_dir)
    coords = np.genfromtxt(data_dir / "EkbCoords.txt", skip_header=1)
    travel_times = np.genfromtxt(data_dir / "EkbTravelTimes.txt")
    demand = np.genfromtxt(data_dir / "EkbDemand.txt")

    node_locs = torch.tensor(np.atleast_2d(coords), dtype=torch.float32)
    street_adj = torch.tensor(np.atleast_2d(travel_times), dtype=torch.float32)
    demand = torch.tensor(np.atleast_2d(demand), dtype=torch.float32)
    street_adj = street_adj * float(travel_time_scale)

    n_nodes = node_locs.shape[0]
    expected = (n_nodes, n_nodes)
    if tuple(street_adj.shape) != expected:
        raise ValueError(
            f"{data_dir}: EkbTravelTimes.txt has shape "
            f"{tuple(street_adj.shape)}, expected {expected}")
    if tuple(demand.shape) != expected:
        raise ValueError(
            f"{data_dir}: EkbDemand.txt has shape {tuple(demand.shape)}, "
            f"expected {expected}")

    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


def load_ekb_routes(data_dir=EKB_DATA_DIR, batched=True):
    """Load EKB seed routes as a padded tensor."""
    data_dir = Path(data_dir)
    try:
        routes = torch.load(
            data_dir / "EkbRoutes.pkl", map_location="cpu",
            weights_only=False)
    except TypeError:
        routes = torch.load(data_dir / "EkbRoutes.pkl", map_location="cpu")
    routes = as_route_tensor(routes).long()
    if batched and routes.ndim == 2:
        routes = routes[None]
    return routes


def ekb_spec(routes=None, city=EKB_CITY_NAME):
    """Build an eval spec from the supplied EKB seed routes."""
    routes = load_ekb_routes(batched=True) if routes is None else as_route_tensor(routes)
    rr = routes[0] if routes.ndim == 3 else routes
    route_lens = (rr > -1).sum(dim=-1)
    return {
        "city": city,
        "n_routes": int(rr.shape[0]),
        "min_route_len": int(route_lens[route_lens > 0].min().item()),
        "max_route_len": int(rr.shape[-1]),
    }


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

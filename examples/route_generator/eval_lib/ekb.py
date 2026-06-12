"""EKB case-study helpers for paper_combined.ipynb."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer

from .context import ARTIFACTS_DIR, DATASETS_DIR
from .helpers import as_route_tensor


EKB_DATA_DIR = DATASETS_DIR / "EKB"
EKB_COORD_CRS = "EPSG:32641"  # UTM zone 41N: Ekaterinburg -> WGS84.
EKB_CITY_NAME = "EKB"
EKB_MAP_PATH = ARTIFACTS_DIR / "paper_results" / "ekb_seed_routes_map.html"

_ROUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
]


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


def _route_nodes(route_tensor):
    return [int(node) for node in route_tensor.tolist() if int(node) >= 0]


def _metric_summary_html(metrics):
    if not metrics:
        return ""
    keys = ["cost", "cost_delta", "cost_delta_pct", "RTT", "WMC", "ATT",
            "adj_vs_seed", "d_un", "redun%"]
    bits = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if key in {"RTT", "ATT"}:
            bits.append(f"{key}={value:.0f}")
        elif key in {"d_un", "redun%", "cost_delta_pct"}:
            bits.append(f"{key}={value:.1f}")
        else:
            bits.append(f"{key}={value:.3f}")
    return " | ".join(bits)


def build_ekb_folium_map(routes, coords, *, metrics=None,
                         source_crs=EKB_COORD_CRS, max_routes=None,
                         tiles="CartoDB positron", show_stops=True,
                         title="EKB seed routes"):
    """Create a folium map with EKB route polylines on a web basemap."""
    import folium

    rr = as_route_tensor(routes)
    if rr.ndim == 3:
        rr = rr[0]
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    latlon = project_ekb_coords(coords, source_crs=source_crs)
    stats = ekb_route_stats(rr)

    fmap = folium.Map(
        location=[float(latlon[:, 0].mean()), float(latlon[:, 1].mean())],
        zoom_start=11,
        tiles=tiles,
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    n_routes = rr.shape[0] if max_routes is None else min(int(max_routes), rr.shape[0])
    for route_idx, route_tensor in enumerate(rr[:n_routes]):
        nodes = _route_nodes(route_tensor)
        if len(nodes) < 2:
            continue
        points = [(float(latlon[node, 0]), float(latlon[node, 1]))
                  for node in nodes]
        color = _ROUTE_COLORS[route_idx % len(_ROUTE_COLORS)]
        tooltip = f"route {route_idx}: {len(nodes)} stops"
        folium.PolyLine(
            points, color=color, weight=3.0, opacity=0.78,
            tooltip=tooltip).add_to(fmap)
        folium.CircleMarker(
            points[0], radius=3.5, color=color, fill=True,
            fill_opacity=0.9, tooltip=f"route {route_idx} start").add_to(fmap)
        folium.CircleMarker(
            points[-1], radius=3.5, color=color, fill=True,
            fill_opacity=0.9, tooltip=f"route {route_idx} end").add_to(fmap)

    if show_stops:
        for node_idx, (lat, lon) in enumerate(latlon):
            folium.CircleMarker(
                [float(lat), float(lon)], radius=1.5, color="#111111",
                weight=0.5, fill=True, fill_opacity=0.45,
                tooltip=f"stop {node_idx}").add_to(fmap)

    stat_line = (
        f"{stats['n_routes']} routes | {stats['unique_stops']} covered stops | "
        f"len {stats['min_len']}-{stats['max_len']} "
        f"(mean {stats['mean_len']:.1f})")
    metric_line = _metric_summary_html(metrics)
    subtitle = stat_line if not metric_line else f"{stat_line}<br>{escape(metric_line)}"
    title_html = f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
                background: rgba(255,255,255,0.92); padding: 8px 10px;
                border: 1px solid #999; border-radius: 4px;
                font-family: Arial, sans-serif; font-size: 13px;">
      <b>{escape(title)}</b><br>{subtitle}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def save_ekb_folium_map(routes, coords, path=EKB_MAP_PATH, **kwargs):
    """Build and save the EKB folium map, returning the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap = build_ekb_folium_map(routes, coords, **kwargs)
    fmap.save(path)
    return path

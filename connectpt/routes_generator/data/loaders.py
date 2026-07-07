"""Raw problem-data loaders for the experiment data sources.

Reads the benchmark / EKB text files and the MACSA scenario dirs into the plain
``tensors`` dict (``node_locs`` / ``street_adj`` / ``demand``) the data sources
wrap in an :class:`~connectpt.routes_generator.data.sources.Instance`. Travel
times follow the Mumford convention (minutes -> seconds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..core.paths import BENCHMARK_DIR, DATASETS_DIR
from .routes import as_route_tensor

# Default route-length bounds for the geometric case studies (EKB / MACSA),
# where the seed network -- not a benchmark spec -- sets the counts.
MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 12

EKB_DATA_DIR = DATASETS_DIR / "EKB"
EKB_COORD_CRS = "EPSG:32641"  # UTM zone 41N: Ekaterinburg -> WGS84
EKB_CITY_NAME = "EKB"

# Standard benchmark instances (route counts / length bounds from the litterature).
BENCHMARK_SPECS = [
    {"city": "Mandl",    "n_routes": 6,  "min_route_len": 2,  "max_route_len": 8},
    {"city": "Mumford0", "n_routes": 12, "min_route_len": 2,  "max_route_len": 15},
    {"city": "Mumford1", "n_routes": 15, "min_route_len": 10, "max_route_len": 30},
    {"city": "Mumford2", "n_routes": 56, "min_route_len": 10, "max_route_len": 22},
    {"city": "Mumford3", "n_routes": 60, "min_route_len": 12, "max_route_len": 25},
]


def benchmark_spec(city: str) -> dict:
    """Route-count / length bounds for a benchmark city."""
    return next(s for s in BENCHMARK_SPECS if s["city"] == city)


def load_benchmark_tensors(city: str) -> dict:
    """Load a benchmark city's coords / travel-times / demand text files."""
    node_locs = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city}Coords.txt", skip_header=1),
        dtype=torch.float32)
    street_adj = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city}TravelTimes.txt"),
        dtype=torch.float32) * 60
    demand = torch.tensor(
        np.genfromtxt(BENCHMARK_DIR / f"{city}Demand.txt"),
        dtype=torch.float32)
    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


def load_ekb_tensors(data_dir=EKB_DATA_DIR, travel_time_scale: float = 60.0) -> dict:
    """Load EKB coords / travel-times / demand in tensor-dataset format."""
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
            f"{data_dir}: EkbTravelTimes.txt has shape {tuple(street_adj.shape)}, "
            f"expected {expected}")
    if tuple(demand.shape) != expected:
        raise ValueError(
            f"{data_dir}: EkbDemand.txt has shape {tuple(demand.shape)}, "
            f"expected {expected}")
    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


def load_ekb_routes(data_dir=EKB_DATA_DIR, batched: bool = True) -> torch.Tensor:
    """Load the EKB seed routes as a padded tensor."""
    data_dir = Path(data_dir)
    try:
        routes = torch.load(data_dir / "EkbRoutes.pkl", map_location="cpu",
                            weights_only=False)
    except TypeError:
        routes = torch.load(data_dir / "EkbRoutes.pkl", map_location="cpu")
    routes = as_route_tensor(routes).long()
    if batched and routes.ndim == 2:
        routes = routes[None]
    return routes


def ekb_spec(routes=None, city: str = EKB_CITY_NAME) -> dict:
    """Build an eval spec (n_routes / length bounds) from the EKB seed routes."""
    routes = load_ekb_routes(batched=True) if routes is None else as_route_tensor(routes)
    rr = routes[0] if routes.ndim == 3 else routes
    route_lens = (rr > -1).sum(dim=-1)
    return {
        "city": city,
        "n_routes": int(rr.shape[0]),
        "min_route_len": int(route_lens[route_lens > 0].min().item()),
        "max_route_len": int(rr.shape[-1]),
    }


def macsa_eval_bounds(scenario) -> tuple[int, int, int]:
    """Route-count / length bounds for evaluating a MACSA scenario."""
    n_nodes = int(scenario["tensors"]["node_locs"].shape[0])
    n_routes = int(scenario["routes"].shape[1])
    seed_route_lens = (scenario["routes"] > -1).sum(dim=-1)
    longest_seed_route = (int(seed_route_lens.max().item())
                          if seed_route_lens.numel() else MIN_ROUTE_LEN)
    max_route_len = min(n_nodes, max(MAX_ROUTE_LEN, longest_seed_route))
    return n_routes, MIN_ROUTE_LEN, max_route_len

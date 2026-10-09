"""Load physical city tensors from an absolute directory.

Coords.txt contains a node count and N coordinate pairs; TravelTimes.txt
contains an N x N matrix in minutes; Demand.txt contains N x N OD demand.
Configuration and initial route loading are assembled by the caller."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .const import TRAVEL_TIME_SCALE


@dataclass(frozen=True)
class InstanceParams:
    name: str
    path: Path
    n_routes: int
    min_route_len: int
    max_route_len: int
    crs: str | None


@dataclass
class Instance:
    """A loaded problem instance: tensors + initial routes + route bounds."""

    label: str
    tensors: dict
    init_routes: torch.Tensor | None
    spec: dict  # n_routes / min_route_len / max_route_len
    coords: Any = None  # node_locs, for route plotting
    street_adj: Any = None  # street adjacency, for route plotting
    meta: dict = field(default_factory=dict)


def load_city_tensors(instance_dir: Path) -> dict:
    """``node_locs`` / ``street_adj`` / ``demand`` from the three text files."""
    d = Path(instance_dir)
    if not d.is_absolute():
        raise ValueError("instance directory must be an absolute Path")
    node_locs = torch.tensor(
        np.atleast_2d(np.genfromtxt(d / "Coords.txt", skip_header=1)), dtype=torch.float32
    )
    street_adj = (
        torch.tensor(np.atleast_2d(np.genfromtxt(d / "TravelTimes.txt")), dtype=torch.float32)
        * TRAVEL_TIME_SCALE
    )
    demand = torch.tensor(np.atleast_2d(np.genfromtxt(d / "Demand.txt")), dtype=torch.float32)

    expected = (node_locs.shape[0],) * 2
    for name, mat in (("TravelTimes.txt", street_adj), ("Demand.txt", demand)):
        if tuple(mat.shape) != expected:
            raise ValueError(f"{d / name}: shape {tuple(mat.shape)}, expected {expected}")
    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


def load_city(instance: InstanceParams) -> Instance:
    """Read physical city data from resolved parameters; routes are bound by run."""
    if not isinstance(instance, InstanceParams):
        raise TypeError("load_city requires InstanceParams")
    name = instance.name
    d = Path(instance.path)
    if not d.is_dir():
        raise FileNotFoundError(
            f"instance folder not found: {d}. An instance is a folder with "
            f"Coords.txt / TravelTimes.txt / Demand.txt."
        )
    tensors = load_city_tensors(d)
    spec = {
        "city": name,
        "n_routes": instance.n_routes,
        "min_route_len": instance.min_route_len,
        "max_route_len": instance.max_route_len,
    }
    meta = {"crs": instance.crs} if instance.crs is not None else {}
    return Instance(
        label=name,
        tensors=tensors,
        init_routes=None,
        spec=spec,
        coords=tensors["node_locs"],
        street_adj=tensors["street_adj"],
        meta=meta,
    )

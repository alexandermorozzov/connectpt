"""Load a problem instance from ``datasets/instances/<name>/``.

One folder per instance, always the same four files:

* ``Coords.txt``      -- first line is the node count N, then N lines of "x y";
* ``TravelTimes.txt`` -- N x N street travel times **in minutes**;
* ``Demand.txt``      -- N x N OD demand;
* ``Routes.txt``      -- the initial network: one route per line, whitespace-
  separated **0-indexed** node ids, ``#`` comments and blank lines ignored.

Route counts and length bounds come from ``cfg/instance/<name>.yaml``; the loader
validates the files against them. A new city is a new folder plus that yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.paths import DATASETS_DIR, resolve_under_root
from .loaders import instance_spec
from .routes import pad_routes_to

INSTANCES_DIR = DATASETS_DIR / "instances"
TRAVEL_TIME_SCALE = 60.0  # files are in minutes, the cost modules work in seconds


@dataclass
class Instance:
    """A loaded problem instance: tensors + initial routes + route bounds."""

    label: str
    tensors: dict
    init_routes: torch.Tensor
    spec: dict                       # n_routes / min_route_len / max_route_len
    coords: Any = None               # node_locs, for route plotting
    street_adj: Any = None           # street adjacency, for route plotting
    meta: dict = field(default_factory=dict)


def load_matrices(instance_dir) -> dict:
    """``node_locs`` / ``street_adj`` / ``demand`` from the three text files."""
    d = Path(instance_dir)
    node_locs = torch.tensor(
        np.atleast_2d(np.genfromtxt(d / "Coords.txt", skip_header=1)),
        dtype=torch.float32)
    street_adj = torch.tensor(
        np.atleast_2d(np.genfromtxt(d / "TravelTimes.txt")),
        dtype=torch.float32) * TRAVEL_TIME_SCALE
    demand = torch.tensor(
        np.atleast_2d(np.genfromtxt(d / "Demand.txt")), dtype=torch.float32)

    expected = (node_locs.shape[0],) * 2
    for name, mat in (("TravelTimes.txt", street_adj), ("Demand.txt", demand)):
        if tuple(mat.shape) != expected:
            raise ValueError(f"{d / name}: shape {tuple(mat.shape)}, expected {expected}")
    return {"node_locs": node_locs, "street_adj": street_adj, "demand": demand}


def load_routes_txt(path, *, n_nodes, spec) -> torch.Tensor:
    """Parse ``Routes.txt`` into a padded ``(1, n_routes, max_route_len)`` tensor.

    Node ids are 0-indexed. Validates the id range, the route count and the
    longest route against ``spec``.
    """
    path = Path(path)
    routes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stops = line.split("#", 1)[0].split()
        if stops:
            routes.append([int(s) for s in stops])

    if len(routes) != int(spec["n_routes"]):
        raise ValueError(
            f"{path}: {len(routes)} routes, but cfg/instance says {spec['n_routes']}")
    flat = [s for r in routes for s in r]
    if min(flat) < 0 or max(flat) >= n_nodes:
        raise ValueError(
            f"{path}: node ids {min(flat)}..{max(flat)} outside 0..{n_nodes - 1} "
            f"(routes must be 0-indexed)")
    longest = max(len(r) for r in routes)
    if longest > int(spec["max_route_len"]):
        raise ValueError(
            f"{path}: longest route has {longest} stops, max_route_len is "
            f"{spec['max_route_len']}")

    padded = torch.full((len(routes), int(spec["max_route_len"])), -1, dtype=torch.long)
    for i, r in enumerate(routes):
        padded[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return padded[None]


def load_instance(instance, *, instances_dir=None) -> Instance:
    """Load an instance preset (or its name) as an :class:`Instance`."""
    if isinstance(instance, str):
        cfg = instance_spec(instance)
        name = instance
    else:
        cfg = instance
        name = str(cfg.name)
    if instances_dir is not None:
        d = Path(instances_dir) / name
    elif cfg.get("path"):
        d = resolve_under_root(cfg.path)
    else:
        d = INSTANCES_DIR / name
    if not d.is_dir():
        raise FileNotFoundError(
            f"instance folder not found: {d}. An instance is a folder with "
            f"Coords.txt / TravelTimes.txt / Demand.txt / Routes.txt.")
    tensors = load_matrices(d)
    spec = {"city": name, **{k: int(cfg[k]) for k in
                             ("n_routes", "min_route_len", "max_route_len")}}
    routes = load_routes_txt(d / "Routes.txt",
                             n_nodes=tensors["node_locs"].shape[0], spec=spec)
    meta = {"crs": cfg["crs"]} if "crs" in cfg else {}
    return Instance(label=name, tensors=tensors, init_routes=routes, spec=spec,
                    coords=tensors["node_locs"], street_adj=tensors["street_adj"],
                    meta=meta)

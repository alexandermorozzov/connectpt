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

from ..run.const import DATASETS_DIR
from ..run.context import resolve_under_root
from omegaconf import OmegaConf
from ..network.routes import read_routes_txt
from .const import EKB_COORD_CRS, TRAVEL_TIME_SCALE

INSTANCE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "cfg" / "instance"


def instance_spec(name: str):
    """The instance spec from ``cfg/instance/<name>.yaml`` -- route count, length
    bounds and any extras (``crs``). One file per instance, same name as its
    folder under ``datasets/instances/``."""
    path = INSTANCE_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no spec for instance {name!r}: expected {path} with n_routes / "
            f"min_route_len / max_route_len"
        )
    return OmegaConf.load(path)


INSTANCES_DIR = DATASETS_DIR / "instances"


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


def _load_matrices(instance_dir) -> dict:
    """``node_locs`` / ``street_adj`` / ``demand`` from the three text files."""
    d = Path(instance_dir)
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


def load_instance(instance, *, instances_dir=None, require_routes=True) -> Instance:
    """Load an instance (``InstanceParams`` or a ``cfg/instance`` name) as an
    :class:`Instance`."""
    from ..run.schemas import InstanceParams, params

    if isinstance(instance, (str, Path)):
        path = resolve_under_root(instance)
        if path.is_dir():
            instances_dir = path.parent
            instance = path.name
        instance = params(InstanceParams, instance_spec(str(instance)))
    name = instance.name
    if instances_dir is not None:
        d = Path(instances_dir) / name
    else:
        d = resolve_under_root(instance.path)
    if not d.is_dir():
        raise FileNotFoundError(
            f"instance folder not found: {d}. An instance is a folder with "
            f"Coords.txt / TravelTimes.txt / Demand.txt / Routes.txt."
        )
    tensors = _load_matrices(d)
    spec = {
        "city": name,
        "n_routes": instance.n_routes,
        "min_route_len": instance.min_route_len,
        "max_route_len": instance.max_route_len,
    }
    routes = (
        read_routes_txt(d / "Routes.txt", n_nodes=tensors["node_locs"].shape[0], spec=spec)
        if require_routes
        else None
    )
    meta = {"crs": instance.crs} if instance.crs is not None else {}
    return Instance(
        label=name,
        tensors=tensors,
        init_routes=routes,
        spec=spec,
        coords=tensors["node_locs"],
        street_adj=tensors["street_adj"],
        meta=meta,
    )

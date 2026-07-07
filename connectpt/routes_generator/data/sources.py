"""Experiment data sources: one ``load() -> Instance`` contract per source.

Every experiment (benchmark / MACSA / EKB) differs only in WHERE its data comes
from; the data source hides that so the search/experiment runner is identical
across them. A new experiment adds a data source config, not a notebook cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .init import resolve_init_routes
from .loaders import (benchmark_spec, ekb_spec, load_benchmark_tensors,
                      load_ekb_routes, load_ekb_tensors, macsa_eval_bounds)
from .routes import as_route_tensor


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


class DataSource:
    def load(self) -> Instance:
        raise NotImplementedError


class BenchmarkDataSource(DataSource):
    """A standard benchmark city (Mandl / Mumford).

    Init routes come from a pinned dump / route file when configured, else are
    generated from scratch by learned construction + the realistic tier.
    """

    def __init__(self, city: str, init_routes_path: str | None = None,
                 init_dump: str | None = None, seed: int = 0):
        self.city = city
        self.init_routes_path = init_routes_path
        self.init_dump = init_dump
        self.seed = seed

    def load(self) -> Instance:
        tensors = load_benchmark_tensors(self.city)
        spec = benchmark_spec(self.city)
        init = resolve_init_routes(
            spec, tensors, init_dump=self.init_dump,
            init_routes_path=self.init_routes_path, seed=self.seed)
        return Instance(
            label=self.city, tensors=tensors, init_routes=init, spec=dict(spec),
            coords=tensors["node_locs"], street_adj=tensors["street_adj"])


class MACSADataSource(DataSource):
    """A MACSA scenario (tensors + seeded routes from a scenario dir)."""

    def __init__(self, scenario: str):
        self.scenario = scenario

    def load(self) -> Instance:
        from ..citygraph_dataset import load_macsa_scenario
        from ..core.paths import MACSA_DATA_DIR

        sc = load_macsa_scenario(MACSA_DATA_DIR / self.scenario)
        n_routes, min_len, max_len = macsa_eval_bounds(sc)
        tensors = sc["tensors"]
        return Instance(
            label=f"MACSA/{self.scenario}", tensors=tensors,
            init_routes=as_route_tensor(sc["routes"]),
            spec={"n_routes": n_routes, "min_route_len": min_len,
                  "max_route_len": max_len},
            coords=tensors["node_locs"], street_adj=tensors["street_adj"])


class EKBDataSource(DataSource):
    """Ekaterinburg case study (tensors + seed routes from disk)."""

    def load(self) -> Instance:
        tensors = load_ekb_tensors()
        routes = load_ekb_routes()
        return Instance(
            label="EKB", tensors=tensors, init_routes=as_route_tensor(routes),
            spec=ekb_spec(routes), coords=tensors["node_locs"],
            street_adj=tensors["street_adj"], meta={"crs": "gis"})


REGISTRY = {
    "ekb": EKBDataSource,
    "macsa": MACSADataSource,
    "benchmark": BenchmarkDataSource,
}


def create_data_source(data_cfg) -> DataSource:
    """Build a DataSource from a ``data:`` config node (``source`` + params)."""
    params = {k: v for k, v in dict(data_cfg).items() if k != "source"}
    return REGISTRY[data_cfg["source"]](**params)

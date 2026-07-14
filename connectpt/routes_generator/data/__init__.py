"""Data layer: problem instances + route-corruption machinery.

Home of the experiment data sources (benchmark / MACSA / EKB) and the shared
route-copy / realistic-tier corruption used to build training datasets and
experiment init-networks. Migrated out of the notebook-side ``eval_lib`` so the
library owns data loading -- the notebook only names a source in its config.
"""
from .routes import as_route_tensor
from .loaders import (benchmark_spec, eval_spec, ekb_spec,
                      load_benchmark_tensors, load_ekb_routes, load_ekb_tensors,
                      macsa_eval_bounds)
from .init import resolve_init_routes
from .sources import (DataSource, Instance, BenchmarkDataSource, EKBDataSource,
                      MACSADataSource, create_data_source)

__all__ = [
    "as_route_tensor",
    "benchmark_spec",
    "eval_spec",
    "ekb_spec",
    "load_benchmark_tensors",
    "load_ekb_routes",
    "load_ekb_tensors",
    "macsa_eval_bounds",
    "resolve_init_routes",
    "DataSource",
    "Instance",
    "BenchmarkDataSource",
    "EKBDataSource",
    "MACSADataSource",
    "create_data_source",
]

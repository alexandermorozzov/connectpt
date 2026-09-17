"""Data layer: problem instances + route-corruption machinery.

An instance is a folder under ``datasets/instances/`` with four text files;
:func:`load_instance` reads it and validates it against ``cfg/instance/<name>.yaml``.
Also home of the route-copy / realistic-tier corruption used to build the
training dataset and the benchmark initial networks.
"""
from .routes import as_route_tensor, pad_routes_to, write_routes_txt
from .loaders import instance_spec, load_benchmark_tensors
from .init import build_init_routes
from .sources import (INSTANCES_DIR, Instance, load_instance, load_matrices,
                      load_routes_txt)

__all__ = [
    "as_route_tensor",
    "pad_routes_to",
    "write_routes_txt",
    "instance_spec",
    "load_benchmark_tensors",
    "build_init_routes",
    "INSTANCES_DIR",
    "Instance",
    "load_instance",
    "load_matrices",
    "load_routes_txt",
]

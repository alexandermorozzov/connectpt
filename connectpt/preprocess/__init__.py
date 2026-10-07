"""
Preprocessing package for public transport network generation.

This package provides modular functions for:
    - downloading and preprocessing transport lines (bus, tram, trolleybus),
    - aggregating and projecting stops,
    - constructing and simplifying transport graphs,
    - computing stop-to-stop time and OD matrices,
    - executing the full preprocessing pipeline with a single entry point.

Typical usage:
---------------
>>> from preprocess import preprocess_data, Modality
>>> result = preprocess_data(blocks, [Modality.BUS, Modality.TRAM])
>>> stops_gdf, time_matrix, graph = result[Modality.BUS]
"""

import importlib

__version__ = importlib.metadata.version("connectpt")

# Submodules load on first access: the city pipeline needs iduedu 1.x, momepy,
# osmnx and others, while connectivity_gdf needs none of them.
_EXPORTS = {
    "Modality": "types",
    "preprocess": "preprocess_data", "get_boundary_gdf": "preprocess_data",
    "get_drive_graph_iduedu": "lines", "get_bus_graph_iduedu": "lines",
    "get_OD": "od", "reform_od_matrix": "od",
    "get_multi_OD": "od_multi",
    "preprocess_stops_names_gdf": "stops", "cluster_stops": "stops",
    "create_stops_gdf_from_routes": "stops",
    "build_new_zones": "polygons",
    "make_block_graph": "block_graph", "get_road_segments_for_block_graph_edges": "block_graph",
    "pt_graph_project_by_stops_name": "block_graph", "block_graph_to_gdfs": "block_graph",
    "create_routes_geodataframe": "block_graph",
    "save_files": "files",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)

# import importlib

#  # TODO поменять название в соответствии с pyproject.toml
# __author__ = "Vasilii Starikov"
# __email__ = "vasilstar97@gmail.com"
# __credits__ = []
# __license__ = "BSD-3"
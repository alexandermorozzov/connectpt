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

from .types import Modality
from .preprocess_data import preprocess, get_boundary_gdf 
from .lines import get_drive_graph_iduedu, get_bus_graph_iduedu
from .od import get_OD, reform_od_matrix
from .od_multi import get_multi_OD
from .stops import preprocess_stops_names_gdf, cluster_stops, create_stops_gdf_from_routes
from .polygons import build_new_zones
from .block_graph import (make_block_graph, 
                          get_road_segments_for_block_graph_edges, 
                          pt_graph_project_by_stops_name, 
                          block_graph_to_gdfs, 
                          create_routes_geodataframe)
from .files import save_files

# import importlib

#  # TODO поменять название в соответствии с pyproject.toml
# __author__ = "Vasilii Starikov"
# __email__ = "vasilstar97@gmail.com"
# __credits__ = []
# __license__ = "BSD-3"
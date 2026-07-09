from typing import Dict, List, Tuple
import warnings
import pandas as pd
import geopandas as gpd
import networkx as nx
import momepy as mp
from shapely.geometry import Polygon

from .types import Modality
from .stops import get_agg_stops
from .lines import get_lines
from .projection import project_stops_on_roads
from .network import roads_to_graph, stop_complete_then_prune, build_time_matrix

warnings.filterwarnings("ignore", category=UserWarning)


def preprocess(blocks : gpd.GeoDataFrame, modalities : list[Modality]) : 
    """End-to-end preprocessing pipeline for stop and line data by modality.

    Given a set of urban blocks, this function:
      1) Builds a boundary polygon from the blocks,
      2) Downloads and aggregates stops per modality,
      3) Downloads and preprocesses line geometries per modality,
      4) Projects stops onto the nearest lines and splits lines at stop locations,
      5) Converts the resulting lines into a graph and marks stop nodes,
      6) Builds a stop-to-stop simplified graph (shortest-path edges only),
      7) Keeps only the largest connected component for robustness,
      8) Exports stop coordinates and a stop-to-stop time matrix.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Polygonal blocks with a valid geometry column. Used to derive the boundary
        (convex hull of the union) within which data are downloaded and processed.
        Will be reprojected to EPSG:4326 internally.
    modalities : list[Modality]
        Transport modalities to process (e.g., [Modality.BUS, Modality.TRAM]).

    Returns
    -------
    dict
        Mapping from `Modality` to a tuple:
        (stops_gdf, time_matrix, simplified_graph_largest), where
        - `stops_gdf` : GeoDataFrame of stop nodes extracted from the final graph,
                        with added `x_coord`/`y_coord` columns (rounded to 2 decimals).
        - `time_matrix`: np.ndarray of string values representing pairwise travel times
                         between stops in minutes (direct-edge matrix).
        - `simplified_graph_largest`: networkx.Graph containing only the largest
                                     connected component of the simplified stop graph.

    Notes
    -----
    - Line processing includes stop projection and segment splitting, then conversion to a graph.
    - The simplified graph retains only edges where the shortest path includes exactly two stops
      (start and end), removing paths that pass through intermediate stops.
    - Only the largest connected component is returned to avoid isolated artifacts.
    - Ensure all intermediate layers use projected CRS (meters) before distance-based steps.

    Examples
    --------
    >>> result = preprocess_data(blocks_gdf, [Modality.BUS, Modality.TRAM])
    >>> bus_stops, bus_time_mx, bus_graph = result[Modality.BUS]
    """
    blocks = blocks.copy()
    blocks.to_crs(4326, inplace=True)
    boundary = blocks.geometry.union_all().convex_hull


    modality_stops = get_agg_stops(boundary, modalities)
    modality_lines = get_lines(boundary, modalities)
        
    result = {}

    for modality in modalities:
        stops_gdf = modality_stops[modality]
        lines_gdf = modality_lines[modality]

        roads_with_stops, filtered_stops = project_stops_on_roads(lines_gdf, stops_gdf)

        if "length_meter" not in roads_with_stops.columns:
            roads_with_stops["length_meter"] = roads_with_stops.geometry.length
        else:
            mask = roads_with_stops["length_meter"].isna()
            roads_with_stops.loc[mask, "length_meter"] = roads_with_stops.loc[mask, "geometry"].length

        roads_with_stops_graph = roads_to_graph(roads_with_stops, filtered_stops)
        # simplified_graph = stop_complete_then_prune(G = roads_with_stops_graph, speed_kmh=20, max_hops = 5, max_detour_factor=2.5, min_weight=150 )
        simplified_graph = stop_complete_then_prune(G = roads_with_stops_graph, speed_kmh=20)
        largest_cc_nodes = max(nx.connected_components(simplified_graph), key=len)  # Узлы самой большой компоненты
        simplified_graph_largest = simplified_graph.subgraph(largest_cc_nodes).copy()  # Создаем подграф
        stops_gdf, _ = mp.nx_to_gdf(simplified_graph_largest)

        stops_gdf["modality"] = modality.value
        
        stops_gdf[['x_coord', 'y_coord']] = stops_gdf.geometry.apply(lambda p: pd.Series([p.x, p.y]))
        stops_gdf['x_coord'] = stops_gdf['x_coord'].round(2)
        stops_gdf['y_coord'] = stops_gdf['y_coord'].round(2)

        time_matrix = build_time_matrix(simplified_graph_largest, attr="time_min")

        result[modality] = (stops_gdf, time_matrix, simplified_graph_largest)

    return result, simplified_graph_largest

def get_boundary_gdf(polygons_gdf):
    union = polygons_gdf.geometry.union_all()
    boundary = Polygon(union.exterior)
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=polygons_gdf.crs)
    return boundary_gdf


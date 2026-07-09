import warnings
import geopandas as gpd
import pygeoops
import osmnx as ox
import momepy as mp
import neatnet
import shapely

from shapely.geometry import Polygon, MultiPolygon
from iduedu import get_drive_graph, graph_to_gdf, get_public_transport_graph
from typing import Dict

from .types import Modality, MODALITY_LINE_TAGS
from .utils import _close_gaps, restore_linestrings

ox.settings.useful_tags_way.append("railway")
warnings.filterwarnings("ignore", category=UserWarning)


def get_drive_graph_iduedu(territory):
    G_drive = get_drive_graph(territory=territory,
                          osm_edge_tags=['highway', 'maxspeed', 'reg', 'name', 'lanes', 'ref'],
                          simplify=True)
    
    G_drive_edges = graph_to_gdf(G_drive, nodes=False, restore_edge_geom=True)
    G_drive_nodes = graph_to_gdf(G_drive, edges=False, restore_edge_geom=True)
    return G_drive, G_drive_edges, G_drive_nodes

def get_bus_graph_iduedu(territory):
    G_pt = get_public_transport_graph(territory=territory,
                                      transport_types='bus',
                                      clip_by_territory=True
                                      )
    
    G_pt_edges = graph_to_gdf(G_pt, nodes=False, restore_edge_geom=True)
    G_pt_nodes = graph_to_gdf(G_pt, edges=False, restore_edge_geom=True)
    return G_pt, G_pt_edges, G_pt_nodes


def _get_electric_lines(polygon: Polygon | MultiPolygon, filter_tags: str) -> gpd.GeoDataFrame:
    # Download and extract electric or rail-based transport lines within a given polygon
    custom_filter = filter_tags
    G = ox.graph_from_polygon(polygon, network_type='all', custom_filter=custom_filter, retain_all=True, truncate_by_edge=True)
    n,e = mp.nx_to_gdf(G)
    edges = restore_linestrings(e, n)
    edges = edges.clip(polygon)
    local_crs = edges.estimate_utm_crs()
    edges.to_crs(local_crs, inplace=True)
    
    return edges


def _get_drive_lines(polygon: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    # Download and extract drivable road network edges within a polygon
    G = get_drive_graph(polygon=polygon)
    edges = graph_to_gdf(G, restore_edge_geom=True)
    edges = edges[edges.geometry.geom_type == 'LineString']

    return edges


def _preprocess_electric_lines(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Preprocess electric or rail-based transport lines to create clean, continuous geometries
    local_crs = edges.estimate_utm_crs()
    edges.to_crs(local_crs, inplace=True)
    buffer_edges = edges.geometry.buffer(25, cap_style="round", join_style="mitre")
    edges_union = buffer_edges.union_all()
    centerline = pygeoops.centerline(edges_union, densify_distance=58, simplifytolerance=-0.15)
    lines_gdf = gpd.GeoDataFrame(geometry=[centerline], crs=local_crs)
    lines_gdf.reset_index(drop=True, inplace=True)
    lines_gdf = lines_gdf.explode("geometry")
    geom_lines = _close_gaps(lines_gdf.geometry, tolerance=2)
    lines_gdf = gpd.GeoDataFrame(geometry=geom_lines)
    
    return lines_gdf


def _preprocess_drive_lines(edges: gpd.GeoDataFrame, boundary: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    # Preprocess driveable network lines to create clean, continuous geometries
    buildings = (ox.features_from_polygon(boundary, tags={"building": True})
                    .query('building != "roof"')
                    .to_crs(edges.crs))
    simplified = neatnet.neatify(edges, exclusion_mask=buildings.geometry , max_segment_length = 1)
    simplified_geom = _close_gaps(simplified, tolerance=0.5)
    lines_gdf = gpd.GeoDataFrame(geometry=simplified_geom)

    return lines_gdf


def get_lines(polygon : Polygon | MultiPolygon, modalities: list[Modality] ) -> Dict[Modality, gpd.GeoDataFrame]:
    """Download and preprocess transport network lines for multiple transport modalities.

    This function retrieves, processes, and unifies transport line geometries (bus, tram, trolleybus, etc.)
    within a given polygon. Depending on the modality, it either downloads drivable roads (for buses)
    or electric network lines (for trams and trolleybuses), then applies the appropriate preprocessing
    steps to produce simplified and continuous line geometries.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
        The geographic boundary within which transport lines are extracted and processed.
    modalities : list of Modality
        List of transport types to process (e.g., `[Modality.BUS, Modality.TRAM]`).

    Returns
    -------
    dict of {Modality: geopandas.GeoDataFrame}
        A dictionary mapping each transport modality to its corresponding processed
        GeoDataFrame of line geometries. Each GeoDataFrame includes:
        - `geometry`: simplified `LineString` geometries,
        - `modality`: modality name (e.g., `"bus"`, `"tram"`).

    Notes
    -----
    - For buses, the function downloads the drivable street network using `_get_drive_lines()`
      and simplifies it via `_preproccess_drive_lines()`.
    - For trams and trolleybuses, it downloads rail or electric lines using `_get_electric_lines()`
      and processes them with `_preproccess_electric_lines()`.
    - All resulting geometries are reprojected to local UTM CRS for consistency and spatial accuracy.

    Examples
    --------
    >>> result = get_lines(city_boundary, [Modality.BUS, Modality.TRAM])
    >>> result[Modality.TRAM].head()
      modality                                           geometry
    0     tram  LINESTRING (445122.1 6139458.2, 445215.4 6139420.7)
    1     tram  LINESTRING (445310.7 6139381.9, 445402.5 6139342.4)
    """
    result: Dict[Modality, gpd.GeoDataFrame] = {}

    for modality in modalities:
        modality_tag = MODALITY_LINE_TAGS[modality]
        if modality_tag is None:
            # For bus, we use driving roads
            roads_gdf = _get_drive_lines(polygon)
            processed_lines = _preprocess_drive_lines(roads_gdf, boundary = polygon)
            processed_lines["modality"] = modality.value
        else:
            lines_gdf = _get_electric_lines(polygon, modality_tag)
            processed_lines = _preprocess_electric_lines(lines_gdf)
            processed_lines["modality"] = modality.value

        result[modality] = processed_lines.reset_index(drop=True)

    return result



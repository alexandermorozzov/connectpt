import logging
from typing import Tuple
import pandas as pd
import geopandas as gpd

from .utils import _project_stop_on_road

logger = logging.getLogger(__name__)


def project_stops_on_roads(
    roads_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    max_distance: float = 70
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Project stops onto the nearest roads and split those roads at stop locations.

    For each stop, the function finds the nearest road geometry. If the stop is within
    `max_distance` of that road, the road segment is split at the stop's projected
    position along the line. Stops farther than `max_distance` from any road are
    removed. Returns the updated road layer (with split segments) and the filtered
    set of stops.

    Parameters
    ----------
    roads_gdf : geopandas.GeoDataFrame
        GeoDataFrame with road geometries (LineString). Must have a valid projected CRS (meters).
    stops_gdf : geopandas.GeoDataFrame
        GeoDataFrame with stop point geometries. Will be reprojected to `roads_gdf.crs`.
    max_distance : float, optional
        Maximum allowed distance (in meters) between a stop and its nearest road for the split
        to occur. Defaults to 70.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        (roads_with_splits, filtered_stops):
        - roads_with_splits: roads with segments split at accepted stop locations.
        - filtered_stops: only the stops within `max_distance` of a road, index reset.

    Notes
    -----
    - The function copies inputs; originals are not modified.
    - Make sure both layers use a projected CRS in meters for correct distance thresholds.
    """
    roads_gdf = roads_gdf.copy()
    stops_gdf = stops_gdf.copy()
    stops_gdf.to_crs(roads_gdf.crs, inplace=True)

    stops_to_remove = []
    
    for stop_i, row in stops_gdf.iterrows():
        stop_gdf = stops_gdf.iloc[[stop_i]]
        sjoin = gpd.sjoin_nearest(stop_gdf, roads_gdf)
        roads_i = sjoin["index_right"].iloc[0]
        
        # Check distance to the nearest road
        road_geom = roads_gdf.loc[roads_i].geometry
        stop_geom = row.geometry
        distance = stop_geom.distance(road_geom)
        
        if distance <= max_distance:
            # If the stop is close enough, project it and split the road
            splitted_roads = _project_stop_on_road(road_geom, stop_geom)
            splitted_roads_gdf = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(splitted_roads, crs=roads_gdf.crs)}, geometry="geometry")
            roads_gdf = roads_gdf.drop(roads_i)
            roads_gdf = pd.concat([roads_gdf, splitted_roads_gdf]).reset_index(drop=True)
        else:
            # If the stop is too far, mark it for removal
            stops_to_remove.append(stop_i)
    
    # Remove stops that are too far from any road
    filtered_stops = stops_gdf.drop(stops_to_remove)
    
    if len(stops_to_remove) > 0:
        logger.warning("Removed %s stops located farther than %s m from roads",
                       len(stops_to_remove), max_distance)
    
    return roads_gdf, filtered_stops.reset_index(drop=True)

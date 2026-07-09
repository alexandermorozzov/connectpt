import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import line_locate_point, union_all
from shapely.ops import linemerge
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon, MultiLineString )
from shapely.ops import (unary_union, substring)
import networkx as nx
from tqdm import tqdm
import osmnx as ox
from shapely import from_wkt, MultiPoint



def _close_gaps(gdf, tolerance) -> gpd.GeoSeries:
    """Function from momepy. Close gaps in LineString geometry where it should be contiguous.

    Snaps both lines to a centroid of a gap in between.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        GeoDataFrame  or GeoSeries containing LineString representation of a network.
    tolerance : float
        nodes within a tolerance will be snapped together

    Returns
    -------
    GeoSeries

    """
    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)

    dissolved = shapely.union_all(buffered)

    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]

    centroids = shapely.centroid(exploded)

    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)


def restore_linestrings(edges_gdf, nodes_gdf):
    # Restore None geometries for roads
    edges = edges_gdf.copy()
    
    mask = edges.geometry.isna() | edges.geometry.is_empty
    
    for idx in edges[mask].index:
        start_node = edges.loc[idx, 'node_start']
        end_node = edges.loc[idx, 'node_end']
        
        start_point = nodes_gdf.loc[start_node, 'geometry']
        end_point = nodes_gdf.loc[end_node, 'geometry']
        
        line = LineString([
            (start_point.x, start_point.y),
            (end_point.x, end_point.y)
        ])
        
        edges.loc[idx, 'geometry'] = line
    
    return edges


def _cut(line: LineString, distance: float) -> list[LineString]:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:

        return [LineString(line)]
    
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)

            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]
        

def _project_stop_on_road(road_geom: LineString, stop_geom: Point) -> list[LineString]:
    # Project stops on road geometry and return LineStrings of splitted road
    distance = line_locate_point(road_geom, stop_geom)
    splitted_road = _cut(road_geom, distance)

    return splitted_road


def _remove_water_objects( water_gdf : gpd.GeoDataFrame, 
                          polygons_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    water_polygons = water_gdf[water_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    water_union = unary_union(water_polygons.geometry) if not water_polygons.empty else None
    if water_union:
        polygons_gdf['geometry'] = polygons_gdf.geometry.difference(water_union)
        polygons_gdf = polygons_gdf[~polygons_gdf.is_empty]
    return polygons_gdf


def _jaccard_similarity(name1, name2):
    if pd.isna(name1) or pd.isna(name2):
        return 0.0
    
    words1 = set(str(name1).strip().split())
    words2 = set(str(name2).strip().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union


def _is_names_similar(name1, name2, threshold=0.65):
    score = _jaccard_similarity(name1, name2)
    if score >= threshold:
        return True
    
    n1 = str(name1).strip().lower()
    n2 = str(name2).strip().lower()
    if n1 in n2 or n2 in n1:
        return True
    
    return False

def _add_nodes_block_graph(G : nx.Graph,
               point_gdf : gpd.GeoDataFrame):
    for idx, row in point_gdf.iterrows():
        G.add_node(
            idx,
            x=row.geometry.x,
            y=row.geometry.y,
            geometry=row.geometry,
            # cluster_id=poly_gdf.loc[idx, 'cluster_id'],
            cluster_id=row.cluster_id,
            name=row.name

        )

def _find_touching_neighbors(poly_gdf : gpd.GeoDataFrame):
    neighbors = gpd.sjoin(
        poly_gdf[['geometry']], 
        poly_gdf[['geometry']], 
        how='left', 
        predicate='touches'
    )

    neighbors = neighbors[neighbors.index != neighbors.index_right].copy()
    neighbors = neighbors[neighbors.index < neighbors.index_right]

    return neighbors
    
def _split_road_by_polygons(road_geom : MultiLineString | LineString, 
                            union_pols : Polygon | MultiPolygon):
    
    intersection = road_geom.intersection(union_pols)
    if intersection.is_empty:
        return []
    
    intersection_parts = list(intersection.geoms) if intersection.geom_type == 'MultiLineString' else [intersection]
    split_dists = [0, road_geom.length]
    segment_info = []
    for part in intersection_parts:
        if part.length == 0:
            continue
        start_dist = road_geom.project(part.interpolate(0))
        end_dist = road_geom.project(part.interpolate(part.length))
        split_dists.extend([start_dist, end_dist])

    split_dists = sorted(set(split_dists))

    for k in range(1, len(split_dists)):
        start_dist = split_dists[k - 1]
        end_dist = split_dists[k]

        if end_dist - start_dist < 1e-8:
            continue

        subst = substring(road_geom, start_dist, end_dist)
        inters = subst.intersection(union_pols)
        inters_geom_type = inters.geom_type
        if inters_geom_type in ('LineString', 'MultiLineString'):
            segment_info.append((True, subst, start_dist, end_dist))
        else:
            segment_info.append((False, subst, start_dist, end_dist))

    return segment_info

def _get_substring_projected(substr : LineString,
                             p_1 : Point,
                             p_2 : Point):
    dist_1 = substr.project(p_1)
    dist_2 = substr.project(p_2)

    if dist_1 < 0 or dist_2 < 0 or dist_1 > substr.length or dist_2 > substr.length:
        return None
    
    start_dist = min(dist_1, dist_2)
    end_dist = max(dist_1, dist_2)

    return substring(substr, start_dist, end_dist)

def _get_order_along_road(road_geom : LineString | MultiLineString,
                           rp_geoms):
    projs = [road_geom.project(p) for p in rp_geoms]
    return np.argsort(projs)

def _find_road_connections(roads : gpd.GeoDataFrame,
                           poly_gdf : gpd.GeoDataFrame,
                           rp_gdf : gpd.GeoDataFrame):

    joined = gpd.sjoin(
        roads, 
        poly_gdf[['geometry']], 
        how='inner', 
        predicate='intersects'
    )

    road_to_polys = joined.groupby(joined.index)['index_right'].agg(list)
    poly_geoms = poly_gdf.geometry
    rp_geoms = rp_gdf.geometry
    poly_boundaries = {idx: geom.boundary for idx, geom in poly_geoms.items()}
    union_cache = {}

    road_connections = []
    
    for rid, poly_list in tqdm(road_to_polys.items(), 
                                  total=len(road_to_polys),
                                  desc="Checking roads for connection", 
                                  unit="road"):
        poly_indices = list(dict.fromkeys(poly_list))
        if len(poly_indices) < 2:
            continue
        road_geom = roads.geometry.loc[rid]

        current_rps = [rp_geoms.loc[idx] for idx in poly_indices]
        order = _get_order_along_road(road_geom, current_rps)
        sorted_poly_indices = [poly_indices[i] for i in order]

        candidate_pairs = []
        n = len(sorted_poly_indices)

        for i in range(n - 1):
            idx1 = sorted_poly_indices[i]
            idx2 = sorted_poly_indices[i + 1]
            candidate_pairs.append((idx1, idx2))
        

        for idx1, idx2 in candidate_pairs:

            pol_1 = poly_gdf.geometry.loc[idx1]
            pol_2 = poly_gdf.geometry.loc[idx2]

            cache_key = tuple(sorted([idx1, idx2]))
            if cache_key not in union_cache:
                union_cache[cache_key] = pol_1.union(pol_2)
            union_pols = union_cache[cache_key]
            
            rp_1 = rp_gdf.geometry.loc[idx1]
            rp_2 = rp_gdf.geometry.loc[idx2]

            boundary1 = poly_boundaries[idx1]
            boundary2 = poly_boundaries[idx2]
            shared_boundary = boundary1.intersection(boundary2)
            is_touching = not(shared_boundary.is_empty)
            
            intersection_road_union = road_geom.intersection(union_pols)
            intersection_road_union_geom_type = intersection_road_union.geom_type

            if intersection_road_union.is_empty or intersection_road_union_geom_type not in ("LineString", "MultiLineString"):
                continue

            roads_between = []

            if intersection_road_union_geom_type == 'LineString':
                road_b = _get_substring_projected(intersection_road_union, rp_1, rp_2)
                if road_b and road_b.geom_type == 'LineString':
                    roads_between += [road_b]
            else:
                if is_touching and intersection_road_union.intersects(shared_boundary):
                    parts = list(intersection_road_union.geoms)
                    for p in parts:
                        if p.intersects(shared_boundary):
                            road_b = _get_substring_projected(p, rp_1, rp_2)
                            if road_b.geom_type == 'Point':
                                continue
                            roads_between += [road_b]
                else:
                    segment_info = _split_road_by_polygons(road_geom, union_pols)
                    filtered_poly_gdf = poly_gdf[(poly_gdf.index != idx1) & (poly_gdf.index != idx2)]
                    for sid in range(2, len(segment_info)):
                        s_1 = segment_info[sid - 2]
                        s_2 = segment_info[sid - 1]
                        s_3 = segment_info[sid]
                        if s_1[0] == s_3[0] == True and s_2[0] == False:
                            s2_intersection = s_2[1].intersection(unary_union(filtered_poly_gdf.geometry))
                            s2_intersection_geom_type = s2_intersection.geom_type
                            if s2_intersection_geom_type not in ['Point', 'GeometryCollection']: 
                                start_dist = s_1[-2]
                                end_dist = s_3[-1]
                                road_b = _get_substring_projected(substring(road_geom, start_dist, end_dist), 
                                                                    rp_1, rp_2)
                                if road_b.geom_type == 'Point':
                                    continue
                                roads_between += [road_b]
            if len(roads_between) > 0:
                road_connections.append({
                    "idx1" : idx1,
                    "idx2" : idx2,
                    "connection_type" : is_touching,
                    "road_segments" : roads_between,
                    "road_id" : rid
                }
                )
    return road_connections

def _get_all_connections(neighbors, road_connections):
    all_connections = {}
    
    for idx1, row in neighbors.iterrows():
        idx2 = row['index_right']
        pair_key = tuple(sorted([idx1, idx2]))
        if pair_key not in all_connections:
            all_connections[pair_key] = {
                'idx1': idx1,
                'idx2': idx2,
                'connection_type': 'touching',
                'road_ids': [],
                'road_segments': []
            }
    
    for conn in road_connections:
        idx1 = conn['idx1']
        idx2 = conn['idx2']
        pair_key = tuple(sorted([idx1, idx2]))
        road_segments = conn['road_segments']
        rid = conn['road_id']
        
        if pair_key not in all_connections:
            all_connections[pair_key] = {
                'idx1': idx1,
                'idx2': idx2,
                'connection_type': 'road_only' if not conn['connection_type'] else 'touching_and_road',
                'road_ids': [],
                'road_segments': []
            }
        
        all_connections[pair_key]['road_ids'].append(rid)
        all_connections[pair_key]['road_segments'] += road_segments
    
    return all_connections

def _add_edges_block_graph(
               G : nx.Graph,
               connections : dict,
               roads : gpd.GeoDataFrame,
               point_gdf : gpd.GeoDataFrame,
               ):
    for pair_key, conn_data in connections.items():
        idx1 = conn_data['idx1']
        idx2 = conn_data['idx2']
        
        total_time = 0
        road_count = 0
        road_geometries = []

        p1 = point_gdf.geometry.loc[idx1]
        p2 = point_gdf.geometry.loc[idx2]

        for r_id in range(len(conn_data['road_ids'])):
            road_id = conn_data['road_ids'][r_id]
            road_segment = conn_data['road_segments'][r_id]
            road_geom = roads.geometry.loc[road_id]
            road_time_min = roads.loc[road_id, 'time_min']

            segment_length = road_segment.length
            total_road_len = road_geom.length

            if total_road_len > 0:
                segment_time = road_time_min * (segment_length / total_road_len)
            else:
                segment_time = 0
            
            total_time += segment_time
            road_count += 1
            road_geometries.append(road_segment)

        if road_count > 0:
            weight = total_time / road_count
            G.add_edge(
                idx1, idx2, 
                geometry=LineString([p1, p2]),
                weight=weight,
                time_min=weight,
                road_count=road_count,
                road_geometry_list=road_geometries,
                connection_type=conn_data['connection_type']
            )

def _get_block_project_dict(dict_routes : dict,
                            poly_gdf : gpd.GeoDataFrame,
                            bg_nodes_gdf : gpd.GeoDataFrame,
                            utm_crs : int | str):
    block_project_dict = {}

    for route, stops_dict in dict_routes.items():
        curr_list_stops = []
        
        for stop_name, point_str in stops_dict.items():
            point_4326 = from_wkt(point_str)
            point_gdf = gpd.GeoDataFrame([{'geometry': point_4326}], crs=4326)
            point_utm = point_gdf.to_crs(utm_crs).iloc[0].geometry
            
            containing_poly = poly_gdf[poly_gdf.geometry.intersects(point_utm)]
            
            if containing_poly.empty:
                print(f"Warning: No polygon found for stop '{stop_name}' on route {route}")
                continue
                
            cluster_id = containing_poly.iloc[0]['cluster_id']
            
            node_in_cluster = bg_nodes_gdf[bg_nodes_gdf['cluster_id'] == cluster_id]
            
            if node_in_cluster.empty:
                print(f"Warning: No node found for cluster {cluster_id}, stop '{stop_name}'")
                continue
                
            curr_list_stops.append(node_in_cluster['node_id'].iloc[0])
        
        block_project_dict[route] = curr_list_stops
    return block_project_dict

def _build_pt_graph(
        block_project_dict : dict,
        block_graph_copy : nx.Graph,
        pt_graph : nx.Graph,
        poly_gdf : gpd.GeoDataFrame,
        tr_edges_gdf : gpd.GeoDataFrame,
        drive_edges_gdf : gpd.GeoDataFrame
):
    for route, stop_node_ids in block_project_dict.items():
        for i in range(1, len(stop_node_ids)):
            start_id = stop_node_ids[i-1]
            end_id = stop_node_ids[i]
            if block_graph_copy.has_edge(start_id, end_id):
                edge_data = block_graph_copy.get_edge_data(start_id, end_id)

                if 'routes' not in edge_data:
                    edge_data['routes'] = []
                if route not in edge_data['routes']:
                    edge_data['routes'].append(route)
                    block_graph_copy[start_id][end_id]['routes'] = edge_data['routes']
                
                pt_graph.add_edge(start_id, end_id, **edge_data)
            else:
                if start_id == end_id:
                    continue
                p1 = block_graph_copy.nodes[start_id]['geometry']
                p2 = block_graph_copy.nodes[end_id]['geometry']
                line = LineString([p1, p2])
                pol_1 = poly_gdf[poly_gdf.geometry.intersects(p1)].iloc[0]
                pol_2 = poly_gdf[poly_gdf.geometry.intersects(p2)].iloc[0]
                pols_union = union_all([pol_1.geometry, pol_2.geometry])

                intersecting = tr_edges_gdf[
                    (tr_edges_gdf.geometry.intersects(pols_union)) &
                    # (tr_edges_gdf['route'] == route) &
                    (tr_edges_gdf['time_min'] > 0)
                ]

                if len(intersecting) > 0:
                    seg_times = 0
                    for rid, row in intersecting.iterrows():
                        row_geom = row.geometry
                        row_geom_length = row_geom.length
                        row_geom_time_min = row['time_min']
                        segment = _get_substring_projected(row_geom, p1, p2)
                        segment_length = segment.length
                        if row_geom_length > 0:
                            segment_time = row_geom_time_min * (segment_length / row_geom_length)
                        else:
                            segment_time = 0
                        seg_times += segment_time
                    time_min = seg_times / len(intersecting)
                else:
                    intersecting = drive_edges_gdf[
                        (drive_edges_gdf.geometry.intersects(pols_union)) &
                        (drive_edges_gdf['time_min'] > 0)
                    ]
                    if len(intersecting) > 0:
                        seg_times = 0
                        for rid, row in intersecting.iterrows():
                            row_geom = row.geometry
                            row_geom_length = row_geom.length
                            row_geom_time_min = row['time_min']
                            segment = _get_substring_projected(row_geom, p1, p2)
                            segment_length = segment.length
                            if row_geom_length > 0:
                                segment_time = row_geom_time_min * (segment_length / row_geom_length)
                            else:
                                segment_time = 0
                            seg_times += segment_time
                        time_min = seg_times / len(intersecting)
                    else:
                        continue
                if time_min > 0:
                    block_graph_copy.add_edge(
                        start_id, end_id,
                        geometry=line,
                        weight=time_min,
                        time_min=time_min,
                        transport_edge=True,
                        routes=[route],
                        road_geometry_list=[],
                        road_count=None
                    )

                    pt_graph.add_edge(
                        start_id, end_id,
                        geometry=line,
                        weight=time_min,
                        time_min=time_min,
                        transport_edge=True,
                        routes=[route],
                        road_geometry_list=[],
                        road_count=None
                    )
                    time_min = 0
    return block_graph_copy, pt_graph



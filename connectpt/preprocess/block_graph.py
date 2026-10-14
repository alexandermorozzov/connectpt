import geopandas as gpd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import linemerge
from tqdm import tqdm
from heapq import heappush, heappop
from itertools import count
import math
from collections import defaultdict
from shapely.geometry import Point
from collections import defaultdict

import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points, substring

from .utils import (
    _add_nodes_block_graph,
    _find_touching_neighbors,
    _find_road_connections,
    _get_all_connections,
    _add_edges_block_graph,
    _get_block_project_dict,
    _build_pt_graph
)

from .lines import get_drive_graph_iduedu, get_bus_graph_iduedu

def _prepare_drive_graph(G_drive, utm_crs):
    import osmnx as ox

    G = ox.project_graph(G_drive, to_crs=utm_crs)
    G = nx.MultiGraph(G)

    for u, v, key, data in G.edges(keys=True, data=True):
        geometry = data.get("geometry")

        if geometry is None:
            geometry = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"])
            ])
            data["geometry"] = geometry


        if data.get("time_min") is not None:
            time_min = float(data["time_min"])

        else:
            raise ValueError(
                f"Road edge ({u}, {v}, {key}) "
                "has no time_min"
            )

        data["time_min"] = time_min

    G.graph["crs"] = utm_crs
    return G

def _get_road_id(edge_data):

    name = edge_data.get("name")

    if isinstance(name, (list, tuple)):
        names = sorted({
            str(x).strip().lower()
            for x in name
            if x is not None and str(x).strip()
        })
        if names:
            return ("name", tuple(names))

    elif name is not None and str(name).strip():
        return ("name", str(name).strip().lower())

    osmid = edge_data.get("osmid")

    if isinstance(osmid, (list, tuple, set)):
        return ("osmid", tuple(sorted(str(x) for x in osmid)))

    if osmid is not None:
        return ("osmid", str(osmid))

    return None

def _find_cluster_access_projections(
    G_drive,
    polygons_gdf,
    cluster_points_gdf
):

    graph_crs = G_drive.graph["crs"]

    polygons = polygons_gdf.to_crs(graph_crs).copy()
    clusters = cluster_points_gdf.to_crs(graph_crs).copy()

    if not polygons["cluster_id"].is_unique:
        raise ValueError(
            "polygons_gdf: cluster_id must be unique"
        )

    if not clusters["cluster_id"].is_unique:
        raise ValueError(
            "cluster_points_gdf: cluster_id must be unique"
        )

    cluster_points = (
        clusters.set_index("cluster_id")
        .geometry.to_dict()
    )

    road_records = []

    for u, v, key, data in G_drive.edges(
        keys=True, data=True
    ):

        geometry = data.get("geometry")

        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type != "LineString":
            continue

        road_records.append({
            "u": u,
            "v": v,
            "key": key,
            "road_id": _get_road_id(data),
            "geometry": geometry
        })

    roads_gdf = gpd.GeoDataFrame(
        road_records,
        geometry="geometry",
        crs=graph_crs
    )

    candidates = gpd.sjoin(
        roads_gdf,
        polygons[["cluster_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    best_candidates = {}

    polygon_geometry = (
        polygons.set_index("cluster_id")
        .geometry.to_dict()
    )

    for row in candidates.itertuples():

        cluster_id = row.cluster_id

        center = cluster_points.get(cluster_id)
        polygon = polygon_geometry[cluster_id]

        if center is None:
            continue

        inside_geometry = row.geometry.intersection(
            polygon
        )

        if inside_geometry.is_empty:
            continue

        line_parts = []

        if inside_geometry.geom_type == "LineString":
            line_parts = [inside_geometry]

        elif inside_geometry.geom_type == "MultiLineString":
            line_parts = list(inside_geometry.geoms)

        elif inside_geometry.geom_type == "GeometryCollection":
            for geom in inside_geometry.geoms:
                if geom.geom_type == "LineString":
                    line_parts.append(geom)
                elif geom.geom_type == "MultiLineString":
                    line_parts.extend(geom.geoms)

        line_parts = [
            geom for geom in line_parts
            if geom.length > 1e-7
        ]

        if not line_parts:
            continue

        best_part = min(
            line_parts,
            key=lambda geom: geom.distance(center)
        )

        projection = nearest_points(
            center, best_part
        )[1]

        road_id = row.road_id


        if road_id is None:
            road_id = (
                "edge",
                str(min(row.u, row.v)),
                str(max(row.u, row.v)),
                str(row.key)
            )

        group_key = (cluster_id, road_id)

        distance = center.distance(projection)

        if (
            group_key not in best_candidates
            or distance
            < best_candidates[group_key]["distance"]
        ):
            best_candidates[group_key] = {
                "cluster_id": cluster_id,
                "road_id": road_id,
                "u": row.u,
                "v": row.v,
                "key": row.key,
                "projection": projection,
                "distance": float(distance)
            }

    return list(best_candidates.values())

def _split_drive_graph_at_projections(
    G_drive,
    projections,
    tolerance=0.01
):


    G = G_drive.copy()

    edge_projections = defaultdict(list)

    access_dict = defaultdict(list)

    for item in projections:
        edge_id = (
            item["u"],
            item["v"],
            item["key"]
        )

        edge_projections[edge_id].append(item)

    new_node_counter = 0

    def create_node_id():
        nonlocal new_node_counter

        while True:
            new_node_counter += 1

            node_id = (
                "access",
                new_node_counter
            )

            if node_id not in G:
                return node_id

    for (u, v, key), items in edge_projections.items():

        if not G.has_edge(u, v, key):
            continue

        edge_data = G[u][v][key].copy()
        geometry = edge_data["geometry"]

        total_length = geometry.length
        total_time = float(edge_data["time_min"])

        if total_length <= 0:
            continue

        positions = []

        for item in items:

            position = geometry.project(
                item["projection"]
            )

            positions.append({
                "position": position,
                "item": item
            })

        positions.sort(
            key=lambda x: x["position"]
        )

        groups = []

        for record in positions:

            position = record["position"]

            if (
                not groups
                or abs(
                    position - groups[-1]["position"]
                ) > tolerance
            ):
                groups.append({
                    "position": position,
                    "items": [record["item"]]
                })

            else:
                groups[-1]["items"].append(
                    record["item"]
                )

        start_point = Point(geometry.coords[0])
        u_point = Point(
            G.nodes[u]["x"], G.nodes[u]["y"]
        )
        v_point = Point(
            G.nodes[v]["x"], G.nodes[v]["y"]
        )

        if start_point.distance(u_point) <= start_point.distance(v_point):
            start_node, end_node = u, v
        else:
            start_node, end_node = v, u

        split_positions = [0.0]
        split_nodes = [start_node]

        for group in groups:

            position = group["position"]

            if position <= tolerance:
                node_id = start_node

            elif total_length - position <= tolerance:
                node_id = end_node

            else:
                point = geometry.interpolate(position)

                node_id = create_node_id()

                G.add_node(
                    node_id,
                    x=point.x,
                    y=point.y,
                    geometry=point
                )

                split_positions.append(position)
                split_nodes.append(node_id)

            for item in group["items"]:
                access_dict[item["cluster_id"]].append({
                    "node_id": node_id,
                    "road_id": item["road_id"],
                    "distance": item["distance"]
                })

        split_positions.append(total_length)
        split_nodes.append(end_node)

        if len(split_nodes) == 2:
            continue

        G.remove_edge(u, v, key)

        for i in range(len(split_nodes) - 1):

            node_a = split_nodes[i]
            node_b = split_nodes[i + 1]

            start = split_positions[i]
            end = split_positions[i + 1]

            segment_geom = substring(
                geometry, start, end
            )

            segment_length = segment_geom.length

            if segment_length <= 0:
                continue

            fraction = segment_length / total_length
            segment_time = total_time * fraction

            new_data = edge_data.copy()

            new_data["geometry"] = segment_geom
            new_data["length"] = segment_length
            new_data["time_min"] = segment_time

            G.add_edge(
                node_a,
                node_b,
                **new_data
            )

    return G, dict(access_dict)

def _prepare_cluster_road_access(
    G_drive,
    polygons_gdf,
    cluster_points_gdf
):


    print("Finding road projections inside polygons...")

    projections = _find_cluster_access_projections(
        G_drive,
        polygons_gdf,
        cluster_points_gdf
    )

    print(f"Found {len(projections)} access projections")

    print("Splitting road graph edges...")

    G_routing, access_dict = (
        _split_drive_graph_at_projections(
            G_drive,
            projections
        )
    )

    print(
        "Total access points:",
        sum(len(v) for v in access_dict.values())
    )

    print(
        "Clusters with access:",
        len(access_dict)
    )

    return G_routing, access_dict

def _find_cluster_access_points(
    G_drive,
    polygons_gdf,
    cluster_points_gdf
):

    graph_crs = G_drive.graph["crs"]

    polygons = polygons_gdf.to_crs(graph_crs).copy()

    cluster_points = (
        cluster_points_gdf.to_crs(graph_crs).copy()
    )

    if not polygons["cluster_id"].is_unique:
        raise ValueError(
            "polygons_gdf должен содержать уникальные cluster_id"
        )

    if not cluster_points["cluster_id"].is_unique:
        raise ValueError(
            "cluster_points_gdf должен содержать уникальные cluster_id"
        )

    node_records = [
        {
            "node_id": node_id,
            "geometry": Point(data["x"], data["y"])
        }
        for node_id, data in G_drive.nodes(data=True)
    ]

    road_nodes_gdf = gpd.GeoDataFrame(
        node_records,
        geometry="geometry",
        crs=graph_crs
    )

    nodes_in_polygons = gpd.sjoin(
        road_nodes_gdf,
        polygons[["cluster_id", "geometry"]],
        how="inner",
        predicate="within"
    )

    cluster_geom = (
        cluster_points
        .set_index("cluster_id")
        .geometry
        .to_dict()
    )

    access_dict = {
        cluster_id: []
        for cluster_id in polygons["cluster_id"]
    }

    for cluster_id, group in nodes_in_polygons.groupby(
        "cluster_id"
    ):

        center = cluster_geom.get(cluster_id)

        if center is None:
            continue

        road_candidates = {}

        for row in group.itertuples():

            node_id = row.node_id
            node_point = row.geometry

            distance = node_point.distance(center)

            for neighbor, parallel_edges in (
                G_drive.adj[node_id].items()
            ):

                for edge_key, edge_data in (
                    parallel_edges.items()
                ):

                    road_id = _get_road_id(edge_data)

                    if road_id is None:
                        road_id = (
                            "edge",
                            str(node_id),
                            str(neighbor),
                            str(edge_key)
                        )

                    if (
                        road_id not in road_candidates
                        or distance
                        < road_candidates[road_id]["distance"]
                    ):

                        road_candidates[road_id] = {
                            "node_id": node_id,
                            "road_id": road_id,
                            "distance": float(distance)
                        }

        access_dict[cluster_id] = sorted(
            road_candidates.values(),
            key=lambda x: x["distance"]
        )

    return access_dict

def _find_best_access_route(
    G_drive,
    accesses_a,
    accesses_b,
    min_length_m=0
):
    if not accesses_a or not accesses_b:
        return None, None, "no_access"

    sources = {
        item["node_id"]
        for item in accesses_a
    }

    targets = {
        item["node_id"]
        for item in accesses_b
    }

    best_valid_path = None
    best_valid_time = float("inf")

    best_any_path = None
    best_any_time = float("inf")

    for source in sources:

        lengths, paths = nx.single_source_dijkstra(
            G_drive,
            source=source,
            weight="time_min",
            cutoff=(
                best_valid_time
                if best_valid_time < float("inf")
                else None
            )
        )

        for target in targets:

            if target not in lengths:
                continue

            route_time = lengths[target]
            path = paths[target]

            if route_time < best_any_time:
                best_any_time = route_time
                best_any_path = path

            if route_time >= best_valid_time:
                continue

            route_length = _get_path_road_length(
                G_drive,
                path
            )

            if route_length >= min_length_m:
                best_valid_path = path
                best_valid_time = route_time

    if best_valid_path is not None:
        return (
            best_valid_path,
            best_valid_time,
            "access_points"
        )

    if best_any_path is not None:
        return (
            best_any_path,
            best_any_time,
            "short_road_route"
        )

    return None, None, "unreachable"

def _path_to_road_segments(G_drive, path_nodes):


    segments = []
    road_geometry_list = []
    total_time = 0.0

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        parallel_edges = G_drive.get_edge_data(u, v)

        if not parallel_edges:
            raise ValueError(f"Missing road edge: {u}, {v}")

        best_key, edge_data = min(
            parallel_edges.items(),
            key=lambda item: item[1]["time_min"]
        )

        geometry = edge_data["geometry"]
        segment_time = edge_data["time_min"]

        if geometry.geom_type == "LineString":
            segments.append(geometry)

        elif geometry.geom_type == "MultiLineString":
            segments.extend(geometry.geoms)

        road_geometry_list.append(
            (geometry, segment_time)
        )

        total_time += segment_time

    route_geometry = MultiLineString(segments)

    return route_geometry, road_geometry_list, total_time

def _update_block_graph_road_routes(
    block_graph,
    G_drive,
    polygons_gdf,
    cluster_points_gdf,
    node_to_cluster
):


    G_routing, access_dict = _prepare_cluster_road_access(
        G_drive=G_drive,
        polygons_gdf=polygons_gdf,
        cluster_points_gdf=cluster_points_gdf
    )

    route_cache = {}

    stats = {
        "access_points": 0,
        "original": 0
    }

    for u, v in tqdm(
        block_graph.edges(),
        total=block_graph.number_of_edges(),
        desc="Building road routes",
        unit="edge"
    ):

        cluster_u = node_to_cluster[u]
        cluster_v = node_to_cluster[v]

        edge_data = block_graph[u][v]

        min_length_m = edge_data["geometry"].length

        pair_key = (
            frozenset((cluster_u, cluster_v)),
            min_length_m
        )

        if pair_key not in route_cache:

            accesses_u = access_dict.get(cluster_u, [])
            accesses_v = access_dict.get(cluster_v, [])

            path, time_min, route_source = _find_best_access_route(
                G_routing,
                accesses_u,
                accesses_v,
                min_length_m=min_length_m
            )

            if path is None:

                route_cache[pair_key] = None

            else:

                geometry, road_list, total_time = (
                    _path_to_road_segments(G_routing, path)
                )

                route_cache[pair_key] = {
                    "road_segments": geometry,
                    "road_geometry_list": road_list,
                    "time_min": total_time,
                    "road_count": len(road_list),
                    "road_length_m": geometry.length
                }

        result = route_cache[pair_key]

        if result is None:


            edge_data["route_found"] = False
            edge_data["route_source"] = "original"

            stats["original"] += 1

            continue

        edge_data.update({
            "weight": result["time_min"],
            "time_min": result["time_min"],
            "road_segments": result["road_segments"],
            "road_geometry_list": result["road_geometry_list"],
            "road_count": result["road_count"],
            "road_length_m": result["road_length_m"],
            "route_found": True,
            "route_source": route_source
        })

        stats["access_points"] += 1

    print("\nRoad routing results:")
    print(f"Access points routes: {stats['access_points']}")
    print(f"Original routes: {stats['original']}")
    print(f"Total edges: {block_graph.number_of_edges()}")

    return block_graph

def _prepare_data_for_block_graph(
    poly_gdf: gpd.GeoDataFrame,
    utm_crs: int | str,
    road_edges_gdf: gpd.GeoDataFrame = None,
    territory: gpd.GeoDataFrame = None,
    G_drive: nx.Graph = None
):

    poly_gdf_utm = poly_gdf.to_crs(utm_crs)

    if G_drive is None and territory is not None:
        territory_4326 = territory.to_crs(4326).geometry.iloc[0]

        G_drive, fetched_edges, G_drive_nodes = (
            get_drive_graph_iduedu(territory_4326)
        )

        if road_edges_gdf is None:
            road_edges_gdf = fetched_edges

    if road_edges_gdf is None:
        if G_drive is None:
            raise ValueError("Provide G_drive or territory or road_edges_gdf")

        road_edges_gdf = ox.graph_to_gdfs(
            G_drive,
            nodes=False
        )

    roads = road_edges_gdf.to_crs(utm_crs).reset_index(
        drop=True
    ).copy()

    point_gdf = poly_gdf_utm.copy()

    point_gdf["geometry"] = (
        point_gdf.geometry.representative_point()
    )

    return poly_gdf_utm, roads, point_gdf, G_drive


def make_block_graph(
    polygons_gdf: gpd.GeoDataFrame,
    utm_crs: int | str,
    road_edges_gdf: gpd.GeoDataFrame = None,
    territory: gpd.GeoDataFrame = None,
    clusters_gdf: gpd.GeoDataFrame = None,
    G_drive: nx.Graph = None
) -> nx.Graph:

    print("Preparing data for block graph making...")

    poly_gdf, roads, point_gdf, G_drive = (
        _prepare_data_for_block_graph(
            polygons_gdf,
            utm_crs,
            road_edges_gdf,
            territory,
            G_drive
        )
    )

    G = nx.Graph()
    G.graph["crs"] = utm_crs

    print("Adding nodes to block graph...")
    _add_nodes_block_graph(G, point_gdf)

    print("Finding touching neighbors polygons...")
    neighbors = _find_touching_neighbors(poly_gdf)

    print("Finding road connections...")
    road_connections = _find_road_connections(
        roads,
        poly_gdf,
        point_gdf
    )

    print("Uniting all connections...")
    all_connections = _get_all_connections(
        neighbors,
        road_connections
    )

    print("Adding edges to block graph...")
    _add_edges_block_graph(
        G,
        all_connections,
        roads,
        point_gdf
    )

    if clusters_gdf is not None:

        if G_drive is None:
            raise ValueError(
                "G_drive is required for road routing"
            )

        print("Preparing drive graph...")

        G_drive_routing = _prepare_drive_graph(
            G_drive,
            utm_crs
        )

        clust_points = clusters_gdf.to_crs(utm_crs).copy()

        if "cluster_id" not in poly_gdf.columns:
            raise ValueError(
                "polygons_gdf must contain cluster_id"
            )

        if "cluster_id" not in clust_points.columns:
            raise ValueError(
                "clusters_gdf must contain cluster_id"
            )

        node_to_cluster = poly_gdf["cluster_id"].to_dict()

        if not set(G.nodes).issubset(node_to_cluster):
            raise ValueError(
                "Block graph node IDs do not match polygon indices. "
                "Build an explicit node_to_cluster mapping."
            )

        print("Finding access points and road routes...")

        G = _update_block_graph_road_routes(
            block_graph=G,
            G_drive=G_drive_routing,
            polygons_gdf=poly_gdf,
            cluster_points_gdf=clust_points,
            node_to_cluster=node_to_cluster
        )

    print("Done!")
    return G

 
def block_graph_to_gdfs(block_graph):
    nodes_data = []
    for node, data in block_graph.nodes(data=True):
        if 'geometry' in data:
            geometry = data['geometry']
        elif 'x' in data and 'y' in data:
            geometry = Point(data['x'], data['y'])
        else:
            print(f"Warning: Node {node} has no geometry or coordinates")
            continue
            
        nodes_data.append({
            'node_id': node,
            'cluster_id': data.get('cluster_id'),
            'name': data.get('name'),
            'geometry': geometry
        })
    
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=block_graph.graph.get('crs'))
    
    edges_data = []
    for u, v, data in block_graph.edges(data=True):
        if 'geometry' not in data:
            u_data = block_graph.nodes[u]
            v_data = block_graph.nodes[v]
            
            if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
                geometry = LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])
            else:
                print(f"Warning: Edge ({u}, {v}) has no geometry")
                continue
        else:
            geometry = data['geometry']
            

        edges_data.append({
            'u': u,
            'v': v,
            'geometry': geometry,
            'weight': data.get('weight', 1),
            'time_min': data.get('time_min', 0),
            'road_segments': data.get('road_segments'),
            'road_geometry_list': data.get('road_geometry_list', []),
            'road_count': data.get('road_count', 0),
            'route_found': data.get('route_found', False),
            "road_length_m": data.get("road_length_m"),
            "route_source": data.get("route_source", "original")
        })
    
    edges_gdf = gpd.GeoDataFrame(edges_data, crs=block_graph.graph.get('crs'))
    
    return nodes_gdf, edges_gdf

def get_road_segments_for_block_graph_edges(block_graph_edges_gdf):
    
    road_segments_list = []
    
    for edge_id, row in block_graph_edges_gdf.iterrows():
        u = row['u']
        v = row['v']
        road_geometry_list = row['road_geometry_list']
        time_min = row['time_min']
        
        if road_geometry_list and isinstance(road_geometry_list, list):
            for segment_idx, segment_tuple in enumerate(road_geometry_list):
                segment_geom = segment_tuple[0]
                segment_time = segment_tuple[-1]
                if segment_geom and not segment_geom.is_empty:
                    road_segments_list.append({
                        'edge_id': edge_id,
                        'u': u,
                        'v': v,
                        'segment_id': f"{edge_id}_{segment_idx}",
                        'geometry': segment_geom,
                        'time_min': segment_time 
                    })
    road_segments_gdf = gpd.GeoDataFrame(road_segments_list, crs=block_graph_edges_gdf.crs)
    
    return road_segments_gdf

def _get_path_road_length(G_drive, path_nodes):

    total_length = 0.0

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):

        parallel_edges = G_drive.get_edge_data(u, v)

        if not parallel_edges:
            raise ValueError(f"Missing road edge: {u}, {v}")

        _, edge_data = min(
            parallel_edges.items(),
            key=lambda item: item[1]["time_min"]
        )

        total_length += edge_data["geometry"].length

    return total_length

def get_road_routes_for_block_graph_edges(block_graph_edges_gdf):


    road_routes_list = []

    for edge_id, row in block_graph_edges_gdf.iterrows():

        route_geometry = row.get("road_segments")

        if route_geometry is None or route_geometry.is_empty:
            continue

        road_routes_list.append({
            "edge_id": edge_id,
            "u": row["u"],
            "v": row["v"],
            "geometry": route_geometry,
            "time_min": row["time_min"],
            "weight": row["weight"],
            "road_count": row.get("road_count", 0),
            "route_found": row.get("route_found", False)
        })

    road_routes_gdf = gpd.GeoDataFrame(
        road_routes_list,
        geometry="geometry",
        crs=block_graph_edges_gdf.crs
    )

    return road_routes_gdf


def _prepare_data_for_pt_graph(
        block_graph : nx.Graph,
        utm_crs : int | str,
        polygons_gdf,
        drive_edges_gdf : gpd.GeoDataFrame = None,
        tr_edges_gdf : gpd.GeoDataFrame = None, 
        territory : gpd.GeoDataFrame = None):
    
    block_graph_copy = block_graph.copy()
    block_graph_copy = ox.project_graph(nx.MultiDiGraph(block_graph_copy), to_crs=utm_crs)
    block_graph_copy =nx.Graph(block_graph_copy)

    for node_id in block_graph_copy.nodes():
        node_data = block_graph_copy.nodes[node_id]
        if 'geometry' not in node_data and 'x' in node_data and 'y' in node_data:
            node_data['geometry'] = Point(node_data['x'], node_data['y'])
            
    bg_nodes_gdf, bg_edges_gdf = block_graph_to_gdfs(block_graph_copy)

    poly_gdf = polygons_gdf.to_crs(utm_crs)

    if drive_edges_gdf is None:
        if territory is None:
            raise ValueError("territory must be provided if drive_edges_gdf is not provided")
        territory_4326 = territory.to_crs(4326).geometry.iloc[0]
        G_drive, G_drive_edges, G_drive_nodes = get_drive_graph_iduedu(territory_4326)
        drive_edges_gdf = G_drive_edges.to_crs(utm_crs)
    
    if tr_edges_gdf is None:
        if territory is None:
            raise ValueError("territory must be provided if tr_edges_gdf is not provided")
        territory_4326 = territory.to_crs(4326).geometry.iloc[0]
        G_pt, G_pt_edges, G_pt_nodes = get_bus_graph_iduedu(territory_4326)
        tr_edges_gdf = G_pt_edges.to_crs(utm_crs)

    return block_graph_copy, bg_nodes_gdf, drive_edges_gdf, tr_edges_gdf, poly_gdf


def pt_graph_project_by_stops_name(
        block_graph : nx.Graph,
        dict_routes : dict,
        polygons_gdf : gpd.GeoDataFrame,
        utm_crs : int | str,
        tr_edges_gdf : gpd.GeoDataFrame = None,
        drive_edges_gdf : gpd.GeoDataFrame =None,
        territory : gpd.GeoDataFrame = None):
    
    print("Preparing data for pt_graph...")
    block_graph_copy, bg_nodes_gdf, drive_edges_gdf, tr_edges_gdf, poly_gdf = _prepare_data_for_pt_graph(block_graph,
                                                                               utm_crs,
                                                                               polygons_gdf,
                                                                               drive_edges_gdf,
                                                                               tr_edges_gdf,
                                                                               territory
                                                                               )

    

    pt_graph = nx.Graph()
    pt_graph.add_nodes_from(block_graph_copy.nodes(data=True))
    pt_graph.graph['crs'] = utm_crs

    print("Comparing stops and zones...")
    block_project_dict = _get_block_project_dict(dict_routes, poly_gdf, bg_nodes_gdf, utm_crs)

    print("Projecting pt routes and building pt graph...")
    block_graph_copy, pt_graph = _build_pt_graph(block_project_dict,
                                                 block_graph_copy,
                                                 pt_graph,
                                                 poly_gdf,
                                                 tr_edges_gdf,
                                                 drive_edges_gdf)

    print("Done!")
    return block_graph_copy, block_project_dict, pt_graph


def create_routes_geodataframe(bg_from_stop_names, route_path_by_stop_names):
    edges_list = []
    
    for route, path_nodes in route_path_by_stop_names.items():
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            
            if bg_from_stop_names.has_edge(u, v):
                edge_data = bg_from_stop_names.get_edge_data(u, v)
                
                geometry = edge_data.get('geometry')
                weight = edge_data.get('weight', edge_data.get('time_min'))
                
                edges_list.append({
                    'u': u,
                    'v': v,
                    'route': route,
                    'weight': weight,
                    'geometry': geometry,
                    'routes_list': edge_data.get('routes', [])  
                })
    
    routes_gdf = gpd.GeoDataFrame(edges_list, crs=bg_from_stop_names.graph.get('crs'))
    
    return routes_gdf

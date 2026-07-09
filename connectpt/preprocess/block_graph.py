import geopandas as gpd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString

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


def _prepare_data_for_block_graph(
    poly_gdf : gpd.GeoDataFrame, 
    utm_crs : int | str, 
    road_edges_gdf : gpd.GeoDataFrame = None, 
    territory : gpd.GeoDataFrame = None ):
    poly_gdf_utm = poly_gdf.to_crs(utm_crs)
    if road_edges_gdf is None:
        if territory is None:
            raise ValueError("territory must be provided if road_edges_gdf is not provided")
        territory_utm = territory.to_crs(4326).geometry.iloc[0]
        G_drive, road_edges_gdf, G_drive_nodes = get_drive_graph_iduedu(territory_utm)
    
    road_edges_gdf.to_crs(utm_crs, inplace=True)
    point_gdf = poly_gdf_utm.copy()
    point_gdf['geometry'] = point_gdf.geometry.representative_point()

    road_edges_gdf_copy = road_edges_gdf.reset_index(drop=True).copy()

    return poly_gdf_utm, road_edges_gdf_copy, point_gdf


def make_block_graph(polygons_gdf : gpd.GeoDataFrame, 
                     utm_crs : int | str,
                     road_edges_gdf : gpd.GeoDataFrame = None, 
                     territory : gpd.GeoDataFrame = None) -> nx.Graph:
    
    print("Preparing data for block graph making...")
    poly_gdf, roads, point_gdf = _prepare_data_for_block_graph(polygons_gdf, utm_crs, road_edges_gdf, territory)

    G = nx.Graph()
    G.graph['crs'] = utm_crs

    print("Adding nodes to block graph...")
    _add_nodes_block_graph(G, point_gdf)
    
    print("Finding touching neighbors polygons...")
    neighbors = _find_touching_neighbors(poly_gdf)
    
    print("Finding road connections...")
    road_connections = _find_road_connections(roads,
                                              poly_gdf,
                                              point_gdf)
    
    print("Uniting all connections..")
    all_connections = _get_all_connections(neighbors, road_connections)
    
    print("Adding edges to block graph...")
    _add_edges_block_graph(
        G,
        all_connections,
        roads, 
        point_gdf
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
            'road_geometry_list': data.get('road_geometry_list', []),
            'road_count': data.get('road_count', 0)
        })
    
    edges_gdf = gpd.GeoDataFrame(edges_data, crs=block_graph.graph.get('crs'))
    
    return nodes_gdf, edges_gdf

def get_road_segments_for_block_graph_edges(block_graph_edges_gdf):
    
    road_segments_list = []
    
    for edge_id, row in block_graph_edges_gdf.iterrows():
        u = row['u']
        v = row['v']
        road_geometry_list = row['road_geometry_list']
        
        if road_geometry_list and isinstance(road_geometry_list, list):
            for segment_idx, segment_geom in enumerate(road_geometry_list):
                if segment_geom and not segment_geom.is_empty:
                    road_segments_list.append({
                        'edge_id': edge_id,
                        'u': u,
                        'v': v,
                        'segment_id': f"{edge_id}_{segment_idx}",
                        'geometry': segment_geom
                    })
    road_segments_gdf = gpd.GeoDataFrame(road_segments_list, crs=block_graph_edges_gdf.crs)
    
    return road_segments_gdf


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

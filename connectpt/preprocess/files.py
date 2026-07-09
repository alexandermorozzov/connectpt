import os
import numpy as np
import torch
import networkx as nx

from .block_graph import block_graph_to_gdfs

def save_files(output_folder : str,
               city_name : str,
               block_graph : nx.Graph = None,
               od_matrix : np.ndarray = None,
               block_project_dict : dict = None,
               save_coords : bool = False,
               save_demand : bool = False,
               save_travel_times : bool = False,
               save_routes : bool = False
               ):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    path_start = output_folder + "//" + city_name
    if block_graph is not None:
        block_graph_nodes, block_graph_edges = block_graph_to_gdfs(block_graph)
    else:
        if save_coords or save_travel_times:
            raise ValueError("For Coords and Travel Times block_graph must be provided")
    if save_coords and (block_graph_nodes is not None):
        block_graph_nodes['x'] = block_graph_nodes.geometry.x
        block_graph_nodes['y'] = block_graph_nodes.geometry.y
        file_name = path_start + "Coords.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(len(block_graph_nodes)) + '\n')
            for idx, row in block_graph_nodes.iterrows():
                x = row['x']
                y = row['y']
                f.write(f"{x} {y}\n")
        print(f"{file_name} saved")

    
    if save_demand and (od_matrix is None):
        raise ValueError("For Demand od_matrix must be provided")

    if (od_matrix is not None) and save_demand:
        file_name = path_start + "Demand.txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            n = od_matrix.shape[0]
            f.write(str(n) + '\n')                    
            np.savetxt(f, od_matrix, fmt='%.6f')   
        print(f"{file_name} saved")
    
    if (block_graph is not None) and save_travel_times:
        file_name = path_start + "TravelTimes.txt"

        node_ids = sorted(block_graph_nodes['node_id'].unique())
        n = len(node_ids)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        distance_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(distance_matrix, 0)
        for _, edge in block_graph_edges.iterrows():
            u = edge['u']
            v = edge['v']
            
            if u in node_to_idx and v in node_to_idx:
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                time_min = edge['time_min']
                
                distance_matrix[idx_u, idx_v] = time_min
                distance_matrix[idx_v, idx_u] = time_min  
        
        with open(file_name, 'w', encoding='utf-8') as f:
            np.savetxt(f, distance_matrix, fmt='%.6f')  
        print(f"{file_name} saved")
    
    if save_routes and (block_project_dict is None):
        raise ValueError("For Routes block_project_dict must be provided")
    
    if save_routes and block_project_dict:
        file_name = path_start + "Routes.pkl"
        routes = list(block_project_dict.values())
        routes = [[int(stop) for stop in route] for route in routes]
        max_length = max(len(route) for route in routes)

        padded_routes = []
        for route in routes:
            padded_route = route + [-1] * (max_length - len(route))
            padded_routes.append(padded_route)
        
        tensor_result = torch.tensor(padded_routes, dtype=torch.long)

        torch.save(tensor_result, file_name)
        print(f"{file_name} saved")


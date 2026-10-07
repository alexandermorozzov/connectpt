"""Граф города, преобразования и сборка батча."""
from itertools import permutations
import logging as log

import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.transforms import BaseTransform, RandomFlip

from .shortest_paths import floyd_warshall

STOP_KEY = "stop"
STREET_KEY = (STOP_KEY, "street", STOP_KEY)
DEMAND_KEY = (STOP_KEY, "demand", STOP_KEY)
# we don't use this in here, but we do elsewhere
ROUTE_KEY = (STOP_KEY, "route", STOP_KEY)


DMD_FEAT_IDX = 0
SHORTESTPATH_FEAT_IDX = 1

OUT_DEMAND_FEAT_IDX = 4
IN_DEMAND_FEAT_IDX = 5

SPEED_MPS = 15.0


def to_batch(
    city,
    device=None,
    *,
    transforms=(),
    edit_features=False,
    scale_dynamically=True,
    extra_node_feats=True,
    center_nodes=True,
):
    """Собрать батч из Instance, матриц города или списка графов.

    Преобразования применяются к копиям. Edit сохраняет прежние два набора
    координат в признаках узлов: pos и результат InsertPosFeatures.
    """
    from .instances import Instance

    if isinstance(city, Instance):
        city = city.tensors
    if isinstance(city, dict):
        city = CityGraphData.from_tensors_with_transformations(
            city,
            scale_dynamically=scale_dynamically,
            extra_node_feats=extra_node_feats,
            center_nodes=center_nodes,
        )
    if isinstance(city, Batch):
        graphs = city.to_data_list()
    elif isinstance(city, HeteroData):
        graphs = [city]
    else:
        graphs = list(city)
    prepared = []
    for graph in graphs:
        if transforms or edit_features:
            graph = graph.clone()
        for transform in transforms:
            graph = transform(graph)
        if edit_features:
            graph[STOP_KEY].x = graph[STOP_KEY].pos.clone()
            graph = InsertPosFeatures()(graph)
        prepared.append(graph)
    return Batch.from_data_list(prepared).to(device)


def replace_nonfinite_drive_times(drive_times):
    finite = torch.where(torch.isfinite(drive_times), drive_times, torch.zeros_like(drive_times))
    if finite.ndim >= 3:
        diameter = finite.flatten(1, 2).max(1).values
    else:
        diameter = finite.flatten().max().reshape(1)
    penalty = (2.0 * diameter.clamp_min(1e-6)).to(
        device=drive_times.device, dtype=drive_times.dtype
    )
    while penalty.ndim < drive_times.ndim:
        penalty = penalty.unsqueeze(-1)
    return torch.where(torch.isfinite(drive_times), drive_times, penalty)


class InsertPosFeatures(BaseTransform):
    def forward(self, data):
        data = data.clone()
        for key in [STOP_KEY]:
            val = data[key]
            if hasattr(val, "x") and val.x is not None:
                val.x = torch.cat((val.pos, val.x), dim=1)
            else:
                val.x = val.pos.clone()
        return data


class RandomFlipCity(RandomFlip):
    def __init__(self, axis=0) -> None:
        super().__init__(axis)

    def forward(self, data: Data) -> Data:
        super().forward(data[STOP_KEY])
        return data


class SpaceScaleTransform(BaseTransform):
    def __init__(self, min_scale=0.75, max_scale=1.25):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, data):
        data = data.clone()
        scale_range = self.max_scale - self.min_scale
        scale = torch.rand(1) * scale_range + self.min_scale
        data[STOP_KEY].pos *= scale
        data[STREET_KEY].edge_attr *= scale
        data.drive_times *= scale
        return data


class DemandScaleTransform(BaseTransform):
    def __init__(self, min_scale=0.5, max_scale=1.5):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, data):
        data = data.clone()
        scale_range = self.max_scale - self.min_scale
        scale = torch.rand(1) * scale_range + self.min_scale
        data[DEMAND_KEY].edge_attr[:, DMD_FEAT_IDX] *= scale
        data.demand *= scale
        if data[STOP_KEY].x.shape[1] > 2:
            data[STOP_KEY].x[:, OUT_DEMAND_FEAT_IDX] *= scale
            data[STOP_KEY].x[:, IN_DEMAND_FEAT_IDX] *= scale
        return data


class CityGraphData(HeteroData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["demand", "drive_times", "street_adj", "nexts"]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    @staticmethod
    def from_tensors(node_locs, street_adj, demand, pos_only=False):
        graph = CityGraphData()
        n_nodes = node_locs.shape[0]
        graph.fixed_routes = torch.zeros((0, n_nodes))
        graph[STOP_KEY].pos = node_locs
        is_edge = (street_adj > 0) & street_adj.isfinite()
        edge_idx = is_edge.nonzero().t()
        graph[STREET_KEY].edge_index = edge_idx
        graph[STREET_KEY].edge_attr = street_adj[is_edge]
        graph.street_adj = street_adj

        # compute all shortest paths
        nexts, times = floyd_warshall(street_adj)
        graph.drive_times = times.squeeze(0)
        graph.nexts = nexts.squeeze(0)

        # compute demand features
        graph.demand = demand
        demand_nonzeros = demand.nonzero()
        graph[DEMAND_KEY].edge_index = demand_nonzeros.t()
        dmd_feats = demand[demand_nonzeros[:, 0], demand_nonzeros[:, 1]]
        drive_times = replace_nonfinite_drive_times(times)[
            0, demand_nonzeros[:, 0], demand_nonzeros[:, 1]
        ]
        graph[DEMAND_KEY].edge_attr = torch.stack((dmd_feats, drive_times))

        if pos_only:
            graph[STOP_KEY].x = torch.zeros((n_nodes, 0))
        else:
            graph[STOP_KEY].x = get_node_features(edge_idx, demand)

        return graph

    @staticmethod
    def from_tensors_with_transformations(
        tensors,
        assumed_speed_mps=SPEED_MPS,
        scale_dynamically=True,
        extra_node_feats=True,
        fully_connected_demand=True,
        center_nodes=True,
        fixed_routes=None,
        pos_only=False,
    ):
        """
        Creates a CityGraphData object from input tensors with transformations similar to from_mumford_data.

        Args:
            node_locs (torch.Tensor): Tensor of node coordinates (n_nodes, 2).
            street_adj (torch.Tensor): Adjacency matrix of street edges in seconds (n_nodes, n_nodes).
            demand (torch.Tensor): Demand matrix (n_nodes, n_nodes).
            assumed_speed_mps (float): Assumed speed in meters per second for scaling.
            scale_dynamically (bool): Whether to dynamically scale node locations.
            extra_node_feats (bool): Whether to include additional node features.
            fully_connected_demand (bool): Whether to create fully connected demand edges.
            center_nodes (bool): Whether to center node locations at 0.
            fixed_routes (torch.Tensor, optional): Tensor of fixed routes (n_routes, n_nodes).
            pos_only (bool): If True, node features are set to zero (from from_tensors).
        """
        node_locs = tensors["node_locs"]
        street_adj = tensors["street_adj"]
        demand = tensors["demand"]

        data = CityGraphData()
        n_nodes = node_locs.shape[0]

        # Center node locations if specified
        orig_pos = node_locs.clone()
        if center_nodes:
            node_locs = node_locs - node_locs.mean(dim=0)

        # Dynamic scaling based on assumed speed
        if scale_dynamically:
            euc_dists = get_euclidean_distances(node_locs)
            edge_drive_dists_m = street_adj * assumed_speed_mps
            has_edge = (street_adj > 0) & street_adj.isfinite()
            has_valid_edge = has_edge & (euc_dists > 0)
            edge_ratios = edge_drive_dists_m[has_valid_edge] / euc_dists[has_valid_edge]
            meters_per_unit = edge_ratios.mean()
            assert meters_per_unit < float("inf")
            log.info(f"Estimated meters per unit: {meters_per_unit}")
            side_len = max((node_locs.max(dim=0)[0] - node_locs.min(dim=0)[0])) * meters_per_unit
            log.info(f"Environment side length: {side_len} meters")
            node_locs = node_locs * meters_per_unit

        # Set fixed routes
        if fixed_routes is None:
            fixed_routes = torch.zeros((0, n_nodes))
        data.fixed_routes = fixed_routes

        # Set node features
        has_edge = (street_adj > 0) & street_adj.isfinite()
        street_idx = torch.stack(torch.where(has_edge))
        if pos_only:
            data[STOP_KEY].x = torch.zeros((n_nodes, 0))
        else:
            if extra_node_feats:
                node_features = get_node_features(street_idx, demand)
                node_features = torch.cat((node_locs, node_features), dim=1)
            else:
                node_features = node_locs
            data[STOP_KEY].x = node_features

        # Set node positions
        data[STOP_KEY].pos = node_locs
        data[STOP_KEY].orig_pos = orig_pos

        # Set street edges
        street_attr = street_adj[has_edge]
        data[STREET_KEY].edge_index = street_idx
        data[STREET_KEY].edge_attr = street_attr
        data.street_adj = street_adj

        # Compute all shortest paths
        nexts, drive_times = floyd_warshall(street_adj)
        data.drive_times = drive_times.squeeze(0)
        data.nexts = nexts.squeeze(0)

        # Compute demand features
        if fully_connected_demand:
            dmd_idx = torch.tensor(list(permutations(range(n_nodes), 2))).T
        else:
            has_demand = demand > 0
            dmd_idx = torch.stack(torch.where(has_demand))

        data[DEMAND_KEY].edge_index = dmd_idx
        demand_feat = demand[dmd_idx[0], dmd_idx[1]]
        safe_drive_times = replace_nonfinite_drive_times(drive_times).squeeze(0)
        drive_time_feat = safe_drive_times[dmd_idx[0], dmd_idx[1]]
        dmd_edge_feat = torch.stack((demand_feat, drive_time_feat), dim=1)
        data[DEMAND_KEY].edge_attr = dmd_edge_feat
        data.demand = demand

        assert data.fixed_routes is not None
        return data

    @property
    def drive_dists(self):
        return self.drive_times * SPEED_MPS

    @property
    def pos(self):
        return self[STOP_KEY].pos


def get_node_features(street_index, demand_matrix):
    n_nodes = demand_matrix.shape[0]
    street_edge_exists = torch.zeros((n_nodes, n_nodes))
    street_edge_exists[street_index[0], street_index[1]] = 1
    out_street_deg = street_edge_exists.sum(dim=0)
    in_street_deg = street_edge_exists.sum(dim=1)
    feats = [out_street_deg, in_street_deg]
    return torch.stack(feats, dim=1)


def get_euclidean_distances(locs):
    n_nodes = locs.shape[0]
    euc_dists = torch.zeros((n_nodes, n_nodes), dtype=locs.dtype)
    upper_tri_idxs = torch.triu_indices(n_nodes, n_nodes, offset=1)
    # populate the above-diagonal entries
    euc_dists[upper_tri_idxs[0], upper_tri_idxs[1]] = torch.pdist(locs)
    # and the below-diagonal entries
    euc_dists = euc_dists + euc_dists.T

    return euc_dists

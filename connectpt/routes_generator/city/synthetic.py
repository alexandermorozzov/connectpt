import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from itertools import permutations
import torch
from scipy.spatial import Voronoi
import networkx as nx
import torch_geometric.utils as pygu
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph, RemoveIsolatedNodes, Compose

from .graph import (
    CityGraphData,
    STOP_KEY,
    STREET_KEY,
    DEMAND_KEY,
    InsertPosFeatures,
    SPEED_MPS,
    get_euclidean_distances,
    get_node_features,
    DMD_FEAT_IDX,
    SHORTESTPATH_FEAT_IDX,
)
from .shortest_paths import floyd_warshall
from .const import (
    RAW_GRAPH_FILENAME,
    MST,
    OUT_KNN,
    IN_KNN,
    VORONOI,
    GRID4,
    GRID8,
    CIRC,
    SMALLWORLD,
    MIXED,
    MAX_DEMAND,
    MIN_DEMAND,
    MAX_POP_STD_M,
    MIN_POP_STD_M,
    MAX_SINK_RADIUS_M,
    MIN_SINK_RADIUS_M,
    CENTERNODES_PER_NODE,
    SIDE_LENGTH_M,
)


class _GraphUnpickler(pickle.Unpickler):
    """Читать города, записанные до переноса CityGraphData в city/graph.py."""

    def find_class(self, module, name):
        if name == "CityGraphData" and module in (
            "connectpt.routes_generator.citygraph_dataset",
            "citygraph_dataset",
        ):
            return CityGraphData
        return super().find_class(module, name)


class _GraphPickle:
    Unpickler = _GraphUnpickler
    load = staticmethod(lambda stream, **kwargs: _GraphUnpickler(stream, **kwargs).load())


def load_graphs(path):
    """Читать список графов, включая старые pickle-датасеты."""
    with Path(path).open("rb") as stream:
        return _GraphUnpickler(stream).load()


class StoredCityGraphs(InMemoryDataset):
    """Read an existing collated dataset without creating files or applying transforms."""
    def __init__(self, path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(path, weights_only=False, pickle_module=_GraphPickle)


def load_graph_collection(path):
    """Read raw or collated legacy cities without modifying their source folder."""
    path = Path(path)
    if path.is_file():
        return load_graphs(path)
    collated = path / 'processed' / 'collated.pt'
    if collated.is_file():
        return StoredCityGraphs(collated)
    return load_graphs(path / RAW_GRAPH_FILENAME)


@dataclass(frozen=True)
class SyntheticParams:
    min_nodes: int
    max_nodes: int
    edge_keep_prob: float
    data_type: str
    directed: bool
    fully_connected_demand: bool
    side_length_m: float
    mumford_style: bool
    pos_only: bool

    def __post_init__(self):
        if not 2 <= self.min_nodes <= self.max_nodes or not 0 <= self.edge_keep_prob <= 1 or self.side_length_m <= 0:
            raise ValueError('invalid synthetic city size, edge probability or side length')
        if self.data_type not in (MST, OUT_KNN, IN_KNN, VORONOI, GRID4, GRID8, CIRC, SMALLWORLD, MIXED):
            raise ValueError(f'unknown synthetic graph type: {self.data_type}')
        if self.data_type == SMALLWORLD and self.directed:
            raise ValueError('smallworld graphs must be undirected')


class CityGraphDataset(InMemoryDataset):
    def __init__(self, root=None, transforms=None):
        # append the required transforms to whatever transforms were passed in
        if transforms is None:
            transforms = []

        # always insert pos features.
        transforms.append(InsertPosFeatures())
        transform = Compose(transforms)

        super().__init__(root, transform, None, None)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False, pickle_module=_GraphPickle
        )

    @property
    def processed_file_names(self):
        return ["collated.pt"]

    def process(self):
        # read in the list from the pickle file
        data_list = load_graphs(Path(self.root) / RAW_GRAPH_FILENAME)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DynamicCityGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        min_nodes,
        max_nodes,
        edge_keep_prob=0.7,
        data_type="mixed",
        directed=False,
        fully_connected_demand=True,
        side_length_m=SIDE_LENGTH_M,
        transforms=None,
        mumford_style=True,
        pos_only=False,
    ):
        super().__init__()
        self.data_type = data_type
        self.edge_keep_prob = edge_keep_prob
        self.directed = directed
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.fully_connected_demand = fully_connected_demand
        self.side_length_m = side_length_m
        self.mumford_style = mumford_style
        self.pos_only = pos_only

        # append the required transforms to whatever transforms were passed in
        if transforms is None:
            transforms = []

        # always insert pos features.
        transforms.append(InsertPosFeatures())
        self.transform = Compose(transforms)

        self.graph_fns = {
            MST: build_mst_graph,
            IN_KNN: build_in4nn_graph,
            OUT_KNN: build_out4nn_graph,
            CIRC: build_circulant_graph,
            VORONOI: build_voronoi_graph,
            GRID4: build_4grid,
            GRID8: build_8grid,
        }

        if data_type == SMALLWORLD:
            assert not directed, "smallworld graphs are undirected"

        if self.directed:
            self.nx_creator = nx.DiGraph
        else:
            self.nx_creator = nx.Graph

    def __iter__(self):
        return self._generator()

    def _generator(self):
        while True:
            graph = self.generate_graph()
            graph = self.transform(graph)
            yield graph

    def generate_graph(self, n_nodes=None, max_n_nodes=None):
        if not max_n_nodes:
            max_n_nodes = self.max_nodes
        if not n_nodes:
            n_nodes = random.randint(self.min_nodes, max_n_nodes)
        assert n_nodes <= max_n_nodes

        # build a street graph with nodes in (-1, 1)
        street_graph = self._generate_street_graph(n_nodes)

        # scale the node positions appropriately
        street_graph.pos = street_graph.pos * self.side_length_m / 2

        graph = CityGraphData()
        fixed_routes = torch.zeros((0, max_n_nodes))
        graph.fixed_routes = fixed_routes
        graph[STOP_KEY].pos = street_graph.pos
        graph[STREET_KEY].edge_index = street_graph.edge_index

        # assign distances to edges
        n_nodes = street_graph.num_nodes
        euc_dists = get_euclidean_distances(street_graph.pos)
        # edge weights are drive times in seconds
        street_idx = graph[STREET_KEY].edge_index
        edge_times = euc_dists[street_idx[0], street_idx[1]] / SPEED_MPS
        graph[STREET_KEY].edge_attr = edge_times[:, None]

        street_edge_mat = torch.full((max_n_nodes, max_n_nodes), float("inf"))
        street_edge_mat[street_idx[0], street_idx[1]] = edge_times
        street_edge_mat.fill_diagonal_(0)
        graph.street_adj = street_edge_mat

        # compute all shortest paths
        nexts, times = floyd_warshall(street_edge_mat)
        graph.drive_times = times.squeeze(0)
        graph.nexts = nexts.squeeze(0)

        # generate demand
        done = False
        while not done:
            if self.mumford_style:
                demand = uniform_demand_on_edges(street_graph.pos, 1.0)
            else:
                demand = inter_loci_demand(street_graph.pos)

            if not self.directed:
                # make the demand matrix symmetric
                demand = demand.triu()
                demand += demand.T.clone()

            scale = demand.max()
            if scale > 0:
                # scale the demands to be 0 to max demand
                demand /= scale

                no_internode_demand = demand == 0
                demand_range = MAX_DEMAND - MIN_DEMAND
                demand = demand * demand_range + MIN_DEMAND
                demand[no_internode_demand] = 0

            done = demand.sum() > 0

        # pad the demand matrix to the max size
        n_pad = max_n_nodes - n_nodes
        demand = torch.nn.functional.pad(demand, (0, n_pad, 0, n_pad))
        graph.demand = demand
        if self.fully_connected_demand:
            dmd_idx = torch.tensor(list(permutations(range(n_nodes), 2))).T
        else:
            dmd_idx = torch.stack(torch.where(demand))
        graph[DEMAND_KEY].edge_index = dmd_idx

        # set the edge features
        dmd_feat = demand[dmd_idx[0], dmd_idx[1]]
        drive_time_attr = times[0, dmd_idx[0], dmd_idx[1]]
        dmd_edge_attr = torch.zeros((dmd_feat.shape[0], 2))
        dmd_edge_attr[:, DMD_FEAT_IDX] = dmd_feat
        dmd_edge_attr[:, SHORTESTPATH_FEAT_IDX] = drive_time_attr
        graph[DEMAND_KEY].edge_attr = dmd_edge_attr

        # set node features
        if self.pos_only:
            # pos features will be inserted by the InsertPosFeatures transform.
            graph[STOP_KEY].x = torch.zeros((n_nodes, 0))
        else:
            graph[STOP_KEY].x = get_node_features(street_idx, demand)

        assert graph is not None
        return graph

    def _generate_street_graph(self, n_nodes):
        if self.data_type == SMALLWORLD:
            return build_smallworld_graph(2, 7)
        else:
            if self.data_type == MIXED:
                choices = [MST, IN_KNN, VORONOI, GRID4, GRID8]
                if self.directed:
                    # if the graphs are directed, out_knn differs from in_knn
                    choices.append(OUT_KNN)

                gen_key = random.choice(choices)
                gen_fn = self.graph_fns[gen_key]
            else:
                gen_fn = self.graph_fns[self.data_type]
            return gen_fn(n_nodes, edge_keep_prob=self.edge_keep_prob, directed=self.directed)


def uniform_demand_on_edges(node_locs, demand_edge_keep_prob):
    # build a demand matrix with uniformly random demands for each node pair
    n_nodes = node_locs.shape[0]
    while True:
        demand = torch.rand((n_nodes, n_nodes))
        demand[n_nodes:] = 0
        demand[:, n_nodes:] = 0
        # zero out approximately demand_frac fraction of demands
        zero_scores = torch.rand(n_nodes, n_nodes)
        demand[zero_scores > demand_edge_keep_prob] = 0
        demand.fill_diagonal_(0)
        if (demand > 0).sum() > demand.numel() * demand_edge_keep_prob / 2:
            break

    return demand


def locus_based_demand(node_locs, max_loci_frac):
    # build a demand matrix with n_loci loci of demand
    loci_frac = max_loci_frac * torch.rand(1)
    n_nodes = node_locs.shape[0]
    n_loci = (n_nodes * loci_frac).ceil().int().item()
    demand = torch.zeros((n_nodes, n_nodes))
    # randomly choose n_loci loci
    loci = torch.randperm(n_nodes)[:n_loci]
    # demand from each node to each of the loci
    demand[:, loci] = torch.rand((n_nodes, n_loci))
    # demand from each of the loci to each node
    demand[loci, :] = torch.rand((n_loci, n_nodes))
    demand.fill_diagonal_(0)
    return demand


def make_oval_groups(node_locs, n_groups, min_radius, max_radius):
    n_nodes = node_locs.shape[0]
    is_in_group = torch.zeros((n_nodes, n_groups), dtype=torch.bool)
    cluster_range = max_radius - min_radius

    # for each pop cluster
    centers = []
    shapes = []
    for gi in range(n_groups):
        # pick a node to be its center that is not a neighbour of any other
        # cluster center and is not part of any other cluster
        is_in_no_group = ~is_in_group.any(dim=1)
        if is_in_no_group.sum() == 0:
            n_groups = gi + 1
            is_in_group = is_in_group[:, :n_groups]
            break
        node = is_in_no_group.float().multinomial(1)
        centers.append(node_locs[node][0])
        # randomly generate an oval around the center node
        radii = torch.rand(2) * cluster_range + min_radius
        shapes.append(radii)
        oval_vals = ((node_locs - node_locs[node]) ** 2 / radii**2).sum(dim=1)
        # all nodes in the oval are now part of the cluster
        is_in_group[:, gi] = oval_vals <= 1
    return is_in_group, centers, shapes


def inter_loci_demand(node_locs):
    # pick number of population clusters
    n_nodes = node_locs.shape[0]
    max_centers = max(int(CENTERNODES_PER_NODE * n_nodes), 1)
    n_pop_centers = torch.randint(max_centers, (1,)) + 1
    is_in_cluster, cluster_centers, cluster_shapes = make_oval_groups(
        node_locs, n_pop_centers, MIN_POP_STD_M, MAX_POP_STD_M
    )
    min_density = 0.3
    density_range = 1 - min_density

    # range of populations on nodes in a cluster, relative to the density
    pop_var = 0.3
    node_pops_by_type = []
    for is_in_this_cluster in is_in_cluster.T:
        # randomly select a "density" for the cluster
        density = density_range * torch.rand(1) + min_density
        # all nodes in the cluster get population randomly sampled near the
        # density
        rel_node_pops = (1 - pop_var) + pop_var * torch.rand(n_nodes)
        type_node_pops = rel_node_pops * density * is_in_this_cluster
        node_pops_by_type.append(type_node_pops)

    # n_nodes x n_pop_centers
    node_pops_by_type = torch.stack(node_pops_by_type, dim=1)

    # same procedure for sinks
    n_sinks = (torch.randint(max_centers, (1,)) + 1).item()
    is_in_sink, sink_centers, sink_shapes = make_oval_groups(
        node_locs, n_sinks, MIN_SINK_RADIUS_M, MAX_SINK_RADIUS_M
    )

    # combine the two as before
    n_pop_clusters = is_in_cluster.shape[1]
    n_sinks = is_in_sink.shape[1]
    attractiveness = torch.rand((n_pop_clusters, n_sinks))

    demand_to_sinks = node_pops_by_type.mm(attractiveness)
    # spread the demand around the nodes in the sink
    node_sink_fracs = is_in_sink / is_in_sink.sum(dim=0)
    demand = demand_to_sinks.mm(node_sink_fracs.T)

    # symmetrize, normalize and return
    demand.fill_diagonal_(0)

    return demand


def build_in4nn_graph(n_nodes, edge_keep_prob, directed):
    return build_knn_graph(n_nodes, 4, edge_keep_prob, "target_to_source", directed)


def build_out4nn_graph(n_nodes, edge_keep_prob, directed):
    return build_knn_graph(n_nodes, 4, edge_keep_prob, "source_to_target", directed)


def _torch_knn_graph(pos, knn, flow="target_to_source", directed=True):
    """Build a k-NN edge index without torch-cluster."""
    n_nodes = pos.shape[0]
    k = min(knn, max(n_nodes - 1, 0))
    if k == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
    else:
        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(float("inf"))
        nearest = dists.topk(k, largest=False, dim=1).indices
        center_nodes = torch.arange(n_nodes, device=pos.device).repeat_interleave(k)
        neighbor_nodes = nearest.reshape(-1)

        if flow == "source_to_target":
            edge_index = torch.stack((neighbor_nodes, center_nodes), dim=0)
        elif flow == "target_to_source":
            edge_index = torch.stack((center_nodes, neighbor_nodes), dim=0)
        else:
            raise ValueError("flow must be 'source_to_target' or 'target_to_source'")

    if not directed:
        edge_index = pygu.to_undirected(edge_index, num_nodes=n_nodes)

    return edge_index


def _apply_knn_graph(street_graph, knn_graph, knn, flow, directed):
    try:
        return knn_graph(street_graph)
    except ImportError as exc:
        if "torch-cluster" not in str(exc):
            raise

    street_graph.edge_index = _torch_knn_graph(street_graph.pos, knn, flow=flow, directed=directed)
    street_graph.edge_attr = None
    return street_graph


def build_knn_graph(n_nodes, knn, edge_keep_prob=1, flow="target_to_source", directed=True):
    knn_graph = KNNGraph(k=knn, flow=flow, force_undirected=not directed)
    rmv_isolated_nodes = RemoveIsolatedNodes()

    while True:
        # generate the node locations
        locs = torch.rand((n_nodes, 2)) * 2 - 1
        # determine the edges
        street_graph = Data(pos=locs)
        pre_rmv = _apply_knn_graph(street_graph, knn_graph, knn, flow, directed)
        street_graph = rmv_isolated_nodes(pre_rmv)

        # drop random edges
        drop_edges(street_graph, edge_keep_prob, directed)

        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(street_graph, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return street_graph


def build_circulant_graph(n_nodes, offsets=[1, 2], edge_keep_prob=1, directed=True):
    graph = nx.circulant_graph(n_nodes, offsets)
    graph = pygu.from_networkx(graph)

    # spread nodes evenly around a circle of radius 1
    graph.pos = get_locs_around_unit_circle(n_nodes)

    while True:
        # drop random edges
        graph_copy = graph.clone()
        drop_edges(graph_copy, edge_keep_prob, directed)

        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(graph_copy, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return graph_copy


def get_locs_around_unit_circle(n_nodes):
    # spread nodes evenly around a circle of radius 1
    angle_step = 2 * torch.pi / n_nodes
    angles = torch.arange(n_nodes) * angle_step
    xlocs = torch.cos(angles)
    ylocs = torch.sin(angles)
    locs = torch.stack((xlocs, ylocs), dim=1)
    return locs


def build_voronoi_graph(n_nodes, *args, **kwargs):
    n_points = int(n_nodes)
    # initial step size for adjusting the number of nodes via binary search.
    # this value seems to work well.
    step = max(n_points // 4, 1)
    while True:
        points = torch.rand((n_points, 2)) * 2 - 1
        vor = Voronoi(points)
        vertices = torch.tensor(vor.vertices, dtype=torch.float32)
        # remove vertices that are outside the (-1, 1) square
        in_square_idxs = torch.where((vertices.abs() < 1).all(dim=1))[0]
        if len(in_square_idxs) != n_nodes:
            # these points don't induce the right number of vertices, so
            # generate new ones with more or less points as appropriate
            if len(in_square_idxs) < n_nodes:
                n_points += step
            else:
                n_points -= step
            step = max(step // 2, 1)
            continue

        # prune excess vertices
        kept_idxs = in_square_idxs  # [:n_nodes]
        node_locs = vertices[kept_idxs]
        # remove edges to pruned vertices and those which extend to infinity
        edges = [ee for ee in vor.ridge_vertices if ee[0] in kept_idxs and ee[1] in kept_idxs]
        edges = torch.tensor(edges, dtype=int).t()
        # map edge's vertex indices to the indices of the kept vertices
        for ii in range(n_nodes):
            old_idx = kept_idxs[ii]
            edges[edges == old_idx] = ii

        # Voronoi graphs are undirected, so make all edges in both directions
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        street_graph = Data(pos=node_locs, edge_index=edges)
        nx_graph = pygu.to_networkx(street_graph, to_undirected=True)
        if is_strongly_connected(nx_graph):
            # we've got a valid one, so we're done!
            break

    return street_graph


def build_mst_graph(n_nodes, directed, *args, **kwargs):
    assert not directed, "MSTs are undirected graphs!"
    locs = torch.rand((n_nodes, 2)) * 2 - 1
    dists = torch.pdist(locs)
    edges = torch.combinations(torch.arange(n_nodes), r=2).t()

    street_graph = Data(pos=locs, edge_index=edges, edge_attr=dists)
    nx_graph = pygu.to_networkx(street_graph, edge_attrs=["edge_attr"], to_undirected=True)

    # run minimum spanning tree on the fully-connected graph
    mst = nx.minimum_spanning_tree(nx_graph, weight="edge_attr", algorithm="kruskal")
    # add minimum edges until desired number of edges is reached
    n_edges = int(3.616 * n_nodes - 34.204)
    # sort edges by distance
    all_edges = sorted(nx_graph.edges(data=True), key=lambda x: x[2]["edge_attr"])
    while len(mst.edges) < n_edges:
        # add the next shortest edge
        ii, jj, attrs = all_edges.pop(0)
        mst.add_edge(ii, jj, **attrs)

    data = pygu.from_networkx(mst)
    data.pos = locs
    return data


def build_4grid(n_nodes, edge_keep_prob, directed):
    return build_grid(n_nodes, "4-connected", edge_keep_prob, directed)


def build_8grid(n_nodes, edge_keep_prob, directed):
    return build_grid(n_nodes, "8-connected", edge_keep_prob, directed)


def build_grid(n_nodes, grid_type="4-connected", edge_keep_prob=1, directed=True):
    """Careful, this is O(n_nodes), though usually much faster than that."""
    assert n_nodes > 3, "n_nodes must be at least 4 in a grid!"

    # fewest # of rows and columns should be more than 2
    min_factor = 3
    factors = [ii for ii in range(min_factor, n_nodes // min_factor + 1) if n_nodes % ii == 0]
    assert len(factors) > 0, "n_nodes must not be prime!"

    x_nodes = factors[random.randrange(len(factors))]
    y_nodes = n_nodes // x_nodes

    side_n_nodes = torch.tensor((x_nodes, y_nodes))
    ranges = side_n_nodes / n_nodes**0.5 + (torch.rand(2) - 0.5) * 0.2
    x_range, y_range = ranges
    # determine the node locations
    x_locs = torch.linspace(-x_range, x_range, x_nodes)
    y_locs = torch.linspace(-y_range, y_range, y_nodes)
    locs = torch.stack(torch.meshgrid(x_locs, y_locs, indexing="ij"), dim=2)
    locs = locs.view(-1, 2)

    # set the edges
    edge_index = grid_index(x_nodes, y_nodes, grid_type)
    graph = Data(pos=locs, edge_index=edge_index)

    while True:
        # drop random edges
        graph_copy = graph.clone()
        drop_edges(graph_copy, edge_keep_prob, directed)

        # check whether the graph is connected, fix it if not
        nx_graph = pygu.to_networkx(graph_copy, to_undirected=not directed)
        if is_strongly_connected(nx_graph):
            return graph_copy


def grid_index(
    height: int,
    width: int,
    grid_type="4-connected",
    device=None,
):
    """Modified from the function of the same name in
    torch_geometric.utils.grid"""
    ww = width
    if grid_type == "8-connected":
        kernel = [-ww - 1, -1, ww - 1, -ww, ww, -ww + 1, 1, ww + 1]
        n_to_cut = 3
    elif grid_type == "4-connected":
        kernel = [-1, -ww, ww, 1]
        n_to_cut = 1
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(n_to_cut, row.size(1) - n_to_cut, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = pygu.coalesce(edge_index, None, height * width)
    return edge_index


def build_smallworld_graph(n_levels, n_nodes_per_level=4, p_attach=0.5):
    """Implements the stochastic hierarchical model described in Ravasz and
    Laslzo 2008."""
    # TODO implement this!
    base_edges = torch.combinations(torch.arange(n_nodes_per_level)).t()
    center_loc = torch.zeros((1, 2))
    periph_locs = get_locs_around_unit_circle(n_nodes_per_level - 1)
    base_locs = torch.cat((center_loc, periph_locs), dim=0)
    base_graph = Data(pos=base_locs, edge_index=base_edges)
    base_graph.edge_index = pygu.to_undirected(base_graph.edge_index)

    base_offsets = get_locs_around_unit_circle(n_nodes_per_level - 1)
    for ii in range(n_levels - 1):
        # make copies of the base graph
        periph_offsets = base_offsets * (ii + 1.25) * 2
        n_new_cluster_nodes = base_graph.num_nodes
        base_degrees = pygu.degree(base_graph.edge_index[0])
        attach_probs = base_degrees / base_degrees.sum()
        new_graph = base_graph.clone()
        i_p_attach = p_attach ** (ii + 1)
        n_to_attach = int(i_p_attach * n_new_cluster_nodes)

        for offset in periph_offsets:
            dup_edges = base_graph.edge_index + new_graph.num_nodes
            dup_locs = base_graph.pos + offset

            # select p_attach fraction new nodes to attach
            new_targets = torch.randperm(n_new_cluster_nodes)[:n_to_attach]
            new_targets += new_graph.num_nodes
            new_sources = attach_probs.multinomial(n_to_attach, replacement=True)
            joining_edges = torch.stack((new_sources, new_targets), dim=0)
            # add new nodes and edges to the graph
            new_graph.edge_index = torch.cat(
                (new_graph.edge_index, dup_edges, joining_edges), dim=1
            )
            new_graph.pos = torch.cat((new_graph.pos, dup_locs), dim=0)

        base_graph = new_graph

        # make graph undirected
        base_graph.edge_index = pygu.to_undirected(base_graph.edge_index)

    # scale node positions to be between -1 and 1.  We can do it this way
    # because the nodes are centered at 0 by construction.
    maxes = base_graph.pos.max(dim=0)[0]
    base_graph.pos = base_graph.pos / maxes

    return base_graph


def is_strongly_connected(nx_graph):
    if isinstance(nx_graph, nx.DiGraph):
        return nx.is_strongly_connected(nx_graph)
    else:
        return nx.is_connected(nx_graph)


def drop_edges(graph_data, edge_keep_prob, directed=False):
    """Randomly drop edges from the graph with probability p."""
    nx_graph = pygu.to_networkx(graph_data, to_undirected=not directed)
    removal_mask = torch.rand(nx_graph.number_of_edges()) > edge_keep_prob
    to_remove = [edge for edge, remove in zip(nx_graph.edges(), removal_mask) if remove]
    nx_graph.remove_edges_from(to_remove)
    updated_graph_data = pygu.from_networkx(nx_graph)
    graph_data.edge_index = updated_graph_data.edge_index
    return graph_data


def drop_nodes(graph_data, n_nodes_to_drop):
    """Randomly drop nodes from the graph."""
    assert n_nodes_to_drop < graph_data.num_nodes, "Can't drop all the nodes!"
    # randomly select nodes to drop
    nx_graph = pygu.to_networkx(graph_data)
    drop_idxs = torch.randperm(graph_data.num_nodes)[:n_nodes_to_drop]

    # remove the nodes from the graph
    for drop_idx in drop_idxs:
        nx_graph.remove_node(drop_idx.item())

    graph_data = pygu.from_networkx(nx_graph)

    return graph_data

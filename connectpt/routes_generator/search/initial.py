"""Initial networks: learned construction, John and Nikolic."""
import logging as log
import torch
import networkx as nx
from tqdm import tqdm
from ..network.routes import to_routes
from ..env.state import RouteGenBatchState
from ..city.shortest_paths import reconstruct_all_paths
from .operators import get_direct_sat_dmd


def john_init(state: RouteGenBatchState, alpha=None, prioritize_direct_connections: bool = True, show_pbar=False):
    """Constructs a network based on the algorithm of John et al. (2014).

    Args:
        state: A representation of the city and the problem constraints.
        alpha: Controls trade-off between transit time and demand. alpha=1
            means only demand matters, alpha=0 means only transit time matters.
    """
    assert state.symmetric_routes, "John et al. (2014) requires symmetric routes"
    assert state.batch_size == 1, "John et al. (2014) does not support batched states"

    # compute weights for all of the edges
    have_edges = (state.street_adj > 0) & state.street_adj.isfinite()
    norm_times = torch.full_like(state.street_adj, float("inf"))
    edge_times = state.street_adj[have_edges]
    norm_times[have_edges] = edge_times / edge_times.max()
    norm_demand = state.demand / state.demand.max()
    if alpha is None:
        # use John's spread of alpha and beta values
        values = torch.linspace(0, 1, 21)
        meshes = torch.meshgrid(values, values, indexing="ij")
        alpha, beta = (mm.flatten()[:, None, None] for mm in meshes)
    else:
        # just use the one passed-in value
        beta = 1 - alpha
    # build the range of edge weight sets based on alpha and beta
    all_edge_costs = beta * norm_times + alpha * (1 - norm_demand)
    all_edge_costs[~have_edges] = float("inf")

    # build a spanning sub-graph of routes
    have_edges = have_edges.squeeze(0)
    networks = []
    for edge_costs in tqdm(all_edge_costs, disable=not show_pbar):
        network = []
        n_nodes = state.n_nodes[0].item()
        are_visited = torch.zeros(n_nodes, dtype=torch.bool, device=state.device)
        are_directly_connected = torch.eye(n_nodes, dtype=torch.bool, device=state.device)
        pair_attempted = torch.eye(n_nodes, dtype=torch.bool, device=state.device)
        # While all nodes are not yet visited:
        while not are_visited.all():
            # select a "seed pair" of vertices to be the start of the route
            scores = edge_costs.clone()
            if len(network) > 0:
                # choose an edge between a covered and an uncovered node
                connects_new = are_visited[:, None] & ~are_visited
                # don't pick pairs that don't connect a new node
                scores[~connects_new] = float("inf")

            # don't choose any pairs that have already been attempted
            scores[pair_attempted] = float("inf")

            flat_seed_pair = torch.argmin(scores).item()
            seed_pair = (flat_seed_pair // n_nodes, flat_seed_pair % n_nodes)
            route = list(seed_pair)
            are_on_route = torch.zeros(n_nodes, dtype=torch.bool, device=state.device)
            are_on_route[[seed_pair]] = True

            # expand the seed pair to form a route by adding adjacent vertices
            while len(route) < state.max_route_len:
                # find valid extension nodes
                are_valid_start_exts = have_edges[route[0]] & ~are_on_route
                are_valid_end_exts = have_edges[route[-1]] & ~are_on_route

                # check if any of them are not yet visited at all
                are_valid_exts = are_valid_start_exts | are_valid_end_exts
                if not are_valid_exts.any():
                    # no valid extensions, so we're done
                    break

                if (are_valid_exts & ~are_visited).any():
                    # we can visit an unvisited node, so we must do so
                    are_valid_start_exts &= ~are_visited
                    are_valid_end_exts &= ~are_visited

                # choose the next node to add by least edge cost
                valid_start_exts = torch.nonzero(are_valid_start_exts).squeeze(-1)
                valid_end_exts = torch.nonzero(are_valid_end_exts).squeeze(-1)
                start_edge_costs = edge_costs[route[0], valid_start_exts]
                end_edge_costs = edge_costs[route[-1], valid_end_exts]
                ext_edge_costs = torch.cat((start_edge_costs, end_edge_costs))
                min_edge_cost = ext_edge_costs.min()
                options = ext_edge_costs == min_edge_cost
                # break ties randomly
                chosen_ext_loc = options.float().multinomial(1).item()
                if chosen_ext_loc < len(start_edge_costs):
                    chosen_node = valid_start_exts[chosen_ext_loc].item()
                    route.insert(0, chosen_node)
                else:
                    end_loc = chosen_ext_loc - len(start_edge_costs)
                    chosen_node = valid_end_exts[end_loc].item()
                    route.append(chosen_node)

                # mark the node as on the route and visited
                are_on_route[chosen_node] = True

            # route is finished
            if len(route) >= state.min_route_len:
                # route is long enough to add
                route = torch.tensor(route)
                # mark the route's nodes as visited and directly connected
                are_directly_connected[route, route] = True
                are_visited |= are_on_route
                network.append(route)

            pair_attempted[seed_pair] = True

            if len(network) == state.n_routes_to_plan:
                # we have enough routes, so we're done
                break

        if len(network) < state.n_routes_to_plan:
            # Use approach of Shih and Mahmassani to add more routes
            # sort node pairs by descending demand
            if prioritize_direct_connections:
                # only consider node pairs that are not directly connected
                nodepairs = torch.nonzero(~are_directly_connected)
                pair_demands = state.demand[0, ~are_directly_connected]
            else:
                pair_demands = state.demand[0]
                nodepairs = torch.combinations(torch.arange(n_nodes), r=2)

            sorted_indices = pair_demands.flatten().argsort(descending=True)
            sorted_pairs = [(ss.item(), dd.item()) for ss, dd in nodepairs[sorted_indices]]

            # build a networkx graph (from_numpy_matrix was renamed to
            # from_numpy_array in networkx >= 3.0)
            edge_costs[state.street_adj[0].isinf()] = 0
            graph = nx.from_numpy_array(edge_costs.numpy())

            # add the existing routes to the graph to get the transit times
            state.add_new_routes([network])

            # For each pair in order, until $|\mathcal{R}| = S$:
            for (src, dst) in sorted_pairs:

                # Use Yen's k-shortest path algorithm (see ref. 22) with k=10
                # to see if a valid route between the nodes exists that
                # doesn't violate any constraints
                from ..city.shortest_paths import yen_k_shortest_paths

                paths = yen_k_shortest_paths(graph, src, dst, 10)

                for path in paths:
                    # check if the path is valid
                    if state.min_route_len <= len(path) <= state.max_route_len and path not in network:
                        # it is valid and shorter than the shortest
                        # transit path between the nodes, add it to
                        # |\mathcal{R}|
                        path_len = nx.path_weight(graph, path, weight="weight")
                        if path_len < state.transit_times[0, src, dst]:
                            # add the path and move on to the next node pair
                            path = torch.tensor(path)
                            network.append(path)
                            are_visited[path] = True

                            state.add_new_routes([[path]])
                            break

                if len(network) == state.n_routes_to_plan:
                    # we have enough routes, so we're done
                    break

        if len(network) < state.n_routes_to_plan:
            # raise RuntimeError('Failed to find enough routes')
            log.warning("John init failed to find enough routes")
        if not are_visited.all():
            # raise RuntimeError('Failed to cover all nodes')
            log.warning("John init failed to cover all nodes")
        else:
            networks.append(network)

        # make sure changes to state don't affect the original state
        state.clear_routes()

    if len(networks) == 0:
        raise RuntimeError("John init failed to find any valid networks")

    networks = to_routes(networks, width=state.max_route_len[0]).to(state.device)
    return networks


def nikolic_init(state: RouteGenBatchState):
    """Constructs a network based on the algorithm of Nikolic and Teodorovic
    (2013).
    """
    shortest_paths, _ = reconstruct_all_paths(state.nexts)

    dev = shortest_paths.device
    batch_size = shortest_paths.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes
    dm_uncovered = state.demand.clone()
    # add dummy column and row
    dm_uncovered = torch.nn.functional.pad(dm_uncovered, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1, device=dev)
    log.info("computing initial network")
    # stop-to-itself routes are all invalid
    terms_are_invalid = torch.eye(max_n_nodes + 1, device=dev, dtype=bool)
    # routes to and from the dummy stop are invalid
    terms_are_invalid[max_n_nodes, :] = True
    terms_are_invalid[:, max_n_nodes] = True
    terms_are_invalid = terms_are_invalid[None].repeat(batch_size, 1, 1)
    for ri in range(n_routes):
        # compute directly-satisfied-demand matrix
        direct_sat_dmd = get_direct_sat_dmd(dm_uncovered, shortest_paths, state.symmetric_routes)
        # set invalid term pairs to -1 so they won't be selected even if there
        # is no uncovered demand.
        direct_sat_dmd[terms_are_invalid] = -1

        # choose maximum DS route and add it to the initial network
        flat_dsd = direct_sat_dmd.flatten(1, 2)
        _, best_flat_idxs = flat_dsd.max(dim=1)
        # add 1 to account for the dummy column and row
        best_i = torch.div(best_flat_idxs, (max_n_nodes + 1), rounding_mode="floor")
        best_j = best_flat_idxs % (max_n_nodes + 1)
        # batch_size x route_len
        routes = shortest_paths[batch_idxs, best_i, best_j]
        best_networks[:, ri, : routes.shape[-1]] = routes

        # mark new routes as in use
        terms_are_invalid[batch_idxs, best_i, best_j] = True
        if state.symmetric_routes:
            terms_are_invalid[batch_idxs, best_j, best_i] = True

        # remove newly-covered demand from uncovered demand matrix
        for ii in range(routes.shape[-1] - 1):
            cur_stops = routes[:, ii][:, None]
            later_stops = routes[:, ii + 1 :]
            dm_uncovered[batch_idxs[:, None], cur_stops, later_stops] = 0
            if state.symmetric_routes:
                maxidx = routes.shape[-1] - 1
                cur_stops = routes[:, maxidx - ii]
                later_stops = routes[:, maxidx - ii - 1]
                dm_uncovered[batch_idxs[:, None], later_stops, cur_stops] = 0

    return best_networks


def build_init_routes(
    spec, tensors, *, generator: str, cost, initial, corruption, policy=None, binding=None,
    seed: int, clean: bool, device
) -> torch.Tensor:
    """Build an initial network with ``INITIALIZERS[generator]``. The ``lc`` network
    also gets the realistic corruption tier unless ``clean``.

    The objects come from ``scripts/make_init_routes.py`` (``cfg/runs/init_routes.yaml``):
    ``cost``, ``initial`` (``InitialParams``), ``corruption`` (``RealisticParams``) and,
    for ``lc``, the construction ``policy``."""
    import random

    from ..network.corruption import corrupt, realistic_tier, CorruptionCity, CorruptionLimits

    parameters = dict(device=device, cost=cost, initial=initial)
    if generator == 'lc':
        if policy is None or binding is None:
            raise ValueError('learned construction requires a policy and environment binding')
        parameters.update(policy=policy, binding=binding)
    routes = INITIALIZERS[generator](spec, tensors, **parameters)
    routes = to_routes(routes, n_routes=spec["n_routes"], width=spec["max_route_len"])[0]
    if generator == "lc" and not clean:
        rng = random.Random(seed)
        routes = corrupt(
            routes,
            rng,
            CorruptionCity(tensors["street_adj"], tensors["demand"], tensors["node_locs"].shape[0]),
            CorruptionLimits(spec["min_route_len"], spec["max_route_len"]),
            realistic_tier(corruption),
        ).routes
    routes = to_routes(routes)
    return routes[None] if routes.ndim == 2 else routes


def learned_initial(spec, tensors, *, device, cost, initial, policy, binding) -> torch.Tensor:
    from torch_geometric.loader import DataLoader
    from ..city.graph import to_batch
    from ..policy.predict import predict

    data = next(iter(DataLoader(to_batch(tensors).to_data_list(), batch_size=1))).to(device)
    return predict(policy, data, cost.get_weights(), cost, binding.factory, binding.params, "sample").routes


def _heuristic_construct(spec, tensors, *, device, generator, cost, initial):
    from ..city.graph import to_batch

    # The preserved heuristic implementations operate on CPU matrices.
    state = RouteGenBatchState(
        to_batch(tensors, torch.device("cpu")),
        cost.constants,
        spec["n_routes"],
        spec["min_route_len"],
        spec["max_route_len"],
        preferences=cost.get_weights(),
    )
    if generator == "john":
        routes = john_init(state, initial.alpha, initial.prioritize_direct_connections)
    else:
        routes = nikolic_init(state)
    return routes.to(device)


def _john_construct(spec, tensors, *, device, cost, initial):
    return _heuristic_construct(spec, tensors, device=device, generator="john", cost=cost, initial=initial)


def _nikolic_construct(spec, tensors, *, device, cost, initial):
    return _heuristic_construct(spec, tensors, device=device, generator="nikolic", cost=cost, initial=initial)


INITIALIZERS = {"lc": learned_initial, "john": _john_construct, "nikolic": _nikolic_construct}

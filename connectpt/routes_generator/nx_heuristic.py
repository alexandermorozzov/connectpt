"""NetworkX-based heuristic route-set constructor.

Ported into connectpt from the `nx_heuristic_dataset_generation.ipynb` example
notebook. Given a city graph it builds a fixed-size set of transit routes, each
of exactly ``max_len`` stops, using a degree-biased shortest-path heuristic:

  * candidate routes are the (extended-to-max-length) shortest paths between
    node pairs;
  * route endpoints are sampled with probability inversely proportional to node
    degree (favouring peripheral nodes);
  * isolated (uncovered) nodes are spliced into existing routes where the graph
    topology allows it.

The public entry point is :func:`build_nx_heuristic_routes`, which returns a
``[1, num_routes, max_len]`` int tensor ready to feed any optimizer via
``test_method``'s ``routes_tensor`` argument.

The route generator itself is objective-agnostic: it does not look at demand or
travel-time weights.
"""
import random

import numpy as np
import networkx as nx

from .citygraph_dataset import STOP_KEY
from .torch_utils import get_batch_tensor_from_routes


def build_street_graph(city_graph):
    """Convert a city graph into an undirected, edge-weighted ``networkx.Graph``.

    ``city_graph`` is a single (non-batched) CityGraphData object: it must
    expose ``.street_adj`` (an ``[n_nodes, n_nodes]`` matrix with non-finite
    entries for missing edges) and ``city_graph[STOP_KEY].pos`` (node coords).
    """
    street_adj = city_graph.street_adj.detach().cpu()
    coords = city_graph[STOP_KEY].pos.detach().cpu().numpy()
    graph_nx = nx.Graph()
    n_nodes = street_adj.shape[0]

    for node_idx in range(n_nodes):
        graph_nx.add_node(
            int(node_idx),
            pos=(float(coords[node_idx, 0]), float(coords[node_idx, 1])),
        )

    for ii in range(n_nodes):
        for jj in range(ii + 1, n_nodes):
            candidates = []
            val_ij = float(street_adj[ii, jj])
            val_ji = float(street_adj[jj, ii])
            if np.isfinite(val_ij):
                candidates.append(val_ij)
            if np.isfinite(val_ji):
                candidates.append(val_ji)
            if candidates:
                graph_nx.add_edge(ii, jj, weight=min(candidates))

    return graph_nx


def node_degree_probabilities(graph_nx):
    inv_deg = {}
    for node, degree in graph_nx.degree():
        inv_deg[node] = 1.0 / degree if degree > 0 else 0.0
    total = sum(inv_deg.values())
    if total <= 0:
        uniform = 1.0 / max(len(inv_deg), 1)
        return {node: uniform for node in inv_deg}
    return {node: value / total for node, value in inv_deg.items()}


def roulette_choice(probabilities, rng):
    nodes = list(probabilities.keys())
    weights = list(probabilities.values())
    return int(rng.choices(nodes, weights=weights, k=1)[0])


def canonical_route(route):
    return tuple(int(node) for node in route)


def route_is_new(route, used_routes):
    route_key = canonical_route(route)
    return route_key not in used_routes and \
        tuple(reversed(route_key)) not in used_routes


def extend_route_to_max_len(graph_nx, route, max_len, rng, max_attempts=40):
    route = [int(node) for node in route]
    if len(route) > max_len:
        return None
    if len(route) == max_len:
        return route

    for _ in range(max_attempts):
        candidate = list(route)
        while len(candidate) < max_len:
            directions = ["start", "end"]
            rng.shuffle(directions)
            extended = False
            for direction in directions:
                endpoint = candidate[0] if direction == "start" else candidate[-1]
                neighbours = [int(node) for node in graph_nx.neighbors(endpoint)
                              if int(node) not in candidate]
                rng.shuffle(neighbours)
                if not neighbours:
                    continue
                if direction == "start":
                    candidate.insert(0, neighbours[0])
                else:
                    candidate.append(neighbours[0])
                extended = True
                break
            if not extended:
                break
        if len(candidate) == max_len:
            return candidate
    return None


def random_max_length_path(graph_nx, max_len, rng, used_routes=None,
                            avoid_used=True, max_attempts=250):
    nodes = [int(node) for node in graph_nx.nodes]
    if len(nodes) < max_len:
        return None
    used_routes = used_routes or set()

    for _ in range(max_attempts):
        start_node = int(rng.choice(nodes))
        route = extend_route_to_max_len(graph_nx, [start_node], max_len, rng,
                                        max_attempts=1)
        if route is None:
            continue
        if avoid_used and not route_is_new(route, used_routes):
            continue
        return route
    return None


def precompute_candidate_paths(graph_nx, min_len, max_len, rng):
    candidate_paths = {}
    for source, path_dict in nx.all_pairs_shortest_path(graph_nx):
        for target, path in path_dict.items():
            if source == target or len(path) > max_len:
                continue
            if len(path) < min_len:
                continue
            max_route = extend_route_to_max_len(graph_nx, path, max_len, rng)
            if max_route is not None and len(max_route) == max_len:
                candidate_paths[(int(source), int(target))] = \
                    canonical_route(max_route)
    return candidate_paths


def choose_fallback_path(candidate_paths, used_routes, used_nodes, pool_size,
                         rng, graph_nx=None, max_len=None, avoid_used=True):
    remaining = []
    for path in candidate_paths.values():
        if avoid_used and not route_is_new(path, used_routes):
            continue
        new_nodes = sum(node not in used_nodes for node in path)
        remaining.append((new_nodes, len(path), path))

    if not remaining and graph_nx is not None and max_len is not None:
        return random_max_length_path(graph_nx, max_len, rng, used_routes,
                                      avoid_used=avoid_used)
    if not remaining:
        return None

    remaining.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_paths = [item[2] for item in remaining[:pool_size]]
    return list(rng.choice(top_paths))


def insert_isolated_nodes(graph_nx, routes, max_len, rng):
    if not routes:
        return routes

    used_nodes = {node for route in routes for node in route}
    isolated_nodes = [node for node in graph_nx.nodes if node not in used_nodes]
    rng.shuffle(isolated_nodes)

    for isolated_node in isolated_nodes:
        route_order = list(range(len(routes)))
        rng.shuffle(route_order)
        inserted = False
        for route_idx in route_order:
            route = routes[route_idx]
            if len(route) >= max_len:
                continue
            for node_pos in range(len(route) - 1):
                left_node = route[node_pos]
                right_node = route[node_pos + 1]
                if graph_nx.has_edge(left_node, isolated_node) and \
                        graph_nx.has_edge(isolated_node, right_node):
                    new_route = (route[:node_pos + 1] + [isolated_node]
                                 + route[node_pos + 1:])
                    if len(new_route) <= max_len:
                        routes[route_idx] = new_route
                        inserted = True
                        break
            if inserted:
                break

    return routes


def generate_initial_routes(graph_nx, num_routes, min_len, max_len, rng,
                            route_attempts=60, fallback_pool_size=10):
    """Generate ``num_routes`` routes, each of exactly ``max_len`` stops.

    ``rng`` must be a :class:`random.Random` instance (the heuristic uses
    ``rng.choices`` for weighted sampling).
    """
    candidate_paths = precompute_candidate_paths(graph_nx, min_len, max_len, rng)

    probabilities = node_degree_probabilities(graph_nx)
    routes = []
    used_nodes = set()
    used_routes = set()

    for _ in range(num_routes):
        selected_route = None
        for _ in range(route_attempts):
            start_node = roulette_choice(probabilities, rng)
            end_node = roulette_choice(probabilities, rng)
            if start_node == end_node:
                continue
            path = candidate_paths.get((start_node, end_node))
            if path is None:
                continue
            if not route_is_new(path, used_routes):
                continue
            selected_route = list(path)
            break

        if selected_route is None:
            selected_route = choose_fallback_path(
                candidate_paths,
                used_routes,
                used_nodes,
                fallback_pool_size,
                rng,
                graph_nx=graph_nx,
                max_len=max_len,
            )
        if selected_route is None:
            break

        routes.append(selected_route)
        used_nodes.update(selected_route)
        used_routes.add(canonical_route(selected_route))

    routes = insert_isolated_nodes(graph_nx, routes, max_len, rng)

    while len(routes) < num_routes:
        selected_route = choose_fallback_path(
            candidate_paths,
            used_routes,
            {node for route in routes for node in route},
            fallback_pool_size,
            rng,
            graph_nx=graph_nx,
            max_len=max_len,
        )
        if selected_route is None:
            selected_route = choose_fallback_path(
                candidate_paths,
                used_routes,
                {node for route in routes for node in route},
                fallback_pool_size,
                rng,
                graph_nx=graph_nx,
                max_len=max_len,
                avoid_used=False,
            )
        if selected_route is None:
            break
        routes.append(selected_route)
        used_routes.add(canonical_route(selected_route))

    if len(routes) < num_routes:
        raise ValueError(
            f"Generated only {len(routes)} / {num_routes} max-length routes")
    bad_lengths = [len(route) for route in routes if len(route) != max_len]
    if bad_lengths:
        raise ValueError(
            f"All routes must have length {max_len}; found lengths {bad_lengths}")

    return routes


def build_nx_heuristic_routes(city_graph, num_routes, min_len, max_len, seed=0,
                              route_attempts=60, fallback_pool_size=10):
    """Build an NX-heuristic route set for ``city_graph``.

    Returns a ``[1, num_routes, max_len]`` int tensor (``-1`` padded), suitable
    as the ``routes_tensor`` argument of ``test_method`` / the notebook runners.
    """
    graph_nx = build_street_graph(city_graph)
    rng = random.Random(seed)
    routes = generate_initial_routes(
        graph_nx, num_routes=num_routes, min_len=min_len, max_len=max_len,
        rng=rng, route_attempts=route_attempts,
        fallback_pool_size=fallback_pool_size)
    return get_batch_tensor_from_routes(routes, max_route_len=max_len)

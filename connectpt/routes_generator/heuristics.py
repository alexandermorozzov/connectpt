"""Low-level route-network mutation operators used by NSGA-II.

Ported into connectpt from AHolliday/transit_learning (learning/heuristics.py).
Changes vs. the original:
  * import paths remapped to connectpt.routes_generator;
  * the Hydra CLI main() and the debug `test_helper` were dropped.

Every public operator has the signature ``op(state, networks) -> networks``
where ``networks`` is a ``(n_networks, n_routes, max_route_len)`` int tensor of
route node indices (-1 padded) and ``state`` is a :class:`RouteGenBatchState`
with either ``batch_size == 1`` or ``batch_size == networks.shape[0]``.
"""
import torch

from .transit_time_estimator import RouteGenBatchState
from . import torch_utils as tu


EPSILON = 1e-6


def add_terminal(state: RouteGenBatchState, networks: torch.Tensor):
    return _add_helper(state, networks, mode='terminal')


def add_inside(state: RouteGenBatchState, networks: torch.Tensor):
    return _add_helper(state, networks, mode='inside')


def _add_helper(state: RouteGenBatchState, networks: torch.Tensor, mode='any'):
    """
    Add a node inside a random route.

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    mode: 'any', 'inside', or 'terminal'
    """
    # is_option: n_networks x n_routes x max_route_len+1 x n_nodes+1
    is_option = _get_add_node_options(state, networks, mode)

    # remove the dummy node from the options
    is_option = is_option[..., :-1].clone()
    action_idcs = choose_from_multidim_options(is_option)

    # don't modify the input tensor
    networks = networks.clone()

    # insert the new nodes into the networks
    for (network, (ri, new_node_si, new_node)) in zip(networks, action_idcs):
        if ri == -1:
            # no valid action, so skip this network
            continue
        _add_node_inplace(network, ri, new_node_si, new_node)

    return networks


# the below two functions are split up to allow unit tests for the probs
def cost_based_grow(state: RouteGenBatchState, networks: torch.Tensor,
                    route_idcs=None):
    probs = _cost_based_grow_probs(state, networks, route_idcs)
    action_idcs = choose_from_multidim_options(probs)

    # insert the new nodes into the networks
    networks = networks.clone()
    for (network, (ri, new_node_si, new_node)) in zip(networks, action_idcs):
        if ri == -1:
            # no valid action, so skip this network
            continue
        _add_node_inplace(network, ri, new_node_si, new_node)
    return networks


def _cost_based_grow_probs(state: RouteGenBatchState, networks: torch.Tensor,
                           route_idcs=None):
    # weight new terminals according to the formula of Husselmann et al. 2024
    is_option = _get_add_node_options(state, networks, mode='terminal')
    # remove the dummy node from the options
    is_option = is_option[..., :-1]
    # pick routes
    route_has_options = is_option.any(dim=-1).any(dim=-1)
    ntwks_no_trimmable_routes = ~ (route_has_options.any(-1))
    route_has_options[ntwks_no_trimmable_routes] = True

    if route_idcs is None:
        # no routes were specified, so select routes randomly
        route_idcs = route_has_options.float().multinomial(1).squeeze(-1)
    ntwk_idcs = torch.arange(networks.shape[0])
    chosen_routes_options = is_option[ntwk_idcs, route_idcs]

    served_pair_counts = get_served_pair_counts(state, networks)
    pairs_are_served = served_pair_counts > 0

    nodes_are_on_routes = tu.get_nodes_on_routes_mask(state.max_n_nodes, networks)
    # remove dummy node
    nodes_are_on_routes = nodes_are_on_routes[..., :-1]
    demand = state.demand[0]
    probs = torch.zeros_like(is_option, dtype=torch.float32)
    route_lens = (networks > -1).sum(-1)

    chosen_routes = networks[ntwk_idcs, route_idcs]

    # first check the start nodes
    start_options = chosen_routes_options[:, 0]
    start_scores = torch.zeros_like(start_options, dtype=torch.float32)
    nodes_are_on_chosen_routes = nodes_are_on_routes[ntwk_idcs, route_idcs]
    if start_options.any():
        new_edge_times = state.street_adj[0, :, chosen_routes[:, 0]]
        new_dmd_mask = (~pairs_are_served) & nodes_are_on_chosen_routes[:, None]
        options_new_demands = demand * new_dmd_mask
        new_dmds = options_new_demands.sum(-1)
        scores = new_dmds / new_edge_times.transpose(0, 1)
        start_scores[start_options] = scores[start_options]

    # then check the end nodes
    end_idcs = route_lens.gather(1, route_idcs[:, None]).squeeze(-1)
    end_options = chosen_routes_options[ntwk_idcs, end_idcs]
    end_scores = torch.zeros_like(end_options, dtype=torch.float32)
    if end_options.any():
        end_terms = chosen_routes[ntwk_idcs, end_idcs-1]
        # end_terms = chosen_routes.gather(1, end_idcs).squeeze(-1)
        new_edge_times = state.street_adj[0, end_terms]
        new_dmd_mask = nodes_are_on_chosen_routes[..., None] & \
                       (~pairs_are_served)
        options_new_demands = demand * new_dmd_mask
        new_dmds = options_new_demands.sum(-2)
        scores = new_dmds / new_edge_times
        end_scores[end_options] = scores[end_options]

    has_scores = (start_scores.sum(-1) > 0) | (end_scores.sum(-1) > 0)
    probs[has_scores, route_idcs[has_scores], 0] = start_scores[has_scores]
    probs[has_scores, route_idcs[has_scores], end_idcs[has_scores]] = \
        end_scores[has_scores]
    probs[~has_scores, route_idcs[~has_scores]] = \
        chosen_routes_options[~has_scores].float()

    return probs


def delete_terminal(state: RouteGenBatchState, networks: torch.Tensor):
    return _delete_helper(state, networks, mode='terminal')


def delete_inside(state: RouteGenBatchState, networks: torch.Tensor):
    return _delete_helper(state, networks, mode='inside')


def _delete_helper(state: RouteGenBatchState, networks: torch.Tensor,
                   mode='any'):
    """
    Delete a node inside a random route.

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    mode: 'any', 'inside', or 'terminal'
    """
    # choose actions
    is_option = _get_delete_node_options(state, networks, mode)

    action_idcs = choose_from_multidim_options(is_option)

    # don't modify the input tensor
    networks = networks.clone()
    # delete the nodes from the networks
    for (network, (ri, del_node_si)) in zip(networks, action_idcs):
        if ri == -1:
            # no valid action, so skip this network
            continue
        _delete_node_inplace(network, ri, del_node_si)
    return networks


# the below two functions are split up to allow unit tests for the probs
def cost_based_trim(state: RouteGenBatchState, networks: torch.Tensor,
                    route_idcs=None):
    probs = _cost_based_trim_probs(state, networks, route_idcs)
    action_idcs = choose_from_multidim_options(probs)

    # delete the nodes from the networks
    networks = networks.clone()
    for (network, (ri, del_node_si)) in zip(networks, action_idcs):
        if ri == -1:
            # no valid action, so skip this network
            continue
        _delete_node_inplace(network, ri, del_node_si)
    return networks


def _cost_based_trim_probs(state: RouteGenBatchState, networks: torch.Tensor,
                           route_idcs=None):
    # get the nodes that can be deleted
    is_option = _get_delete_node_options(state, networks, 'terminal')
    # pick routes
    route_has_options = is_option.any(dim=-1)
    ntwks_no_trimmable_routes = ~ (route_has_options.any(dim=-1))
    # set all options to true if there are no trimmable routes to let \
     # multinomial() work
    route_has_options[ntwks_no_trimmable_routes] = True

    if route_idcs is None:
        route_idcs = route_has_options.float().multinomial(1).squeeze(-1)
    served_pair_counts = get_served_pair_counts(state, networks)
    nodes_are_on_routes = tu.get_nodes_on_routes_mask(state.max_n_nodes, networks)
    # trim the dummy node row
    nodes_are_on_routes = nodes_are_on_routes[..., :-1]
    route_lens = (networks > -1).sum(-1)

    # weight terminals according to the formula of Husselmann et al. 2024
    # remove the first node from each route, where possible
    demand = state.demand[0]
    probs = torch.zeros_like(is_option, dtype=torch.float32)
    n_networks = networks.shape[0]
    ntwk_idcs = torch.arange(n_networks)
    chosen_routes = networks[ntwk_idcs, route_idcs]
    nodes_are_on_chosen_routes = nodes_are_on_routes[ntwk_idcs, route_idcs]
    chosen_routes_options = is_option[ntwk_idcs, route_idcs]

    start_probs = torch.zeros(n_networks)
    have_start_options = chosen_routes_options[:, 0]
    if have_start_options.any():
        start_terms = chosen_routes[:, 0]
        edge_times = state.street_adj[0, start_terms, chosen_routes[:, 1]]
        is_lost_dmd = (served_pair_counts[ntwk_idcs, start_terms] == 1) & \
                       nodes_are_on_chosen_routes
        dmd_delta = (demand[start_terms] * is_lost_dmd).sum(-1)
        pp = edge_times / (dmd_delta + EPSILON)
        start_probs[have_start_options] = pp[have_start_options]

    end_probs = torch.zeros(n_networks)
    end_idcs = route_lens.gather(1, route_idcs[:, None]).squeeze(-1) - 1
    have_end_options = chosen_routes_options[ntwk_idcs, end_idcs]
    if have_end_options.any():
        end_terms = chosen_routes[ntwk_idcs, end_idcs]
        pre_et_nodes = chosen_routes[ntwk_idcs, end_idcs-1]
        edge_times = state.street_adj[0, pre_et_nodes, end_terms]
        is_lost_dmd = (served_pair_counts[ntwk_idcs, :, end_terms] == 1) & \
                       nodes_are_on_chosen_routes
        dmd_delta = (demand[:, end_terms].t() * is_lost_dmd).sum(-1)
        pp = edge_times / (dmd_delta + EPSILON)
        end_probs[have_end_options] = pp[have_end_options]

    has_scores = (start_probs > 0) | (end_probs > 0)
    probs[has_scores, route_idcs[has_scores], 0] = start_probs[has_scores]
    probs[has_scores, route_idcs[has_scores], end_idcs[has_scores]] = \
        end_probs[has_scores]
    probs[~has_scores, route_idcs[~has_scores]] = \
        chosen_routes_options[~has_scores].float()

    return probs


def get_directly_satisfied_demand(state: RouteGenBatchState,
                                  networks: torch.Tensor,
                                  edge_connection_counts=None):
    if edge_connection_counts is None:
        edge_connection_counts = get_served_pair_counts(state, networks)
    edges_are_satisfied = edge_connection_counts > 0
    satisfied_demands = torch.zeros(networks.shape[0])
    for ni in range(networks.shape[0]):
        satisfied_demands[ni] = state.demand[0, edges_are_satisfied[ni]].sum()
    return satisfied_demands


def get_served_pair_counts(state: RouteGenBatchState, networks: torch.Tensor,
                           remove_padding=True):
    edge_connection_counts = torch.zeros((networks.shape[0],
                                          state.max_n_nodes + 1,
                                          state.max_n_nodes + 1), dtype=int)
    ntwk_idcs = torch.arange(networks.shape[0])
    for ri in range(networks.shape[1]):
        route_batch = networks[:, ri]
        for si in range(route_batch.shape[1] - 1):
            start_node = route_batch[:, si]
            following_nodes = route_batch[:, si + 1:]
            # start_idx = min(start_node, end_node)
            # end_idx = max(start_node, end_node)
            edge_connection_counts[ntwk_idcs[:, None], start_node[:, None],
                                   following_nodes] += 1

    if state.symmetric_routes:
        flipped_ecc_copy = edge_connection_counts.permute(0, 2, 1).clone()
        edge_connection_counts += flipped_ecc_copy
    # remove dummy node row and column and return
    if remove_padding:
        edge_connection_counts = edge_connection_counts[:, :-1, :-1]

    return edge_connection_counts


def invert_nodes(state: RouteGenBatchState, networks: torch.Tensor):
    """
    Choose two points on a route and invert the order of the nodes in between.

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    """
    adj_mat = get_formatted_adj_mat(state, networks)

    # serial implementation
    n_networks = networks.shape[0]
    n_routes = networks.shape[1]
    max_route_len = state.max_route_len.max().item()
    is_option = torch.zeros((n_networks, n_routes, max_route_len-1,
                             max_route_len-1), dtype=bool)
    route_lens = (networks != -1).sum(dim=-1)

    for ni, network in enumerate(networks):
        for ri, route in enumerate(network):
            route_len = route_lens[ni, ri]
            for si, inode in enumerate(route[:-1]):
                if inode == -1:
                    break
                before_node = route[si-1] if si > 0 else -1
                for sj, jnode in enumerate(route[si+1:], start=si+1):
                    if jnode == -1:
                        break
                    if sj == route_len - 1 and si == 0:
                        # don't allow flipping whole route, that does nothing
                        continue
                    after_node = route[sj+1] if sj+1 < route_len else -1
                    if (before_node == -1 or adj_mat[ni, before_node, jnode]) and \
                        (after_node == -1 or adj_mat[ni, inode, after_node]):
                        is_option[ni, ri, si, sj-1] = True

    # flatten the is_option tensor
    action_idcs = choose_from_multidim_options(is_option)

    # don't modify the input tensor
    networks = networks.clone()
    # invert the nodes in the networks
    for (network, act_idx) in zip(networks, action_idcs):
        if act_idx[0] == -1:
            # no valid action, so skip this network
            continue
        route = network[act_idx[0]]
        node1_loc = act_idx[1]
        node2_loc = act_idx[2] + 1
        section = route[node1_loc:node2_loc+1].clone()
        route[node1_loc:node2_loc+1] = section.flip(0)

    return networks


def exchange_routes(state: RouteGenBatchState, networks: torch.Tensor):
    """
    Chose a node visited by two routes, and swap the route parts after it.

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    """
    n_networks = networks.shape[0]
    n_routes = networks.shape[1]
    max_route_len = state.max_route_len.max().item()

    is_option = torch.zeros((n_networks, n_routes, n_routes, max_route_len,
                             max_route_len), dtype=bool)
    route_lens = (networks != -1).sum(dim=-1)
    route_limits = torch.stack([state.min_route_len, state.max_route_len],
                               dim=-1).cpu()
    if state.batch_size == 1:
        route_limits = route_limits.expand((n_networks, 2))

    # set up some variables for the loop
    route_idcs = torch.arange(n_routes)
    n_nodes = state.max_n_nodes

    # serial implementation
    for ni, ntwk in enumerate(networks):
        # by using a different dummy value on both sides, dummy stops don't
         # show up as shared
        min_len, max_len = route_limits[ni]
        diff_dummy = ntwk.clone()
        diff_dummy[diff_dummy == -1] = -2
        stops_are_shared = \
            ntwk[:, None, :, None] == diff_dummy[None, :, None, :]
        possible_exchange = stops_are_shared
        # don't exchange routes at both of their starts, that does nothing
        possible_exchange[:, :, 0, 0] = False
        possible_exchange[route_idcs, route_idcs, route_lens[ni]-1,
                          route_lens[ni]-1] = False
        any_shared = possible_exchange.any(dim=-1).any(dim=-1)
        for ri, rj in torch.stack(torch.where(any_shared), 1):
            if rj <= ri:
                continue
            rilen = route_lens[ni, ri]
            rjlen = route_lens[ni, rj]
            i_psbl, j_psbl = torch.where(possible_exchange[ri, rj])
            possible_locs = torch.stack((i_psbl, j_psbl), 1)
            new_1_lens = possible_locs[:, 0] + rjlen - possible_locs[:, 1]
            new_2_lens = possible_locs[:, 1] + rilen - possible_locs[:, 0]
            lens_are_valid = (min_len <= new_1_lens) & \
                             (new_1_lens <= max_len) & \
                             (min_len <= new_2_lens) & \
                             (new_2_lens <= max_len)
            is_option[ni, ri, rj, i_psbl, j_psbl] = lens_are_valid
            # don't exchange routes at their ends either, that does nothing
            is_option[ni, ri, rj, rilen-1, rjlen-1] = False

            # don't allow exchanges that would result in looping routes
            iroute = ntwk[ri]
            jroute = ntwk[rj]
            has_nodes = torch.zeros(n_nodes, dtype=bool)
            for si, sj in possible_locs[lens_are_valid]:
                # check that i part 1 and j part 2 have no nodes in common
                i_part1 = iroute[:si]
                j_part2 = jroute[min(sj+1, rjlen):rjlen]
                has_nodes[i_part1] = True
                overlaps = has_nodes[j_part2].any()
                is_option[ni, ri, rj, si, sj] &= not overlaps
                if not overlaps:
                    # check that j part 1 and i part 2 have no nodes in common
                    j_part1 = jroute[:sj]
                    i_part2 = iroute[min(si+1, rilen):rilen]
                    has_nodes.fill_(False)
                    has_nodes[j_part1] = True
                    overlaps = has_nodes[i_part2].any()
                    is_option[ni, ri, rj, si, sj] &= not overlaps

    action_idcs = choose_from_multidim_options(is_option)

    # don't modify the input tensor
    networks = networks.clone()
    # exchange the routes in the networks
    for (ntwk, act_idx, ntwk_rlens) in zip(networks, action_idcs, route_lens):
        if act_idx[0] == -1:
            # no valid action, so skip this network
            continue
        route1 = ntwk[act_idx[0]]
        route2 = ntwk[act_idx[1]]
        route1_copy = route1.clone()
        route2_copy = route2.clone()
        route1_len = ntwk_rlens[act_idx[0]]
        route2_len = ntwk_rlens[act_idx[1]]
        loc1 = act_idx[2]
        loc2 = act_idx[3]
        route2_segment = route2_copy[loc2:route2_len]
        route1[loc1:] = -1
        route1[loc1:loc1 + len(route2_segment)] = route2_segment
        route1_segment = route1_copy[loc1:route1_len]
        route2[loc2:] = -1
        route2[loc2:loc2 + len(route1_segment)] = route1_segment

    # Husselmann here runs the repair procedure if it's not valid, and
     # then samples another option if the repair fails, continuing until all
     # options have been exhausted.  Can we do that?  But I don't think any
     # constraint violation could be created by this operator, so it should
     # be fine to skip the repair step.

    return networks


def replace_node(state: RouteGenBatchState, networks: torch.Tensor):
    """
    Choose a node on a route and replace it with another node.

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    """
    adj_mat = get_formatted_adj_mat(state, networks)

    # n_networks x n_routes x max_route_len x n_nodes
    n_nodes = state.max_n_nodes
    is_option = torch.zeros(networks.shape + (n_nodes+1,), dtype=bool)
    are_nodes_on_route = tu.get_nodes_on_routes_mask(state.max_n_nodes, networks)
    visited_atleast_twice = are_nodes_on_route.sum(dim=1) > 1
    route_lens = (networks != -1).sum(dim=-1)

    for ni, network in enumerate(networks):
        for ri, route in enumerate(network):
            ianor = are_nodes_on_route[ni, ri]
            rilen = route_lens[ni, ri]
            # the replacement node cannot already be on the route
            is_option[ni, ri, :rilen] = ~ianor[None]
            route = route[:rilen]
            # the node being replaced must be visited at least twice
            is_option[ni, ri, :rilen] &= visited_atleast_twice[ni, route, None]
            # replacement node must be adjacent to next node
            is_option[ni, ri, :rilen-1] &= adj_mat[ni, route[1:]]
            # replacement node must be adjacent to previous node
            is_option[ni, ri, 1:rilen] &= adj_mat[ni, route[:-1]]

    # remove dummy node loc from options
    is_option = is_option[..., :-1]

    action_idcs = choose_from_multidim_options(is_option)

    # don't modify the input tensor
    networks = networks.clone()
    # replace the nodes in the networks
    for (network, (ri, si, new_node)) in zip(networks, action_idcs):
        if ri == -1:
            # no valid action, so skip this network
            continue
        network[ri, si] = new_node

    return networks


def donate_node(state: RouteGenBatchState, networks: torch.Tensor):
    """
    Remove a node from one route and add it to another route

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    """
    # get removable and addable nodes
    # n_networks x n_routes x max_route_len+1 x n_nodes+1
    add_options = _get_add_node_options(state, networks)
    # n_networks x n_routes x max_route_len
    del_options = _get_delete_node_options(state, networks)
    # find overlaps between the two

    # expand del_options with a new dim that is a mask of the node at that loc
    # number of nodes plus 1 for the dummy node
    n_nodes_p1 = add_options.shape[-1]
    del_opt_exp = del_options[:, :, :, None].expand((-1, -1, -1, n_nodes_p1))
    nodes_arange = torch.arange(n_nodes_p1)
    network_is_node_mask = \
        nodes_arange[None, None, None, :] == networks[:, :, :, None]
    # n_networks x n_routes x max_route_len x n_nodes+1
    del_nodes = del_opt_exp & network_is_node_mask
    # remove the dummy node rows from the options
    add_options = add_options[..., :-1]
    del_nodes = del_nodes[..., :-1]

    # don't modify the input tensor
    out_networks = networks.clone()

    # n_networks x n_routes x n_nodes
    nodes_are_deletable = del_nodes.any(2)
    # n_networks x n_routes x n_nodes
    nodes_are_addable = add_options.any(2)

    # pick a node to move from one route to another on each network, and move it
    n_routes = networks.shape[1]
    max_route_len = state.max_route_len.max().item()
    for ni in range(networks.shape[0]):
        is_option = torch.zeros((n_routes, n_routes, max_route_len,
                                 max_route_len+1), dtype=bool)
        # n_networks x n_routes x n_routes x n_nodes
        nodes_are_movable = nodes_are_deletable[ni, :, None] & \
                            nodes_are_addable[ni, None, :]
        # n_routes x n_routes
        have_movable_nodes = nodes_are_movable.any(dim=-1)
        # don't allow moving nodes to the same route
        have_movable_nodes.fill_diagonal_(False)

        for ri, rj in torch.stack(torch.where(have_movable_nodes), 1):
            # max_route_len x n_nodes
            iroute_del_masks = del_nodes[ni, ri]
            # max_route_len+1 x n_nodes
            jroute_add_options = add_options[ni, rj]
            move_allowed = iroute_del_masks[:, None] & \
                            jroute_add_options[None]
            is_option[ri, rj] = move_allowed.any(dim=-1)

        # choose a node to move
        action_idcs = choose_from_multidim_options(is_option[None])
        # move the node
        ri, rj, si, sj = action_idcs.squeeze(0)
        if ri > -1:
            # we got a valid option, so make the swap
            _delete_node_inplace(out_networks[ni], ri, si)
            new_node = networks[ni, ri, si].item()
            _add_node_inplace(out_networks[ni], rj, sj, new_node)

    return out_networks


def _get_add_node_options(state, networks, mode='any'):
    """
    is_option: A n_networks x n_routes x max_route_len + 1 x n_nodes + 1 tensor

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    mode: 'any', 'inside', or 'terminal'
    """
    adj_mat = get_formatted_adj_mat(state, networks)
    are_nodes_on_route = tu.get_nodes_on_routes_mask(state.max_n_nodes, networks)

    # n_networks x n_routes x max_route_len + 1 x n_nodes + 1
    network_idcs = torch.arange(networks.shape[0])[:, None, None]
    network_idcs = network_idcs.expand_as(networks)
    max_route_len = state.max_route_len.max().item()
    is_option = ~are_nodes_on_route[:, :, None]
    # only nodes not already on routes are options
    is_option = is_option.repeat(1, 1, max_route_len+1, 1)
    stop_adjs = adj_mat[network_idcs, networks]
    # for inside locations, only nodes adjacent to both neighbours are options
    is_option[..., :-1, :] &= stop_adjs
    is_option[..., 1:, :] &= stop_adjs
    # for terminal locations, only nodes adjacent to the existing terminal are
     # options

    gather_idcs = _get_route_end_scatter_idcs(networks, is_option)
    end_adjs = stop_adjs.gather(2, gather_idcs)
    is_end_option = end_adjs & ~are_nodes_on_route[:, :, None]
    scatter_idcs = (gather_idcs + 1).squeeze(-1)
    is_option.scatter_(2, scatter_idcs, is_end_option)

    # avoid adding nodes that would make the route too long
    route_lens = (networks != -1).sum(dim=-1)
    max_route_lens = state.max_route_len.cpu()
    if state.batch_size == 1:
        max_route_lens = max_route_lens[None].expand((networks.shape[0], 1))
    short_enough_to_lengthen = route_lens < max_route_lens
    is_option &= short_enough_to_lengthen[..., None, None]

    if 'mode' != 'any':
        # add one to index the location *after* the last stop
        post_route_scatter_idcs = \
            _get_route_end_scatter_idcs(networks, is_option) + 1
        is_option = update_options_for_mode(is_option, post_route_scatter_idcs,
                                            mode)

    return is_option


def _add_node_inplace(network, route_idx, stop_idx, new_node):
    route = network[route_idx]
    route_copy = route.clone()
    route[stop_idx] = new_node
    route[stop_idx+1:] = route_copy[stop_idx:-1]


def _get_delete_node_options(state, networks, mode='any'):
    """
    returns a n_networks x n_routes x max_route_len tensor

    state: a RouteGenBatchState object with either batch_size = 1 or
        batch_size = networks.shape[0]
    networks: a torch.Tensor of shape (batch_size, num_routes, route_len)
    mode: 'any', 'inside', or 'terminal'
    """
    adj_mat = get_formatted_adj_mat(state, networks)
    # clone before in-place writes: get_formatted_adj_mat may return an
    # expanded (memory-shared) view, which newer torch refuses to mutate.
    adj_mat = adj_mat.clone()
    # set dummy node rows to True, so they're adjacent to everything
    adj_mat[:, -1, :] = True
    adj_mat[:, :, -1] = True

    # parallel implementation of identifying the nodes that can be deleted
    network_idcs = torch.arange(networks.shape[0])[:, None, None]

    # n_networks x n_routes x max_route_len
    n_networks = networks.shape[0]
    max_route_len = state.max_route_len.max().item()
    is_option = torch.ones((n_networks, networks.shape[1], max_route_len),
                           dtype=bool)
    # nodes surrounding non-terminal stops must be neighbours to delete stop
    network_idcs = network_idcs.expand_as(networks[..., :-2])
    surrounding_stops_are_neighbours = \
        adj_mat[network_idcs, networks[:, :, :-2], networks[:, :, 2:]]
    # because the dummy node is adjacent to everything, this marks the terminal
     # nodes deletable.
    is_option[..., 1:-1] &= surrounding_stops_are_neighbours

    # avoid deleting stops that would leave a node unvisited
    # n_networks x n_routes x n_nodes+1
    are_nodes_on_route = tu.get_nodes_on_routes_mask(state.max_n_nodes, networks)
    visited_atleast_twice = are_nodes_on_route.sum(dim=1) > 1
    n_routes = networks.shape[1]
    valt = visited_atleast_twice[:, None].expand((-1, n_routes, -1))
    gather_networks = networks.clone()
    n_nodes = state.max_n_nodes
    gather_networks[gather_networks == -1] = n_nodes
    stops_visited_atleast_twice = valt.gather(2, gather_networks)
    is_option &= stops_visited_atleast_twice

    # disallow deletion of nodes that would make the route too short
    route_lens = (networks != -1).sum(dim=-1)
    min_route_len = state.min_route_len.cpu()
    if state.batch_size == 1:
        min_route_len = min_route_len[None].expand((networks.shape[0], 1))
    long_enough_to_shorten = route_lens > min_route_len
    is_option &= long_enough_to_shorten[:, :, None]

    if mode != 'any':
        end_scatter_idcs = _get_route_end_scatter_idcs(networks, is_option)
        is_option = update_options_for_mode(is_option, end_scatter_idcs, mode)

    return is_option


def update_options_for_mode(is_option, end_scatter_idcs, mode):
    if mode == 'inside':
        # remove the options for terminal nodes
        is_option[:, :, 0] = False
        is_option.scatter_(2, end_scatter_idcs, False)
    elif mode == 'terminal':
        # keep only the options for terminal nodes
        term_is_option = torch.zeros_like(is_option)
        term_is_option[:, :, 0] = is_option[:, :, 0]
        end_is_option = torch.gather(is_option, 2, end_scatter_idcs)
        term_is_option.scatter_(2, end_scatter_idcs, end_is_option)
        is_option = term_is_option
    return is_option


def _delete_node_inplace(network, route_idx, stop_idx):
    route = network[route_idx]
    route_copy = route.clone()
    route[stop_idx:-1] = route_copy[stop_idx+1:]
    route[-1] = -1


def get_formatted_adj_mat(state, networks):
    adj_mat = state.street_adj.isfinite()
    cpu_dev = torch.device('cpu')
    if state.device != cpu_dev:
        adj_mat = adj_mat.to(cpu_dev)
    # add padding for the dummy node
    adj_mat = torch.nn.functional.pad(adj_mat, (0, 1, 0, 1), value=False)
    # add the batch dimension to the adjacency matrix if needed
    if adj_mat.shape[0] == 1:
        expanded_shape = (networks.shape[0],) + adj_mat.shape[1:]
        adj_mat = adj_mat.expand(expanded_shape)
    return adj_mat


def choose_from_multidim_options(is_option: torch.Tensor):
    # flatten the is_option tensor to make it a choice over all possible
     # nodes on all routes
    action_probs = is_option.contiguous().view(is_option.shape[0], -1).float()
    # set the probabilities of invalid actions to 1 to avoid errors
    no_valid_action = action_probs.sum(dim=1) == 0
    action_probs[no_valid_action] = 1
    # pick one node to add to one route for each network
    action_idcs = action_probs.multinomial(1).squeeze(-1)
    # unravel the action indices to the original shape
    action_idcs = tu.unravel_indices(action_idcs, is_option.shape[1:])
    # set the action indices to -1 if there is no valid action
    action_idcs[no_valid_action] = -1
    return action_idcs


def _get_route_end_scatter_idcs(networks, is_option):
    """Returns a tensor of shape (n_networks, n_routes, 1, D) that has the
        length of each route in the 3rd dimension, expanded over the D
        dimensions.

    networks: a tensor of shape (n_networks, n_routes, max_route_len)
        containing the routes
    is_option: a tensor of shape (n_networks, n_routes, X, D)
        where X >= max_route_len and D is some number of subsequent dimensions.
    """
    route_lens = (networks != -1).sum(dim=-1)
    route_end_scatter_idcs = (route_lens - 1)
    for _ in range(is_option.ndim - 2):
        route_end_scatter_idcs = route_end_scatter_idcs[..., None]
    scatter_shape = (-1, -1, -1,) + is_option.shape[3:]
    route_end_scatter_idcs = route_end_scatter_idcs.expand(scatter_shape)
    return route_end_scatter_idcs

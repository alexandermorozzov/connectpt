# Copyright 2023 Andrew Holliday
#
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Transit Learning is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.


"""A route network: the ``[B, R, L]`` route tensor with ``-1`` padding.

:func:`to_routes` is the one way to turn any route form into that tensor;
:func:`route_lists` is the way back. Also ``Routes.txt`` and the travel times
along routes.
"""
from pathlib import Path

import torch

from ..common.tensors import get_update_at_mask


def to_routes(x, n_routes=None, width=None) -> torch.Tensor:
    """Any route form -> a long ``[B, R, L]`` tensor with ``-1`` padding.

    ``x`` is a tensor ``[L]``, ``[R, L]`` or ``[B, R, L]``, or the same as nested
    lists (a route is a list of node ids or a 1D tensor). A list that holds no
    route at all is empty networks: ``[]`` -> ``[1, 0, 0]``, ``[[], []]`` ->
    ``[2, 0, 0]``. A tensor keeps its device; lists go to the device of their
    first route tensor, else the CPU. ``n_routes`` and ``width`` pad or cut R and
    L; a cut that drops a stop raises ``ValueError``.
    """
    if isinstance(x, torch.Tensor):
        if not 1 <= x.ndim <= 3:
            raise ValueError(f"a route tensor has 1 to 3 dims, got {tuple(x.shape)}")
        t = x.long().reshape((1,) * (3 - x.ndim) + tuple(x.shape))
    else:
        t = _from_lists(x)
    return _fit(_fit(t, 1, n_routes), 2, width)


def route_lists(t) -> list:
    """A route tensor of any rank -> nested lists of node ids, ``-1`` dropped."""

    def strip(x):
        if x and isinstance(x[0], list):
            return [strip(v) for v in x]
        return [v for v in x if v >= 0]

    return strip(t.tolist())


def _depth(x):
    """Levels down to the first node id (a node id is 0, a route 1, a network
    2); ``None`` if ``x`` holds no route."""
    if isinstance(x, torch.Tensor):
        return x.ndim
    if not isinstance(x, (list, tuple)):
        return 0
    for item in x:
        depth = _depth(item)
        if depth is not None:
            return depth + 1
    return None


def _from_lists(x) -> torch.Tensor:
    depth = _depth(x)
    if depth is None:
        networks = x if len(x) else [x]
    elif 1 <= depth <= 3:
        networks = [[x]] if depth == 1 else [x] if depth == 2 else x
    else:
        raise ValueError(f"routes nest {depth} levels deep, expected 1 to 3")
    device = next((r.device for net in networks for r in net if isinstance(r, torch.Tensor)), None)
    n_routes = max((len(net) for net in networks), default=0)
    width = max((len(r) for net in networks for r in net), default=0)
    out = torch.full((len(networks), n_routes, width), -1, dtype=torch.long, device=device)
    for bi, net in enumerate(networks):
        for ri, route in enumerate(net):
            if len(route):
                out[bi, ri, : len(route)] = torch.as_tensor(route)
    return out


def _fit(t, dim, size):
    """Pad dim ``dim`` of ``t`` with ``-1`` or cut it to ``size``."""
    if size is None:
        return t
    size, have = int(size), t.shape[dim]
    if size < 0:
        raise ValueError(f"route dimension {dim} must be nonnegative, got {size}")
    if size < have:
        if (t.narrow(dim, size, have - size) >= 0).any():
            raise ValueError(f"cutting route dim {dim} from {have} to {size} drops stops")
        return t.narrow(dim, 0, size)
    if size > have:
        shape = list(t.shape)
        shape[dim] = size - have
        return torch.cat((t, t.new_full(shape, -1)), dim)
    return t


def read_routes_txt(path, *, n_nodes, spec) -> torch.Tensor:
    """Parse ``Routes.txt`` into a ``[1, n_routes, max_route_len]`` tensor.

    One route per line, whitespace-separated **0-indexed** node ids; ``#``
    comments and blank lines are ignored. Validates the id range, the route
    count and the longest route against ``spec``.
    """
    path = Path(path)
    routes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stops = line.split("#", 1)[0].split()
        if stops:
            routes.append([int(s) for s in stops])

    if len(routes) != int(spec["n_routes"]):
        raise ValueError(f"{path}: {len(routes)} routes, but cfg/instance says {spec['n_routes']}")
    flat = [s for r in routes for s in r]
    if min(flat) < 0 or max(flat) >= n_nodes:
        raise ValueError(
            f"{path}: node ids {min(flat)}..{max(flat)} outside 0..{n_nodes - 1} "
            f"(routes must be 0-indexed)"
        )
    longest = max(len(r) for r in routes)
    if longest > int(spec["max_route_len"]):
        raise ValueError(
            f"{path}: longest route has {longest} stops, max_route_len is "
            f"{spec['max_route_len']}"
        )
    return to_routes(routes, width=int(spec["max_route_len"]))


def write_routes_txt(path, routes, header=None) -> int:
    """Write the first network of ``routes`` as ``Routes.txt``: one route per
    line, 0-indexed node ids. Returns the number of routes written."""
    lines = [f"# {header}"] if header else []
    lines.append("# 0-indexed; one route per line")
    for stops in route_lists(to_routes(routes)[0]):
        if stops:
            lines.append(" ".join(str(s) for s in stops))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - (2 if header else 1)


def get_route_leg_times(batch_routes, drive_times_matrix, mean_stop_time_s=0):
    add_route_dim = batch_routes.ndim == 2
    if add_route_dim:
        # add a route dimension
        batch_routes = batch_routes[:, None]

    batch_idxs = torch.arange(batch_routes.shape[0], device=batch_routes.device)
    # batch_size x n_routes x route_len - 1
    leg_times = drive_times_matrix[
        batch_idxs[:, None, None], batch_routes[..., :-1], batch_routes[..., 1:]
    ]
    if type(mean_stop_time_s) is torch.Tensor and mean_stop_time_s.ndim == 1:
        # make number of dimensions match leg_times
        mean_stop_time_s = mean_stop_time_s[..., None, None]

    leg_times += mean_stop_time_s
    invalid_leg_mask = (batch_routes == -1)[..., 1:]
    leg_times = get_update_at_mask(leg_times, invalid_leg_mask)
    if add_route_dim:
        # remove the extra dimension we added from the output
        leg_times.squeeze_(1)
    return leg_times


def get_route_edge_matrix(
    batch_routes,
    drive_times_matrix,
    mean_stop_time_s,
    symmetric_routes=True,
    low_mem_mode=False,
    return_used_routes=False,
):
    """
    batch_routes: (batch_size, n_routes, max_route_len) tensor of route indices
    drive_times_matrix: (batch_size, n_nodes, n_nodes) tensor of shortest-path
        drive times between nodes
    mean_stop_time_s: mean time spent at each stop in seconds
    symmetric_routes: if True, treat routes as going both ways along their
        stops.

    Returns a (batch_size, max_route_len, max_route_len) tensor of
        the shortest time on a route between each pair of stops.
    """
    batch_size = batch_routes.shape[0]
    max_route_len = batch_routes.shape[-1]
    n_routes = batch_routes.shape[1]
    n_street_nodes = drive_times_matrix.shape[-1]
    dev = batch_routes.device
    # add a dummy row and column
    route_edge_mat = torch.full(
        (n_street_nodes + 1, n_street_nodes + 1),
        float("inf"),
        dtype=drive_times_matrix.dtype,
        device=dev,
    )
    route_edge_mat.fill_diagonal_(0)
    if n_routes == 0:
        outval = route_edge_mat.repeat(batch_size, 1, 1)[:, :-1, :-1]
        if return_used_routes:
            return outval, torch.zeros_like(outval, dtype=int)[:, 0]
        else:
            return outval

    route_edge_mat = route_edge_mat.repeat(batch_size, n_routes, 1, 1)

    batch_idxs = torch.arange(batch_size, device=dev)
    # batch_size x n_routes x route_len - 1
    leg_times = get_route_leg_times(batch_routes, drive_times_matrix, mean_stop_time_s)
    # batch_size x n_routes x route_len x route_len
    interstop_times = get_route_interstop_times(leg_times)

    invalid_stop = batch_routes == -1
    invalid_dst = invalid_stop[..., None].expand(-1, -1, -1, max_route_len)
    interstop_times[invalid_dst] = float("inf")
    invalid_src = invalid_stop[..., None, :].expand(-1, -1, max_route_len, -1)
    interstop_times[invalid_src] = float("inf")
    route_idxs = torch.arange(n_routes, device=dev)

    if low_mem_mode:
        # Avoid OOM errors for large tensors.  Doing it by trying to count the
        # actual memory used doesn't work, so pass a flag instead.
        for bi in batch_idxs:
            route_edge_mat[
                bi,
                route_idxs[:, None, None],
                batch_routes[bi, ..., None],
                batch_routes[bi, ..., None, :],
            ] = interstop_times[bi]
    else:
        route_edge_mat[
            batch_idxs[:, None, None, None],
            route_idxs[None, :, None, None],
            batch_routes[..., None],
            batch_routes[..., None, :],
        ] = interstop_times

    route_edge_mat, min_idxs = route_edge_mat[..., :-1, :-1].min(dim=1)

    if symmetric_routes:
        # add routes going back the other way
        transpose_dtm = drive_times_matrix.permute(0, 2, 1)
        reverse_edges, reverse_used_routes = get_route_edge_matrix(
            batch_routes,
            transpose_dtm,
            mean_stop_time_s,
            symmetric_routes=False,
            return_used_routes=True,
        )
        reverse_edges = reverse_edges.permute(0, 2, 1)
        route_edge_mat = torch.minimum(route_edge_mat, reverse_edges)

    if return_used_routes:
        # used routes are just the min idxs, but with -1 where there's no route
        used_routes = min_idxs
        used_routes[route_edge_mat == float("inf")] = -1
        # set diagonal elements of each batch matrix to -1
        node_idxs = torch.arange(n_street_nodes, device=dev)
        used_routes[:, node_idxs, node_idxs] = -1

        if symmetric_routes:
            use_reverse = reverse_edges == route_edge_mat
            reverse_used_routes = reverse_used_routes.permute(0, 2, 1)
            used_routes[use_reverse] = reverse_used_routes[use_reverse]

        return route_edge_mat, used_routes

    return route_edge_mat


def get_route_interstop_times(leg_times):
    """leg_times: An n_routes x route_length - 1 tensor.

    Returns an n_routes x route_length x route_length tensor of the time it
    takes to get from one stop to another.
    """
    cum_dists = leg_times.cumsum(-1)
    # preappend some zeros so the next operation will give time
    # from node i to other node j
    cum_dists = torch.nn.functional.pad(cum_dists, (1, 0))
    inter_stop_dists = cum_dists[..., None, :] - cum_dists[..., :, None]
    inter_stop_dists[inter_stop_dists < 0] = float("inf")
    return inter_stop_dists


def get_nodes_on_routes_mask(n_nodes, networks):
    """Returns an n_networks x n_routes x n_nodes+1 tensor of booleans,
    where T_ijk = True where node i is on route j in network k, False
    otherwise."""
    if networks.ndim == 2:
        networks = networks[None]
    n_networks = networks.shape[0]
    n_routes = networks.shape[1]

    are_nodes_on_route = torch.zeros(
        (n_networks, n_routes, n_nodes + 1), dtype=bool, device=networks.device
    )
    # replace -1 dummy elements with n_nodes, so they work with scatter
    scatter_networks = networks.clone()
    scatter_networks[scatter_networks == -1] = n_nodes
    # set the nodes on the routes to True
    are_nodes_on_route.scatter_(2, scatter_networks, True)
    # set the dummy node to False, it's not "on" any node
    are_nodes_on_route[..., -1] = False
    return are_nodes_on_route


def get_unselected_routes(networks, selected_idxs):
    # create a mask that is 0 at all nodes of selected routes, 1 elsewhere
    # add route and node dimensions to the selected indices
    scatter_idxs = selected_idxs[..., None, None]
    # expand the node dimension to the size of the tensor
    exp_target = [-1] * (scatter_idxs.ndim - 1) + [networks.shape[-1]]
    scatter_idxs = scatter_idxs.expand(exp_target)
    mask = torch.ones_like(networks).scatter(-2, scatter_idxs, 0)
    # assemble a tensor of indices to gather the unselected routes
    unselected_shape = list(networks.shape)
    unselected_shape[-2] -= 1
    unselected_idxs = mask.nonzero()[:, -2].reshape(*unselected_shape)
    # get the unselected routes and return them
    unselected_routes = networks.gather(-2, unselected_idxs)
    return unselected_routes

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


"""All-pairs shortest paths of a street graph and aggregation along them."""
import torch

from ..common.tensors import get_variable_slice_mask


class AllShortestPaths:
    def __init__(self, nexts, dists, precompute=False):
        self.dists = dists
        self.nexts = nexts
        self._cached_paths = {}
        if precompute:
            seq_tensor, route_lens = reconstruct_all_paths(nexts)
            for from_i in range(nexts.shape[-1]):
                tos = {}
                for to_i in range(nexts.shape[-1]):
                    if to_i == from_i:
                        continue
                    path = seq_tensor[from_i, to_i, : route_lens[from_i, to_i]]
                    tos[to_i] = path
                self._cached_paths[from_i] = tos

    def get_length(self, source, dest):
        return self.dists[source, dest]

    def get_path(self, source, dest):
        if (source, dest) in self._cached_paths:
            return self._cached_paths[(source, dest)]
        else:
            path = reconstruct_path(source, dest, self.nexts)
            self._cached_paths[(source, dest)] = path
            return path


def floyd_warshall(edge_cost_tensor, return_raw_tensors=True):
    """A pytorch-vectorized implementation of the Floyd-Warshall algorithm.

    Input tensor should be a matrix of edge costs, where element (u,v) is the
    cost of traversing the edge from u to v if there is one, and is otherwise
    infinity. It may have a leading batch dimension.

    We vectorize the two inner loops of the algorithm.  The outermost loop must
    be sequential: the calculation at i assumes results from iteration i-1 are
    present.

    For the pseudocode of the unvectorized algorithm, see:
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#Path_reconstruction
    """
    has_batch_dim = edge_cost_tensor.ndim == 3
    if not has_batch_dim:
        # add a batch dimension
        edge_cost_tensor = edge_cost_tensor[None]

    # put the batch dimension last
    edge_cost_tensor = edge_cost_tensor.permute(1, 2, 0)
    batch_size = edge_cost_tensor.shape[2]

    device = edge_cost_tensor.device
    dists = edge_cost_tensor.clone().detach()

    # set up next-node array
    num_nodes = dists.shape[0]
    nexts = torch.full((num_nodes, num_nodes, batch_size), -1, dtype=torch.long, device=device)
    node_idxs = torch.arange(num_nodes, device=device, dtype=torch.long)
    nexts[node_idxs, node_idxs] = node_idxs[:, None]

    has_edge_mat = dists < float("inf")
    dest_idxs = node_idxs[None, :, None].expand(num_nodes, num_nodes, batch_size)
    nexts = torch.where(has_edge_mat, dest_idxs, nexts)

    # the core loop
    for kk in range(num_nodes):
        # vectorize two levels
        from_k_dists = dists[kk]
        to_k_dists = dists[:, kk]
        through_k_dists = to_k_dists[:, None] + from_k_dists[None]
        update_mask = dists > through_k_dists
        dists[update_mask] = through_k_dists[update_mask]

        # tile the k-th column into N x N
        new_nexts = nexts[:, kk : kk + 1].repeat(1, num_nodes, 1)
        nexts[update_mask] = new_nexts[update_mask]

    # handle different batch-size cases for the returned value
    nexts = nexts.permute(2, 0, 1)
    dists = dists.permute(2, 0, 1)

    if return_raw_tensors:
        return nexts, dists
    else:
        if not has_batch_dim:
            retval = AllShortestPaths(nexts.squeeze(0), dists.squeeze(0))
        else:
            retval = [AllShortestPaths(nn, dd) for nn, dd in zip(nexts, dists)]
        return retval


def reconstruct_path(source, dest, nexts):
    """Given a source, dest, and the nexts matrix returned by floyd-warshall,
    returns the shortest path from source to dest."""
    if nexts[source, dest] < 0:
        return None
    path = [source]
    curr_node = source
    while curr_node != dest:
        curr_node = nexts[curr_node, dest]
        path.append(curr_node)
    # return a list of integers, not tensors
    path = [ee.item() if type(ee) is torch.Tensor else ee for ee in path]
    return path


def reconstruct_all_paths(nexts):
    n_nodes = nexts.shape[-1]
    node_idxs = torch.arange(n_nodes, device=nexts.device)[:, None]
    if nexts.ndim == 3:
        batch_size = nexts.shape[0]
        node_idxs = node_idxs.expand(batch_size, -1, -1)
    path_tensor, path_lens = aggregate_node_features(
        nexts, node_idxs, agg_mode="concat", return_node_counts=True
    )
    if nexts.ndim == 3 or nexts.shape[-1] != 1:
        path_tensor.squeeze_(-1)

    # set all padding elements to -1
    invalid_mask = get_variable_slice_mask(path_tensor, -1, froms=path_lens)
    path_tensor[invalid_mask] = -1

    return path_tensor, path_lens


def count_path_nodes(nexts):
    """Node counts of the shortest paths encoded by ``nexts``.

    Returns exactly ``reconstruct_all_paths(nexts)[1]`` (0 on the diagonal and
    for unreachable pairs) but walks the ``nexts`` matrix with a vectorized
    pointer chase instead of materializing the full (B, N, N, max_len) path
    tensor -- much cheaper in both time and memory. The walk follows the same
    pointers, so tie-broken equal-cost paths yield identical counts.
    """
    has_batch_dim = nexts.ndim == 3
    nx = nexts if has_batch_dim else nexts[None]
    batch_size, n_nodes, _ = nx.shape
    dev = nx.device
    node_idxs = torch.arange(n_nodes, device=dev)
    dst = node_idxs[None, None, :].expand(batch_size, n_nodes, n_nodes)

    flat_nx = nx.reshape(batch_size, -1)
    cur = nx.clone()  # first hop from source toward dst
    # source + the first hop = 2 nodes; each further step below adds one.
    # (the diagonal / unreachable entries are overwritten to 0 at the end.)
    counts = torch.full((batch_size, n_nodes, n_nodes), 2, dtype=torch.long, device=dev)
    active = (cur != dst) & (cur >= 0)
    for _ in range(n_nodes):
        if not active.any():
            break
        counts[active] += 1
        step_idx = (cur.clamp(min=0) * n_nodes + dst).reshape(batch_size, -1)
        nxt = flat_nx.gather(1, step_idx).reshape(batch_size, n_nodes, n_nodes)
        cur = torch.where(active, nxt, cur)
        active = active & (cur != dst) & (cur >= 0)

    # match reconstruct_all_paths: 0 for unreachable pairs and the diagonal
    counts[nx == -1] = 0
    counts[:, node_idxs, node_idxs] = 0
    if not has_batch_dim:
        counts = counts.squeeze(0)
    return counts


def aggregate_node_features(nexts, features, agg_mode="sum", return_node_counts=False):
    """
    features is a tensor of node features that will be aggregated along the
    shortest path.

    agg_mode: either "sum", "mean", or "concat".
    """
    # translate node features to edge features: each "edge" feature is the
    # feature of the edge's target node.
    has_batch_dim = features.ndim == 3
    if not has_batch_dim:
        # no batch dimension, so add one
        features = features[None]

    n_nodes = features.shape[1]

    # repeat the new dimension n_nodes times, giving ef_ij = nf_j
    edge_feats = features[:, None].expand(-1, n_nodes, -1, -1).clone()
    node_idxs = torch.arange(n_nodes)
    # fill the diagonal with 0s
    edge_feats[:, node_idxs, node_idxs] = 0

    edge_agg_mode = "concat" if agg_mode == "concat" else "sum"
    aggs, edge_counts = aggregate_edge_features(
        nexts, edge_feats, edge_agg_mode, return_edge_counts=True
    )
    # the final aggregates are missing the first node's feature, so add it
    if agg_mode == "concat":
        # transpose the (to, from) axes of edge features
        first_feats = edge_feats.permute(0, 2, 1, 3)
        # add a sequence dimension
        first_feats = first_feats[:, :, :, None]
        # pre-append the first node's feature along the sequence dimension
        aggs = torch.cat((first_feats, aggs), dim=-2)
    else:
        # add the first node's feature dimension
        aggs = aggs + features[:, :, None]

    if agg_mode == "mean" or return_node_counts:
        node_counts = edge_counts + 1
        if not has_batch_dim:
            nexts = nexts[None]
        if agg_mode == "mean":
            aggs = aggs / node_counts[..., None]
        # force node counts to be 0 for invalid paths and self-loops.
        node_counts[nexts == -1] = 0
        node_counts[:, node_idxs, node_idxs] = 0

    if not has_batch_dim:
        # remove the batch dim we added above
        aggs.squeeze_(0)
        if agg_mode == "mean" or return_node_counts:
            node_counts.squeeze_(0)

    if return_node_counts:
        return aggs, node_counts
    else:
        return aggs


def aggregate_edge_features(nexts, features, agg_mode="sum", return_edge_counts=False):
    """
    features is a tensor of edge features that will be aggregated along the
        shortest path.  Must be 3-dimensional (n_nodes x n_nodes x feature dim)
        or 4-dimensional (batch_dim x n_nodes x n_nodes x feature dim)

    agg_mode: "sum", "mean", or "concat".
    """
    nexts_has_batch_dim = nexts.ndim == 3
    if nexts_has_batch_dim:
        # make batch dimension come last
        nexts = nexts.permute(1, 2, 0)
        batch_size = nexts.shape[2]
    else:
        nexts = nexts[..., None]
        batch_size = 1

    features_has_batch_dim = features.ndim == 4
    if features_has_batch_dim:
        # move batch dim to just before feature dim
        features = features.permute(1, 2, 0, 3)
    else:
        # add a batch dimension
        features = features[..., None]

    dev = features.device
    n_nodes = nexts.shape[0]
    node_idxs = torch.arange(n_nodes, device=dev)
    batch_idxs = torch.arange(batch_size, device=dev).expand(n_nodes, n_nodes, -1)
    cur_node, dst_node = torch.meshgrid(node_idxs, node_idxs, indexing="ij")
    if nexts.ndim == 3:
        dst_node = dst_node[..., None].expand(-1, -1, batch_size)
        cur_node = cur_node[..., None].expand(-1, -1, batch_size)
    if agg_mode == "concat":
        aggs = []
    else:
        aggs = torch.zeros_like(features)
    if agg_mode == "mean" or return_edge_counts:
        edge_counts = torch.zeros_like(nexts)

    for ni in range(n_nodes):
        next_node = nexts[cur_node, dst_node, batch_idxs]
        next_feats = features[cur_node, next_node, batch_idxs]
        if agg_mode == "concat":
            # the final edge feature is always a self-loop, since no
            # non-looping path has more than n_nodes - 1 edges, so skip it.
            if ni < n_nodes - 1:
                if ((cur_node != dst_node) & (nexts != -1)).any():
                    # also, to save space, only append features if any are
                    # valid next nodes
                    aggs.append(next_feats)
        else:
            aggs = aggs + next_feats
        if agg_mode == "mean" or return_edge_counts:
            edge_counts += (cur_node != dst_node) & (nexts != -1)

        cur_node = next_node

    if agg_mode == "mean":
        # make 0s into 1s to avoid division by 0
        denom = edge_counts + (edge_counts == 0)
        retval = aggs / denom[..., None]
    elif agg_mode == "concat":
        if n_nodes == 1:
            # the "sequence" is just the single self-loop feature with a seq dim
            retval = features[None]
        elif len(aggs) == 0:
            retval = torch.zeros((0, n_nodes, n_nodes, batch_size, features.shape[-1]), device=dev)
        else:
            retval = torch.stack(aggs)
    else:
        retval = aggs

    if features_has_batch_dim:
        if agg_mode == "concat":
            # starts as sequence, from, to, batch, feature
            # make it batch, from, to, sequence, feature
            retval = retval.permute(3, 1, 2, 0, 4)
        else:
            retval = retval.permute(2, 0, 1, 3)
        if return_edge_counts:
            edge_counts = edge_counts.permute(2, 0, 1)
    else:
        retval = retval.squeeze(-1)
        if return_edge_counts:
            edge_counts = edge_counts.squeeze(-1)

    if return_edge_counts:
        return retval, edge_counts
    else:
        return retval


def get_path_edge_index(path, get_dense_edges=True, device=None):
    # assemble the edge indices
    if device is None and type(path) is torch.Tensor:
        device = path.device
    if type(path) is list:
        path = torch.tensor(path, device=device)
    if get_dense_edges:
        return torch.combinations(path).T
    else:
        # each stop has an edge only to the next stop on the route
        from_indices = path[:-1]
        to_indices = path[1:]
        return torch.tensor([from_indices, to_indices], dtype=int)

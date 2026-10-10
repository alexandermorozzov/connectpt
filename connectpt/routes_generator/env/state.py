import logging as log
import copy
from contextlib import contextmanager
from typing import Union
from collections.abc import Sequence

from numpy import ndarray
import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from ..city.graph import STOP_KEY
from ..city.shortest_paths import count_path_nodes, floyd_warshall, reconstruct_all_paths
from ..common.tensors import get_variable_slice_mask
from ..network.routes import get_route_edge_matrix, get_route_leg_times, to_routes

import numpy as np


from ..common.const import EPSILON
from .const import RouteActionKind


def enforce_correct_batch(matrix, batch_size):
    if matrix.ndim == 2:
        matrix = matrix[None]
    if matrix.shape[0] > 1:
        assert batch_size == matrix.shape[0]
    elif batch_size > 1:
        shape = (batch_size,) + (-1,) * (matrix.ndim - 1)
        matrix = matrix.expand(*shape)
    return matrix


class ExtraStateData(HeteroData):
    """A class for holding data, some of it computed, that is specific to one
    scenario in a state."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in [
            "base_valid_terms_mat",
            "valid_terms_mat",
            "mean_stop_time",
            "fixed_routes_file",
            "transfer_time_s",
            "total_route_time",
            "n_routes_to_plan",
            "min_route_len",
            "max_route_len",
            "n_nodes_in_scenario",
            "directly_connected",
            "route_mat",
            "transit_times",
            "has_path",
            "current_routes",
            "current_route_time",
            "current_route_times_from_start",
            "shortest_path_sequences",
            "route_nexts",
            "n_transfers",
            "context_node_covered_mask",
            "context_edge_covered_mask",
            "context_leg_use_count",
            "route_slot_context",
            "active_route_idx",
            "fixed_routes",
            "node_coords",
            "adjustment_target",
            "adjustment_weight",
            "adjustment_seed",
        ]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class RouteGenBatchState:
    def __init__(
        self,
        graph_data,
        constants,
        n_routes_to_plan,
        min_route_len=2,
        max_route_len=None,
        valid_terms_mat=None,
        *,
        preferences,
    ):
        # do initialization needed to make properties work
        if not isinstance(graph_data, Batch):
            if not isinstance(graph_data, list):
                graph_data = [graph_data]
            graph_data = Batch.from_data_list(graph_data)

        # set members that live directly on this object
        self.graph_data = graph_data
        # right now this must have the same value for all scenarios
        self.symmetric_routes = constants.symmetric_routes
        self._finished_routes = [[] for _ in range(graph_data.num_graphs)]
        self._route_update_deferred = False

        # the object isn't ready to give this property yet, so find it here
        dev = graph_data[STOP_KEY].x.device
        max_n_nodes = max([dd.num_nodes for dd in graph_data.to_data_list()])
        if valid_terms_mat is None:
            # all terminal pairs (i,j) are valid except if i = j
            valid_terms_mat = ~torch.eye(max_n_nodes, device=dev, dtype=bool)
            valid_terms_mat = valid_terms_mat.repeat(self.batch_size, 1, 1)

        cost_weights = dict(preferences)
        for key, val in cost_weights.items():
            # expand the cost weights to match the batch
            if type(val) is not Tensor:
                val = torch.tensor([val], device=dev)
            else:
                val = val.to(dev)
            if val.numel() == 1:
                val = val.expand(graph_data.num_graphs)
            cost_weights[key] = val

        # make a tensor of n_routes_to_plan, expanded to the right shape
        if type(n_routes_to_plan) is not Tensor:
            n_routes_to_plan = torch.tensor(n_routes_to_plan, device=dev)
        if n_routes_to_plan.numel() == 1:
            n_routes_to_plan = n_routes_to_plan.expand(graph_data.num_graphs)

        extra_datas = []
        transit_times = 1 - torch.eye(max_n_nodes, device=self.device)
        transit_times[transit_times > 0] = float("inf")
        for ii, dd in enumerate(graph_data.to_data_list()):
            extra_data = ExtraStateData()

            extra_data.transit_times = transit_times
            extra_data.route_mat = transit_times.clone()
            dircon = torch.eye(max_n_nodes, device=self.device, dtype=bool)

            extra_data.directly_connected = dircon
            extra_data.has_path = extra_data.directly_connected.clone()
            extra_data.base_valid_terms_mat = valid_terms_mat[ii]
            extra_data.valid_terms_mat = valid_terms_mat[ii]
            extra_data.mean_stop_time = torch.tensor(constants.mean_stop_time_s, device=dev)
            extra_data.transfer_time_s = torch.tensor(constants.avg_transfer_wait_time_s, device=dev)
            extra_data.total_route_time = torch.zeros((), device=dev)
            # make this a tensor so it's stackable
            extra_data.n_routes_to_plan = n_routes_to_plan[ii]

            if isinstance(min_route_len, Tensor):
                if min_route_len.numel() == 1:
                    extra_data.min_route_len = min_route_len
                else:
                    extra_data.min_route_len = min_route_len[ii]
            else:
                extra_data.min_route_len = torch.tensor(min_route_len, device=dev)
            extra_data.min_route_len.squeeze_()

            if max_route_len is None:
                max_route_len = dd.num_nodes
            if isinstance(max_route_len, Tensor):
                if max_route_len.numel() == 1:
                    extra_data.max_route_len = max_route_len
                else:
                    extra_data.max_route_len = max_route_len[ii]
            else:
                extra_data.max_route_len = torch.tensor(max_route_len, device=dev)
            extra_data.max_route_len.squeeze_()

            extra_data.n_nodes_in_scenario = torch.tensor(dd.num_nodes, device=dev)
            extra_data.current_routes = torch.full((max_n_nodes,), -1, device=dev)
            extra_data.current_route_time = torch.zeros((), device=dev)
            extra_data.current_route_times_from_start = torch.zeros((max_n_nodes,), device=dev)
            extra_data.shortest_path_sequences = torch.zeros((0, 0, 0), device=dev)
            extra_data.route_nexts = torch.zeros((max_n_nodes, max_n_nodes), device=dev, dtype=torch.long)
            extra_data.n_transfers = extra_data.route_nexts.clone()
            extra_data.context_node_covered_mask = torch.zeros((max_n_nodes,), device=dev, dtype=torch.bool)
            extra_data.context_edge_covered_mask = torch.zeros((max_n_nodes, max_n_nodes), device=dev, dtype=torch.bool)
            extra_data.context_leg_use_count = torch.zeros((max_n_nodes, max_n_nodes), device=dev, dtype=torch.long)
            extra_data.route_slot_context = torch.full((0, 0), -1, device=dev, dtype=torch.long)
            extra_data.active_route_idx = torch.tensor(-1, device=dev, dtype=torch.long)

            extra_data.norm_node_features = torch.zeros(
                (
                    dd.num_nodes,
                    0,
                ),
                device=dev,
            )

            extra_data.cost_weights = {}
            for key, val in cost_weights.items():
                # expand the cost weights to match the batch
                extra_data.cost_weights[key] = val[ii]

            extra_data.adjustment_target = torch.tensor(0.0, device=dev)
            extra_data.adjustment_weight = torch.tensor(0.0, device=dev)
            extra_data.adjustment_seed = torch.full((0, 0), -1, device=dev, dtype=torch.long)

            extra_datas.append(extra_data)
            if "fixed_routes" in dd and dd.fixed_routes.numel() > 0:
                fixed_routes = dd.fixed_routes
                if fixed_routes.ndim == 3 and fixed_routes.shape[0] == 1:
                    fixed_routes = fixed_routes.squeeze(0)
                elif fixed_routes.ndim == 1:
                    fixed_routes = fixed_routes[None]
                extra_data.fixed_routes = fixed_routes
            else:
                extra_data.fixed_routes = torch.zeros(0)

        self.extra_data = Batch.from_data_list(extra_datas)

        # this initializes route data
        self.clear_routes()

    def shortest_path_action(self, path_indices):
        routes_are_done = path_indices[:, 0] == -1
        path_seqs = self.get_shortest_path_sequences()
        new_parts = path_seqs[self.batch_indices, path_indices[:, 0], path_indices[:, 1]]
        new_len = self.current_routes.shape[-1]
        n_pad = max(new_len - new_parts.shape[-1], 0)
        new_parts = torch.nn.functional.pad(new_parts, (0, n_pad), value=-1)

        starting = (self.current_routes[:, 0] == -1) & ~routes_are_done
        extending = ~starting & ~routes_are_done
        ends_at_cur_start = path_indices[:, 1] == self.current_routes[:, 0]
        last_nodes = self.current_routes[self.batch_indices, self.current_route_n_stops - 1]
        starts_at_cur_end = path_indices[:, 0] == last_nodes
        valid_action = starting | ends_at_cur_start | starts_at_cur_end | routes_are_done
        assert valid_action.all(), "invalid action!"

        chose_prev = extending & ends_at_cur_start
        chose_next = extending & ~ends_at_cur_start

        first_parts = (
            new_parts * (starting | chose_prev)[:, None] + self.current_routes * (chose_next | routes_are_done)[:, None]
        )
        second_parts = (
            new_parts * chose_next[:, None]
            + self.current_routes * chose_prev[:, None]
            + -1 * ~(chose_next | chose_prev)[:, None]
        )
        # cut off the first node of the second parts, since they overlap
        second_parts = second_parts[..., 1:]
        # combine the two parts without dummy nodes in between
        updated_routes = torch.full((self.batch_size, self.max_n_nodes), -1, device=self.device)
        updated_routes[..., : first_parts.shape[-1]] = first_parts
        # insert the second parts at the appropriate masks
        first_part_lens = (first_parts > -1).sum(dim=-1)
        scnd_part_lens = (second_parts > -1).sum(dim=-1)
        new_route_lens = first_part_lens + scnd_part_lens
        scnd_part_mask = get_variable_slice_mask(updated_routes, dim=1, froms=first_part_lens, tos=new_route_lens)
        scnd_part_len_mask = get_variable_slice_mask(second_parts, dim=1, tos=scnd_part_lens)
        updated_routes[scnd_part_mask] = second_parts[scnd_part_len_mask]
        assert updated_routes.max() < self.max_n_nodes

        updated_routes = updated_routes.clamp(min=-1)

        updated_routes[self.is_done()] = -1

        # then, update the lists of routes that are still being planned
        planning_already_done = self.is_done()
        new_finished_context_routes = torch.full(
            (self.batch_size, 1, self.max_n_nodes), -1, dtype=torch.long, device=self.device
        )
        has_new_finished_route = False
        for bi in range(self.batch_size):
            if not planning_already_done[bi] and routes_are_done[bi]:
                route = updated_routes[bi]
                route = route.clone()[route > -1]
                if len(route) > 0:
                    # the route is valid, so add it to the finished set
                    # otherwise, corresponds to a "no-op" route
                    self._finished_routes[bi].append(route)
                    new_finished_context_routes[bi, 0, : len(route)] = route
                    has_new_finished_route = True
                    self.extra_data.total_route_time[bi] += self.current_route_time[bi]
            if planning_already_done[bi] or routes_are_done[bi]:
                updated_routes[bi] = -1

        if has_new_finished_route:
            self._add_routes_to_context_masks(new_finished_context_routes)

        self.extra_data.current_routes = updated_routes

        # finally, update all the internal stuff based on the current routes.
        # transforms it to a tensor if necessary
        self.extra_data.current_route_time = self.get_total_route_time(updated_routes)

        ncr_times_from_start = torch.zeros_like(self.current_routes, dtype=torch.float32)
        ncr_leg_times = get_route_leg_times(self.current_routes, self.drive_times, self.mean_stop_time)
        # keep zeros at the start for the first stop
        ncr_times_from_start[:, 1:] = ncr_leg_times.cumsum(dim=1)
        self.extra_data.current_route_times_from_start = ncr_times_from_start

        self._add_routes_to_tensors(updated_routes[:, None])

    def apply_route_actions(self, action_kinds, path_indices):
        """Apply typed route-planning actions.

        The legacy action representation only had terminal pairs plus
        ``(-1, -1)`` halt.  Trim actions can remove already-added edges, so we
        rebuild route tensors after them to avoid stale connectivity.
        """
        if action_kinds is None:
            self.shortest_path_action(path_indices)
            return

        action_kinds = action_kinds.to(device=self.device, dtype=torch.long)
        path_indices = path_indices.to(device=self.device, dtype=torch.long)
        if action_kinds.ndim != 1 or action_kinds.shape[0] != self.batch_size:
            raise ValueError("Expected action_kinds to have shape (batch_size,), got " f"{action_kinds.shape}")
        if path_indices.shape != (self.batch_size, 2):
            raise ValueError("Expected path_indices to have shape (batch_size, 2), got " f"{path_indices.shape}")

        is_halt = action_kinds == RouteActionKind.HALT
        uses_legacy_actions = (action_kinds == RouteActionKind.EXTEND) | is_halt
        if uses_legacy_actions.all():
            legacy_actions = path_indices.clone()
            legacy_actions[is_halt] = -1
            self.shortest_path_action(legacy_actions)
            return

        valid_action_kinds = (
            uses_legacy_actions | (action_kinds == RouteActionKind.TRIM_START) | (action_kinds == RouteActionKind.TRIM_END)
        )
        if not valid_action_kinds.all():
            raise ValueError("Unknown route action kind")

        updated_routes = self.current_routes.clone()
        planning_already_done = self.is_done()
        can_act = ~planning_already_done
        extend_mask = (action_kinds == RouteActionKind.EXTEND) & can_act
        trim_start_mask = (action_kinds == RouteActionKind.TRIM_START) & can_act
        trim_end_mask = (action_kinds == RouteActionKind.TRIM_END) & can_act
        halt_mask = is_halt & can_act

        if extend_mask.any():
            ext_routes = self._get_routes_after_shortest_path_actions(path_indices, active_mask=extend_mask)
            updated_routes[extend_mask] = ext_routes[extend_mask]

        if trim_start_mask.any() or trim_end_mask.any():
            self._apply_trim_actions_to_routes(updated_routes, path_indices, trim_start_mask, trim_end_mask)

        for bi in range(self.batch_size):
            if not planning_already_done[bi] and halt_mask[bi]:
                route = updated_routes[bi].clone()
                route = route[route > -1]
                if len(route) > 0:
                    self._finished_routes[bi].append(route)
            if planning_already_done[bi] or halt_mask[bi]:
                updated_routes[bi] = -1

        finished_routes = [[route.clone() for route in batch_routes] for batch_routes in self._finished_routes]
        self._replace_planned_routes(finished_routes, updated_routes)

    def _get_routes_after_shortest_path_actions(self, path_indices, active_mask=None):
        if active_mask is None:
            active_mask = torch.ones((self.batch_size,), dtype=torch.bool, device=self.device)

        path_seqs = self.get_shortest_path_sequences()
        safe_indices = path_indices.clamp(min=0)
        new_parts = path_seqs[self.batch_indices, safe_indices[:, 0], safe_indices[:, 1]]
        new_len = self.current_routes.shape[-1]
        n_pad = max(new_len - new_parts.shape[-1], 0)
        new_parts = torch.nn.functional.pad(new_parts, (0, n_pad), value=-1)

        starting = (self.current_routes[:, 0] == -1) & active_mask
        extending = ~starting & active_mask
        ends_at_cur_start = path_indices[:, 1] == self.current_routes[:, 0]
        last_nodes = self.current_routes[self.batch_indices, self.current_route_n_stops - 1]
        starts_at_cur_end = path_indices[:, 0] == last_nodes
        valid_action = ~active_mask | starting | ends_at_cur_start | starts_at_cur_end
        assert valid_action.all(), "invalid action!"

        chose_prev = extending & ends_at_cur_start
        chose_next = extending & ~ends_at_cur_start

        first_parts = (
            new_parts * (starting | chose_prev)[:, None] + self.current_routes * (chose_next | ~active_mask)[:, None]
        )
        second_parts = (
            new_parts * chose_next[:, None]
            + self.current_routes * chose_prev[:, None]
            + -1 * ~(chose_next | chose_prev)[:, None]
        )
        second_parts = second_parts[..., 1:]

        updated_routes = torch.full((self.batch_size, self.max_n_nodes), -1, device=self.device)
        updated_routes[..., : first_parts.shape[-1]] = first_parts
        first_part_lens = (first_parts > -1).sum(dim=-1)
        scnd_part_lens = (second_parts > -1).sum(dim=-1)
        new_route_lens = first_part_lens + scnd_part_lens
        scnd_part_mask = get_variable_slice_mask(updated_routes, dim=1, froms=first_part_lens, tos=new_route_lens)
        scnd_part_len_mask = get_variable_slice_mask(second_parts, dim=1, tos=scnd_part_lens)
        updated_routes[scnd_part_mask] = second_parts[scnd_part_len_mask]
        assert updated_routes.max() < self.max_n_nodes

        updated_routes = updated_routes.clamp(min=-1)
        updated_routes[~active_mask] = self.current_routes[~active_mask]
        return updated_routes

    def _apply_trim_actions_to_routes(self, updated_routes, path_indices, trim_start_mask, trim_end_mask):
        for bi in range(self.batch_size):
            if not (trim_start_mask[bi] or trim_end_mask[bi]):
                continue

            route = updated_routes[bi]
            route = route[route > -1]
            if len(route) == 0:
                raise ValueError("Cannot trim an empty current route")

            # Positional trim: path_indices[bi, 0] is the route POSITION index
            # (unambiguous even when a node repeats on the route).
            pos = int(path_indices[bi, 0].item())
            pos = max(0, min(pos, len(route) - 1))
            if trim_start_mask[bi]:
                trimmed = route[pos:]  # keep route[pos:]
            else:
                trimmed = route[: pos + 1]  # keep route[:pos+1]

            # Absolute floor: a route needs >= 2 stops to be a path. The model's
            # trim-candidate mask enforces min_route_len unless below-min trims
            # are explicitly enabled, so this guard only trips on degenerate
            # (<2 stop) trims.
            if len(trimmed) < 2:
                raise ValueError("Trim action would leave fewer than 2 stops")

            updated_routes[bi] = -1
            updated_routes[bi, : len(trimmed)] = trimmed

    def _replace_planned_routes(self, finished_routes, current_routes):
        # one transit-data rebuild for the whole replace + re-seed sequence
        with self.defer_route_data_update():
            self._clear_routes_helper()
            if self.extra_data.fixed_routes.numel() > 0:
                self._add_routes_to_tensors(self.extra_data.fixed_routes)
                self._add_routes_to_context_masks(self.extra_data.fixed_routes)

            has_finished_routes = any(len(routes) > 0 for routes in finished_routes)
            if has_finished_routes:
                finished_tensor = to_routes(finished_routes).to(self.device)
                self.add_new_routes(finished_tensor)
            else:
                self._update_route_data()

            if current_routes.device != self.device:
                current_routes = current_routes.to(self.device)
            if (current_routes > -1).any():
                self.set_current_routes(current_routes)

    def _clear_routes_helper(self, batch_index=None):
        if batch_index is None:
            batch_index = self.batch_indices
        elif type(batch_index) is not Tensor:
            batch_index = torch.tensor(batch_index, device=self.device)

        for bi in batch_index:
            self._finished_routes[bi] = []

        self.extra_data.valid_terms_mat[batch_index] = self.extra_data.base_valid_terms_mat[batch_index].clone()
        self.extra_data.total_route_time[batch_index] = 0

        directly_connected = torch.eye(self.max_n_nodes, device=self.device, dtype=bool)
        self.extra_data.directly_connected[batch_index] = directly_connected.expand(len(batch_index), -1, -1)
        transit_times = 1 - torch.eye(self.max_n_nodes, device=self.device)
        transit_times[transit_times > 0] = float("inf")
        self.extra_data.route_mat[batch_index] = transit_times
        self.extra_data.transit_times[batch_index] = transit_times
        self.extra_data.has_path[batch_index] = self.directly_connected[batch_index]
        self.extra_data.current_routes[batch_index] = -1
        self.extra_data.current_route_time[batch_index] = 0
        self.extra_data.current_route_times_from_start[batch_index] = 0
        self.extra_data.route_nexts[batch_index] = 0
        self.extra_data.n_transfers[batch_index] = 0
        self.extra_data.context_node_covered_mask[batch_index] = False
        self.extra_data.context_edge_covered_mask[batch_index] = False
        self.extra_data.context_leg_use_count[batch_index] = 0

    def _add_routes_to_tensors(
        self, batch_new_routes, only_routes_with_demand_are_valid=False, invalid_directly_connected=False
    ):
        """Takes a tensor of new routes. The first dimension is the batch"""
        # incorporate new routes into the route graphs
        batch_new_routes = self._prepare_batch_route_tensor(batch_new_routes)
        # add new routes to the route matrix.
        new_route_mat = get_route_edge_matrix(
            batch_new_routes, self.drive_times, self.mean_stop_time, self.symmetric_routes
        )
        self.extra_data.route_mat = torch.minimum(self.route_mat, new_route_mat)

        self.directly_connected[self.route_mat < float("inf")] = True

        # allow connection to any node 'upstream' of a demand dest, or
        # 'downstream' of a demand src.
        if only_routes_with_demand_are_valid:
            float_is_demand = (self.demand > 0).to(torch.float32)
            connected_T = self.nodes_are_connected(2).transpose(1, 2)
            connected_T = connected_T.to(torch.float32)
            valid_upstream = float_is_demand.bmm(connected_T)
            self.extra_data.valid_terms_mat[valid_upstream.to(bool)] = True
            valid_downstream = connected_T.bmm(float_is_demand)
            self.extra_data.valid_terms_mat[valid_downstream.to(bool)] = True

        if invalid_directly_connected:
            self.extra_data.valid_terms_mat[self.directly_connected] = False

        if self.symmetric_routes:
            self.extra_data.valid_terms_mat = self.valid_terms_mat & self.valid_terms_mat.transpose(1, 2)

        self._update_route_data()

    def _add_routes_to_context_masks(self, batch_routes):
        """Track nodes/edges covered by finished or fixed context routes."""
        batch_routes = self._prepare_batch_route_tensor(batch_routes)
        if batch_routes.numel() == 0:
            return

        valid_nodes = batch_routes >= 0
        if valid_nodes.any():
            batch_idxs = torch.arange(self.batch_size, device=self.device)[:, None, None]
            batch_idxs = batch_idxs.expand_as(batch_routes)
            safe_nodes = batch_routes.clamp(min=0)
            self.extra_data.context_node_covered_mask[
                batch_idxs[valid_nodes],
                safe_nodes[valid_nodes],
            ] = True

        if batch_routes.shape[-1] < 2:
            return

        from_nodes = batch_routes[..., :-1]
        to_nodes = batch_routes[..., 1:]
        valid_edges = (from_nodes >= 0) & (to_nodes >= 0)
        if not valid_edges.any():
            return

        edge_batch_idxs = torch.arange(self.batch_size, device=self.device)[:, None, None]
        edge_batch_idxs = edge_batch_idxs.expand_as(from_nodes)
        safe_from = from_nodes.clamp(min=0)
        safe_to = to_nodes.clamp(min=0)
        self.extra_data.context_edge_covered_mask[
            edge_batch_idxs[valid_edges],
            safe_from[valid_edges],
            safe_to[valid_edges],
        ] = True
        self.extra_data.context_leg_use_count.index_put_(
            (
                edge_batch_idxs[valid_edges],
                safe_from[valid_edges],
                safe_to[valid_edges],
            ),
            torch.ones_like(safe_from[valid_edges]),
            accumulate=True,
        )
        if self.symmetric_routes:
            non_loop_edges = valid_edges & (safe_from != safe_to)
            self.extra_data.context_edge_covered_mask[
                edge_batch_idxs[non_loop_edges],
                safe_to[non_loop_edges],
                safe_from[non_loop_edges],
            ] = True
            self.extra_data.context_leg_use_count.index_put_(
                (
                    edge_batch_idxs[non_loop_edges],
                    safe_to[non_loop_edges],
                    safe_from[non_loop_edges],
                ),
                torch.ones_like(safe_to[non_loop_edges]),
                accumulate=True,
            )

    def replace_routes(
        self, batch_new_routes, only_routes_with_demand_are_valid=False, invalid_directly_connected=False
    ):
        with self.defer_route_data_update():
            self._clear_routes_helper()
            if self.extra_data.fixed_routes.numel() > 0:
                self._add_routes_to_tensors(self.extra_data.fixed_routes)
                self._add_routes_to_context_masks(self.extra_data.fixed_routes)
            self.add_new_routes(batch_new_routes, only_routes_with_demand_are_valid, invalid_directly_connected)

    def clear_routes(self):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
            self._add_routes_to_context_masks(self.extra_data.fixed_routes)
        self._update_route_data()

    def reset_dones(self):
        batch_index = torch.where(self.is_done())[0]
        if len(batch_index) > 0:
            self._clear_routes_helper(batch_index)

    def set_current_routes(self, batch_current_routes):
        """Seed the state with an in-progress route for each batch element."""
        if self.has_current_route.any():
            raise RuntimeError("Cannot seed current routes when the state already has an " "active route")

        if type(batch_current_routes) is list:
            is_single_route = (
                len(batch_current_routes) == 0
                or isinstance(batch_current_routes[0], (int, np.integer))
                or (isinstance(batch_current_routes[0], Tensor) and batch_current_routes[0].ndim == 0)
            )
            if is_single_route:
                batch_current_routes = [batch_current_routes]
            batch_current_routes = to_routes([[route] for route in batch_current_routes])[:, 0].to(self.device)

        if batch_current_routes.ndim == 1:
            batch_current_routes = batch_current_routes.unsqueeze(0)
        elif batch_current_routes.ndim == 3 and batch_current_routes.shape[1] == 1:
            batch_current_routes = batch_current_routes.squeeze(1)
        elif batch_current_routes.ndim != 2:
            raise ValueError(
                "Expected current routes to have shape " f"(batch, max_route_len), got {batch_current_routes.shape}"
            )

        if batch_current_routes.shape[0] == 1 and self.batch_size > 1:
            batch_current_routes = batch_current_routes.expand(self.batch_size, -1)
        elif batch_current_routes.shape[0] != self.batch_size:
            raise ValueError(
                "Current-route batch size does not match state batch size: "
                f"{batch_current_routes.shape[0]} vs {self.batch_size}"
            )

        batch_current_routes = to_routes(batch_current_routes, width=self.max_n_nodes)[0].to(self.device)
        invalid_nodes = (batch_current_routes < -1) | (batch_current_routes >= self.max_n_nodes)
        invalid_nodes &= batch_current_routes != -1
        if invalid_nodes.any():
            raise ValueError("Current routes contain invalid node indices")

        self.extra_data.current_routes = batch_current_routes
        self.extra_data.current_route_time = self.get_total_route_time(batch_current_routes)

        ncr_times_from_start = torch.zeros_like(self.current_routes, dtype=torch.float32)
        ncr_leg_times = get_route_leg_times(self.current_routes, self.drive_times, self.mean_stop_time)
        ncr_times_from_start[:, 1:] = ncr_leg_times.cumsum(dim=1)
        self.extra_data.current_route_times_from_start = ncr_times_from_start

        self._add_routes_to_tensors(batch_current_routes[:, None])

    def add_new_routes(
        self, batch_new_routes, only_routes_with_demand_are_valid=False, invalid_directly_connected=False
    ):
        batch_new_routes = self._prepare_batch_route_tensor(batch_new_routes)

        self._add_routes_to_tensors(batch_new_routes, only_routes_with_demand_are_valid, invalid_directly_connected)
        self._add_routes_to_context_masks(batch_new_routes)
        self._add_routes_to_list(batch_new_routes)

        # finally, update the total route times with the new routes
        leg_times = get_route_leg_times(batch_new_routes, self.graph_data.drive_times, self.mean_stop_time)
        total_new_time = leg_times.sum(dim=(1, 2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_new_routes, transpose_dtm, self.mean_stop_time)
            total_new_time += return_leg_times.sum(dim=(1, 2))

        self.extra_data.total_route_time += total_new_time

    def _prepare_batch_route_tensor(self, batch_routes):
        if isinstance(batch_routes, torch.Tensor) and batch_routes.ndim == 2 and self.batch_size > 1:
            batch_routes = batch_routes[:, None]
        batch_routes = to_routes(batch_routes).to(self.device)
        if batch_routes.shape[0] == 1 and self.batch_size > 1:
            batch_routes = batch_routes.expand(self.batch_size, -1, -1)
        elif batch_routes.shape[0] != self.batch_size:
            raise ValueError(
                "Route batch size does not match state batch size: " f"{batch_routes.shape[0]} vs {self.batch_size}"
            )
        return batch_routes

    def _add_routes_to_list(self, batch_routes):
        for bi in range(self.batch_size):
            for route in batch_routes[bi]:
                if type(route) is list:
                    route = torch.tensor(route, device=self.device)
                length = (route > -1).sum()
                if length == 0:
                    # empty placeholder route, e.g. an unfilled slot in a
                    # partially initialized network
                    continue
                if length < 2:
                    # A route with a single stop is degenerate: drop it from
                    # the finished-route list (the cost objective scores the
                    # resulting network as constraint-violating). This happens
                    # routinely as a transient while heuristic search (BCO /
                    # SA / GA / HH / NSGA-II) explores candidate networks, so
                    # it is logged at debug level rather than spamming WARNING.
                    log.debug("dropping degenerate route with fewer than 2 stops")
                    continue
                self._finished_routes[bi].append(route[:length])

    @contextmanager
    def defer_route_data_update(self):
        """Batch several route-tensor mutations into ONE _update_route_data.

        Several mutation paths (replace + seed current route) rebuild the
        transit data twice back-to-back although nothing reads the
        intermediate result. _update_route_data is a pure function of
        route_mat / transfer times, so running it once after the last
        mutation yields exactly the tensors the eager per-mutation updates
        would have produced -- bit-identical results, half the Floyd-Warshall
        work on those paths.
        """
        if self._route_update_deferred:
            yield  # already inside an outer deferral; its exit will update
            return
        self._route_update_deferred = True
        try:
            yield
        finally:
            self._route_update_deferred = False
            self._update_route_data()

    def _update_route_data(self):
        if self._route_update_deferred:
            return
        # do things that have to be done whether we added or removed routes
        fw_mat = self.route_mat + self.transfer_time_s[:, None, None]
        nexts, transit_times = floyd_warshall(fw_mat)
        # node counts of the shortest paths; identical to
        # reconstruct_all_paths(nexts)[1] but without materializing the full
        # (B, N, N, max_len) path tensor (only the counts are needed here).
        path_edge_counts = count_path_nodes(nexts)
        # subtract one transfer time from each transit time to avoid counting
        # a transfer for the first edge of a journey
        transit_times -= self.transfer_time_s[:, None, None]

        self.extra_data.route_nexts = nexts
        self.extra_data.transit_times = transit_times
        self.extra_data.has_path = transit_times < float("inf")
        # number of transfers is number of nodes except start and end
        n_transfers = (path_edge_counts - 2).clamp(min=0)
        # set number of transfers where there is no path to 0
        n_transfers[~self.has_path] = 0
        self.extra_data.n_transfers = n_transfers

    def set_normalized_features(self, norm_stop_features):
        self.extra_data.norm_node_features = norm_stop_features

    def clone(self):
        """return a deep copy of this state."""
        return copy.deepcopy(self)

    def to_device(self, device):
        dev_state = self.clone()
        dev_state.graph_data = dev_state.graph_data.to(device)
        dev_state.extra_data = dev_state.extra_data.to(device)
        for ii in range(self.batch_size):
            for jj, route in enumerate(dev_state._finished_routes[ii]):
                dev_state._finished_routes[ii][jj] = route.to(device)

        return dev_state

    def snapshot_for_buffer(self, device=None):
        """Lean snapshot suitable for storing in a rollout buffer.

        Same-device path shares ``graph_data`` (which is read-only during
        rollouts) and deep-copies only ``extra_data`` plus the per-element
        finished-route lists, avoiding the GPU peak of a full deepcopy.

        Cross-device path deep-copies once on the source device and then
        moves the copy in place to the target device, replacing the
        previous ``clone() + to_device()`` pattern that did two deepcopies.

        The shortest-path sequence tensor is a lazy cache. Buffer consumers
        recompute it as needed, so omit it from the copy to avoid transferring
        a large padded tensor only to clear it immediately afterward.
        """
        target = self.device if device is None else torch.device(device)
        shortest_paths = self.extra_data.shortest_path_sequences
        self.extra_data.shortest_path_sequences = shortest_paths.new_empty((self.batch_size, 0, 0, 0))
        try:
            if target == self.device:
                new = copy.copy(self)
                new.graph_data = self.graph_data
                new.extra_data = copy.deepcopy(self.extra_data)
                new._finished_routes = [list(routes) for routes in self._finished_routes]
                return new

            new = copy.deepcopy(self)
        finally:
            self.extra_data.shortest_path_sequences = shortest_paths

        new.graph_data = new.graph_data.to(target)
        new.extra_data = new.extra_data.to(target)
        new._finished_routes = [[route.to(target) for route in routes] for routes in new._finished_routes]
        return new

    @staticmethod
    def batch_from_list(state_list):
        """return a batch state from a list of states."""
        if len(state_list) == 1:
            return state_list[0]

        graph_datas = sum([ss.graph_data.to_data_list() for ss in state_list], [])
        extra_datas = sum([ss.extra_data.to_data_list() for ss in state_list], [])
        batch_graph_data = Batch.from_data_list(graph_datas)
        batch_extra_data = Batch.from_data_list(extra_datas)
        batch_state = copy.copy(state_list[0])
        batch_state.graph_data = batch_graph_data
        batch_state.extra_data = batch_extra_data
        copied_routes = [[copy.copy(bfr) for bfr in ss._finished_routes] for ss in state_list]
        batch_state._finished_routes = sum(copied_routes, [])
        return batch_state

    def batch_to_list(self):
        """return a list of RouteGenBatchState objects, one for each element in
        this batch."""
        if self.batch_size == 1:
            return [self]

        graph_datas = self.graph_data.to_data_list()
        extra_datas = self.extra_data.to_data_list()
        state_list = []
        for gd, ed, routes in zip(graph_datas, extra_datas, self._finished_routes):
            state = copy.copy(self)
            state.graph_data = Batch.from_data_list([gd])
            state.extra_data = Batch.from_data_list([ed])
            state._finished_routes = [copy.copy(routes)]
            state_list.append(state)

        return state_list

    def index_select(self, idx: Union[slice, Tensor, ndarray, Sequence]):
        """return a new state with only the given indices."""
        state = copy.copy(self)
        graph_data = self.graph_data.index_select(idx)
        extra_data = self.extra_data.index_select(idx)
        if isinstance(graph_data, list):
            graph_data = Batch.from_data_list(graph_data)
        if isinstance(extra_data, list):
            extra_data = Batch.from_data_list(extra_data)
        state.graph_data = graph_data
        state.extra_data = extra_data
        if isinstance(idx, slice):
            state._finished_routes = self._finished_routes[idx]
        else:
            if isinstance(idx, Tensor):
                idx = idx.detach().cpu().tolist()
            state._finished_routes = [self._finished_routes[ii] for ii in idx]
        return state

    def is_done(self):
        return self.n_routes_left_to_plan == 0

    def get_total_route_time(self, batch_routes):
        batch_routes = self._prepare_batch_route_tensor(batch_routes)

        leg_times = get_route_leg_times(batch_routes, self.graph_data.drive_times, self.mean_stop_time)
        route_time = leg_times.sum(dim=(1, 2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_routes, transpose_dtm, self.mean_stop_time)
            route_time += return_leg_times.sum(dim=(1, 2))

        return route_time

    def _count_route_legs(self, batch_routes):
        """Count adjacent stop-to-stop legs in a batch of route tensors."""
        batch_routes = self._prepare_batch_route_tensor(batch_routes)

        counts = torch.zeros(
            (self.batch_size, self.max_n_nodes, self.max_n_nodes),
            device=self.device,
            dtype=torch.long,
        )
        if batch_routes.shape[-1] < 2:
            return counts

        routes = batch_routes.to(device=self.device, dtype=torch.long)
        from_nodes = routes[..., :-1]
        to_nodes = routes[..., 1:]
        valid = (from_nodes >= 0) & (to_nodes >= 0)
        if not valid.any():
            return counts

        batch_idxs = torch.arange(self.batch_size, device=self.device)[:, None, None]
        batch_idxs = batch_idxs.expand_as(from_nodes)
        safe_from = from_nodes.clamp(min=0)
        safe_to = to_nodes.clamp(min=0)
        counts.index_put_(
            (
                batch_idxs[valid],
                safe_from[valid],
                safe_to[valid],
            ),
            torch.ones_like(safe_from[valid]),
            accumulate=True,
        )
        if self.symmetric_routes:
            non_loop_edges = valid & (safe_from != safe_to)
            counts.index_put_(
                (
                    batch_idxs[non_loop_edges],
                    safe_to[non_loop_edges],
                    safe_from[non_loop_edges],
                ),
                torch.ones_like(safe_to[non_loop_edges]),
                accumulate=True,
            )
        return counts

    def get_n_disconnected_demand_edges(self):
        # count the number of demand edges that are disconnected
        nopath = ~self.has_path
        needed_path_missing = nopath & (self.demand > 0)
        n_disconnected_demand_edges = needed_path_missing.sum(dim=(1, 2))
        if self.symmetric_routes:
            # connecting one of the two connects both, so count each 2 as
            # just 1.
            n_disconnected_demand_edges = n_disconnected_demand_edges / 2
        return n_disconnected_demand_edges

    def get_shortest_path_sequences(self):
        if self.extra_data.shortest_path_sequences.numel() == 0:
            path_seqs, _ = reconstruct_all_paths(self.nexts)
            self.extra_data.shortest_path_sequences = path_seqs
        else:
            path_seqs = self.extra_data.shortest_path_sequences
        return path_seqs

    def set_cost_weights(self, new_cost_weights):
        for key, val in new_cost_weights.items():
            # expand the cost weights to match the batch
            if type(val) is not Tensor:
                val = torch.tensor(val, device=self.device)
            else:
                val = val.to(self.device)
            if val.ndim == 0:
                val = val[None]
            if val.numel() == 1 and self.batch_size > 1:
                val = val.expand(self.batch_size)
            self.extra_data.cost_weights[key][...] = val

    @property
    def routes(self):
        """Returns the collection of all routes, including the one currently
        being planned if it has any stops."""
        # copy the lists of routes for each batch element.
        routes = [copy.copy(fr) for fr in self._finished_routes]
        if self.current_routes is not None:
            for batch_idx, current_route in enumerate(self.current_routes):
                if (current_route > -1).sum() > 1:
                    # the route being planned has at least 2 stops, so it is
                    # functional.  Add it.
                    current_route = current_route[current_route > -1]
                    routes[batch_idx].append(current_route)

        return routes

    @property
    def batch_indices(self):
        return torch.arange(self.batch_size, device=self.device)

    @property
    def n_demand_edges(self):
        n_demand_edges = (self.demand > 0).sum(dim=(1, 2))
        if self.symmetric_routes:
            n_demand_edges = (n_demand_edges / 2).ceil()
        return n_demand_edges

    def get_cost_weights_tensor(self, key_order, normalize=False):
        """THE single point where the named cost-weight dict becomes a
        positional vector. ``key_order`` is required: every caller states
        which keys it wants and in which order — the cost/reward math passes
        COST_WEIGHT_KEY_ORDER, models pass their own checkpoint-locked
        ``preference_goals`` spec. There is deliberately no default and no
        implicit ordering (dict iteration / sorting never decides slots)."""
        cost_weights_list = []
        for key in key_order:
            if key not in self.cost_weights:
                raise KeyError(f"State does not contain cost weight {key!r}")
            if type(self.cost_weights[key]) is Tensor:
                cw = self.cost_weights[key].to(self.device)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(self.cost_weights[key], device=self.device)[None]
            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(self.batch_size, -1)
        if cost_weights.shape[0] > self.batch_size:
            cost_weights = cost_weights[: self.batch_size]
        if normalize:
            denom = cost_weights.sum(dim=-1, keepdim=True).clamp_min(EPSILON)
            cost_weights = cost_weights / denom
        return cost_weights

    @property
    def node_covered_mask(self):
        have_out_paths = self.directly_connected.any(dim=1)
        if self.symmetric_routes:
            are_covered = have_out_paths
        else:
            are_covered = have_out_paths & self.directly_connected.any(dim=2)
        return are_covered

    # don't expose the extra data directly, just provide this interface.
    @property
    def norm_node_features(self):
        if "norm_node_features" in self.extra_data:
            return self.extra_data.norm_node_features
        else:
            return None

    @property
    def norm_cost_weights(self):
        if "norm_cost_weights" in self.extra_data:
            return self.extra_data.norm_cost_weights
        else:
            return None

    @property
    def current_routes(self):
        return self.extra_data.current_routes

    @property
    def current_route_times_from_start(self):
        return self.extra_data.current_route_times_from_start

    @property
    def current_route_time(self):
        return self.extra_data.current_route_time

    @property
    def current_route_n_stops(self):
        return (self.current_routes > -1).sum(dim=-1)

    @property
    def has_current_route(self):
        return self.current_route_n_stops > 0

    @property
    def n_routes_to_plan(self):
        return self.extra_data.n_routes_to_plan

    @property
    def valid_terms_mat(self):
        return self.extra_data.valid_terms_mat

    @property
    def mean_stop_time(self):
        return self.extra_data.mean_stop_time

    @property
    def transfer_time_s(self):
        return self.extra_data.transfer_time_s

    @property
    def total_route_time(self):
        # time of finished routes plus time of in-progress routes
        return self.extra_data.total_route_time + self.current_route_time

    @property
    def min_route_len(self):
        return self.extra_data.min_route_len

    @property
    def max_route_len(self):
        return self.extra_data.max_route_len

    @property
    def n_nodes(self):
        return self.extra_data.n_nodes_in_scenario

    @property
    def alpha(self):
        return self.extra_data.cost_weights["demand_time_weight"]

    @property
    def cost_weights(self):
        return self.extra_data.cost_weights

    @property
    def adjustment_target(self):
        """Per-graph adjustment-degree target."""
        return self.extra_data.adjustment_target

    @property
    def adjustment_weight(self):
        """Per-graph adjustment-penalty weight."""
        return self.extra_data.adjustment_weight

    @property
    def adjustment_seed(self):
        """Per-graph seed routes used to compute the live current adj."""
        return self.extra_data.adjustment_seed

    def set_adjustment_seed(self, routes):
        """Set the objective reference without enabling policy conditioning."""
        seed = to_routes(routes).to(self.device)
        if seed.shape[0] == 1 and self.batch_size > 1:
            seed = seed.expand(self.batch_size, -1, -1)
        if seed.shape[0] != self.batch_size:
            raise ValueError("adjustment seed batch size differs from state")
        self.extra_data.adjustment_seed = seed.clone()

    def adjustment_routes(self):
        """Read the current network in reference-slot order; preserve every stop."""
        if self.route_slot_context.numel():
            width = max(self.route_slot_context.shape[-1], self.current_routes.shape[-1])
            routes = to_routes(self.route_slot_context, width=width).clone()
            active = self.active_route_idx >= 0
            indices = self.batch_indices[active]
            current = torch.nn.functional.pad(self.current_routes, (0, width - self.current_routes.shape[-1]), value=-1)
            context_count = (self.route_slot_context >= 0).any(-1).sum(-1)
            context_count = context_count - (self.route_slot_context[self.batch_indices, self.active_route_idx.clamp_min(0)] >= 0).any(-1).long()
            finalized = self.n_finished_routes > context_count
            finished = to_routes(self._finished_routes, n_routes=int(self.n_routes_to_plan.max()), width=width).to(self.device)
            last_finished = finished[self.batch_indices, (self.n_finished_routes.long() - 1).clamp_min(0)]
            current = torch.where(finalized[:, None], last_finished, current)
            routes[indices, self.active_route_idx[active]] = current[active]
            return routes
        routes = to_routes(self._finished_routes, n_routes=int(self.n_routes_to_plan.max()), width=self.current_routes.shape[-1]).to(self.device)
        active = (self.current_routes >= 0).any(-1) & (self.n_finished_routes < self.n_routes_to_plan)
        routes[self.batch_indices[active], self.n_finished_routes[active].long()] = self.current_routes[active]
        return routes

    def set_adjustment_conditioning(self, target, weight=None, seed_routes=None):
        """Store conditioning values; the feature set selects their columns."""
        bs = self.batch_size
        dev = self.device
        tt = torch.as_tensor(target, device=dev, dtype=torch.float32)
        if tt.numel() == 1:
            tt = tt.expand(bs)
        self.extra_data.adjustment_target = tt.reshape(bs).clone()
        if weight is None:
            self.extra_data.adjustment_weight = torch.zeros(bs, device=dev)
        else:
            ww = torch.as_tensor(weight, device=dev, dtype=torch.float32)
            if ww.numel() == 1:
                ww = ww.expand(bs)
            self.extra_data.adjustment_weight = ww.reshape(bs).clone()
        if seed_routes is not None:
            sr = torch.as_tensor(seed_routes, device=dev, dtype=torch.long)
            if sr.dim() == 2:  # [n_routes, L] -> broadcast to batch
                sr = sr.unsqueeze(0).expand(bs, -1, -1)
            self.extra_data.adjustment_seed = sr.reshape(bs, sr.shape[-2], sr.shape[-1]).clone()

    def set_route_slot_context(self, route_batch, active_route_idx):
        """Keep route-slot ordering for closed-loop adjustment conditioning."""
        routes = torch.as_tensor(route_batch, device=self.device, dtype=torch.long)
        if routes.ndim == 2:
            routes = routes.unsqueeze(0)
        if routes.shape[0] == 1 and self.batch_size > 1:
            routes = routes.expand(self.batch_size, -1, -1)
        elif routes.shape[0] != self.batch_size:
            raise ValueError(
                "Route-slot context batch size does not match state batch "
                f"size: {routes.shape[0]} vs {self.batch_size}"
            )
        idx = torch.as_tensor(active_route_idx, device=self.device, dtype=torch.long)
        if idx.numel() == 1:
            idx = idx.expand(self.batch_size)
        elif idx.numel() != self.batch_size:
            raise ValueError("active_route_idx must be scalar or have one value per batch")
        self.extra_data.route_slot_context = routes.clone()
        self.extra_data.active_route_idx = idx.reshape(self.batch_size).clone()

    @property
    def directly_connected(self):
        return self.extra_data.directly_connected

    @property
    def nexts(self):
        return self.graph_data.nexts

    @property
    def route_mat(self):
        return self.extra_data.route_mat

    @property
    def route_nexts(self):
        return self.extra_data.route_nexts

    @property
    def transit_times(self):
        return self.extra_data.transit_times

    @property
    def has_path(self):
        return self.extra_data.has_path

    @property
    def n_transfers(self):
        return self.extra_data.n_transfers

    @property
    def context_node_covered_mask(self):
        return self.extra_data.context_node_covered_mask

    @property
    def context_edge_covered_mask(self):
        return self.extra_data.context_edge_covered_mask

    @property
    def context_leg_use_count(self):
        if "context_leg_use_count" in self.extra_data:
            return self.extra_data.context_leg_use_count
        return self.context_edge_covered_mask.to(dtype=torch.long)

    @property
    def route_slot_context(self):
        return self.extra_data.route_slot_context

    @property
    def active_route_idx(self):
        return self.extra_data.active_route_idx

    @property
    def current_leg_use_count(self):
        return self._count_route_legs(self.current_routes)

    @property
    def total_leg_use_count(self):
        return self.context_leg_use_count + self.current_leg_use_count

    @property
    def excess_leg_use_count(self):
        return (self.total_leg_use_count - 1).clamp_min(0)

    @property
    def street_adj(self):
        return self.graph_data.street_adj

    @property
    def demand(self):
        return self.graph_data.demand

    @property
    def drive_times(self):
        return self.graph_data.drive_times

    @property
    def n_finished_routes(self):
        nrsf = [len(rrs) for rrs in self._finished_routes]
        return torch.tensor(nrsf, dtype=torch.float32, device=self.device)

    @property
    def batch_size(self):
        return self.graph_data.num_graphs

    @property
    def n_routes_left_to_plan(self):
        return self.n_routes_to_plan - self.n_finished_routes

    def get_n_routes_features(self):
        so_far = self.n_finished_routes
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        return (both + 1).log()

    def nodes_are_connected(self, n_transfers=2):
        dircon_float = self.directly_connected.to(torch.float32)
        connected = dircon_float
        for _ in range(n_transfers):
            # connected by 2 or fewer transfers
            connected = connected.bmm(dircon_float)
        return connected.bool()

    @property
    def device(self):
        return self.graph_data[STOP_KEY].x.device

    @property
    def max_n_nodes(self):
        return max(self.n_nodes)

    @property
    def route_n_stops(self):
        route_n_stops = [[len(rr) for rr in br] for br in self.routes]
        return torch.tensor(route_n_stops, device=self.device)

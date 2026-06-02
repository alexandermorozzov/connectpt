import logging as log
import copy
import math
from collections import deque
from typing import Union
from collections.abc import Sequence

from numpy import ndarray
import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData
import networkx as nx
from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig

from .citygraph_dataset import STOP_KEY
from . import torch_utils as tu

import geopandas as gpd
import numpy as np


MEAN_STOP_TIME_S = 0
AVG_TRANSFER_WAIT_TIME_S = 300
UNSAT_PENALTY_EXTRA_S = 3000
EPSILON = 1e-6

ROUTE_ACTION_EXTEND = 0
ROUTE_ACTION_TRIM_START = 1
ROUTE_ACTION_TRIM_END = 2
ROUTE_ACTION_HALT = 3

COST_WEIGHT_KEY_ORDER = (
    'demand_time_weight',
    'route_time_weight',
    'median_connectivity_weight',
)

# Short, human-readable names for the three cost components, index-aligned
# with COST_WEIGHT_KEY_ORDER. Used to enable / disable individual components.
COST_COMPONENT_NAMES = ('demand', 'route', 'connectivity')

# Accepted spellings for each cost component, mapped to its index. Lets
# configs / notebooks refer to a component by short name, by its weight-key
# name, or by a couple of common aliases (ATT / RTT).
_COST_COMPONENT_ALIASES = {
    'demand': 0, 'demand_time_weight': 0, 'demand_cost': 0,
    'demand_time': 0, 'att': 0,
    'route': 1, 'route_time_weight': 1, 'route_cost': 1,
    'route_time': 1, 'rtt': 1,
    'connectivity': 2, 'median_connectivity_weight': 2,
    'median_connectivity': 2, 'connectivity_cost': 2,
}


def resolve_cost_component_index(item):
    """Map a cost-component identifier to its index in
    ``COST_WEIGHT_KEY_ORDER`` (0=demand, 1=route, 2=connectivity).

    Accepts an integer index, a short name (``demand`` / ``route`` /
    ``connectivity``), a weight-key name, or an ATT/RTT alias.
    """
    if isinstance(item, bool):
        raise TypeError(f"invalid cost-component identifier: {item!r}")
    if isinstance(item, (int, np.integer)):
        idx = int(item)
        if not 0 <= idx < len(COST_COMPONENT_NAMES):
            raise ValueError(f"cost-component index out of range: {idx}")
        return idx
    key = str(item).strip().lower()
    if key not in _COST_COMPONENT_ALIASES:
        raise ValueError(
            f"unknown cost component {item!r}; expected one of "
            f"{COST_COMPONENT_NAMES} (or a weight-key / ATT-RTT alias)")
    return _COST_COMPONENT_ALIASES[key]


def resolve_enabled_cost_components(enabled_components=None,
                                    disabled_components=None):
    """Return a length-3 tuple of bools (index-aligned with
    ``COST_COMPONENT_NAMES``) describing which cost components are active.

    Pass at most one of ``enabled_components`` / ``disabled_components``;
    ``None`` for both keeps all three components enabled — the default,
    fully backward-compatible behaviour.
    """
    if enabled_components is not None and disabled_components is not None:
        raise ValueError(
            "pass only one of enabled_components / disabled_components")
    if enabled_components is None and disabled_components is None:
        return (True, True, True)
    if disabled_components is not None:
        items = disabled_components
        if isinstance(items, (str, int)):
            items = [items]
        mask = [True, True, True]
        for item in items:
            mask[resolve_cost_component_index(item)] = False
    else:
        items = enabled_components
        if isinstance(items, (str, int)):
            items = [items]
        mask = [False, False, False]
        for item in items:
            mask[resolve_cost_component_index(item)] = True
    if not any(mask):
        raise ValueError("at least one cost component must stay enabled")
    return tuple(mask)


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
        if key in ['base_valid_terms_mat', 
                   'valid_terms_mat',
                   'mean_stop_time',
                   'fixed_routes_file',
                   'transfer_time_s',
                   'total_route_time', 
                   'n_routes_to_plan', 
                   'min_route_len',
                   'max_route_len', 
                   'n_nodes_in_scenario',
                   'directly_connected', 
                   'route_mat',
                   'transit_times',
                   'has_path', 
                   'current_routes',
                   'current_route_time',
                   'current_route_times_from_start',
                   'shortest_path_sequences',
                   'route_nexts',
                   'n_transfers',
                   'context_node_covered_mask',
                   'context_edge_covered_mask',
                   'context_leg_use_count',
                   'route_slot_context',
                   'active_route_idx',
                   'fixed_routes',
                   'node_coords',
                   'adjustment_target',
                   'adjustment_weight',
                   'adjustment_use_current',
                   'adjustment_seed']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class RouteGenBatchState:
    def __init__(self, graph_data, cost_obj, n_routes_to_plan, min_route_len=2,
                 max_route_len=None, valid_terms_mat=None, cost_weights=None):
        # do initialization needed to make properties work
        if not isinstance(graph_data, Batch):
            if not isinstance(graph_data, list):
                graph_data = [graph_data]
            graph_data = Batch.from_data_list(graph_data)

        # set members that live directly on this object
        self.graph_data = graph_data
        # right now this must have the same value for all scenarios
        self.symmetric_routes = cost_obj.symmetric_routes
        self._finished_routes = [[] for _ in range(graph_data.num_graphs)]
        self._redundancy_features_enabled = False
        self._adjustment_gap = 0.1
        self._adjustment_mode = 'paper'

        # the object isn't ready to give this property yet, so find it here
        dev = graph_data[STOP_KEY].x.device
        max_n_nodes = max([dd.num_nodes for dd in graph_data.to_data_list()])
        if valid_terms_mat is None:
            # all terminal pairs (i,j) are valid except if i = j
            valid_terms_mat = ~torch.eye(max_n_nodes, device=dev, dtype=bool)
            valid_terms_mat = valid_terms_mat.repeat(self.batch_size, 1, 1)

        if cost_weights is None:
            # get the cost weights
            cost_weights = cost_obj.get_weights(device=dev)
        for key, val in cost_weights.items():
            # expand the cost weights to match the batch
            if type(val) is not Tensor:
                val = torch.tensor([val], device=dev)
            if val.numel() == 1:
                val = val.expand(graph_data.num_graphs)
            cost_weights[key] = val

        # make a tensor of n_routes_to_plan, expanded to the right shape
        if type(n_routes_to_plan) is not Tensor:
            n_routes_to_plan = torch.tensor(n_routes_to_plan, device=dev)
        if n_routes_to_plan.numel() == 1:
            n_routes_to_plan = n_routes_to_plan.expand(graph_data.num_graphs)

        extra_datas = []
        transit_times = (1 - torch.eye(max_n_nodes, device=self.device))
        transit_times[transit_times > 0] = float('inf')
        for ii, dd in enumerate(graph_data.to_data_list()):
            extra_data = ExtraStateData()

            
            extra_data.transit_times = transit_times
            extra_data.route_mat = transit_times.clone()
            dircon = torch.eye(max_n_nodes, device=self.device, dtype=bool)

            extra_data.directly_connected = dircon
            extra_data.has_path = extra_data.directly_connected.clone()
            extra_data.base_valid_terms_mat = valid_terms_mat[ii]
            extra_data.valid_terms_mat = valid_terms_mat[ii]
            extra_data.mean_stop_time = \
                torch.tensor(cost_obj.mean_stop_time_s, device=dev)
            extra_data.transfer_time_s = \
                torch.tensor(cost_obj.avg_transfer_wait_time_s, device=dev)
            extra_data.total_route_time = torch.zeros((), device=dev)
            # make this a tensor so it's stackable
            extra_data.n_routes_to_plan = n_routes_to_plan[ii]

            if isinstance(min_route_len, Tensor):
                if min_route_len.numel() == 1:
                    extra_data.min_route_len = min_route_len
                else:
                    extra_data.min_route_len = min_route_len[ii]
            else:
                extra_data.min_route_len = torch.tensor(min_route_len,
                                                        device=dev)
            extra_data.min_route_len.squeeze_()

            if max_route_len is None:
                max_route_len = dd.num_nodes
            if isinstance(max_route_len, Tensor):
                if max_route_len.numel() == 1:
                    extra_data.max_route_len = max_route_len
                else:
                    extra_data.max_route_len = max_route_len[ii]
            else:
                extra_data.max_route_len = torch.tensor(max_route_len,
                                                        device=dev)
            extra_data.max_route_len.squeeze_()

            extra_data.n_nodes_in_scenario = torch.tensor(dd.num_nodes,
                                                          device=dev)
            extra_data.current_routes = \
                torch.full((max_n_nodes,), -1, device=dev)
            extra_data.current_route_time = torch.zeros((), device=dev)
            extra_data.current_route_times_from_start = \
                torch.zeros((max_n_nodes,), device=dev)
            extra_data.shortest_path_sequences = torch.zeros(
                (0, 0, 0), device=dev)
            extra_data.route_nexts = \
                torch.zeros((max_n_nodes, max_n_nodes), device=dev, 
                            dtype=torch.long)            
            extra_data.n_transfers = extra_data.route_nexts.clone()
            extra_data.context_node_covered_mask = \
                torch.zeros((max_n_nodes,), device=dev, dtype=torch.bool)
            extra_data.context_edge_covered_mask = torch.zeros(
                (max_n_nodes, max_n_nodes), device=dev, dtype=torch.bool)
            extra_data.context_leg_use_count = torch.zeros(
                (max_n_nodes, max_n_nodes), device=dev, dtype=torch.long)
            extra_data.route_slot_context = torch.full(
                (0, 0), -1, device=dev, dtype=torch.long)
            extra_data.active_route_idx = torch.tensor(
                -1, device=dev, dtype=torch.long)

            extra_data.norm_node_features = torch.zeros(
                (dd.num_nodes, 0,), device=dev)

            extra_data.cost_weights = {}
            for key, val in cost_weights.items():
                # expand the cost weights to match the batch
                extra_data.cost_weights[key] = val[ii]

            # Optional adjustment-degree conditioning (gated). Sentinel -1 means
            # "not conditioned" -> get_global_state_features keeps the legacy
            # 12-feature vector (so the frozen construction model is unaffected).
            extra_data.adjustment_target = torch.tensor(-1.0, device=dev)
            extra_data.adjustment_weight = torch.tensor(-1.0, device=dev)
            # Optional "current adjustment degree vs seed" feature (closed-loop):
            # adjustment_use_current >= 0 enables it; adjustment_seed holds the
            # per-graph reference (seed) routes used to compute the live adj.
            extra_data.adjustment_use_current = torch.tensor(-1.0, device=dev)
            extra_data.adjustment_seed = torch.full((0, 0), -1.0, device=dev)

            extra_datas.append(extra_data)
            if hasattr(dd, 'fixed_routes') and dd.fixed_routes.numel() > 0:
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
        new_parts = path_seqs[self.batch_indices, path_indices[:, 0], 
                              path_indices[:, 1]]
        new_len = self.current_routes.shape[-1]
        n_pad = max(new_len - new_parts.shape[-1], 0)
        new_parts = torch.nn.functional.pad(new_parts, (0, n_pad), value=-1)

        starting = (self.current_routes[:, 0] == -1) & ~routes_are_done
        extending = ~starting & ~routes_are_done
        ends_at_cur_start = path_indices[:, 1] == self.current_routes[:, 0]
        last_nodes = self.current_routes[self.batch_indices, 
                                         self.current_route_n_stops - 1]
        starts_at_cur_end = path_indices[:, 0] == last_nodes
        valid_action = starting | ends_at_cur_start | starts_at_cur_end | \
            routes_are_done
        assert valid_action.all(), "invalid action!"

        chose_prev = extending & ends_at_cur_start
        chose_next = extending & ~ends_at_cur_start

        first_parts = new_parts * (starting | chose_prev)[:, None] + \
            self.current_routes * (chose_next | routes_are_done)[:, None]
        second_parts = new_parts * chose_next[:, None] + \
            self.current_routes * chose_prev[:, None] + \
            -1 * ~(chose_next | chose_prev)[:, None]
        # cut off the first node of the second parts, since they overlap
        second_parts = second_parts[..., 1:]
        # combine the two parts without dummy nodes in between
        updated_routes = torch.full((self.batch_size, self.max_n_nodes), -1, 
                                    device=self.device)
        updated_routes[..., :first_parts.shape[-1]] = first_parts
        # insert the second parts at the appropriate masks
        first_part_lens = (first_parts > -1).sum(dim=-1)
        scnd_part_lens = (second_parts > -1).sum(dim=-1)
        new_route_lens = first_part_lens + scnd_part_lens
        scnd_part_mask = tu.get_variable_slice_mask(
            updated_routes, dim=1, froms=first_part_lens, tos=new_route_lens)
        scnd_part_len_mask = tu.get_variable_slice_mask(
            second_parts, dim=1, tos=scnd_part_lens)
        updated_routes[scnd_part_mask] = second_parts[scnd_part_len_mask]
        assert updated_routes.max() < self.max_n_nodes

        updated_routes = updated_routes.clamp(min=-1)

        updated_routes[self.is_done()] = -1

        # then, update the lists of routes that are still being planned
        planning_already_done = self.is_done()
        new_finished_context_routes = torch.full(
            (self.batch_size, 1, self.max_n_nodes), -1,
            dtype=torch.long, device=self.device)
        has_new_finished_route = False
        for bi in range(self.batch_size):
            if not planning_already_done[bi] and routes_are_done[bi]:
                route = updated_routes[bi]
                route = route.clone()[route > -1]
                if len(route) > 0:
                    # the route is valid, so add it to the finished set
                     # otherwise, corresponds to a "no-op" route
                    self._finished_routes[bi].append(route)
                    new_finished_context_routes[bi, 0, :len(route)] = route
                    has_new_finished_route = True
                    self.extra_data.total_route_time[bi] += \
                        self.current_route_time[bi]
            if planning_already_done[bi] or routes_are_done[bi]:
                updated_routes[bi] = -1

        if has_new_finished_route:
            self._add_routes_to_context_masks(new_finished_context_routes)

        self.extra_data.current_routes = updated_routes

        # finally, update all the internal stuff based on the current routes.
        # transforms it to a tensor if necessary
        self.extra_data.current_route_time = \
            self.get_total_route_time(updated_routes)

        ncr_times_from_start = torch.zeros_like(self.current_routes, 
                                                dtype=torch.float32)
        ncr_leg_times = tu.get_route_leg_times(self.current_routes,
                                               self.drive_times,
                                               self.mean_stop_time)
        # keep zeros at the start for the first stop
        ncr_times_from_start[:, 1:] = ncr_leg_times.cumsum(dim=1)
        self.extra_data.current_route_times_from_start = \
            ncr_times_from_start

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
            raise ValueError(
                "Expected action_kinds to have shape (batch_size,), got "
                f"{action_kinds.shape}"
            )
        if path_indices.shape != (self.batch_size, 2):
            raise ValueError(
                "Expected path_indices to have shape (batch_size, 2), got "
                f"{path_indices.shape}"
            )

        is_halt = action_kinds == ROUTE_ACTION_HALT
        uses_legacy_actions = (action_kinds == ROUTE_ACTION_EXTEND) | is_halt
        if uses_legacy_actions.all():
            legacy_actions = path_indices.clone()
            legacy_actions[is_halt] = -1
            self.shortest_path_action(legacy_actions)
            return

        valid_action_kinds = uses_legacy_actions | \
            (action_kinds == ROUTE_ACTION_TRIM_START) | \
            (action_kinds == ROUTE_ACTION_TRIM_END)
        if not valid_action_kinds.all():
            raise ValueError("Unknown route action kind")

        updated_routes = self.current_routes.clone()
        planning_already_done = self.is_done()
        can_act = ~planning_already_done
        extend_mask = (action_kinds == ROUTE_ACTION_EXTEND) & can_act
        trim_start_mask = (action_kinds == ROUTE_ACTION_TRIM_START) & can_act
        trim_end_mask = (action_kinds == ROUTE_ACTION_TRIM_END) & can_act
        halt_mask = is_halt & can_act

        if extend_mask.any():
            ext_routes = self._get_routes_after_shortest_path_actions(
                path_indices, active_mask=extend_mask
            )
            updated_routes[extend_mask] = ext_routes[extend_mask]

        if trim_start_mask.any() or trim_end_mask.any():
            self._apply_trim_actions_to_routes(
                updated_routes, path_indices, trim_start_mask, trim_end_mask
            )

        for bi in range(self.batch_size):
            if not planning_already_done[bi] and halt_mask[bi]:
                route = updated_routes[bi].clone()
                route = route[route > -1]
                if len(route) > 0:
                    self._finished_routes[bi].append(route)
            if planning_already_done[bi] or halt_mask[bi]:
                updated_routes[bi] = -1

        finished_routes = [
            [route.clone() for route in batch_routes]
            for batch_routes in self._finished_routes
        ]
        self._replace_planned_routes(finished_routes, updated_routes)

    def _get_routes_after_shortest_path_actions(self, path_indices,
                                                active_mask=None):
        if active_mask is None:
            active_mask = torch.ones((self.batch_size,), dtype=torch.bool,
                                     device=self.device)

        path_seqs = self.get_shortest_path_sequences()
        safe_indices = path_indices.clamp(min=0)
        new_parts = path_seqs[self.batch_indices, safe_indices[:, 0],
                              safe_indices[:, 1]]
        new_len = self.current_routes.shape[-1]
        n_pad = max(new_len - new_parts.shape[-1], 0)
        new_parts = torch.nn.functional.pad(new_parts, (0, n_pad), value=-1)

        starting = (self.current_routes[:, 0] == -1) & active_mask
        extending = ~starting & active_mask
        ends_at_cur_start = path_indices[:, 1] == self.current_routes[:, 0]
        last_nodes = self.current_routes[self.batch_indices,
                                         self.current_route_n_stops - 1]
        starts_at_cur_end = path_indices[:, 0] == last_nodes
        valid_action = ~active_mask | starting | ends_at_cur_start | \
            starts_at_cur_end
        assert valid_action.all(), "invalid action!"

        chose_prev = extending & ends_at_cur_start
        chose_next = extending & ~ends_at_cur_start

        first_parts = new_parts * (starting | chose_prev)[:, None] + \
            self.current_routes * (chose_next | ~active_mask)[:, None]
        second_parts = new_parts * chose_next[:, None] + \
            self.current_routes * chose_prev[:, None] + \
            -1 * ~(chose_next | chose_prev)[:, None]
        second_parts = second_parts[..., 1:]

        updated_routes = torch.full((self.batch_size, self.max_n_nodes), -1,
                                    device=self.device)
        updated_routes[..., :first_parts.shape[-1]] = first_parts
        first_part_lens = (first_parts > -1).sum(dim=-1)
        scnd_part_lens = (second_parts > -1).sum(dim=-1)
        new_route_lens = first_part_lens + scnd_part_lens
        scnd_part_mask = tu.get_variable_slice_mask(
            updated_routes, dim=1, froms=first_part_lens, tos=new_route_lens)
        scnd_part_len_mask = tu.get_variable_slice_mask(
            second_parts, dim=1, tos=scnd_part_lens)
        updated_routes[scnd_part_mask] = second_parts[scnd_part_len_mask]
        assert updated_routes.max() < self.max_n_nodes

        updated_routes = updated_routes.clamp(min=-1)
        updated_routes[~active_mask] = self.current_routes[~active_mask]
        return updated_routes

    def _apply_trim_actions_to_routes(self, updated_routes, path_indices,
                                      trim_start_mask, trim_end_mask):
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
                trimmed = route[pos:]          # keep route[pos:]
            else:
                trimmed = route[:pos + 1]       # keep route[:pos+1]

            if len(trimmed) < self.min_route_len[bi]:
                raise ValueError("Trim action would violate min_route_len")

            updated_routes[bi] = -1
            updated_routes[bi, :len(trimmed)] = trimmed

    def _replace_planned_routes(self, finished_routes, current_routes):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
            self._add_routes_to_context_masks(self.extra_data.fixed_routes)

        has_finished_routes = any(len(routes) > 0 for routes in finished_routes)
        if has_finished_routes:
            finished_tensor = tu.get_batch_tensor_from_routes(
                finished_routes, self.device
            )
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

        self.extra_data.valid_terms_mat[batch_index] = \
            self.extra_data.base_valid_terms_mat[batch_index].clone()
        self.extra_data.total_route_time[batch_index] = 0

        directly_connected = torch.eye(self.max_n_nodes, device=self.device, 
                                       dtype=bool)
        self.extra_data.directly_connected[batch_index] = \
            directly_connected.expand(len(batch_index), -1, -1)
        transit_times = (1 - torch.eye(self.max_n_nodes, device=self.device))
        transit_times[transit_times > 0] = float('inf')
        self.extra_data.route_mat[batch_index] = transit_times
        self.extra_data.transit_times[batch_index] = transit_times
        self.extra_data.has_path[batch_index] = \
            self.directly_connected[batch_index]
        self.extra_data.current_routes[batch_index] = -1
        self.extra_data.current_route_time[batch_index] = 0
        self.extra_data.current_route_times_from_start[batch_index] = 0
        self.extra_data.route_nexts[batch_index] = 0
        self.extra_data.n_transfers[batch_index] = 0
        self.extra_data.context_node_covered_mask[batch_index] = False
        self.extra_data.context_edge_covered_mask[batch_index] = False
        self.extra_data.context_leg_use_count[batch_index] = 0

    def _add_routes_to_tensors(self, batch_new_routes,
                               only_routes_with_demand_are_valid=False, 
                               invalid_directly_connected=False):
        """Takes a tensor of new routes. The first dimension is the batch"""
        # incorporate new routes into the route graphs
        if type(batch_new_routes) is list:
            batch_new_routes = tu.get_batch_tensor_from_routes(batch_new_routes,
                                                               self.device)
        if batch_new_routes.device != self.device:
            batch_new_routes = batch_new_routes.to(self.device)
        if batch_new_routes.ndim == 2:
            batch_new_routes = batch_new_routes[:, None]
        # add new routes to the route matrix.
        new_route_mat = tu.get_route_edge_matrix(batch_new_routes, 
            self.drive_times, self.mean_stop_time, self.symmetric_routes)
        self.extra_data.route_mat = \
            torch.minimum(self.route_mat, new_route_mat)

        self.directly_connected[self.route_mat < float('inf')] = True

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
            self.extra_data.valid_terms_mat = self.valid_terms_mat & \
                self.valid_terms_mat.transpose(1, 2)

        self._update_route_data()

    def _add_routes_to_context_masks(self, batch_routes):
        """Track nodes/edges covered by finished or fixed context routes."""
        if type(batch_routes) is list:
            batch_routes = tu.get_batch_tensor_from_routes(batch_routes,
                                                           self.device)
        if batch_routes.device != self.device:
            batch_routes = batch_routes.to(self.device)
        batch_routes = batch_routes.to(dtype=torch.long)
        if batch_routes.ndim == 2:
            batch_routes = batch_routes[:, None]
        if batch_routes.numel() == 0:
            return
        if batch_routes.shape[0] == 1 and self.batch_size > 1:
            batch_routes = batch_routes.expand(self.batch_size, -1, -1)
        elif batch_routes.shape[0] != self.batch_size:
            raise ValueError(
                "Context route batch size does not match state batch size: "
                f"{batch_routes.shape[0]} vs {self.batch_size}"
            )

        valid_nodes = batch_routes >= 0
        if valid_nodes.any():
            batch_idxs = torch.arange(
                self.batch_size, device=self.device)[:, None, None]
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

        edge_batch_idxs = torch.arange(
            self.batch_size, device=self.device)[:, None, None]
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
    
    def replace_routes(self, batch_new_routes, 
                only_routes_with_demand_are_valid=False, 
                invalid_directly_connected=False):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
            self._add_routes_to_context_masks(self.extra_data.fixed_routes)
        self.add_new_routes(batch_new_routes, 
                            only_routes_with_demand_are_valid,
                            invalid_directly_connected)

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
            raise RuntimeError(
                "Cannot seed current routes when the state already has an "
                "active route"
            )

        if type(batch_current_routes) is list:
            is_single_route = len(batch_current_routes) == 0 or \
                isinstance(batch_current_routes[0], (int, np.integer)) or \
                (isinstance(batch_current_routes[0], Tensor) and
                 batch_current_routes[0].ndim == 0)
            if is_single_route:
                batch_current_routes = [batch_current_routes]
            batch_current_routes, _ = tu.get_tensor_from_varlen_lists(
                batch_current_routes,
                self.device,
            )

        if batch_current_routes.ndim == 1:
            batch_current_routes = batch_current_routes.unsqueeze(0)
        elif batch_current_routes.ndim == 3 and batch_current_routes.shape[1] == 1:
            batch_current_routes = batch_current_routes.squeeze(1)
        elif batch_current_routes.ndim != 2:
            raise ValueError(
                "Expected current routes to have shape "
                f"(batch, max_route_len), got {batch_current_routes.shape}"
            )

        if batch_current_routes.shape[0] == 1 and self.batch_size > 1:
            batch_current_routes = batch_current_routes.expand(self.batch_size,
                                                               -1)
        elif batch_current_routes.shape[0] != self.batch_size:
            raise ValueError(
                "Current-route batch size does not match state batch size: "
                f"{batch_current_routes.shape[0]} vs {self.batch_size}"
            )

        if batch_current_routes.shape[-1] > self.max_n_nodes:
            raise ValueError(
                "Current route is longer than the scenario node capacity: "
                f"{batch_current_routes.shape[-1]} vs allowed "
                f"{self.max_n_nodes}"
            )

        if batch_current_routes.shape[-1] < self.max_n_nodes:
            pad_len = self.max_n_nodes - batch_current_routes.shape[-1]
            batch_current_routes = torch.nn.functional.pad(
                batch_current_routes, (0, pad_len), value=-1
            )

        batch_current_routes = batch_current_routes.to(
            device=self.device,
            dtype=torch.long,
        )
        invalid_nodes = (batch_current_routes < -1) | \
            (batch_current_routes >= self.max_n_nodes)
        invalid_nodes &= batch_current_routes != -1
        if invalid_nodes.any():
            raise ValueError("Current routes contain invalid node indices")

        self.extra_data.current_routes = batch_current_routes
        self.extra_data.current_route_time = \
            self.get_total_route_time(batch_current_routes)

        ncr_times_from_start = torch.zeros_like(self.current_routes,
                                                dtype=torch.float32)
        ncr_leg_times = tu.get_route_leg_times(self.current_routes,
                                               self.drive_times,
                                               self.mean_stop_time)
        ncr_times_from_start[:, 1:] = ncr_leg_times.cumsum(dim=1)
        self.extra_data.current_route_times_from_start = ncr_times_from_start

        self._add_routes_to_tensors(batch_current_routes[:, None])

    def add_new_routes(self, batch_new_routes,
                       only_routes_with_demand_are_valid=False, 
                       invalid_directly_connected=False):
        if type(batch_new_routes) is list:
            batch_new_routes = tu.get_batch_tensor_from_routes(batch_new_routes,
                                                               self.device)
        if batch_new_routes.device != self.device:
            batch_new_routes = batch_new_routes.to(self.device)
        if batch_new_routes.ndim == 2:
            batch_new_routes = batch_new_routes[:, None]

        self._add_routes_to_tensors(batch_new_routes, 
                                    only_routes_with_demand_are_valid, 
                                    invalid_directly_connected)
        self._add_routes_to_context_masks(batch_new_routes)
        self._add_routes_to_list(batch_new_routes)

        # finally, update the total route times with the new routes
        leg_times = tu.get_route_leg_times(batch_new_routes, 
                                           self.graph_data.drive_times,
                                           self.mean_stop_time)
        total_new_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:                                           
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = tu.get_route_leg_times(batch_new_routes, 
                                                      transpose_dtm,
                                                      self.mean_stop_time)
            total_new_time += return_leg_times.sum(dim=(1,2))

        self.extra_data.total_route_time += total_new_time        

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
                    log.debug('dropping degenerate route with fewer than 2 stops')
                    continue
                self._finished_routes[bi].append(route[:length])
    
    def _update_route_data(self):
        # do things that have to be done whether we added or removed routes
        fw_mat = self.route_mat + self.transfer_time_s[:, None, None]
        nexts, transit_times = tu.floyd_warshall(fw_mat)
        _, path_edge_counts = tu.reconstruct_all_paths(nexts)
        # subtract one transfer time from each transit time to avoid counting
         # a transfer for the first edge of a journey
        transit_times -= self.transfer_time_s[:, None, None]

        self.extra_data.route_nexts = nexts
        self.extra_data.transit_times = transit_times
        self.extra_data.has_path = transit_times < float('inf')
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
        self.extra_data.shortest_path_sequences = shortest_paths.new_empty(
            (self.batch_size, 0, 0, 0))
        try:
            if target == self.device:
                new = copy.copy(self)
                new.graph_data = self.graph_data
                new.extra_data = copy.deepcopy(self.extra_data)
                new._finished_routes = [list(routes)
                                        for routes in self._finished_routes]
                return new

            new = copy.deepcopy(self)
        finally:
            self.extra_data.shortest_path_sequences = shortest_paths

        new.graph_data = new.graph_data.to(target)
        new.extra_data = new.extra_data.to(target)
        new._finished_routes = [
            [route.to(target) for route in routes]
            for routes in new._finished_routes
        ]
        return new
    
    @staticmethod
    def batch_from_list(state_list):
        """return a batch state from a list of states."""
        if len(state_list) == 1:
            return state_list[0]
        
        graph_datas = sum([ss.graph_data.to_data_list() for ss in state_list],
                          [])
        extra_datas = sum([ss.extra_data.to_data_list() for ss in state_list], 
                          [])
        batch_graph_data = Batch.from_data_list(graph_datas)
        batch_extra_data = Batch.from_data_list(extra_datas)
        batch_state = copy.copy(state_list[0])
        batch_state.graph_data = batch_graph_data
        batch_state.extra_data = batch_extra_data
        copied_routes = [[copy.copy(bfr) for bfr in ss._finished_routes]
                         for ss in state_list]
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
        for gd, ed, routes in zip(graph_datas, extra_datas, 
                                  self._finished_routes):
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
        if batch_routes.ndim == 2:
            # add a routes dimension
            batch_routes = batch_routes[:, None]

        leg_times = tu.get_route_leg_times(batch_routes, 
                                           self.graph_data.drive_times,
                                           self.mean_stop_time)
        route_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = tu.get_route_leg_times(batch_routes, 
                                                      transpose_dtm,
                                                      self.mean_stop_time)
            route_time += return_leg_times.sum(dim=(1,2))

        return route_time
    
    def get_global_state_features(self, include_redundancy=None):
        cost_weights = self.cost_weights_tensor
        diameter = self.drive_times.flatten(1,2).max(1).values
        mean_route_time = self.total_route_time / (
            self.n_routes_to_plan * diameter)

        so_far = self.n_finished_routes
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        n_routes_log_feats = (both + 1).log()
        # use fractions so it's independent of n_routes_to_plan
        n_routes_frac_feats = both / (so_far + left)[:, None]

        n_disconnected_demand_edges = self.get_n_disconnected_demand_edges()
        # as with n_routes feats, use both log and fractional
        log_uncovered = (n_disconnected_demand_edges + 1).log()
        frac_uncovered = n_disconnected_demand_edges / self.n_demand_edges
        uncovered_feats = torch.stack((log_uncovered, 
                                       frac_uncovered
                                     ), dim=-1)
        curr_route_n_stops = self.current_route_n_stops[:, None]

        served_demand = (self.has_path * self.demand).sum(dim=(1, 2))
        tt = self.transit_times.clone()
        tt[~self.has_path] = 0
        total_demand_time = (self.demand * tt).sum(dim=(1,2))
        mean_demand_time = total_demand_time / (served_demand + EPSILON)
        mean_demand_time_frac = mean_demand_time / diameter

        global_features = torch.cat((
            cost_weights, mean_route_time[:, None], n_routes_log_feats,
            n_routes_frac_feats, uncovered_feats, curr_route_n_stops,
            mean_demand_time_frac[:, None],
        ), dim=-1)

        if include_redundancy is None:
            include_redundancy = self._redundancy_features_enabled
        if include_redundancy:
            redundancy_features = self.get_redundancy_global_features()
            global_features = torch.cat(
                (global_features, redundancy_features), dim=-1)

        # Gated adjustment conditioning: append conditioning features only when
        # set (sentinel >= 0). Default keeps the legacy 12-feature vector so the
        # frozen construction model is unaffected. The appended width is:
        #   +1 (target only)      when target is set and weight is NOT,
        #   +2 (target, weight)   when both are set.
        # This must match the model's n_adjustment_cond_feats (1 or 2).
        adj_target = self.adjustment_target.reshape(-1)
        if bool((adj_target >= 0).any().item()):
            feats = [adj_target.clamp(min=0.0)]
            adj_weight = self.adjustment_weight.reshape(-1)
            if bool((adj_weight >= 0).any().item()):
                # log1p-normalize the weight to ~[0,1] over range [0, 8].
                feats.append(torch.log1p(adj_weight.clamp(min=0.0)) /
                             math.log1p(8.0))
            adj_use_cur = self.adjustment_use_current.reshape(-1)
            if bool((adj_use_cur >= 0).any().item()):
                # closed-loop: live adjustment degree of the current network
                # (incl. the in-progress route) vs the stored seed routes.
                feats.append(self._compute_current_adjustment())
            adj_feats = torch.stack(feats, dim=-1)
            global_features = torch.cat((global_features, adj_feats), dim=-1)
        return global_features

    def set_redundancy_feature_mode(self, enabled):
        """Select whether optional redundancy summaries join global features."""
        self._redundancy_features_enabled = bool(enabled)

    def _count_route_legs(self, batch_routes):
        """Count adjacent stop-to-stop legs in a batch of route tensors."""
        if batch_routes.ndim == 2:
            batch_routes = batch_routes[:, None]
        if batch_routes.shape[0] == 1 and self.batch_size > 1:
            batch_routes = batch_routes.expand(self.batch_size, -1, -1)
        elif batch_routes.shape[0] != self.batch_size:
            raise ValueError(
                "Route batch size does not match state batch size: "
                f"{batch_routes.shape[0]} vs {self.batch_size}"
            )

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

        batch_idxs = torch.arange(
            self.batch_size, device=self.device)[:, None, None]
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

    def get_redundancy_global_features(self):
        """Return normalized network and active-route redundancy summaries."""
        context_counts = self.context_leg_use_count.to(dtype=torch.float32)
        current_counts = self.current_leg_use_count.to(dtype=torch.float32)
        total_counts = context_counts + current_counts
        excess_counts = (total_counts - 1).clamp_min(0)

        total_traversals = total_counts.sum(dim=(1, 2))
        network_redundancy = excess_counts.sum(dim=(1, 2)) / \
            total_traversals.clamp_min(1)

        n_used_legs = (total_counts > 0).sum(dim=(1, 2))
        mean_excess_leg_use = excess_counts.sum(dim=(1, 2)) / \
            n_used_legs.clamp_min(1)
        max_context_leg_use = context_counts.amax(dim=(1, 2))

        current_traversals = current_counts.sum(dim=(1, 2))
        current_route_self_repeat = \
            (current_counts - 1).clamp_min(0).sum(dim=(1, 2)) / \
            current_traversals.clamp_min(1)
        current_route_context_overlap = \
            (current_counts * (context_counts > 0)).sum(dim=(1, 2)) / \
            current_traversals.clamp_min(1)

        return torch.stack((
            network_redundancy,
            mean_excess_leg_use,
            max_context_leg_use,
            current_route_self_repeat,
            current_route_context_overlap,
        ), dim=-1)

    def _compute_current_adjustment(self):
        """Per-graph adjustment degree of the CURRENT network vs the stored
        seed routes ([batch] in [0,1]). Used as a closed-loop conditioning
        feature."""
        from .bee_colony import get_adjustment_degrees
        if self.route_slot_context.numel() > 0:
            cur = self.route_slot_context.clone()
            for bi, route_idx in enumerate(self.active_route_idx.tolist()):
                if route_idx < 0:
                    continue
                cur[bi, route_idx] = -1
                route = self.current_routes[bi]
                copy_len = min(cur.shape[-1], route.shape[-1])
                cur[bi, route_idx, :copy_len] = route[:copy_len]
        else:
            cur = tu.get_batch_tensor_from_routes(self.routes, self.device)
        seed = self.adjustment_seed.to(self.device)
        if seed.dim() == 2:           # unbatched [n_routes, L]
            seed = seed.unsqueeze(0)
        nr = min(cur.shape[1], seed.shape[1])
        ll = min(cur.shape[-1], seed.shape[-1])
        adj = get_adjustment_degrees(
            cur[:, :nr, :ll], seed[:, :nr, :ll].long(),
            self.symmetric_routes, gap=self._adjustment_gap,
            mode=self._adjustment_mode)
        # get_adjustment_degrees returns per-route [batch, n_routes]; the
        # feature is the per-graph mean adjustment.
        return adj.reshape(adj.shape[0], -1).mean(dim=1).float()

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
            path_seqs, _ = tu.reconstruct_all_paths(self.nexts)
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

    @property
    def cost_weights_tensor(self):
        cost_weights_list = []
        for key in sorted(self.cost_weights.keys()):
            if type(self.cost_weights[key]) is Tensor:
                cw = self.cost_weights[key].to(self.device)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(self.cost_weights[key], 
                                  device=self.device)[None]

            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(self.batch_size, -1)
        if cost_weights.shape[0] > self.batch_size:
            cost_weights = cost_weights[:self.batch_size]
        return cost_weights

    def get_cost_weights_tensor(self, key_order=COST_WEIGHT_KEY_ORDER,
                                normalize=False):
        cost_weights_list = []
        for key in key_order:
            if key not in self.cost_weights:
                raise KeyError(f"State does not contain cost weight {key!r}")
            if type(self.cost_weights[key]) is Tensor:
                cw = self.cost_weights[key].to(self.device)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(self.cost_weights[key],
                                  device=self.device)[None]
            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(self.batch_size, -1)
        if cost_weights.shape[0] > self.batch_size:
            cost_weights = cost_weights[:self.batch_size]
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
        if hasattr(self.extra_data, 'norm_node_features'):
            return self.extra_data.norm_node_features
        else:
            return None
        
    @property
    def norm_cost_weights(self):
        if hasattr(self.extra_data, 'norm_cost_weights'):
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
        return self.extra_data.cost_weights['demand_time_weight']

    @property
    def cost_weights(self):
        return self.extra_data.cost_weights

    @property
    def adjustment_target(self):
        """Per-graph adjustment-degree target (-1 if not conditioned)."""
        return self.extra_data.adjustment_target

    @property
    def adjustment_weight(self):
        """Per-graph adjustment-penalty weight (-1 if not conditioned)."""
        return self.extra_data.adjustment_weight

    @property
    def adjustment_use_current(self):
        """Per-graph flag (>=0) enabling the live current-adj feature."""
        return self.extra_data.adjustment_use_current

    @property
    def adjustment_seed(self):
        """Per-graph seed routes used to compute the live current adj."""
        return self.extra_data.adjustment_seed

    @property
    def is_adjustment_conditioned(self):
        return bool((self.adjustment_target.reshape(-1)[0] >= 0).item())

    def set_adjustment_conditioning(self, target, weight=None,
                                    seed_routes=None, gap=0.1, mode='paper'):
        """Enable adjustment conditioning: store per-graph target (always) and,
        optionally, weight and seed routes, so they are appended to
        get_global_state_features (gated).

        - ``weight`` None  -> weight stays sentinel -1, NOT fed into features
          (use for "target only"; the penalty weight is a fixed scalar outside).
        - ``seed_routes`` given ([batch, n_routes, L]) -> enables the live
          "current adjustment degree vs seed" feature (closed-loop): each
          forward recomputes adj(current network, seed) and appends it.
        """
        bs = self.batch_size
        dev = self.device
        self._adjustment_gap = float(gap)
        self._adjustment_mode = str(mode)
        tt = torch.as_tensor(target, device=dev, dtype=torch.float32)
        if tt.numel() == 1:
            tt = tt.expand(bs)
        self.extra_data.adjustment_target = tt.reshape(bs).clone()
        if weight is None:
            # leave weight at sentinel -1 -> excluded from global features
            self.extra_data.adjustment_weight = torch.full(
                (bs,), -1.0, device=dev)
        else:
            ww = torch.as_tensor(weight, device=dev, dtype=torch.float32)
            if ww.numel() == 1:
                ww = ww.expand(bs)
            self.extra_data.adjustment_weight = ww.reshape(bs).clone()
        if seed_routes is None:
            self.extra_data.adjustment_use_current = torch.full(
                (bs,), -1.0, device=dev)
        else:
            sr = torch.as_tensor(seed_routes, device=dev, dtype=torch.long)
            if sr.dim() == 2:           # [n_routes, L] -> broadcast to batch
                sr = sr.unsqueeze(0).expand(bs, -1, -1)
            self.extra_data.adjustment_seed = sr.reshape(bs, sr.shape[-2],
                                                         sr.shape[-1]).clone()
            self.extra_data.adjustment_use_current = torch.ones(bs, device=dev)

    def set_route_slot_context(self, route_batch, active_route_idx):
        """Keep route-slot ordering for closed-loop adjustment conditioning."""
        routes = torch.as_tensor(
            route_batch, device=self.device, dtype=torch.long)
        if routes.ndim == 2:
            routes = routes.unsqueeze(0)
        if routes.shape[0] == 1 and self.batch_size > 1:
            routes = routes.expand(self.batch_size, -1, -1)
        elif routes.shape[0] != self.batch_size:
            raise ValueError(
                "Route-slot context batch size does not match state batch "
                f"size: {routes.shape[0]} vs {self.batch_size}")
        idx = torch.as_tensor(
            active_route_idx, device=self.device, dtype=torch.long)
        if idx.numel() == 1:
            idx = idx.expand(self.batch_size)
        elif idx.numel() != self.batch_size:
            raise ValueError(
                "active_route_idx must be scalar or have one value per batch")
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
        if hasattr(self.extra_data, 'context_leg_use_count'):
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
        return torch.tensor(nrsf, dtype=torch.float32,
                            device=self.device)
    
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


@dataclass
class CostHelperOutput:
    total_demand_time: Tensor
    total_route_time: Tensor
    trips_at_transfers: Tensor
    total_demand: Tensor
    unserved_demand: Tensor
    total_transfers: Tensor
    trip_times: Tensor
    n_disconnected_demand_edges: Tensor
    n_stops_oob: Tensor
    n_duplicate_stops: Tensor
    batch_routes: Tensor
    unserved_demand_matrix: Tensor 
    per_route_riders: Optional[Tensor] = None
    cost: Optional[Tensor] = None
    median_connectivity: Optional[Tensor] = None
    median_connectivity_weighted: Optional[Tensor] = None

    @property
    def mean_demand_time(self):
        served_demand = self.total_demand - self.unserved_demand
        # avoid division by 0
        return self.total_demand_time / (served_demand + EPSILON)

    def get_metrics(self):
        """return a dictionary with the metrics we usually report."""
        frac_tat = self.trips_at_transfers / self.total_demand[:, None]
        percent_tat = frac_tat * 100
        metrics = {
            'cost': self.cost,
            'ATT': self.mean_demand_time / 60,
            'RTT': self.total_route_time / 60,
            '$d_0$': percent_tat[:, 0],
            '$d_1$': percent_tat[:, 1],
            '$d_2$': percent_tat[:, 2],
            '$d_{un}$': percent_tat[:, 3],
            '# disconnected node pairs': 
                self.n_disconnected_demand_edges.float(),
            '# stops out of bounds': self.n_stops_oob.float(),
            'median_connectivity': self.median_connectivity / 60 if self.median_connectivity is not None else None,
            'median_connectivity_weighted': self.median_connectivity_weighted / 60 if self.median_connectivity_weighted is not None else None,

        }
        return metrics

    def get_metrics_tensor(self):
        """return a tensor with the metrics we usually report."""
        metrics = self.get_metrics()
        metrics = torch.stack([metrics[k] for k in metrics], dim=-1)
        return metrics
    
    def are_constraints_violated(self):
        return (self.n_stops_oob > 0) | \
               (self.n_disconnected_demand_edges > 0) | \
               (self.n_duplicate_stops > 0) # | \
               # (self.n_skipped_stops > 0)
    

class CostModule(torch.nn.Module):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False):
        super().__init__()
        self.mean_stop_time_s = mean_stop_time_s
        self.avg_transfer_wait_time_s = avg_transfer_wait_time_s
        self.symmetric_routes = symmetric_routes
        self.low_memory_mode = low_memory_mode

    def get_metric_names(self):
        dummy_obj = CostHelperOutput(
            torch.zeros(1), torch.zeros(1), torch.zeros(1, 4), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),)
        return dummy_obj.get_metrics().keys()
    
    def _compute_all_pairs_times_floyd(self, state):

        transit_times = state.transit_times.clone()
        B, N, _ = transit_times.shape

        dist = transit_times.clone()

        for k in range(N):
            dist = torch.minimum(dist, dist[:, :, k].unsqueeze(2) + dist[:, k, :].unsqueeze(1))

        return dist  # [B, N, N]


    def _cost_helper(self, state, return_per_route_riders=False):
        """
        Compute cost components including median connectivity using transit_times converted to NetworkX graph.
        
        Args:
            state: Object containing graph_data, transit_times, demand, etc.
            return_per_route_riders: If True, compute per-route riders (default: False)
        
        Returns:
            CostHelperOutput: Object containing computed cost components
        """
        drive_times_matrix = state.drive_times
        demand_matrix = state.demand
        dev = drive_times_matrix.device

        # assemble route graph
        batch_routes = tu.get_batch_tensor_from_routes(state.routes, dev)
        route_lens = (batch_routes > -1).sum(-1)

        log.debug("summing cost components")

        zero = torch.zeros_like(route_lens)
        route_len_delta = (state.min_route_len[:, None] - route_lens)
        route_len_delta = route_len_delta.maximum(zero)
        # don't penalize placeholer "dummy" routes in the tensor
        route_len_delta[route_lens == 0] = 0
        if state.max_route_len is not None:
            route_len_over = (route_lens - state.max_route_len[:, None])
            route_len_delta = route_len_delta + route_len_over.maximum(zero)
        
        # if there is a current route, it's already included in route_len_delta
        n_unstarted_routes = \
            state.n_routes_left_to_plan - \
                (state.has_current_route).to(torch.float32)
        n_stops_oob = route_len_delta.sum(-1) + \
            n_unstarted_routes * state.min_route_len

        assert (n_stops_oob >= 0).all(), "negative stops-out-of-bounds!"

        # calculate the amount of demand at each number of transfers
        trips_at_transfers = torch.zeros(state.batch_size, 4, device=dev)
        # trips with no path get '3' transfers so they'll be included in d_un,
         # not d_0
        n_transfers = state.n_transfers.clone()
        nopath = ~state.has_path
        n_transfers[nopath] = 3
        for ii in range(3):
            d_i = (demand_matrix * (n_transfers == ii)).sum(dim=(1, 2))
            trips_at_transfers[:, ii] = d_i
        
        d_un = (demand_matrix * (n_transfers > 2)).sum(dim=(1, 2))
        trips_at_transfers[:, 3] = d_un

        # calculate some more quantities of interest
        trip_times = state.transit_times.clone()
        trip_times[nopath] = 0
        demand_time = demand_matrix * trip_times
        total_dmd_time = demand_time.sum(dim=(1, 2))
        demand_transfers = demand_matrix * state.n_transfers
        total_transfers = demand_transfers.sum(dim=(1, 2))
        unserved_demand = (demand_matrix * nopath).sum(dim=(1, 2))
        total_demand = demand_matrix.sum(dim=(1,2))

        n_duplicate_stops = count_duplicate_stops(state.max_n_nodes, 
                                                batch_routes)
        
        # Получаем матрицу кратчайших путей
        all_pairs = self._compute_all_pairs_times_floyd(state)
        # Берём максимум по строкам (по оси 1)
        row_max = demand_matrix.max(dim=1, keepdim=True).values  # [N, 1] или [B, N, 1] в батче

        # Чтобы избежать деления на 0
        row_max = torch.where(row_max == 0, torch.tensor(1., device=row_max.device), row_max)

        # Делим каждую строку на её максимум
        demand_weight = demand_matrix / row_max
        B, N, _ = all_pairs.shape

        # Убираем диагональ (расстояние от узла к себе)
        eye = torch.eye(N, device=all_pairs.device).bool().unsqueeze(0)  # [1, N, N]
        masked = all_pairs.masked_fill(eye, float('nan'))                # [B, N, N]

        # Меняем inf → nan, чтобы их исключить из медианы
        masked = masked.masked_fill(~masked.isfinite(), float('nan'))    # теперь только достижимые пути

        # === 1. Обычная медианная связанность ===
        node_medians = torch.nanmedian(masked, dim=2).values              # [B, N]
        tmp = torch.nanmean(node_medians, dim=1)
        median_connectivity = torch.where(torch.isnan(tmp), torch.tensor(0., device=tmp.device), tmp)

        # === 2. Взвешенная медианная связанность ===
        # Маска та же, но домножаем расстояния на веса спроса
        # demand_weight: [N, N] → добавим ось батча
        weighted_masked = masked * demand_weight             # [B, N, N]
        node_medians_w = torch.nanmedian(weighted_masked, dim=2).values   # [B, N]
        tmp_w = torch.nanmean(node_medians_w, dim=1)
        median_connectivity_weighted = torch.where(torch.isnan(tmp_w), torch.tensor(0., device=tmp_w.device), tmp_w)

        unserved_demand_matrix = demand_matrix * nopath

        output = CostHelperOutput(
            total_dmd_time, state.total_route_time, trips_at_transfers, 
            total_demand, unserved_demand, total_transfers, trip_times,
            state.get_n_disconnected_demand_edges(), n_stops_oob, 
            n_duplicate_stops, batch_routes,
            unserved_demand_matrix, 
            median_connectivity=median_connectivity,
            median_connectivity_weighted=median_connectivity_weighted
        )

        if return_per_route_riders:
            _, used_routes = \
                tu.get_route_edge_matrix(batch_routes, drive_times_matrix,
                                        self.mean_stop_time_s, 
                                        self.symmetric_routes, 
                                        self.low_memory_mode, 
                                        return_used_routes=True)

            used_routes.unsqueeze_(-1)
            route_seqs = tu.aggregate_edge_features(state.route_nexts, 
                                                    used_routes, 'concat')
            route_seqs.squeeze_(-1)
            per_route_riders = torch.zeros(batch_routes.shape[:2], device=dev)
            for bi in range(state.batch_size):
                for ri in range(batch_routes.shape[1]):
                    srcs, dsts, _ = torch.where(route_seqs[bi] == ri)
                    ri_demand = demand_matrix[bi, srcs, dsts].sum()
                    per_route_riders[bi, ri] = ri_demand

            output.per_route_riders = per_route_riders
        
        return output


class MyCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S,
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False, use_weighted_connectivity=False,
                 demand_time_weight=0.33, route_time_weight=0.33,
                 median_connectivity_weight=0.33,
                 constraint_violation_weight=5, variable_weights=False,
                 ignore_stops_oob=False, pp_fraction=0.33,
                 op_fraction=0.33, mcw_fraction=0.33,
                 enabled_components=None, disabled_components=None):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.use_weighted_connectivity = use_weighted_connectivity
        self.demand_time_weight = demand_time_weight
        self.route_time_weight = route_time_weight
        self.median_connectivity_weight = median_connectivity_weight
        self.constraint_violation_weight = constraint_violation_weight
        self.variable_weights = variable_weights
        if self.variable_weights:
            self.pp_fraction = pp_fraction
            self.op_fraction = op_fraction
            self.mcw_fraction = mcw_fraction
            assert pp_fraction + op_fraction + mcw_fraction <= 1, \
                "fractions of extreme samples must sum to <= 1"
        self.ignore_stops_oob = ignore_stops_oob
        # Which of the three cost components (demand / route / connectivity)
        # are active. Disabled components are dropped from the weighted cost
        # and from sampled / fixed cost-weight tensors; when all three are
        # enabled (the default) every masking helper below is a no-op, so
        # behaviour is bit-for-bit unchanged.
        self.set_enabled_components(enabled_components, disabled_components)

    @property
    def cost_component_names(self):
        return COST_WEIGHT_KEY_ORDER

    # ------------------------------------------------------------------
    # Enable / disable individual cost components
    # ------------------------------------------------------------------
    def set_enabled_components(self, enabled_components=None,
                               disabled_components=None):
        """Configure which cost components are active (see
        ``resolve_enabled_cost_components``)."""
        self._enabled_components = resolve_enabled_cost_components(
            enabled_components, disabled_components)
        self._all_components_enabled = all(self._enabled_components)

    @property
    def enabled_component_mask(self):
        """Tuple of three bools, index-aligned with ``COST_COMPONENT_NAMES``."""
        return self._enabled_components

    @property
    def all_components_enabled(self):
        return self._all_components_enabled

    @property
    def enabled_component_indices(self):
        return tuple(i for i, on in enumerate(self._enabled_components) if on)

    @property
    def enabled_component_names(self):
        return tuple(COST_COMPONENT_NAMES[i]
                     for i in self.enabled_component_indices)

    @property
    def disabled_component_names(self):
        return tuple(COST_COMPONENT_NAMES[i]
                     for i, on in enumerate(self._enabled_components)
                     if not on)

    @property
    def n_enabled_components(self):
        return int(sum(self._enabled_components))

    def _enabled_mask_row(self, device=None, dtype=torch.float32):
        return torch.tensor(
            [1.0 if on else 0.0 for on in self._enabled_components],
            device=device, dtype=dtype)

    def apply_enabled_mask(self, weights, normalize=True):
        """Zero the weights of disabled cost components along the last axis
        and (optionally) renormalize the survivors to sum to 1.

        ``weights`` is a ``[..., 3]`` tensor. This is a no-op when every
        component is enabled, so default runs are unaffected. Rows whose
        enabled weights are all zero fall back to a uniform distribution
        over the enabled components instead of producing a degenerate
        all-zero (pure-constraint) cost."""
        if self._all_components_enabled:
            return weights
        mask = self._enabled_mask_row(weights.device, weights.dtype)
        weights = weights * mask
        if normalize:
            total = weights.sum(dim=-1, keepdim=True)
            uniform = mask / mask.sum()
            weights = torch.where(
                total > EPSILON, weights / total.clamp_min(EPSILON),
                uniform)
        return weights

    def _mask_weight_dict(self, wdict, normalize=True):
        """Apply ``apply_enabled_mask`` to a dict of per-component weight
        tensors keyed by ``COST_WEIGHT_KEY_ORDER``."""
        if self._all_components_enabled:
            return wdict
        keys = list(COST_WEIGHT_KEY_ORDER)
        stacked = torch.stack([wdict[k] for k in keys], dim=-1)
        stacked = self.apply_enabled_mask(stacked, normalize=normalize)
        out = dict(wdict)
        for i, k in enumerate(keys):
            out[k] = stacked[..., i]
        return out

    def sample_variable_weights(self, batch_size, device=None):
        if not self.variable_weights:
            dtw = torch.full((batch_size,), self.demand_time_weight, 
                             device=device)
            rtw = torch.full((batch_size,), self.route_time_weight, 
                             device=device)
            mcw = torch.full((batch_size,), self.median_connectivity_weight, 
                             device=device)
        else:
            random_number = torch.rand(batch_size, device=device)
            dtw = torch.zeros(batch_size, device=device)
            rtw = torch.zeros(batch_size, device=device)
            mcw = torch.zeros(batch_size, device=device)
            
            # Set weights for extreme cases
            is_pp = random_number < self.pp_fraction
            dtw[is_pp] = 1.0  # Passenger-perspective: demand_time_weight = 1
            is_op = (self.pp_fraction <= random_number) & (random_number < self.pp_fraction + self.op_fraction)
            rtw[is_op] = 1.0  # Operator-perspective: route_time_weight = 1
            is_mcw = (self.pp_fraction + self.op_fraction <= random_number) & (random_number < self.pp_fraction + self.op_fraction + self.mcw_fraction)
            mcw[is_mcw] = 1.0  # Connectivity-perspective: median_connectivity_weight = 1
            
            # For intermediate cases, generate random weights that sum to 1
            extremes_fraction = self.pp_fraction + self.op_fraction + self.mcw_fraction
            is_intermediate = random_number >= extremes_fraction
            n_intermediate = is_intermediate.sum()
            
            if n_intermediate > 0:
                # Generate random weights using uniform distribution on the simplex
                weights = torch.rand(n_intermediate, 3, device=device)
                weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize to sum to 1
                dtw[is_intermediate] = weights[:, 0]
                rtw[is_intermediate] = weights[:, 1]
                mcw[is_intermediate] = weights[:, 2]

        return self._mask_weight_dict({
            'demand_time_weight': dtw,
            'route_time_weight': rtw,
            'median_connectivity_weight': mcw,
        })
    
    def get_weights(self, device=None):
        dtm = self.demand_time_weight
        if type(dtm) is not Tensor:
            dtm = torch.tensor([dtm], device=device)
        rtm = self.route_time_weight
        if type(rtm) is not Tensor:
            rtm = torch.tensor([rtm], device=device)
        mcw = self.median_connectivity_weight
        if type(mcw) is not Tensor:
            mcw = torch.tensor([mcw], device=device)

        return self._mask_weight_dict({
            'demand_time_weight': dtm,
            'route_time_weight': rtm,
            'median_connectivity_weight': mcw,
        })
    
    def set_weights(self, demand_time_weight=None, route_time_weight=None, 
                                    median_connectivity_weight=None,  constraint_violation_weight=None):
        if demand_time_weight is not None:
            self.demand_time_weight = demand_time_weight
        if route_time_weight is not None:
            self.route_time_weight = route_time_weight
        if median_connectivity_weight is not None:  
            self.median_connectivity_weight = median_connectivity_weight
        if constraint_violation_weight is not None:
            self.constraint_violation_weight = constraint_violation_weight

    def get_preference_weights(self, state, normalize=False):
        if self._all_components_enabled:
            if hasattr(state, "get_cost_weights_tensor"):
                return state.get_cost_weights_tensor(
                    COST_WEIGHT_KEY_ORDER, normalize=normalize)

            weights = []
            for key in COST_WEIGHT_KEY_ORDER:
                weights.append(state.cost_weights[key])
            weights = torch.stack(weights, dim=-1)
            if normalize:
                weights = weights / weights.sum(
                    dim=-1, keepdim=True).clamp_min(EPSILON)
            return weights

        # A disabled component must not carry preference mass — mask the
        # raw weights, then renormalize the survivors when requested.
        if hasattr(state, "get_cost_weights_tensor"):
            weights = state.get_cost_weights_tensor(
                COST_WEIGHT_KEY_ORDER, normalize=False)
        else:
            weights = torch.stack(
                [state.cost_weights[key] for key in COST_WEIGHT_KEY_ORDER],
                dim=-1)
        return self.apply_enabled_mask(weights, normalize=normalize)

    def _expand_batch_value(self, value, batch_size, device):
        if type(value) is not Tensor:
            value = torch.tensor([value], device=device)
        else:
            value = value.to(device)
            if value.ndim == 0:
                value = value[None]
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size)
        if value.shape[0] > batch_size:
            value = value[:batch_size]
        return value

    def _compute_cost_components_from_result(
            self, state, cho, constraint_weight=None, no_norm=False):
        cost_weights = state.cost_weights
        if 'demand_time_weight' in cost_weights:
            demand_time_weight = cost_weights['demand_time_weight']
        else:
            demand_time_weight = self.demand_time_weight
        if 'route_time_weight' in cost_weights:
            route_time_weight = cost_weights['route_time_weight']
        else:
            route_time_weight = self.route_time_weight
        if 'median_connectivity_weight' in cost_weights:
            median_connectivity_weight = cost_weights['median_connectivity_weight']
        else:
            median_connectivity_weight = self.median_connectivity_weight
                
        if constraint_weight is None:
            constraint_weight = self.constraint_violation_weight

        # if we have more weights than routes, truncate the weights
        demand_time_weight = self._expand_batch_value(
            demand_time_weight, state.batch_size, state.device)
        route_time_weight = self._expand_batch_value(
            route_time_weight, state.batch_size, state.device)
        median_connectivity_weight = self._expand_batch_value(
            median_connectivity_weight, state.batch_size, state.device)
        constraint_weight = self._expand_batch_value(
            constraint_weight, state.batch_size, state.device)

        # normalize all time values by the maximum drive time in the graph
        time_normalizer = state.drive_times.flatten(1,2).max(1).values

        n_routes = state.n_routes_to_plan

        # fraction of demand not covered by routes, and fraction of routes
        frac_uncovered = cho.n_disconnected_demand_edges / state.n_demand_edges
        if state.max_route_len is None:
            denom = n_routes * state.max_route_len
        else:
            denom = n_routes * state.min_route_len
        # avoid division by 0
        denom[denom == 0] = 1
        # unserved demand is treated as taking twice the diameter of the graph
         # to get where it's going
        served_demand = cho.total_demand - cho.unserved_demand
        demand_cost = cho.mean_demand_time * served_demand + \
            cho.unserved_demand * time_normalizer * 2
        demand_cost /= cho.total_demand
        # demand_cost = cho.mean_demand_time
        route_cost = cho.total_route_time

        if self.use_weighted_connectivity:
            median_connectivity = cho.median_connectivity_weighted
        else:
            median_connectivity = cho.median_connectivity

        # average trip time, total route time, and trips-at-n-transfers
        if not no_norm:
            # normalize cost components
            demand_cost = demand_cost / time_normalizer
            route_cost = route_cost / (time_normalizer * n_routes + 1e-6)
            median_connectivity =  median_connectivity/ (time_normalizer)
            # cho.median_connectivity = median_connectivity
        cost_components = torch.stack(
            (demand_cost, route_cost, median_connectivity), dim=-1)
        weights = torch.stack(
            (demand_time_weight, route_time_weight,
             median_connectivity_weight), dim=-1)
        # Drop disabled components from the weighted cost (no-op when all
        # three are enabled). The raw cost_components above are left intact
        # so get_cost_components() still reports every component.
        weights = self.apply_enabled_mask(weights)

        # compute the weight for the violated-constraint penalty, as an
         # upper bound on how bad the demand and route cost components may be
        # demand_constraint_weight = 2
        # edge_times = state.street_adj.isfinite() * state.drive_times
        # max_edge_time = edge_times.flatten(1,2).max(-1)[0]
        # route_constraint_weight = 2 * max_edge_time * state.n_nodes
        # route_constraint_weight /= time_normalizer
        # dynamic_cv_weight = demand_constraint_weight * demand_time_weight + \
        #     route_constraint_weight * route_time_weight
        # constraint_weight *= dynamic_cv_weight 

        const_viol_cost = frac_uncovered + 0.1 * (frac_uncovered > 0)
        if not self.ignore_stops_oob:
            frac_stops_oob = cho.n_stops_oob / denom
            const_viol_cost += frac_stops_oob + 0.1 * (frac_stops_oob > 0)

        constraint_cost = const_viol_cost * constraint_weight
        return cost_components, weights, constraint_cost

    def get_cost_components(self, state, result=None, include_constraint=True,
                            no_norm=False):
        if result is None:
            result = self._cost_helper(state)
        components, _, constraint_cost = \
            self._compute_cost_components_from_result(
                state, result, no_norm=no_norm)
        if include_constraint:
            components = components + constraint_cost[:, None]
        return components

    def forward(self, state, constraint_weight=None, no_norm=False, 
                return_per_route_riders=False):
        cho = self._cost_helper(state, return_per_route_riders)
        components, weights, constraint_cost = \
            self._compute_cost_components_from_result(
                state, cho, constraint_weight=constraint_weight,
                no_norm=no_norm)
        cost = (components * weights).sum(dim=-1) + constraint_cost
        cho.cost = cost

        assert cost.isfinite().all(), "invalid cost was computed!"
        assert (cost >= 0).all(), "cost is negative!"

        return cho
    

class MultiObjectiveCostModule(MyCostModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable_weights = True
        # don't sample any edge cases
        self.pp_fraction = 0.33
        self.op_fraction = 0.33
        # median-connectivity fraction: required by the inherited
        # sample_variable_weights; this class predated its addition.
        self.mcw_fraction = 0.33

    def sample_weights(self, batch_size, device=None):
        weights = self.sample_variable_weights(batch_size, device)
        weights['demand_time_weight'][0] = 1.0
        weights['demand_time_weight'][1] = 0.0
        weights['demand_time_weight'][2] = 0.0

        weights['route_time_weight'][0] = 0.0
        weights['route_time_weight'][1] = 1.0
        weights['route_time_weight'][2] = 0.0

        weights['median_connectivity_weight'][0] = 0.0
        weights['median_connectivity_weight'][1] = 0.0
        weights['median_connectivity_weight'][2] = 1.0
        return self._mask_weight_dict(weights)

    def forward(self, state, return_per_route_riders=False):
        cho = super().forward(state, return_per_route_riders=return_per_route_riders)
        return cho
    
    def get_cost(self, cho):
        if self.use_weighted_connectivity ==True:
            costs = torch.stack((cho.mean_demand_time, cho.total_route_time, cho.median_connectivity_weighted), 
                                dim=-1)
        else:
            costs = torch.stack((cho.mean_demand_time, cho.total_route_time), 
                                dim=-1)
        any_violations = cho.are_constraints_violated()
        return costs, any_violations


class NikolicCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False,
                 unsatisfied_penalty_extra_s=UNSAT_PENALTY_EXTRA_S, 
                 ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.unsatisfied_penalty_extra_s = unsatisfied_penalty_extra_s

    def forward(self, state):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        cho = self._cost_helper(state)
        # Note that unlike Nikolic, we count trips that take >2 transfers as 
         # satisfied.
        tot_sat_demand = cho.total_demand - cho.unserved_demand
        w_2 = cho.total_demand_time / tot_sat_demand
        no_sat_dmd = torch.isclose(tot_sat_demand, 
                                   torch.zeros_like(tot_sat_demand))
        # if no demand is satisfied, set w_2 to the average time of all trips 
         # plus the penalty
        w_2[no_sat_dmd] = cho.trip_times[no_sat_dmd].mean(dim=(-2,-1))
        w_2 += self.unsatisfied_penalty_extra_s

        cost = cho.total_demand_time + w_2 * cho.unserved_demand

        assert not ((cost == 0) & (cho.total_demand > 0)).any()

        log.debug("finished nikolic")
        assert cost.isfinite().all(), "invalid cost was computed!"

        cho.cost = cost
        
        return cho

    def get_weights(self, device=None):
        return {}
    

def get_cost_module_from_cfg(cost_cfg: DictConfig, low_memory_mode=False,
                             symmetric_routes=True):
    # setup the cost function
    if cost_cfg.type == 'nikolic':
        cost_obj = NikolicCostModule(low_memory_mode=low_memory_mode, 
                                     **cost_cfg.kwargs)
    elif cost_cfg.type == 'mine':
        cost_obj = MyCostModule(low_memory_mode=low_memory_mode, 
                                **cost_cfg.kwargs)
    elif cost_cfg.type == 'multi':
        cost_obj = MultiObjectiveCostModule(
            low_memory_mode=low_memory_mode,
            **cost_cfg.kwargs)                                            
    return cost_obj


def check_for_duplicate_routes(routes_tensor):
    """check if any routes are duplicates of each other.

    In theory we want to avoid duplicate routes.  But no other work seems to 
    care.  Mumford doesn't check for them.b

    routes_tensor: a tensor of shape (batch_size, n_routes, route_len)"""
    routes_tensor = routes_tensor[:, None]
    same_stops = routes_tensor == routes_tensor.transpose(1, 2)
    routes_are_identical = same_stops.all(dim=-1)
    any_routes_are_identical = routes_are_identical.any(-1).any(-1)
    return any_routes_are_identical


def network_is_connected(routes_tensor, n_nodes):
    if routes_tensor.ndim == 2:
        # add a batch dimension
        routes_tensor = routes_tensor[None]

    are_connected = torch.zeros(routes_tensor.shape[0], dtype=bool)
    for bi in range(routes_tensor.shape[0]):
        elem_routes = routes_tensor[bi]
        adj_matrix = torch.eye(n_nodes+1, dtype=bool)
        for route in elem_routes:
            for src, dst in zip(route[:-1], route[1:]):
                adj_matrix[src, dst] = True
        # make symmetrical
        adj_matrix = adj_matrix | adj_matrix.t()
        # remove the dummy node entries of adj_matrix
        adj_matrix = adj_matrix[:-1, :-1]

        visited = [False] * n_nodes
        start_node = 0  # Start from an arbitrary node
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if not visited[node]:
                visited[node] = True
                for neighbour, is_connected in enumerate(adj_matrix[node]):
                    if is_connected and not visited[neighbour]:
                        queue.append(neighbour)
        
        are_connected[bi] = all(visited)
    
    return are_connected


def count_duplicate_stops(n_nodes, networks):
    if networks.ndim == 2:
        batch_size = 1
        networks = networks[None]
    else:
        batch_size = networks.shape[0]
    # count duplicate stops in routes
    route_lens = (networks > -1).sum(dim=-1)
    nodes_on_routes = tu.get_nodes_on_routes_mask(n_nodes, networks)
    n_nodes_covered = nodes_on_routes[..., :-1].sum(-1)
    n_ntwk_duplicates = (route_lens - n_nodes_covered).sum(-1)

    return n_ntwk_duplicates


def count_skipped_stops(network, shortest_path_lens):
    path_n_skips = shortest_path_lens - 2
    path_n_skips.clamp_(min=0)
    leg_n_skips = tu.get_route_leg_times(network, path_n_skips)
    total_n_skips = leg_n_skips.sum(dim=(1, 2))
    return total_n_skips

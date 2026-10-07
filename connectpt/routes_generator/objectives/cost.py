"""Vector objectives and immutable parameters for transit networks."""
import logging as log
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch
from torch import Tensor
from omegaconf import OmegaConf

from ..city.shortest_paths import aggregate_edge_features
from ..network.routes import get_nodes_on_routes_mask, get_route_edge_matrix, get_route_leg_times, to_routes
from .preferences import COST_WEIGHT_KEY_ORDER, COST_COMPONENT_NAMES, PreferenceSampler, scalarize
from .criteria import GOAL_NAMES, criterion

MEAN_STOP_TIME_S = 0
AVG_TRANSFER_WAIT_TIME_S = 300
EPSILON = 1e-6


def _finite_time_diameter(times, eps=EPSILON):
    """Per-batch max finite travel time for normalization."""
    finite_times = torch.where(torch.isfinite(times), times, torch.zeros_like(times))
    if finite_times.ndim >= 3:
        diameter = finite_times.flatten(1, 2).max(1).values
    else:
        diameter = finite_times.flatten().max().reshape(1)
    return diameter.clamp_min(eps)


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
    median_connectivity_weighted: Optional[Tensor] = None
    goals: Optional[Tensor] = None
    penalties: Optional[Tensor] = None
    preferences: Optional[Tensor] = None
    penalty_weights: Optional[list] = None
    components: Optional[Tensor] = None
    constraint_cost: Optional[Tensor] = None

    @property
    def cost(self):
        return scalarize(self.goals, self.preferences, self.penalties, self.penalty_weights)

    def add_penalty(self, value, weight=1.0):
        self.penalties = torch.cat((self.penalties, value.reshape(-1, 1)), dim=-1)
        self.penalty_weights = [*self.penalty_weights, weight]

    @property
    def mean_demand_time(self):
        served_demand = self.total_demand - self.unserved_demand
        # avoid division by 0
        return self.total_demand_time / (served_demand + EPSILON)

    def get_metrics(self):
        from .metrics import evaluate_network

        return evaluate_network(self)

    def get_metrics_tensor(self):
        """return a tensor with the metrics we usually report."""
        metrics = self.get_metrics()
        metrics = torch.stack([metrics[k] for k in metrics], dim=-1)
        return metrics

    def are_constraints_violated(self):
        return (self.n_stops_oob > 0) | (self.n_disconnected_demand_edges > 0) | (self.n_duplicate_stops > 0)  # | \


@dataclass(frozen=True)
class CostConstants:
    symmetric_routes: bool
    mean_stop_time_s: float
    avg_transfer_wait_time_s: float


@dataclass(frozen=True)
class CostParams:
    mean_stop_time_s: float
    avg_transfer_wait_time_s: float
    demand_time_weight: float
    route_time_weight: float
    median_connectivity_weight: float
    variable_weights: bool
    pp_fraction: float
    op_fraction: float
    mcw_fraction: float
    ignore_stops_oob: bool
    adjustment_degree_target: float
    adjustment_degree_objective: str
    adjustment_degree_gap: float
    adjustment_degree_mode: str
    objectives: list
    penalties: dict
    monitor: list


def build_cost(cost_cfg, **overrides):
    if isinstance(cost_cfg, CostParams):
        values = asdict(cost_cfg)
    else:
        values = OmegaConf.to_container(cost_cfg, resolve=True)
        if values.pop("type") != "mine":
            raise ValueError("unknown cost type")
        values = values["kwargs"]
        values = asdict(CostParams(**values))
    low_memory = overrides.pop("low_memory_mode", False)
    symmetric = overrides.pop("symmetric_routes", True)
    weights = dict(values["penalties"])
    for key, name in (("constraint_violation_weight", "constraints"), ("adjustment_degree_weight", "adjustment")):
        if key in overrides:
            weights[name] = overrides.pop(key)
    values.update(overrides)
    values["penalties"] = weights
    CostParams(**values)
    return MyCostModule(low_memory_mode=low_memory, symmetric_routes=symmetric, **values)


class CostModule(torch.nn.Module):
    def __init__(
        self,
        mean_stop_time_s=MEAN_STOP_TIME_S,
        avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
        symmetric_routes=True,
        low_memory_mode=False,
    ):
        super().__init__()
        self.mean_stop_time_s = mean_stop_time_s
        self.avg_transfer_wait_time_s = avg_transfer_wait_time_s
        self.symmetric_routes = symmetric_routes
        self.low_memory_mode = low_memory_mode

    @property
    def constants(self):
        return CostConstants(self.symmetric_routes, self.mean_stop_time_s, self.avg_transfer_wait_time_s)

    def with_params(self, **overrides):
        values = {f.name: getattr(self, f.name) for f in fields(CostParams) if f.name != "penalties"}
        values["penalties"] = self.penalty_weights
        return build_cost(
            CostParams(**values),
            symmetric_routes=self.symmetric_routes,
            low_memory_mode=self.low_memory_mode,
            **overrides,
        )

    def get_metric_names(self):
        from .metrics import METRIC_NAMES

        return METRIC_NAMES

    def _compute_all_pairs_times_floyd(self, state):
        # state.transit_times is ALREADY the all-pairs shortest transit-time
        # matrix (tau_Rij, transfer penalties included) -- it is produced by a
        # Floyd-Warshall pass in RouteGenBatchState._update_route_data. Running
        # Floyd-Warshall on it again does NOT add transfer penalties; it shaves
        # one transfer penalty per intermediate junction (transit_times only
        # satisfies the triangle inequality up to a +transfer_time slack after
        # the boarding-transfer subtraction), undercounting transfer time by
        # 10-30%. So we return it directly -- the same matrix ATT / the passenger
        # cost use, exactly as in the paper.
        return state.transit_times.clone()  # [B, N, N]

    @staticmethod
    def _demand_weighted_node_times(values, weights, dim=-1, eps=1e-9):
        """Per-node demand-weighted mean travel time:
            C_i = sum_j (D_ij / sum_j D_ij) * tau_ij
        computed along ``dim`` over reachable destinations only. Entries where
        ``values`` (tau) is non-finite -- the NaN-masked diagonal and
        unreachable pairs -- are dropped (weight 0). Nodes whose total demand to
        reachable destinations is ~0 return NaN so the caller's nan-aware
        reduction over nodes skips them. The weights are normalised by their own
        row sum, so the result is a genuine travel time (a convex combination of
        the tau values), not a demand-attenuated quantity."""
        valid = torch.isfinite(values)
        w = torch.where(valid, weights, torch.zeros_like(weights))
        v = torch.where(valid, values, torch.zeros_like(values))
        num = (w * v).sum(dim=dim)
        den = w.sum(dim=dim)
        return torch.where(den > eps, num / den.clamp(min=eps), torch.full_like(num, float("nan")))

    @staticmethod
    def _demand_weighted_pair_times(values, weights, dim=-1, eps=1e-9):
        """Per-pair terms of :meth:`_demand_weighted_node_times`:
            c_ij = (D_ij / sum_j D_ij) * tau_ij
        NaN where ``values`` is non-finite or the row demand is ~0."""
        valid = torch.isfinite(values)
        w = torch.where(valid, weights, torch.zeros_like(weights))
        den = w.sum(dim=dim, keepdim=True)
        return torch.where(valid & (den > eps), w * values / den.clamp(min=eps), torch.full_like(values, float("nan")))

    def _connectivity_values(self, state):
        all_pairs = self._compute_all_pairs_times_floyd(state)
        N = all_pairs.shape[-1]
        eye = torch.eye(N, device=all_pairs.device).bool().unsqueeze(0)
        max_T = _finite_time_diameter(state.drive_times)
        unreached_penalty = (2.0 * max_T).view(-1, 1, 1)
        unreachable = (~state.has_path) | (~all_pairs.isfinite())
        conn_vals = torch.where(unreachable, unreached_penalty.expand_as(all_pairs), all_pairs)
        return conn_vals.masked_fill(eye, float("nan"))

    def connectivity_pair_times(self, state):
        return self._demand_weighted_pair_times(self._connectivity_values(state), state.demand, dim=2)

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
        batch_routes = to_routes(state.routes).to(dev)
        route_lens = (batch_routes > -1).sum(-1)

        log.debug("summing cost components")

        zero = torch.zeros_like(route_lens)
        route_len_delta = state.min_route_len[:, None] - route_lens
        route_len_delta = route_len_delta.maximum(zero)
        # don't penalize placeholer "dummy" routes in the tensor
        route_len_delta[route_lens == 0] = 0
        if state.max_route_len is not None:
            route_len_over = route_lens - state.max_route_len[:, None]
            route_len_delta = route_len_delta + route_len_over.maximum(zero)

        # if there is a current route, it's already included in route_len_delta
        n_unstarted_routes = state.n_routes_left_to_plan - (state.has_current_route).to(torch.float32)
        n_stops_oob = route_len_delta.sum(-1) + n_unstarted_routes * state.min_route_len

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
        total_demand = demand_matrix.sum(dim=(1, 2))

        n_duplicate_stops = count_duplicate_stops(state.max_n_nodes, batch_routes)

        # all-pairs transit times (tau_Rij, transfer penalties included)
        all_pairs = self._compute_all_pairs_times_floyd(state)
        B, N, _ = all_pairs.shape

        conn_vals = self._connectivity_values(state)
        node_cw = self._demand_weighted_node_times(conn_vals, demand_matrix, dim=2)
        values = torch.nanmedian(node_cw, dim=1).values
        median_connectivity_weighted = torch.where(torch.isnan(values), torch.tensor(0.0, device=values.device), values)

        unserved_demand_matrix = demand_matrix * nopath

        output = CostHelperOutput(
            total_dmd_time,
            state.total_route_time,
            trips_at_transfers,
            total_demand,
            unserved_demand,
            total_transfers,
            trip_times,
            state.get_n_disconnected_demand_edges(),
            n_stops_oob,
            n_duplicate_stops,
            batch_routes,
            unserved_demand_matrix,
            median_connectivity_weighted=median_connectivity_weighted,
        )

        if return_per_route_riders:
            _, used_routes = get_route_edge_matrix(
                batch_routes,
                drive_times_matrix,
                self.mean_stop_time_s,
                self.symmetric_routes,
                self.low_memory_mode,
                return_used_routes=True,
            )

            used_routes.unsqueeze_(-1)
            route_seqs = aggregate_edge_features(state.route_nexts, used_routes, "concat")
            route_seqs.squeeze_(-1)
            per_route_riders = torch.zeros(batch_routes.shape[:2], device=dev)
            for bi in range(state.batch_size):
                for ri in range(batch_routes.shape[1]):
                    srcs, dsts, _ = torch.where(route_seqs[bi] == ri)
                    ri_demand = demand_matrix[bi, srcs, dsts].sum()
                    per_route_riders[bi, ri] = ri_demand

            output.per_route_riders = per_route_riders

        return output


class MyCostModule(CostModule, PreferenceSampler):
    def __init__(
        self,
        mean_stop_time_s=MEAN_STOP_TIME_S,
        avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
        symmetric_routes=True,
        low_memory_mode=False,
        demand_time_weight=0.33,
        route_time_weight=0.33,
        median_connectivity_weight=0.33,
        constraint_violation_weight=5,
        variable_weights=False,
        ignore_stops_oob=False,
        pp_fraction=0.33,
        op_fraction=0.33,
        mcw_fraction=0.33,
        objectives=("ATT", "RTT", "WMC"),
        penalties=None,
        monitor=(),
        adjustment_degree_weight=0.0,
        adjustment_degree_target=0.2,
        adjustment_degree_objective="raw",
        adjustment_degree_gap=0.1,
        adjustment_degree_mode="paper",
    ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s, symmetric_routes, low_memory_mode)
        if penalties is not None:
            expected = {"constraints", "adjustment", "edit_step", "forced_halt"}
            if set(penalties) != expected:
                raise ValueError(f"penalties must contain {sorted(expected)}")
            constraint_violation_weight = penalties["constraints"]
            adjustment_degree_weight = penalties["adjustment"]
        # Unified adjustment-degree penalty (single source of truth, shared by
        # BCO / metaheuristics / training). Gated: with weight 0 OR no seed set
        # the term is skipped entirely and cost is bit-for-bit unchanged.
        self.adjustment_degree_weight = float(adjustment_degree_weight)
        self.adjustment_degree_target = float(adjustment_degree_target)
        self.adjustment_degree_objective = adjustment_degree_objective
        self.adjustment_degree_gap = float(adjustment_degree_gap)
        self.adjustment_degree_mode = adjustment_degree_mode
        self.demand_time_weight = demand_time_weight
        self.route_time_weight = route_time_weight
        self.median_connectivity_weight = median_connectivity_weight
        self.constraint_violation_weight = constraint_violation_weight
        self.variable_weights = variable_weights
        self.pp_fraction = pp_fraction
        self.op_fraction = op_fraction
        self.mcw_fraction = mcw_fraction
        if self.variable_weights:
            self.pp_fraction = pp_fraction
            self.op_fraction = op_fraction
            self.mcw_fraction = mcw_fraction
            assert pp_fraction + op_fraction + mcw_fraction <= 1, "fractions of extreme samples must sum to <= 1"
        self.ignore_stops_oob = ignore_stops_oob
        # Which of the three cost components (demand / route / connectivity)
        # are active. Disabled components are dropped from the weighted cost
        # and from sampled / fixed cost-weight tensors; when all three are
        # enabled (the default) every masking helper below is a no-op, so
        # behaviour is bit-for-bit unchanged.
        self.objectives = tuple(objectives)
        self.monitor = tuple(monitor)
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must be nonempty and unique")
        for name in (*self.objectives, *self.monitor):
            criterion(name)
        self.penalty_weights = dict(
            penalties
            if penalties is not None
            else {
                "constraints": constraint_violation_weight,
                "adjustment": adjustment_degree_weight,
                "edit_step": 0.0,
                "forced_halt": 0.0,
            }
        )
        self._enabled_components = tuple(name in self.objectives for name in ("ATT", "RTT", "WMC"))
        self._all_components_enabled = all(self._enabled_components)

    @property
    def cost_component_names(self):
        return COST_WEIGHT_KEY_ORDER

    # ------------------------------------------------------------------
    # Enable / disable individual cost components
    # ------------------------------------------------------------------

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
        return tuple(COST_COMPONENT_NAMES[i] for i in self.enabled_component_indices)

    @property
    def disabled_component_names(self):
        return tuple(COST_COMPONENT_NAMES[i] for i, on in enumerate(self._enabled_components) if not on)

    @property
    def n_enabled_components(self):
        return int(sum(self._enabled_components))

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

    def _compute_cost_components_from_result(self, state, cho, constraint_weight=None, no_norm=False):
        cost_weights = state.cost_weights
        demand_time_weight = cost_weights["demand_time_weight"]
        route_time_weight = cost_weights["route_time_weight"]
        median_connectivity_weight = cost_weights["median_connectivity_weight"]

        if constraint_weight is None:
            constraint_weight = self.constraint_violation_weight

        # if we have more weights than routes, truncate the weights
        demand_time_weight = self._expand_batch_value(demand_time_weight, state.batch_size, state.device)
        route_time_weight = self._expand_batch_value(route_time_weight, state.batch_size, state.device)
        median_connectivity_weight = self._expand_batch_value(
            median_connectivity_weight, state.batch_size, state.device
        )
        constraint_weight = self._expand_batch_value(constraint_weight, state.batch_size, state.device)

        cost_components = torch.stack(
            [criterion(name).compute(state, cho, no_norm=no_norm) for name in GOAL_NAMES], dim=-1
        )
        weights = torch.stack((demand_time_weight, route_time_weight, median_connectivity_weight), dim=-1)
        # Drop disabled components from the weighted cost (no-op when all
        # three are enabled). The raw cost_components above are left intact
        # so get_cost_components() still reports every component.
        weights = self.apply_enabled_mask(weights)

        if bool((constraint_weight != 0).any()):
            constraint_cost = (
                criterion("constraints", ignore_stops_oob=self.ignore_stops_oob).compute(state, cho) * constraint_weight
            )
        else:
            constraint_cost = state.drive_times.new_zeros(state.batch_size)
        return cost_components, weights, constraint_cost

    def get_cost_components(self, state, result=None, include_constraint=True, no_norm=False):
        if result is None:
            result = self._cost_helper(state)
        components, _, constraint_cost = self._compute_cost_components_from_result(state, result, no_norm=no_norm)
        if include_constraint:
            components = components + constraint_cost[:, None]
        return components

    def forward(self, state, constraint_weight=None, no_norm=False, return_per_route_riders=False):
        result = self._cost_helper(state, return_per_route_riders)
        components, weights, constraints = self._compute_cost_components_from_result(
            state, result, constraint_weight=constraint_weight, no_norm=no_norm
        )
        indices = [GOAL_NAMES.index(name) for name in self.objectives]
        result.goals = components[:, indices]
        result.preferences = weights[:, indices]
        result.components = components
        result.constraint_cost = constraints
        penalty_weight = self._expand_batch_value(
            self.constraint_violation_weight if constraint_weight is None else constraint_weight,
            state.batch_size,
            state.device,
        )
        raw_constraint = (
            criterion("constraints", ignore_stops_oob=self.ignore_stops_oob).compute(state, result)
            if bool((penalty_weight != 0).any())
            else constraints
        )
        result.penalties = raw_constraint[:, None]
        result.penalty_weights = [penalty_weight]
        if self.penalty_weights["adjustment"] > 0 and state.adjustment_seed.numel():
            penalty = criterion(
                "adjustment",
                gap=self.adjustment_degree_gap,
                mode=self.adjustment_degree_mode,
                target=self.adjustment_degree_target,
                objective=self.adjustment_degree_objective,
            ).compute(state)
            result.add_penalty(penalty, self.penalty_weights["adjustment"])
        else:
            result.add_penalty(constraints.new_zeros(state.batch_size), 1.0)
        assert result.cost.isfinite().all(), "invalid cost was computed!"
        assert (result.cost >= 0).all(), "cost is negative!"
        return result


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
        adj_matrix = torch.eye(n_nodes + 1, dtype=bool)
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
    nodes_on_routes = get_nodes_on_routes_mask(n_nodes, networks)
    n_nodes_covered = nodes_on_routes[..., :-1].sum(-1)
    n_ntwk_duplicates = (route_lens - n_nodes_covered).sum(-1)

    return n_ntwk_duplicates


def count_skipped_stops(network, shortest_path_lens):
    path_n_skips = shortest_path_lens - 2
    path_n_skips.clamp_(min=0)
    leg_n_skips = get_route_leg_times(network, path_n_skips)
    total_n_skips = leg_n_skips.sum(dim=(1, 2))
    return total_n_skips

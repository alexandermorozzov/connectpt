"""Physical network components shared by cost and standalone criteria."""
import logging as log
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
from ..city.shortest_paths import aggregate_edge_features
from ..network.routes import get_route_edge_matrix, get_nodes_on_routes_mask, to_routes
from .preferences import scalarize

from ..common.const import EPSILON


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


class PhysicalCost(torch.nn.Module):
    def __init__(
        self,
        mean_stop_time_s,
        avg_transfer_wait_time_s,
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


def count_duplicate_stops(n_nodes, networks):
    if networks.ndim == 2:
        networks = networks[None]
    # count duplicate stops in routes
    route_lens = (networks > -1).sum(dim=-1)
    nodes_on_routes = get_nodes_on_routes_mask(n_nodes, networks)
    n_nodes_covered = nodes_on_routes[..., :-1].sum(-1)
    n_ntwk_duplicates = (route_lens - n_nodes_covered).sum(-1)

    return n_ntwk_duplicates

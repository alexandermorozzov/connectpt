"""Vector objectives and immutable parameters for transit networks."""
from collections import deque
from dataclasses import asdict, dataclass, fields

import torch

from ..network.routes import get_route_leg_times
from .preferences import PreferenceSampler
from .criteria import criterion
from .const import GOAL_NAMES, COST_WEIGHT_KEY_ORDER, COST_COMPONENT_NAMES
from .components import PhysicalCost
from .const import ACTION_PENALTIES, STATE_PENALTIES
from ..common.parameters import CostConstants
from .components import CostHelperOutput

__all__ = ["build_cost", "MyCostModule", "CostParams", "CostConstants", "CostHelperOutput"]


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
    objectives: list
    penalties: dict
    monitor: list


def build_cost(parameters: CostParams, *, symmetric_routes: bool, low_memory_mode: bool, **overrides):
    if not isinstance(parameters, CostParams):
        raise TypeError("build_cost requires CostParams")
    values = asdict(parameters)
    weights = dict(overrides.pop("penalties", values["penalties"]))
    for key, name in (("constraint_violation_weight", "constraints"), ("adjustment_degree_weight", "adjustment")):
        if key in overrides:
            weights[name] = overrides.pop(key)
    values.update(overrides)
    values["penalties"] = weights
    CostParams(**values)
    return MyCostModule(low_memory_mode=low_memory_mode, symmetric_routes=symmetric_routes, **values)


class MyCostModule(PhysicalCost):
    def with_params(self, **overrides):
        values = {f.name: getattr(self, f.name) for f in fields(CostParams) if f.name != "penalties"}
        values["penalties"] = self.penalty_weights
        return build_cost(
            CostParams(**values),
            symmetric_routes=self.symmetric_routes,
            low_memory_mode=self.low_memory_mode,
            **overrides,
        )

    def __init__(
        self,
        *,
        mean_stop_time_s,
        avg_transfer_wait_time_s,
        symmetric_routes,
        low_memory_mode,
        demand_time_weight,
        route_time_weight,
        median_connectivity_weight,
        variable_weights,
        ignore_stops_oob,
        pp_fraction,
        op_fraction,
        mcw_fraction,
        objectives,
        penalties,
        monitor,
        adjustment_degree_target,
        adjustment_degree_objective,
    ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s, symmetric_routes, low_memory_mode)
        expected = set(ACTION_PENALTIES) | set(STATE_PENALTIES)
        if not expected <= set(penalties):
            raise ValueError(f"penalties must contain {sorted(expected)}")
        constraint_violation_weight = penalties["constraints"]
        adjustment_degree_weight = penalties["adjustment"]
        # Unified adjustment-degree penalty (single source of truth, shared by
        # BCO / metaheuristics / training). Gated: with weight 0 OR no seed set
        # the term is skipped entirely and cost is bit-for-bit unchanged.
        self.adjustment_degree_weight = float(adjustment_degree_weight)
        self.adjustment_degree_target = float(adjustment_degree_target)
        self.adjustment_degree_objective = adjustment_degree_objective
        self.demand_time_weight = demand_time_weight
        self.route_time_weight = route_time_weight
        self.median_connectivity_weight = median_connectivity_weight
        self.constraint_violation_weight = constraint_violation_weight
        self.variable_weights = variable_weights
        self.pp_fraction = pp_fraction
        self.op_fraction = op_fraction
        self.mcw_fraction = mcw_fraction
        if self.variable_weights and not pp_fraction + op_fraction + mcw_fraction <= 1:
            raise ValueError("fractions of extreme samples must sum to <= 1")
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
        self.goal_criteria = {name: criterion(name) for name in dict.fromkeys((*GOAL_NAMES, *self.objectives))}
        self.monitor_criteria = {name: criterion(name) for name in self.monitor}
        self.penalty_weights = dict(penalties)
        parameters = {
            "constraints": {"ignore_stops_oob": self.ignore_stops_oob},
            "adjustment": {"target": self.adjustment_degree_target, "objective": self.adjustment_degree_objective},
        }
        self.penalty_criteria = {
            name: criterion(name, **parameters.get(name, {}))
            for name in penalties if name not in ACTION_PENALTIES
        }
        self._enabled_components = tuple(name in self.objectives for name in ("ATT", "RTT", "WMC"))
        self._all_components_enabled = all(self._enabled_components)
        self.preference_sampler = PreferenceSampler(
            objectives=self.objectives, demand_time_weight=demand_time_weight,
            route_time_weight=route_time_weight, median_connectivity_weight=median_connectivity_weight,
            variable_weights=variable_weights, pp_fraction=pp_fraction,
            op_fraction=op_fraction, mcw_fraction=mcw_fraction,
        )

    def get_weights(self, device=None):
        return self.preference_sampler.get_weights(device)

    def sample_variable_weights(self, batch_size, device=None):
        return self.preference_sampler.sample_variable_weights(batch_size, device)

    def get_preference_weights(self, state, normalize=False):
        return self.preference_sampler.get_preference_weights(state, normalize)

    def sample(self, batch_size, device=None):
        return self.preference_sampler.sample(batch_size, device)

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

    def get_cost_components(self, state, result=None, include_constraint=True, no_norm=False):
        result = self(state, no_norm=no_norm) if result is None else result
        components = torch.stack([result.raw_goals[name] for name in GOAL_NAMES], -1) if no_norm else result.components
        return components + result.constraint_cost[:, None] if include_constraint else components

    def forward(self, state, constraint_weight=None, no_norm=False, return_per_route_riders=False):
        from .evaluate import evaluate
        return evaluate(state, self, constraint_weight=constraint_weight, no_norm=no_norm,
                        return_per_route_riders=return_per_route_riders)


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


def count_skipped_stops(network, shortest_path_lens):
    path_n_skips = shortest_path_lens - 2
    path_n_skips.clamp_(min=0)
    leg_n_skips = get_route_leg_times(network, path_n_skips)
    total_n_skips = leg_n_skips.sum(dim=(1, 2))
    return total_n_skips

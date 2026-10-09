"""Named, pure criteria used as objectives, penalties or monitors."""
from abc import ABC, abstractmethod


from .preferences import GOAL_NAMES
from .penalties import PENALTIES


class Criterion(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def compute(self, state):
        """Return one value per graph."""


class NetworkCriterion(Criterion):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def compute(self, state, result=None, *, no_norm=False):
        from .components import PhysicalCost
        from .components import _finite_time_diameter

        values = (
            result
            if result is not None
            else PhysicalCost(
                state.mean_stop_time[0].item(), state.transfer_time_s[0].item(), state.symmetric_routes
            )._cost_helper(state)
        )
        diameter = _finite_time_diameter(state.drive_times)
        if self.name == "ATT":
            served = values.total_demand - values.unserved_demand
            value = values.mean_demand_time * served + values.unserved_demand * diameter * 2
            value /= values.total_demand
            return value if no_norm else value / diameter
        if self.name == "RTT":
            value = values.total_route_time
            return value if no_norm else value / (diameter * state.n_routes_to_plan + 1e-6)
        if self.name == "WMC":
            value = values.median_connectivity_weighted
            return value if no_norm else value / diameter
        if self.name == "n_disconnected":
            return values.n_disconnected_demand_edges
        return (values.trips_at_transfers / values.total_demand[:, None])[
            :, ("d0", "d1", "d2", "d_un").index(self.name)
        ] * 100


class PenaltyCriterion(Criterion):
    def __init__(self, name, **parameters):
        self._name = name
        self.parameters = parameters

    @property
    def name(self):
        return self._name

    def compute(self, state, result=None):
        from .penalties import constraint_penalty, adjustment_degree, adjustment_penalty
        from .penalties import edit_step_penalty, forced_halt_penalty
        from ..network.routes import to_routes

        if self.name == "constraints":
            if result is None:
                from .components import PhysicalCost

                result = PhysicalCost(
                    state.mean_stop_time[0].item(), state.transfer_time_s[0].item(), state.symmetric_routes
                )._cost_helper(state)
            return constraint_penalty(state, result, self.parameters["ignore_stops_oob"])
        if self.name == "adjustment":
            current = to_routes(state.routes).to(state.device)
            seed = state.adjustment_seed.to(state.device)
            nr, width = min(current.shape[1], seed.shape[1]), min(current.shape[-1], seed.shape[-1])
            degrees = adjustment_degree(
                current[:, :nr, :width],
                seed[:, :nr, :width],
                state.symmetric_routes,
                gap=self.parameters["gap"],
                mode=self.parameters["mode"],
            )
            network_degree = degrees.sum(-1) / max(float(current.shape[1]), 1.0)
            return adjustment_penalty(network_degree, self.parameters["objective"], self.parameters["target"])
        if self.name == "edit_step":
            return edit_step_penalty(self.parameters["action"], self.parameters["active"], 1.0)
        return forced_halt_penalty(self.parameters["active"], self.parameters["forced"], 1.0)


REGISTRY = {
    **dict.fromkeys((*GOAL_NAMES, "d0", "d1", "d2", "d_un", "n_disconnected"), NetworkCriterion),
    **dict.fromkeys(PENALTIES, PenaltyCriterion),
}


def criterion(name, **parameters):
    if name not in REGISTRY:
        raise ValueError(f"unknown criterion {name!r}")
    return REGISTRY[name](name, **parameters)

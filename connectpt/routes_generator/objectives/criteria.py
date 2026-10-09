"""Criterion factories consume shared physical results, never rescore a state."""
from dataclasses import dataclass

from .components import _finite_time_diameter
from .penalties import constraint_penalty, adjustment_penalty


@dataclass(frozen=True)
class Criterion:
    name: str
    value: object
    normalize: object = None

    def compute(self, state, result, *, no_norm=False):
        value = self.value(state, result)
        return value if no_norm or self.normalize is None else self.normalize(state, value)


def _att(state, result):
    served = result.total_demand - result.unserved_demand
    value = result.mean_demand_time * served + result.unserved_demand * _finite_time_diameter(state.drive_times) * 2
    return value / result.total_demand


def _time(state, value):
    return value / _finite_time_diameter(state.drive_times)


def _rtt(state, value):
    return value / (_finite_time_diameter(state.drive_times) * state.n_routes_to_plan + 1e-6)


def _constraints(*, ignore_stops_oob):
    return Criterion("constraints", lambda state, result: constraint_penalty(state, result, ignore_stops_oob))


def _adjustment(*, target, objective):
    import torch
    adjustment_penalty(torch.zeros(1), objective, target)
    return Criterion("adjustment", lambda state, result: adjustment_penalty(result.adjustment, objective, target)
                     if state.adjustment_seed.numel() else torch.zeros_like(result.adjustment))


def _transfer(name, index):
    return lambda: Criterion(name, lambda state, result: (result.trips_at_transfers / result.total_demand[:, None])[:, index] * 100)


REGISTRY = {
    "ATT": lambda: Criterion("ATT", _att, _time),
    "RTT": lambda: Criterion("RTT", lambda state, result: result.total_route_time, _rtt),
    "WMC": lambda: Criterion("WMC", lambda state, result: result.median_connectivity_weighted, _time),
    **{name: _transfer(name, index) for index, name in enumerate(("d0", "d1", "d2", "d_un"))},
    "n_disconnected": lambda: Criterion("n_disconnected", lambda state, result: result.n_disconnected_demand_edges),
    "constraints": _constraints,
    "adjustment": _adjustment,
}


def criterion(name, **parameters):
    if name not in REGISTRY:
        raise ValueError(f"unknown criterion {name!r}")
    return REGISTRY[name](**parameters)

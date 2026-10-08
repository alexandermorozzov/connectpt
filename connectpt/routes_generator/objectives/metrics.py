"""Score a route set into a full metric row.

The unified objective computes normalized RTT / WMC / cost components during the
search; this module turns the engine's raw metrics dict (plus the route tensors)
into the full metric row every results table reports -- including the derived
metrics the engine does not emit (adjustment degree vs the seed network, edge
redundancy). All tables use the same evaluator.

Adjustment-degree gap/mode come from the composed ``cost.kwargs`` block.
"""
from __future__ import annotations

import numpy as np
import torch

from .penalties import adjustment_degree
from .preferences import scalarize
from ..network.corruption import redundancy_fraction
from ..network.routes import to_routes


def _adjustment_values(adjustment=None):
    """Resolve adjustment settings at the call boundary, never at import time."""
    values = adjustment
    return values.adjustment_degree_gap, values.adjustment_degree_mode


def metric_value(metrics, key, default=np.nan):
    """Read one metric as a plain float (mean over the batch), or ``default``."""
    if key not in metrics or metrics[key] is None:
        return default
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 0:
            return default
        return float(value.float().mean().item())
    return float(value)


def adj_vs_init(routes, init_routes, adjustment=None):
    """Mean adjustment degree of a route set vs the init network."""
    r = to_routes(routes)
    s = to_routes(init_routes)
    if r.ndim == 2:
        r = r[None]
    if s.ndim == 2:
        s = s[None]
    nr = min(r.shape[1], s.shape[1])
    w = min(r.shape[-1], s.shape[-1])
    gap, mode = _adjustment_values(adjustment)
    return float(adjustment_degree(r[:, :nr, :w], s[:, :nr, :w], True, gap=gap, mode=mode).mean().item())


def route_adjustment_degrees(routes, reference_routes, adjustment=None):
    """Per-route adjustment degree vs a reference network (numpy vector).

    Same gap/mode as :func:`adj_vs_init`, but without the mean -- route figures
    colour each route by how much it moved, and the mean of this vector is the
    reported ``adj_vs_seed``. Both route sets are padded to a common width so a
    candidate longer than the seed still aligns.
    """
    r = to_routes(routes).long()
    s = to_routes(reference_routes).long()
    if r.ndim == 3:
        r = r[0]
    if s.ndim == 3:
        s = s[0]
    width = max(int(r.shape[-1]), int(s.shape[-1]))
    r, s = to_routes(r, width=width), to_routes(s, width=width)
    gap, mode = _adjustment_values(adjustment)
    return adjustment_degree(r, s, True, gap=gap, mode=mode)[0].detach().cpu().numpy()


def redundancy_pct(routes):
    """% of edge traversals that re-cover an already-covered edge."""
    R = to_routes(routes)
    if R.ndim == 3:
        R = R[0]
    return 100.0 * redundancy_fraction(R)


def conn_metric(m):
    """The optimized weighted connectivity."""
    v = metric_value(m, "median_connectivity_weighted")
    return v


def select_metrics(row: dict, keep) -> dict:
    """Filter a metric row to the ``keep`` keys (preserving present ones)."""
    keep = list(keep)
    return {k: row[k] for k in keep if k in row}


class Metrics(dict):
    """Named network measurements, in the units of the result tables."""


METRIC_NAMES = (
    "cost",
    "ATT",
    "RTT",
    "$d_0$",
    "$d_1$",
    "$d_2$",
    "$d_{un}$",
    "# disconnected node pairs",
    "# stops out of bounds",
    "median_connectivity_weighted",
)


def evaluate_network(state_or_result, cost=None, *, routes=None, seed=None, adjustment=None):
    """Single source for network metrics; accept an already evaluated vector."""
    if routes is not None:
        m = state_or_result
        rt = to_routes(routes)
        return Metrics(
            {
                "ATT": metric_value(m, "ATT"),
                "RTT": metric_value(m, "RTT"),
                # WMC uses the lower median of demand-weighted node times.
                "WMC": conn_metric(m),
                "adj_vs_seed": adj_vs_init(rt, seed, adjustment),
                # normalized cost components -- the RTT / WMC the algorithm actually
                # optimizes (route_cost and weighted_median_connectivity, /time_normalizer).
                "rtt_cost": metric_value(m, "cost_route_component"),
                "wmc_cost": metric_value(m, "cost_connectivity_component"),
                "cost": metric_value(m, "cost"),
                "d0": metric_value(m, "$d_0$"),
                "d1": metric_value(m, "$d_1$"),
                "d2": metric_value(m, "$d_2$"),
                "d_un": metric_value(m, "$d_{un}$"),
                "redun%": redundancy_pct(rt),
            }
        )

    result = cost(state_or_result) if cost is not None else state_or_result
    percent = (result.trips_at_transfers / result.total_demand[:, None]) * 100
    return Metrics(
        zip(
            METRIC_NAMES,
            (
                result.cost,
                result.mean_demand_time / 60,
                result.total_route_time / 60,
                percent[:, 0],
                percent[:, 1],
                percent[:, 2],
                percent[:, 3],
                result.n_disconnected_demand_edges.float(),
                result.n_stops_oob.float(),
                result.median_connectivity_weighted / 60,
            ),
        )
    )


def connectivity_matrix(state, cost):
    """Per-pair demand-weighted transit times in seconds, with the WMC penalty."""
    return cost.connectivity_pair_times(state)


def _expand_batch_value(value, batch_size: int, device):
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32)
    else:
        value = torch.tensor(value, device=device, dtype=torch.float32)
    if value.numel() == 1:
        value = value.expand(batch_size)
    elif value.shape[0] > batch_size:
        value = value[:batch_size]
    return value


def _cost_weight(state, cost_obj, name: str):
    return _expand_batch_value(state.cost_weights[name], state.batch_size, state.device)


def cost_breakdown(state, cost_obj):
    cho = cost_obj._cost_helper(state)
    demand_time_weight = _cost_weight(state, cost_obj, "demand_time_weight")
    route_time_weight = _cost_weight(state, cost_obj, "route_time_weight")
    median_connectivity_weight = _cost_weight(state, cost_obj, "median_connectivity_weight")
    constraint_weight = _expand_batch_value(
        cost_obj.constraint_violation_weight,
        state.batch_size,
        state.device,
    )

    time_normalizer = state.drive_times.flatten(1, 2).max(1).values
    n_routes = state.n_routes_to_plan
    served_demand = cho.total_demand - cho.unserved_demand
    demand_component = (
        cho.mean_demand_time * served_demand + cho.unserved_demand * time_normalizer * 2
    ) / cho.total_demand
    route_component = cho.total_route_time
    connectivity_component = cho.median_connectivity_weighted

    demand_component = demand_component / time_normalizer
    route_component = route_component / (time_normalizer * n_routes + 1e-6)
    connectivity_component = connectivity_component / time_normalizer

    demand_term = demand_component * demand_time_weight
    route_term = route_component * route_time_weight
    connectivity_term = connectivity_component * median_connectivity_weight

    frac_uncovered = cho.n_disconnected_demand_edges / state.n_demand_edges
    penalty_component = frac_uncovered + 0.1 * (frac_uncovered > 0)
    if not cost_obj.ignore_stops_oob:
        denom = n_routes * state.min_route_len
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        frac_stops_oob = cho.n_stops_oob / denom
        penalty_component = penalty_component + frac_stops_oob + 0.1 * (frac_stops_oob > 0)
    penalty_term = penalty_component * constraint_weight
    cost_sum = scalarize(
        torch.stack((demand_component, route_component, connectivity_component), -1),
        torch.stack((demand_time_weight, route_time_weight, median_connectivity_weight), -1),
        penalty_component[:, None],
        [constraint_weight],
    )

    def mean_cpu(x):
        return x.float().mean().detach().cpu()

    return {
        "cost_demand_component": mean_cpu(demand_component),
        "cost_route_component": mean_cpu(route_component),
        "cost_connectivity_component": mean_cpu(connectivity_component),
        "cost_penalty_component": mean_cpu(penalty_component),
        "cost_demand_term": mean_cpu(demand_term),
        "cost_route_term": mean_cpu(route_term),
        "cost_connectivity_term": mean_cpu(connectivity_term),
        "cost_penalty_term": mean_cpu(penalty_term),
        "cost_sum_check": mean_cpu(cost_sum),
        "cost_demand_weight": mean_cpu(demand_time_weight),
        "cost_route_weight": mean_cpu(route_time_weight),
        "cost_connectivity_weight": mean_cpu(median_connectivity_weight),
        "cost_constraint_weight": mean_cpu(constraint_weight),
    }


def evaluate_routes(routes, tensors, spec, *, cost, device, seed_routes=None):
    """Evaluate a fixed network using already resolved objective parameters."""
    from ..city.graph import to_batch
    from ..env.state import RouteGenBatchState

    state = RouteGenBatchState(
        to_batch(tensors, device),
        cost.constants,
        spec["n_routes"],
        spec["min_route_len"],
        spec["max_route_len"],
        preferences=cost.get_weights(device),
    )
    scored = to_routes(routes).to(device)
    state.add_new_routes(scored)
    if seed_routes is not None:
        state.set_adjustment_seed(seed_routes)
    values = evaluate_network(state, cost)
    values.update(cost_breakdown(state, cost))
    return values, scored

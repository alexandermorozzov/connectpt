"""Per-component cost breakdown for a scored route set.

Recomputes the individual optimization terms (demand-time, route-time, weighted
median connectivity) and the penalty term from a cost module + routes, on the
same normalized scale the scalar cost uses, plus the two reported demand-weighted
WMC variants. Used to annotate result tables with a cost decomposition.

This is the library home of what used to be
``eval_lib.helpers.compute_cost_breakdown`` / ``add_cost_breakdown_to_metrics``;
the arithmetic is identical.
"""
from __future__ import annotations

import torch

from ..transit_time_estimator import RouteGenBatchState
from ..torch_utils import get_batch_tensor_from_routes


def _as_route_tensor(routes):
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()


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
    fallback = getattr(cost_obj, name)
    return _expand_batch_value(state.cost_weights.get(name, fallback), state.batch_size, state.device)


def compute_cost_breakdown(dataloader, eval_cfg, cost_obj, routes_tensor, device):
    data = next(iter(dataloader))
    if device is not None and device.type != "cpu":
        data = data.cuda()
    state = RouteGenBatchState(
        data,
        cost_obj,
        eval_cfg.n_routes,
        eval_cfg.min_route_len,
        eval_cfg.max_route_len,
    )
    routes_tensor = _as_route_tensor(routes_tensor).to(state.device)
    state.add_new_routes(routes_tensor)

    cho = cost_obj._cost_helper(state)
    demand_time_weight = _cost_weight(state, cost_obj, "demand_time_weight")
    route_time_weight = _cost_weight(state, cost_obj, "route_time_weight")
    median_connectivity_weight = _cost_weight(state, cost_obj, "median_connectivity_weight")
    constraint_weight = _expand_batch_value(
        getattr(cost_obj, "constraint_violation_weight", 0.0),
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
    connectivity_component = cho.median_connectivity_weighted if cost_obj.use_weighted_connectivity else cho.median_connectivity

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
    cost_sum = demand_term + route_term + connectivity_term + penalty_term

    def mean_cpu(x):
        return x.float().mean().detach().cpu()

    # Both demand-weighted WMC variants (modified-Cp), reported alongside the
    # optimized WMC. /60 to match the WMC column scale (get_metrics divides too).
    _wmean = getattr(cho, "mean_weighted_connectivity", None)
    _wmed = getattr(cho, "median_weighted_connectivity", None)
    return {
        "WMC_mean": mean_cpu(_wmean / 60) if _wmean is not None else float("nan"),
        "WMC_median": mean_cpu(_wmed / 60) if _wmed is not None else float("nan"),
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


def add_cost_breakdown_to_metrics(metrics, dataloader, eval_cfg, cost_obj, routes_tensor, device):
    metrics = dict(metrics)
    metrics.update(compute_cost_breakdown(dataloader, eval_cfg, cost_obj, routes_tensor, device))
    return metrics

"""Export prepared evaluation values, and assemble fixed initial-network states."""
from ..city.graph import to_batch
from ..env.state import RouteGenBatchState
from ..objectives.evaluate import evaluate
from ..objectives.metrics import evaluate_network


def export_evaluation(result, *, index=None, reduction=None):
    """Export one batch index or an explicit mean; never evaluate routes/cost."""
    if (index is None) == (reduction is None):
        raise ValueError("choose one index or an explicit reduction")
    if reduction not in (None, "mean"):
        raise ValueError("supported reduction is mean")
    metrics = evaluate_network(result)
    values = {
        "ATT": metrics["ATT"], "RTT": metrics["RTT"], "WMC": metrics["median_connectivity_weighted"],
        "cost": result.cost, "adj_vs_seed": result.adjustment,
        "rtt_cost": result.normalized["RTT"], "wmc_cost": result.normalized["WMC"],
        **{name: metrics[key] for name, key in zip(("d0", "d1", "d2", "d_un"), ("$d_0$", "$d_1$", "$d_2$", "$d_{un}$"))},
        "redun%": result.redundancy * 100,
    }
    return {name: float(value[index].detach().cpu()) if index is not None else float(value.mean().detach().cpu())
            for name, value in values.items()}


def evaluate_initial(instance, cost, *, device):
    """Build an Initial state using the search cost and its original reference."""
    state = RouteGenBatchState(to_batch(instance, device), cost.constants,
                              instance.spec["n_routes"], instance.spec["min_route_len"], instance.spec["max_route_len"],
                              preferences=cost.get_weights(device))
    state.add_new_routes(instance.init_routes.to(device))
    state.set_adjustment_seed(instance.init_routes)
    return evaluate(state, cost)

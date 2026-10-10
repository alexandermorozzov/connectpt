"""Search one existing network at a seeded preference point."""
from dataclasses import dataclass
from time import perf_counter

import torch
from torch_geometric.loader import DataLoader

from ..city.graph import to_batch
from ..env.state import RouteGenBatchState
from ..network.routes import to_routes
from ..objectives.metrics import Metrics, evaluate_network
from ..objectives.evaluate import Evaluation
from ..common.random import seed_everything
from .colony import Colony


@dataclass
class SearchResult:
    routes: torch.Tensor
    metrics: Metrics
    history: torch.Tensor
    evaluation: Evaluation


def run_search(
    instance,
    method,
    plan,
    alpha,
    adjustment,
    seed,
    iterations,
    *,
    cost,
    device,
    silent,
    logger_list=(),
    mutation_counts_out=None,
):
    overrides = dict(adjustment)
    if alpha is not None:
        overrides.update(route_time_weight=float(alpha), median_connectivity_weight=float(1 - alpha))
    cost = cost.with_params(**overrides)
    # Iterator creation consumes the same torch base seed as the original runner.
    loader = DataLoader(to_batch(instance.tensors).to_data_list(), batch_size=1)
    seed_everything(seed)
    histories, routes, evaluations, collected = [], [], [], None
    for graph in loader:
        graph = graph.to(device)
        started = perf_counter()
        spec = instance.spec
        state = RouteGenBatchState(
            graph,
            cost.constants,
            spec["n_routes"],
            spec["min_route_len"],
            spec["max_route_len"],
            preferences=cost.get_weights(),
        )
        initial = to_routes(instance.init_routes, n_routes=int(state.n_routes_to_plan.min().item()))
        initial = initial.expand(state.batch_size, -1, -1).to(device)
        state, evaluation, history = Colony(
            state,
            cost,
            initial,
            plan=plan,
            n_iterations=int(iterations),
            passes_per_it=method.passes_per_it,
            mod_steps_per_pass=method.mod_steps_per_pass,
            shorten_prob=method.shorten_prob,
            force_linking_unlinked=method.force_linking_unlinked,
            process_neural_bees_sequentially=method.process_neural_bees_sequentially,
            silent=silent,
            logger_list=logger_list,
            mutation_counts_out=mutation_counts_out,
        ).run()
        elapsed = perf_counter() - started
        metrics = evaluate_network(evaluation)
        evaluations.append(evaluation)
        metrics["average wall-clock duration"] = torch.full_like(metrics["ATT"], elapsed)
        metrics["average # iterations"] = torch.full_like(metrics["ATT"], history.shape[1])
        if collected is None:
            collected = metrics
        else:
            for key, value in metrics.items():
                collected[key] = torch.cat((collected[key], value))
        routes.append(to_routes(state.routes).to(device))
        histories.append(history)
    return SearchResult(torch.cat(routes), collected, torch.cat(histories), Evaluation.concatenate(evaluations))

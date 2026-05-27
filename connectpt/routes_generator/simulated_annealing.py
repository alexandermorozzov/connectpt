"""Simulated annealing with reheating for transit route network optimization.

Ported from the ancestor repository AHolliday/transit_learning
(``learning/simulated_annealing.py``). The Hydra CLI ``main()`` entry point was
dropped; this module exposes the algorithm as a library function whose
signature matches the ``method_fn`` contract of
:func:`connectpt.routes_generator.utils.test_method`, i.e.
``method_fn(state, cost_obj, init_network, *, sum_writer, silent, **kwargs)``
returning ``(state, cost_history)``.

The algorithm is purely heuristic -- it never uses a neural model. Candidate
moves are produced by the bee-style type-1 / type-2 mutators
(:func:`get_bee_1_variants` / :func:`get_bee_2_variants`).
"""
import math
import logging as log

from tqdm import tqdm
import torch

from .initialization import get_direct_sat_dmd
from .torch_utils import reconstruct_all_paths, get_unselected_routes
from .bee_colony import get_bee_1_variants, get_bee_2_variants


# Temperature Schedules
def exponential_cooling(current_temp, rate):
    return current_temp * rate


def linear_cooling(current_temp, rate):
    return current_temp - rate


COOLING_SCHEDULES = {
    'exponential': exponential_cooling,
    'linear': linear_cooling,
}


def _resolve_schedule(schedule):
    """Allow ``schedule`` to be passed either as a callable or by name."""
    if callable(schedule):
        return schedule
    try:
        return COOLING_SCHEDULES[schedule]
    except KeyError:
        raise ValueError(
            f"Invalid temperature schedule {schedule!r}; expected a callable "
            f"or one of {sorted(COOLING_SCHEDULES)}")


def simulated_annealing_with_reheating(state, cost_obj, init_network,
                                       initial_temp, final_temp, n_iterations,
                                       schedule, cooling_rate, shorten_prob,
                                       reheating_threshold=None,
                                       reheating_factor=None, sum_writer=None,
                                       silent=False,
                                       early_stop_patience=None,
                                       early_stop_min_delta=0.0):
    """Performs the simulated annealing algorithm with reheating.

    Args:
        state: The initial state of the system, not including the routes.
        cost_obj: The cost-function object.
        init_network: The initial network to start from.
        initial_temp: The initial temperature.
        final_temp: The final temperature.
        n_iterations: The number of iterations to run the algorithm for.
        schedule: The temperature schedule to use -- either a callable
            ``(temp, rate) -> temp`` or one of the names in
            :data:`COOLING_SCHEDULES` (``'linear'`` / ``'exponential'``).
        cooling_rate: The cooling rate to use.
        shorten_prob: The probability of shortening a route in the type-2
            mutator.
        reheating_threshold: The number of iterations without improvement
            before reheating.
        reheating_factor: The factor by which to increase the temperature when
            reheating.
        sum_writer: The summary writer to use for logging.
        silent: Whether to suppress output.

    Returns:
        ``(state, cost_history)`` -- the state holding the best solution found
        and the per-iteration cost history tensor.
    """
    schedule = _resolve_schedule(schedule)
    dev = state.device
    # keep the initial network on the algorithm's device
    init_network = init_network.to(dev)
    # get all shortest paths
    shortest_paths, _ = reconstruct_all_paths(state.nexts)
    demand = torch.nn.functional.pad(state.demand, (0, 1, 0, 1))

    current_network = init_network

    def get_cost(network):
        state.replace_routes(network)
        return cost_obj(state).cost

    best_network = current_network
    best_cost = get_cost(best_network)
    cost_norm = best_cost
    current_cost = best_cost.clone()
    current_temp = initial_temp
    no_improvement_iterations = 0
    # Separate counter for early-stopping: never reset by reheating, only by
    # an actual `min_delta`-strict improvement of the best-ever cost.
    iters_since_best_improved = 0
    use_early_stop = (early_stop_patience is not None
                      and early_stop_patience > 0)
    if reheating_threshold is None:
        # we never reheat
        reheating_threshold = n_iterations

    cost_history = torch.zeros((state.batch_size, n_iterations + 1), device=dev)
    cost_history[0] = current_cost
    if sum_writer is not None:
        sum_writer.add_scalar('cost', current_cost, 0)
        sum_writer.add_scalar('best cost', best_cost, 0)
        sum_writer.add_scalar('temperature', current_temp, 0)

    # set up some matrices that will be used to choose modifications
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths,
                                        cost_obj.symmetric_routes)
    street_node_neighbours = (state.street_adj.isfinite() &
                              (state.street_adj > 0))

    for ii in tqdm(range(n_iterations), disable=silent):
        new_network = modify(state, current_network, direct_sat_dmd,
                             shorten_prob, street_node_neighbours,
                             shortest_paths, dev)
        new_cost = get_cost(new_network)
        cost_diff = new_cost - current_cost

        accept_worse_prob = math.exp(-cost_diff / (cost_norm * current_temp))
        improved_best = False
        if cost_diff < 0 or torch.rand(1) < accept_worse_prob:
            current_network = new_network
            current_cost = new_cost
            if current_cost < best_cost:
                # Mark a min_delta-strict best improvement.
                improved_best = bool(
                    (best_cost - current_cost).item() > early_stop_min_delta)
                best_network = current_network
                best_cost = current_cost
                no_improvement_iterations = 0
            else:
                no_improvement_iterations += 1
        else:
            no_improvement_iterations += 1

        if improved_best:
            iters_since_best_improved = 0
        else:
            iters_since_best_improved += 1

        current_temp = max(schedule(current_temp, cooling_rate),
                           final_temp)

        # Reheating condition
        if no_improvement_iterations > reheating_threshold:
            current_temp = min(reheating_factor * current_temp,
                               initial_temp)
            no_improvement_iterations = 0  # Reset counter after reheating

        cost_history[:, ii + 1] = current_cost
        if sum_writer is not None:
            sum_writer.add_scalar('cost', current_cost, ii + 1)
            sum_writer.add_scalar('best cost', best_cost, ii + 1)
            sum_writer.add_scalar('temperature', current_temp, ii + 1)
            sum_writer.add_scalar('worsen prob', min(accept_worse_prob, 1), ii)

        if use_early_stop and iters_since_best_improved >= early_stop_patience:
            # Trim the right zero-padded tail so the viz curve ends at the
            # last real iter rather than dropping to 0 afterwards.
            cost_history = cost_history[:, :ii + 2].clone()
            if not silent:
                log.info(
                    f"[SA] early stop at iter {ii + 1}/{n_iterations}: "
                    f"no >{early_stop_min_delta:g} best improvement in "
                    f"{iters_since_best_improved} iters")
            break

    state.replace_routes(best_network)
    return state, cost_history


def modify(state, network, direct_sat_dmd, shorten_prob,
           street_node_neighbours, shortest_paths, device):
    """Modify the current solution by adding or removing a random connection."""
    new_network = network.clone()
    # choose a random route to modify
    route_idx = torch.randint(network.shape[-2], (1,), device=device)
    chosen_route = network[..., route_idx, :]

    if torch.rand(1) < 0.5:
        # apply type-1 mutator
        # add a "bee" dimension to route_idx, as get_unselected_routes needs it
        unsel_routes = get_unselected_routes(network, route_idx)
        remaining_state = state.clone()
        remaining_state.replace_routes(unsel_routes)
        new_route = get_bee_1_variants(remaining_state, chosen_route,
                                       direct_sat_dmd, shortest_paths)
    else:
        # apply type-2 mutator
        new_route = get_bee_2_variants(chosen_route, shorten_prob,
                                       street_node_neighbours)

    new_network[..., route_idx, :] = new_route
    return new_network

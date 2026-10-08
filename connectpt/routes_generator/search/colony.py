"""Bee-colony selection and acceptance, in the original RNG order."""
import logging as log
import torch
from tqdm import tqdm
from ..env.construction import normalize_nodes
from ..env.state import RouteGenBatchState
from ..city.shortest_paths import reconstruct_all_paths
from ..run.logger import log as log_metrics
from ..objectives.metrics import evaluate_network
from ..objectives.preferences import scalarize
from ..objectives.penalties import adjustment_degree, adjustment_penalty
from .records import METRIC_FIELDS, SearchBestRecord
from .operators import get_direct_sat_dmd


def _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name, bee_names):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault(bucket_name, {})
    for name in bee_names:
        bucket.setdefault(name, 0)
    return bucket


def _record_attempted_mutations(mutation_counts_out, attempted_counts):
    """Record attempted mutations per bee name.

    ``attempted_counts`` maps a bee name (the plan group's name, e.g. the
    declarative BeeSpec name) to how many bees ran that operator this step --
    ``plan.summary()``. Stats are keyed by bee name, not by legacy type slot.
    """
    if mutation_counts_out is None:
        return
    attempted = _ensure_mutation_stats_bucket(mutation_counts_out, "attempted", attempted_counts.keys())
    for name, inc in attempted_counts.items():
        attempted[name] += int(inc)
        # Flat alias at the top level so callers reading the bare name see the
        # running attempted total.
        mutation_counts_out[name] = attempted[name]


def _record_mutation_mask(mutation_counts_out, bucket_name, mutation_types, mask, bee_names):
    """Add ``mask`` hits to ``bucket_name``, keyed by bee name.

    ``bee_names`` is the plan's ordered group names; ``mutation_types`` holds
    each bee's 1-based group slot, so slot ``i`` maps to ``bee_names[i - 1]``.
    """
    if mutation_counts_out is None:
        return
    bucket = _ensure_mutation_stats_bucket(mutation_counts_out, bucket_name, bee_names)
    if mutation_types is None:
        return

    for slot, name in enumerate(bee_names, start=1):
        type_mask = mutation_types == slot
        if not type_mask.any():
            continue
        bucket[name] += int(mask[:, type_mask].sum().item())


def _record_accepted_mutations(mutation_counts_out, mutation_types, accepted_mask, bee_names):
    _record_mutation_mask(mutation_counts_out, "accepted", mutation_types, accepted_mask, bee_names)


def _ensure_selection_stats_bucket(mutation_counts_out):
    if mutation_counts_out is None:
        return None
    bucket = mutation_counts_out.setdefault("selection", {})
    bucket.setdefault("parent_copies", 0)
    bucket.setdefault("nonbest_parent_copies", 0)
    bucket.setdefault("worse_parent_copies", 0)
    return bucket


def _record_selection_stats(mutation_counts_out, parent_idxs, bee_objective_costs):
    """Record which population members survive the selection step."""
    if mutation_counts_out is None:
        return
    bucket = _ensure_selection_stats_bucket(mutation_counts_out)
    parent_costs = bee_objective_costs.gather(1, parent_idxs)
    best_costs = bee_objective_costs.min(dim=1, keepdim=True).values

    bucket["parent_copies"] += int(parent_idxs.numel())
    bucket["nonbest_parent_copies"] += int((parent_costs > best_costs).sum().item())
    bucket["worse_parent_copies"] += int((parent_costs > bee_objective_costs).sum().item())


def _choose_route_indices(bee_networks, n_routes):
    """Sample one route index per bee for mutation, uniformly."""
    return torch.randint(
        high=int(n_routes.item()),
        size=bee_networks.shape[:2],
        device=bee_networks.device,
    )


def run_colony(
    state,
    cost_obj,
    init_network,
    *,
    n_bees,
    passes_per_it,
    mod_steps_per_pass,
    shorten_prob,
    n_iterations,
    plan,
    silent,
    force_linking_unlinked,
    logger_list,
    mutation_counts_out,
    adjustment_degree_weight,
    adjustment_degree_gap,
    adjustment_degree_mode,
    adjustment_degree_objective,
    adjustment_degree_target,
    process_neural_bees_sequentially,
    iteration_callback=None,
):
    """Mutate, accept strict improvements and select recruiter networks."""
    plan.validate(n_bees)

    batch_size = state.batch_size

    dev = state.device
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes

    # get all shortest paths
    shortest_paths, _ = reconstruct_all_paths(state.nexts)

    demand = torch.nn.functional.pad(state.demand, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1, device=dev)
    n_init_routes = init_network.shape[1]
    best_networks[:, :n_init_routes, : init_network.shape[-1]] = init_network
    reference_networks = best_networks.clone()
    use_adjustment_penalty = adjustment_degree_weight > 0
    if use_adjustment_penalty:
        state.set_adjustment_seed(reference_networks)
    route_count = max(float(n_routes.item()), 1.0)

    # expand state to networks

    # compute can-be-directly-satisfied demand matrix
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths, cost_obj.symmetric_routes)

    # set up required matrices
    street_node_neighbours = state.street_adj.isfinite() & (state.street_adj > 0)
    bee_idxs = torch.arange(n_bees, device=dev)

    log.debug("starting BCO")

    # set up the cost function to work with batches of bees
    # "multiply" the batch for the bees
    exp_states = sum([[substate] * n_bees for substate in state.batch_to_list()], [])
    bee_states = RouteGenBatchState.batch_from_list(exp_states)
    # the construction ("bee") model prepares the bee states; heuristic-only
    # plans have no such model and skip planning setup (legacy semantics).
    planning_model = plan.planning_model
    if planning_model is not None:
        normalize_nodes(bee_states, planning_model.node_norm, planning_model.feature_set)
    # Sequential neural-bee processing runs the model on one bee at a time and
    # needs a state whose batch equals the number of graphs (not graphs *
    # n_bees), so a single bee's network is not broadcast up to the full bee
    # batch by the route-replace path. Build it once and reuse it per bee.
    single_bee_states = None
    if process_neural_bees_sequentially:
        single_bee_states = RouteGenBatchState.batch_from_list(state.batch_to_list())
        if planning_model is not None:
            normalize_nodes(single_bee_states, planning_model.node_norm, planning_model.feature_set)

    metric_names = cost_obj.get_metric_names()

    def batched_cost_fn(bee_networks):
        bee_states.replace_routes(bee_networks.flatten(0, 1))

        result = cost_obj(bee_states)

        costs = scalarize(result.goals, result.preferences, result.penalties, result.penalty_weights).reshape(
            batch_size, n_bees
        )
        metrics = evaluate_network(result)
        other_metrics = torch.stack(list(metrics.values()), dim=-1)
        other_metrics = other_metrics.reshape(batch_size, n_bees, -1)
        return costs, other_metrics

    # initialize bees' networks
    # batch size x n_bees x n_routes x max_n_nodes
    bee_networks = best_networks[:, None].repeat(1, n_bees, 1, 1)
    # evaluate and record the reward of the initial network
    bee_raw_costs, bee_metrics = batched_cost_fn(bee_networks)
    bee_adjustment_degrees = torch.zeros_like(bee_raw_costs)
    bee_adjustment_penalties = adjustment_penalty(
        bee_adjustment_degrees,
        objective=adjustment_degree_objective,
        target=adjustment_degree_target,
    )
    # adj already included in bee_raw_costs via cost_obj (unified penalty)
    bee_objective_costs = bee_raw_costs
    best_objective_costs, best_idxs = bee_objective_costs.min(1)
    best_raw_costs = bee_raw_costs[batch_idxs, best_idxs]
    best_metrics = bee_metrics[batch_idxs, best_idxs]
    best_adjustment_degrees = bee_adjustment_degrees[batch_idxs, best_idxs]
    best_adjustment_penalties = bee_adjustment_penalties[batch_idxs, best_idxs]

    def log_best(step):
        record = SearchBestRecord(
            **{METRIC_FIELDS[name]: vals.mean() for name, vals in zip(metric_names, best_metrics.unbind(-1))}
        )
        if use_adjustment_penalty:
            record.adjustment_degree = best_adjustment_degrees.mean()
            record.adjustment_penalty = best_adjustment_penalties.mean()
        log_metrics(logger_list, record, step)

    log_best(0)

    cost_history = torch.zeros((batch_size, n_iterations + 1), device=dev)
    cost_history[:, 0] = best_raw_costs

    for iteration in tqdm(range(n_iterations), disable=silent):
        for pi in range(passes_per_it):
            for mi in range(mod_steps_per_pass):

                chosen_route_idxs = _choose_route_indices(bee_networks, n_routes)
                gather_idx = chosen_route_idxs[..., None, None]
                gather_idx = gather_idx.expand(-1, -1, -1, max_n_nodes)
                old_modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)
                _record_attempted_mutations(mutation_counts_out, plan.summary())
                new_bee_networks, mutation_types = plan.get_mutants(
                    bee_networks,
                    chosen_route_idxs,
                    direct_sat_dmd=direct_sat_dmd,
                    shorten_prob=shorten_prob,
                    street_node_neighbours=street_node_neighbours,
                    shortest_paths=shortest_paths,
                    force_linking_unlinked=force_linking_unlinked,
                    env_state=bee_states,
                    cost=cost_obj,
                    single_bee_env_state=single_bee_states,
                    adj_condition_target=adjustment_degree_target,
                    adj_condition_weight=adjustment_degree_weight,
                    process_neural_bees_sequentially=process_neural_bees_sequentially,
                )

                new_bee_raw_costs, new_bee_metrics = batched_cost_fn(new_bee_networks)

                if use_adjustment_penalty:
                    reference_routes = (
                        reference_networks[:, None].expand(-1, n_bees, -1, -1).gather(2, gather_idx).squeeze(2)
                    )
                    old_route_adjustments = adjustment_degree(
                        old_modified_routes,
                        reference_routes,
                        cost_obj.symmetric_routes,
                        gap=adjustment_degree_gap,
                        mode=adjustment_degree_mode,
                    )
                    new_modified_routes = new_bee_networks.gather(2, gather_idx).squeeze(2)
                    new_route_adjustments = adjustment_degree(
                        new_modified_routes,
                        reference_routes,
                        cost_obj.symmetric_routes,
                        gap=adjustment_degree_gap,
                        mode=adjustment_degree_mode,
                    )
                    new_bee_adjustment_degrees = (
                        bee_adjustment_degrees + (new_route_adjustments - old_route_adjustments) / route_count
                    )
                else:
                    new_bee_adjustment_degrees = bee_adjustment_degrees

                new_bee_adjustment_penalties = adjustment_penalty(
                    new_bee_adjustment_degrees,
                    objective=adjustment_degree_objective,
                    target=adjustment_degree_target,
                )
                # adj already included in new_bee_raw_costs via cost_obj
                new_bee_objective_costs = new_bee_raw_costs

                # greedy acceptance: keep only strict improvements
                accepted_idxs = (new_bee_objective_costs - bee_objective_costs) < 0

                _record_accepted_mutations(
                    mutation_counts_out,
                    mutation_types,
                    accepted_idxs,
                    plan.group_names(),
                )
                bee_networks[accepted_idxs] = new_bee_networks[accepted_idxs]
                bee_raw_costs[accepted_idxs] = new_bee_raw_costs[accepted_idxs]
                bee_objective_costs[accepted_idxs] = new_bee_objective_costs[accepted_idxs]
                bee_adjustment_degrees[accepted_idxs] = new_bee_adjustment_degrees[accepted_idxs]
                bee_adjustment_penalties[accepted_idxs] = new_bee_adjustment_penalties[accepted_idxs]
                bee_metrics[accepted_idxs] = new_bee_metrics[accepted_idxs]

            # do "backward pass"

            # update the best solution found so far
            current_best_objective_cost, current_best_idx = bee_objective_costs.min(1)
            is_improvement = current_best_objective_cost < best_objective_costs
            improvement_idx = current_best_idx[is_improvement]
            best_objective_costs[is_improvement] = current_best_objective_cost[is_improvement]
            best_networks[is_improvement] = bee_networks[is_improvement, improvement_idx]
            best_raw_costs[is_improvement] = bee_raw_costs[is_improvement, improvement_idx]
            best_adjustment_degrees[is_improvement] = bee_adjustment_degrees[is_improvement, improvement_idx]
            best_adjustment_penalties[is_improvement] = bee_adjustment_penalties[is_improvement, improvement_idx]
            cost_history[:, iteration + 1] = best_raw_costs
            new_best = bee_metrics[is_improvement, improvement_idx]
            best_metrics[is_improvement] = new_best

            # Live checkpoint: whenever the incumbent improves, hand the current
            # best network (+ its costs/metrics) to an optional callback so a
            # caller can persist it -- so an early stop never loses the best.
            if iteration_callback is not None and bool(is_improvement.any()):
                try:
                    iteration_callback(
                        int(iteration),
                        best_networks,
                        best_objective_costs,
                        best_adjustment_degrees,
                        best_metrics,
                        metric_names,
                    )
                except Exception as _cb_exc:
                    log.warning(f"bee_colony iteration_callback failed: {_cb_exc}")

            # decide whether each bee is a recruiter or follower
            max_bee_costs, _ = bee_objective_costs.max(dim=1)
            min_bee_costs, _ = bee_objective_costs.min(dim=1)
            spread = max_bee_costs - min_bee_costs
            # avoid division by 0
            spread[spread == 0] = 1

            qualities = (max_bee_costs[:, None] - bee_objective_costs) / spread[:, None]
            min_quality, _ = qualities.min(1)
            follow_probs = (-qualities + min_quality[:, None]).exp()
            are_recruiters = follow_probs < torch.rand(n_bees, device=dev)

            if not are_recruiters.any():
                # no recruiters, so make everyone keep their own network.
                continue

            # decide which recruiter each follower will follow
            denom = (qualities * are_recruiters).sum(1)
            # avoid division by 0
            denom[denom == 0] = 1
            recruit_probs = qualities / denom[:, None]
            recruit_probs[~are_recruiters] = 0
            # set probs where there is no probability to 1, so multinomial
            # doesn't complain.
            no_valid_recruiters = recruit_probs.sum(-1) == 0
            recruit_probs[no_valid_recruiters] = 1
            recruiters = recruit_probs.multinomial(n_bees, replacement=True)
            recruiters[no_valid_recruiters] = bee_idxs
            recruiters[are_recruiters] = bee_idxs[None].expand(batch_size, -1)[are_recruiters]
            _record_selection_stats(mutation_counts_out, recruiters, bee_objective_costs)

            # update the networks and costs of the followers
            bee_networks = bee_networks[batch_idxs[:, None], recruiters]
            bee_raw_costs = bee_raw_costs[batch_idxs[:, None], recruiters]
            bee_objective_costs = bee_objective_costs[batch_idxs[:, None], recruiters]
            bee_adjustment_degrees = bee_adjustment_degrees[batch_idxs[:, None], recruiters]
            bee_adjustment_penalties = bee_adjustment_penalties[batch_idxs[:, None], recruiters]
            bee_metrics = bee_metrics[batch_idxs[:, None], recruiters]

        log_best(iteration + 1)

    # return the best solution
    state.replace_routes(best_networks)
    # reset the unified adjustment-degree penalty so the shared cost_obj does
    # not carry this run's seed into any later evaluation.
    if use_adjustment_penalty:
        state.set_adjustment_seed(torch.full((state.batch_size, 0, 0), -1.0, device=dev))
    return state, cost_history

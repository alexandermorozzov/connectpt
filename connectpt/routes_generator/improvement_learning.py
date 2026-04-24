import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from .citygraph_dataset import (
    STOP_KEY,
    DemandScaleTransform,
    InsertPosFeatures,
    SpaceScaleTransform,
)
from .transit_time_estimator import (
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_HALT,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_TRIM_START,
    RouteGenBatchState,
)
from .torch_utils import get_batch_tensor_from_routes

ROUTE_ACTION_NAMES = {
    ROUTE_ACTION_EXTEND: "extend",
    ROUTE_ACTION_TRIM_START: "trim_start",
    ROUTE_ACTION_TRIM_END: "trim_end",
    ROUTE_ACTION_HALT: "halt",
}
ROUTE_ACTION_STAT_NAMES = tuple(ROUTE_ACTION_NAMES.values())


def _trim_route_padding(routes, max_route_len=None):
    """Drop trailing all-padding route columns from a loaded route tensor."""
    if routes.ndim == 1:
        routes = routes.unsqueeze(0)
    elif routes.ndim == 3:
        routes = routes[0]

    used_columns = (routes >= 0).any(dim=0).nonzero().flatten()
    if used_columns.numel() == 0:
        trim_len = 0
    else:
        trim_len = int(used_columns[-1].item()) + 1

    if max_route_len is not None:
        trim_len = min(trim_len, int(max_route_len))
    trim_len = max(trim_len, 1)
    return routes[:, :trim_len]


def load_lc_result_routes(results_dir, n_graphs, max_route_len=None):
    """Load per-graph LC route tensors from examples/lc_results."""
    results_dir = Path(results_dir)
    route_sets = []
    for graph_idx in range(n_graphs):
        route_path = results_dir / f"graph_{graph_idx:04d}" / \
            f"lc_lc_graph_{graph_idx:04d}_routes_routes.pkl"
        if not route_path.exists():
            matches = sorted(
                (results_dir / f"graph_{graph_idx:04d}").glob(
                    "lc_*_routes_routes.pkl")
            )
            if not matches:
                raise FileNotFoundError(route_path)
            route_path = matches[0]
        with route_path.open("rb") as file:
            routes = pickle.load(file)
        if isinstance(routes, list):
            routes = routes[0]
        if routes.ndim == 3 and routes.shape[0] == 1:
            routes = routes.squeeze(0)
        routes = _trim_route_padding(routes, max_route_len)
        route_sets.append(routes.long())

    max_n_routes = max(route_set.shape[0] for route_set in route_sets)
    max_route_len = max(route_set.shape[1] for route_set in route_sets)
    padded_routes = torch.full(
        (len(route_sets), max_n_routes, max_route_len),
        -1, dtype=torch.long)
    for graph_idx, routes in enumerate(route_sets):
        padded_routes[
            graph_idx, :routes.shape[0], :routes.shape[1]
        ] = routes
    return padded_routes


def load_raw_graphs_and_lc_routes(raw_graphs_path, lc_results_dir):
    with Path(raw_graphs_path).open("rb") as file:
        graphs = pickle.load(file)
    max_route_len = max(int(graph[STOP_KEY].num_nodes) for graph in graphs)
    seed_routes = load_lc_result_routes(
        lc_results_dir, len(graphs), max_route_len=max_route_len)
    return graphs, seed_routes


def pad_seed_routes_to_n_routes(route_batch, target_n_routes=None):
    """Append empty route slots so improvement can plan extra routes."""
    if target_n_routes is None:
        return route_batch

    target_n_routes = int(target_n_routes)
    current_n_routes = int(route_batch.shape[1])
    if target_n_routes < current_n_routes:
        raise ValueError(
            "target_n_routes must be at least the number of seed routes: "
            f"{target_n_routes} < {current_n_routes}"
        )
    if target_n_routes == current_n_routes:
        return route_batch

    empty_shape = (
        route_batch.shape[0],
        target_n_routes - current_n_routes,
        route_batch.shape[2],
    )
    empty_routes = torch.full(
        empty_shape, -1, dtype=route_batch.dtype, device=route_batch.device)
    return torch.cat((route_batch, empty_routes), dim=1)


def _get_graph_batch_max_n_nodes(graph_batch):
    stop_data = graph_batch[STOP_KEY]
    if hasattr(stop_data, "ptr") and stop_data.ptr is not None:
        node_counts = stop_data.ptr[1:] - stop_data.ptr[:-1]
        return int(node_counts.max().item())
    return int(stop_data.num_nodes)


def _trim_route_batch_to_graph_capacity(route_batch, graph_batch):
    max_n_nodes = _get_graph_batch_max_n_nodes(graph_batch)
    if route_batch.shape[-1] <= max_n_nodes:
        return route_batch

    route_tail = route_batch[..., max_n_nodes:]
    if (route_tail >= 0).any():
        raise ValueError(
            "Loaded route has real stops beyond the scenario node "
            "capacity. This is not just padding: "
            f"route width {route_batch.shape[-1]}, "
            f"batch max nodes {max_n_nodes}."
        )
    return route_batch[..., :max_n_nodes]


def make_improvement_batch(graphs, seed_routes, indices, device, training=False,
                           space_scale=None, demand_scale=None,
                           insert_pos=None, target_n_routes=None):
    if space_scale is None:
        space_scale = SpaceScaleTransform(0.95, 1.05)
    if demand_scale is None:
        demand_scale = DemandScaleTransform(0.9, 1.1)
    if insert_pos is None:
        insert_pos = InsertPosFeatures()

    batch_graphs = []
    for idx in indices:
        graph = graphs[int(idx)].clone()
        if training:
            graph = space_scale(graph)
            graph = demand_scale(graph)
        graph[STOP_KEY].x = graph[STOP_KEY].pos.clone()
        graph = insert_pos(graph)
        batch_graphs.append(graph)

    graph_batch = Batch.from_data_list(batch_graphs).to(device)
    route_batch = seed_routes[torch.as_tensor(indices, dtype=torch.long)].to(
        device)
    route_batch = _trim_route_batch_to_graph_capacity(route_batch,
                                                      graph_batch)
    route_batch = pad_seed_routes_to_n_routes(route_batch, target_n_routes)
    return graph_batch, route_batch


def _clone_cost_weights(cost_weights):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in cost_weights.items()
    }


def _make_route_context_state(cost_obj, graph_batch, route_batch, route_idx,
                              min_route_len, max_route_len, cost_weights,
                              invalid_directly_connected=False):
    route_batch = _trim_route_batch_to_graph_capacity(route_batch,
                                                      graph_batch)
    n_routes = route_batch.shape[1]
    state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))

    # The route being improved is active, while every other seeded route is
    # already visible in the transit network context.
    context_routes = route_batch.clone()
    context_routes[:, route_idx] = -1
    state.add_new_routes(
        context_routes,
        invalid_directly_connected=invalid_directly_connected)
    state.set_current_routes(route_batch[:, route_idx])
    return state


def _get_planned_current_routes(state, fallback_routes,
                                context_route_counts=None):
    planned_routes = []
    if context_route_counts is not None:
        context_route_counts = context_route_counts.detach().cpu().tolist()
    for batch_idx, batch_routes in enumerate(state.routes):
        context_count = 0
        if context_route_counts is not None:
            context_count = int(context_route_counts[batch_idx])
        if len(batch_routes) <= context_count:
            route = state.current_routes[batch_idx]
            if not (route > -1).any():
                route = fallback_routes[batch_idx]
            route = route[route > -1]
        else:
            route = batch_routes[-1]
        planned_routes.append(route.clone())
    return planned_routes


def _assemble_routes_by_original_order(routes_by_route, device):
    batch_size = len(routes_by_route[0])
    batch_routes = []
    for batch_idx in range(batch_size):
        batch_routes.append([
            route_set[batch_idx] for route_set in routes_by_route
        ])
    return get_batch_tensor_from_routes(batch_routes, device)


def _get_default_max_route_edit_steps(max_route_len):
    if max_route_len is None:
        return None
    if torch.is_tensor(max_route_len):
        max_route_len = int(max_route_len.max().item())
    return 2 * int(max_route_len)


def rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                           min_route_len, max_route_len, greedy=False,
                           cost_weights=None, return_actions=False,
                           force_nonhalt_first_step=False,
                           max_route_edit_steps=None,
                           return_step_data=False,
                           sequential_empty_routes=True,
                           invalid_directly_connected_for_empty=True):
    if cost_weights is None:
        cost_weights = cost_obj.sample_variable_weights(graph_batch.num_graphs,
                                                        graph_batch[STOP_KEY].x.device)
    if max_route_edit_steps is None:
        max_route_edit_steps = _get_default_max_route_edit_steps(max_route_len)
    route_batch = _trim_route_batch_to_graph_capacity(route_batch,
                                                      graph_batch)

    n_routes = route_batch.shape[1]
    seed_state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))
    seed_state.add_new_routes(route_batch)
    seed_result = cost_obj(seed_state)

    route_logits = []
    route_entropies = []
    route_actions = []
    route_action_kinds = []
    route_step_logits = []
    route_step_entropies = []
    routes_by_route = []
    context_route_len = int(route_batch.shape[-1])
    if max_route_len is not None:
        if torch.is_tensor(max_route_len):
            context_route_len = max(
                context_route_len, int(max_route_len.max().item()))
        else:
            context_route_len = max(context_route_len, int(max_route_len))
    working_route_batch = torch.full(
        (route_batch.shape[0], route_batch.shape[1], context_route_len),
        -1, dtype=route_batch.dtype, device=route_batch.device)
    working_route_batch[..., :route_batch.shape[-1]] = route_batch
    seed_route_is_empty = (route_batch > -1).sum(dim=-1) == 0
    for route_idx in range(n_routes):
        context_route_batch = route_batch
        use_sequential_context = False
        if sequential_empty_routes and seed_route_is_empty[:, route_idx].all():
            context_route_batch = working_route_batch
            use_sequential_context = True

        route_state = _make_route_context_state(
            cost_obj, graph_batch, context_route_batch, route_idx,
            min_route_len, max_route_len, cost_weights,
            invalid_directly_connected=(
                invalid_directly_connected_for_empty and
                use_sequential_context))
        context_route_counts = route_state.n_finished_routes.detach().clone()
        if getattr(model, "supports_trim_actions", False):
            actions, logits, entropy = model.plan_new_route(
                route_state, greedy=greedy,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_steps=max_route_edit_steps)
        else:
            actions, logits, entropy = model.plan_new_route(
                route_state, greedy=greedy)
        route_logits.append(logits)
        route_entropies.append(entropy)
        planned_current_routes = \
            _get_planned_current_routes(
                route_state, route_batch[:, route_idx], context_route_counts)
        routes_by_route.append(planned_current_routes)
        if use_sequential_context:
            planned_tensor = get_batch_tensor_from_routes(
                [[planned_current_routes[batch_idx]]
                 for batch_idx in range(len(planned_current_routes))],
                route_batch.device,
                max_route_len=working_route_batch.shape[-1])
            working_route_batch[:, route_idx] = planned_tensor[:, 0]
        if return_actions:
            route_actions.append(actions.detach().clone())
            if hasattr(model, "last_route_action_kinds"):
                route_action_kinds.append(
                    model.last_route_action_kinds.detach().clone())
            else:
                route_action_kinds.append(None)
            if return_step_data:
                if hasattr(model, "last_route_step_logits"):
                    route_step_logits.append(
                        model.last_route_step_logits.detach().clone())
                    route_step_entropies.append(
                        model.last_route_step_entropies.detach().clone())
                else:
                    route_step_logits.append(None)
                    route_step_entropies.append(None)

    final_routes = _assemble_routes_by_original_order(
        routes_by_route, graph_batch[STOP_KEY].x.device)
    state = RouteGenBatchState(
        graph_batch, cost_obj, n_routes, min_route_len, max_route_len,
        cost_weights=_clone_cost_weights(cost_weights))
    state.add_new_routes(final_routes)
    final_result = cost_obj(state)
    output = (
        state,
        seed_result,
        final_result,
        torch.stack(route_logits, dim=1),
        torch.stack(route_entropies, dim=1),
    )
    if return_actions:
        output = output + (route_actions, route_action_kinds)
        if return_step_data:
            output = output + (route_step_logits, route_step_entropies)
    return output


def _route_change_mask(seed_routes, generated_routes):
    seed_routes = seed_routes.detach()
    generated_routes = generated_routes.detach().to(seed_routes.device)

    batch_size = seed_routes.shape[0]
    n_seed_routes = seed_routes.shape[1]
    n_generated_routes = generated_routes.shape[1]
    n_compare_routes = max(n_seed_routes, n_generated_routes)
    if n_compare_routes == 0:
        return torch.zeros((batch_size, 0), dtype=torch.bool,
                           device=seed_routes.device)

    max_route_len = max(seed_routes.shape[-1], generated_routes.shape[-1])
    seed_cmp = torch.full(
        (batch_size, n_compare_routes, max_route_len), -1,
        dtype=seed_routes.dtype, device=seed_routes.device)
    generated_cmp = torch.full_like(seed_cmp, -1)
    seed_cmp[:, :n_seed_routes, :seed_routes.shape[-1]] = seed_routes
    generated_cmp[:, :n_generated_routes, :generated_routes.shape[-1]] = \
        generated_routes
    return (seed_cmp != generated_cmp).any(dim=-1)


def _new_action_count_totals():
    totals = {
        f"{name}_count": 0 for name in ROUTE_ACTION_STAT_NAMES
    }
    totals["total_count"] = 0
    totals["route_plan_count"] = 0
    return totals


def _finalize_action_stats(counts):
    stats = dict(counts)
    total_count = max(int(stats["total_count"]), 1)
    route_plan_count = max(int(stats["route_plan_count"]), 1)
    stats["trim_count"] = \
        stats["trim_start_count"] + stats["trim_end_count"]
    stats["edit_count"] = stats["extend_count"] + stats["trim_count"]

    for name in ROUTE_ACTION_STAT_NAMES:
        stats[f"{name}_rate"] = stats[f"{name}_count"] / total_count
    stats["trim_rate"] = stats["trim_count"] / total_count
    stats["edit_rate"] = stats["edit_count"] / total_count
    stats["avg_actions_per_route"] = \
        stats["total_count"] / route_plan_count
    return stats


def _merge_action_stats(total_counts, batch_stats):
    for name in ROUTE_ACTION_STAT_NAMES:
        total_counts[f"{name}_count"] += batch_stats[f"{name}_count"]
    total_counts["total_count"] += batch_stats["total_count"]
    total_counts["route_plan_count"] += batch_stats["route_plan_count"]


def summarize_route_action_stats(route_actions, route_action_kinds=None):
    """Count executed route actions, ignoring post-halt batch padding."""
    counts = _new_action_count_totals()
    if route_actions is None:
        return _finalize_action_stats(counts)

    for route_idx, actions_for_route in enumerate(route_actions):
        if actions_for_route.numel() == 0:
            continue

        batch_size = actions_for_route.shape[0]
        counts["route_plan_count"] += batch_size

        if route_action_kinds is not None and \
                route_action_kinds[route_idx] is not None:
            kinds_for_route = route_action_kinds[route_idx]
        else:
            kinds_for_route = torch.full(
                actions_for_route.shape[:2], ROUTE_ACTION_EXTEND,
                dtype=torch.long, device=actions_for_route.device)
            kinds_for_route[actions_for_route[..., 0] < 0] = \
                ROUTE_ACTION_HALT

        ended = torch.zeros((batch_size,), dtype=torch.bool,
                            device=kinds_for_route.device)
        for step_idx in range(kinds_for_route.shape[1]):
            active = ~ended
            if not active.any():
                break

            active_kinds = kinds_for_route[:, step_idx][active]
            for action_kind, action_name in ROUTE_ACTION_NAMES.items():
                counts[f"{action_name}_count"] += int(
                    (active_kinds == action_kind).sum().item()
                )
            counts["total_count"] += int(active.sum().item())

            just_ended = active_kinds == ROUTE_ACTION_HALT
            active_indices = torch.where(active)[0]
            ended[active_indices] = just_ended

    return _finalize_action_stats(counts)


def _backward_on_sampled_actions(model, cost_obj, graph_batch, route_batch,
                                 min_route_len, max_route_len, cost_weights,
                                 route_actions, route_action_kinds,
                                 advantages, entropy_weight):
    """Replay sampled actions and backprop before mutating the state.

    The sampled rollout mutates RouteGenBatchState many times.  Backpropagating
    through a graph that still references those mutable LongTensor indices can
    trigger version-counter errors.  This helper replays fixed actions and calls
    backward on each step before applying that step to the replay state.
    """
    n_routes = route_batch.shape[1]
    supports_route_actions = getattr(model, "supports_trim_actions", False)
    objective_values = []

    for route_idx in range(n_routes):
        state = _make_route_context_state(
            cost_obj, graph_batch, route_batch, route_idx, min_route_len,
            max_route_len, cost_weights)
        state = model.setup_planning(state)
        actions_for_route = route_actions[route_idx]
        kinds_for_route = route_action_kinds[route_idx]
        ended = torch.zeros((state.batch_size,), dtype=torch.bool,
                            device=state.device)

        for step_idx in range(actions_for_route.shape[1]):
            if ended.all():
                break

            sampled_actions = actions_for_route[:, step_idx].clone()
            if supports_route_actions:
                sampled_kinds = kinds_for_route[:, step_idx].clone()
                active = ~ended
                score_mask = active
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, _, logits, entropy = model.step_route_action(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask],
                        action_kinds=sampled_kinds[score_mask])
                    objective = \
                        (advantages[score_mask] * logits).mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                apply_kinds = sampled_kinds.detach().clone()
                apply_actions = sampled_actions.detach().clone()
                apply_kinds[ended] = ROUTE_ACTION_HALT
                apply_actions[ended] = -1
                just_ended = apply_kinds == ROUTE_ACTION_HALT
                apply_actions[just_ended] = -1
                with torch.no_grad():
                    state.apply_route_actions(apply_kinds, apply_actions)
                ended = ended | just_ended
            else:
                active = ~ended
                score_mask = active
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, logits, entropy = model.step(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask])
                    objective = \
                        (advantages[score_mask] * logits).mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                apply_actions = sampled_actions.detach().clone()
                apply_actions[ended] = -1
                just_ended = apply_actions[:, 0] == -1
                with torch.no_grad():
                    state.shortest_path_action(apply_actions)
                ended = ended | just_ended

    if len(objective_values) == 0:
        return torch.zeros((), device=graph_batch[STOP_KEY].x.device)
    return torch.stack(objective_values).mean()


def _backward_on_sampled_actions_ppo(
        model, cost_obj, graph_batch, route_batch, min_route_len,
        max_route_len, cost_weights, route_actions, route_action_kinds,
        old_step_logits, advantages, entropy_weight, clip_epsilon):
    """Replay sampled actions with a PPO clipped objective."""
    n_routes = route_batch.shape[1]
    supports_route_actions = getattr(model, "supports_trim_actions", False)
    objective_values = []
    ratio_sum = 0.0
    ratio_count = 0
    clipped_count = 0

    for route_idx in range(n_routes):
        state = _make_route_context_state(
            cost_obj, graph_batch, route_batch, route_idx, min_route_len,
            max_route_len, cost_weights)
        state = model.setup_planning(state)
        actions_for_route = route_actions[route_idx]
        kinds_for_route = route_action_kinds[route_idx]
        old_logits_for_route = old_step_logits[route_idx]
        ended = torch.zeros((state.batch_size,), dtype=torch.bool,
                            device=state.device)

        for step_idx in range(actions_for_route.shape[1]):
            if ended.all():
                break

            sampled_actions = actions_for_route[:, step_idx].clone()
            old_logits = old_logits_for_route[:, step_idx].to(state.device)

            if supports_route_actions:
                sampled_kinds = kinds_for_route[:, step_idx].clone()
                active = ~ended
                forced_halt = (sampled_kinds == ROUTE_ACTION_HALT) & \
                    (old_logits == 0)
                score_mask = active & ~forced_halt
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, _, logits, entropy = model.step_route_action(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask],
                        action_kinds=sampled_kinds[score_mask])
                    ratios = (logits - old_logits[score_mask]).exp()
                    clipped_ratios = ratios.clamp(
                        1 - clip_epsilon, 1 + clip_epsilon)
                    step_advantages = advantages[score_mask]
                    clip_obj = torch.minimum(
                        ratios * step_advantages,
                        clipped_ratios * step_advantages)
                    objective = clip_obj.mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                    ratio_sum += float(ratios.detach().sum().item())
                    ratio_count += int(ratios.numel())
                    clipped = (ratios.detach() < 1 - clip_epsilon) | \
                        (ratios.detach() > 1 + clip_epsilon)
                    clipped_count += int(clipped.sum().item())

                apply_kinds = sampled_kinds.detach().clone()
                apply_actions = sampled_actions.detach().clone()
                apply_kinds[ended] = ROUTE_ACTION_HALT
                apply_actions[ended] = -1
                just_ended = apply_kinds == ROUTE_ACTION_HALT
                apply_actions[just_ended] = -1
                with torch.no_grad():
                    state.apply_route_actions(apply_kinds, apply_actions)
                ended = ended | just_ended
            else:
                active = ~ended
                forced_halt = (sampled_actions[:, 0] < 0) & (old_logits == 0)
                score_mask = active & ~forced_halt
                if score_mask.any():
                    score_idxs = torch.where(score_mask)[0]
                    score_state = state.index_select(score_idxs)
                    score_state = model.setup_planning(score_state)
                    _, logits, entropy = model.step(
                        score_state, greedy=False,
                        actions=sampled_actions[score_mask])
                    ratios = (logits - old_logits[score_mask]).exp()
                    clipped_ratios = ratios.clamp(
                        1 - clip_epsilon, 1 + clip_epsilon)
                    step_advantages = advantages[score_mask]
                    clip_obj = torch.minimum(
                        ratios * step_advantages,
                        clipped_ratios * step_advantages)
                    objective = clip_obj.mean()
                    objective = objective + entropy_weight * entropy.mean()
                    (-objective).backward()
                    objective_values.append(objective.detach())

                    ratio_sum += float(ratios.detach().sum().item())
                    ratio_count += int(ratios.numel())
                    clipped = (ratios.detach() < 1 - clip_epsilon) | \
                        (ratios.detach() > 1 + clip_epsilon)
                    clipped_count += int(clipped.sum().item())

                apply_actions = sampled_actions.detach().clone()
                apply_actions[ended] = -1
                just_ended = apply_actions[:, 0] == -1
                with torch.no_grad():
                    state.shortest_path_action(apply_actions)
                ended = ended | just_ended

    if len(objective_values) == 0:
        objective = torch.zeros((), device=graph_batch[STOP_KEY].x.device)
    else:
        objective = torch.stack(objective_values).mean()

    stats = {
        "objective": float(objective.detach().item()),
        "ratio_sum": ratio_sum,
        "ratio_count": ratio_count,
        "clipped_count": clipped_count,
    }
    return objective, stats


def _get_cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _make_optimizer_from_cfg(model, cfg):
    optimizer_name = _get_cfg_value(cfg, "optimizer", "Adam")
    optimizer_type = getattr(torch.optim, optimizer_name)
    return optimizer_type(
        model.parameters(),
        lr=float(_get_cfg_value(cfg, "lr", 1e-4)),
        weight_decay=float(_get_cfg_value(cfg, "decay", 0.0)),
        maximize=True,
    )


def _merge_step_action_stats(counts, actions, action_kinds, active):
    if not active.any():
        return
    if action_kinds is None:
        action_kinds = torch.full(
            (actions.shape[0],), ROUTE_ACTION_EXTEND,
            dtype=torch.long, device=actions.device)
        action_kinds[actions[:, 0] < 0] = ROUTE_ACTION_HALT

    active_kinds = action_kinds[active]
    for action_kind, action_name in ROUTE_ACTION_NAMES.items():
        counts[f"{action_name}_count"] += int(
            (active_kinds == action_kind).sum().item()
        )
    counts["total_count"] += int(active.sum().item())


def _compute_ppo_returns_and_advantages(rewards, value_estimates, dones,
                                        final_value_estimates, gamma,
                                        use_gae, gae_lambda):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    if use_gae:
        lastgaelam = torch.zeros_like(final_value_estimates)
        for step_idx in reversed(range(rewards.shape[0])):
            if step_idx == rewards.shape[0] - 1:
                next_values = final_value_estimates
            else:
                next_values = value_estimates[step_idx + 1]
            next_nonterminal = (~dones[step_idx]).to(torch.float32)
            delta = rewards[step_idx] + gamma * next_values * \
                next_nonterminal - value_estimates[step_idx]
            lastgaelam = delta + gamma * gae_lambda * \
                next_nonterminal * lastgaelam
            advantages[step_idx] = lastgaelam
        returns = advantages + value_estimates
    else:
        next_return = final_value_estimates
        for step_idx in reversed(range(rewards.shape[0])):
            next_nonterminal = (~dones[step_idx]).to(torch.float32)
            next_return = rewards[step_idx] + gamma * \
                next_nonterminal * next_return
            returns[step_idx] = next_return
        advantages = returns - value_estimates

    return returns, advantages


def _clear_state_lazy_tensors(state):
    """Clear lazy caches whose padded length can differ between states."""
    state.extra_data.shortest_path_sequences = torch.zeros(
        (state.batch_size, 0, 0, 0),
        dtype=torch.long,
        device=state.device,
    )
    return state


def _state_batch_signature(state):
    """Return shape signature for tensors that PyG must collate exactly."""
    extra_data = state.extra_data
    signature_keys = (
        "base_valid_terms_mat",
        "valid_terms_mat",
        "directly_connected",
        "route_mat",
        "transit_times",
        "has_path",
        "current_routes",
        "current_route_times_from_start",
        "shortest_path_sequences",
        "route_nexts",
        "n_transfers",
    )
    signature = []
    for key in signature_keys:
        if not hasattr(extra_data, key):
            continue
        value = getattr(extra_data, key)
        if torch.is_tensor(value):
            # Drop the batch dimension; all remaining dimensions must match
            # for RouteGenBatchState.batch_from_list() to collate safely.
            signature.append((key, tuple(value.shape[1:])))
    return tuple(signature)


def _split_state_indices_by_signature(states, idxs):
    groups = {}
    for idx in idxs.detach().cpu().tolist():
        _clear_state_lazy_tensors(states[idx])
        signature = _state_batch_signature(states[idx])
        groups.setdefault(signature, []).append(idx)
    return [
        torch.tensor(group, dtype=idxs.dtype, device=idxs.device)
        for group in groups.values()
    ]


def _collect_lc_improvement_cfg_ppo_rollout(
        model, cost_obj, make_next_state, value_module, horizon,
        reward_scale, diff_reward, supports_route_actions, device,
        max_route_edit_steps=None, force_nonhalt_first_step=False):
    states = []
    rewards = []
    value_estimates = []
    logits = []
    actions = []
    action_kinds = []
    dones = []
    active_masks = []
    score_masks = []
    action_counts = _new_action_count_totals()
    episode_start_costs = []
    episode_final_costs = []

    state = None
    prev_cost = None
    current_start_cost = None
    last_cost = None
    route_step_idx = 0

    with torch.no_grad():
        for _ in range(horizon):
            if state is None or state.is_done().all():
                state, current_start_cost = make_next_state()
                prev_cost = current_start_cost.clone()
                last_cost = current_start_cost.clone()
                route_step_idx = 0
                action_counts["route_plan_count"] += state.batch_size

            done_before = state.is_done()
            active = ~done_before
            state_for_buffer = state.clone()
            _clear_state_lazy_tensors(state_for_buffer)
            states.append(state_for_buffer.to_device("cpu"))
            value_estimates.append(value_module.from_state(state))

            force_halt = max_route_edit_steps is not None and \
                route_step_idx >= max_route_edit_steps
            if force_halt:
                step_actions = torch.full(
                    (state.batch_size, 2), -1, dtype=torch.long,
                    device=state.device)
                step_kinds = torch.full(
                    (state.batch_size,), ROUTE_ACTION_HALT,
                    dtype=torch.long, device=state.device)
                step_logits = torch.zeros(
                    (state.batch_size,), dtype=torch.float32,
                    device=state.device)
                if supports_route_actions:
                    state.apply_route_actions(step_kinds, step_actions)
                else:
                    state.shortest_path_action(step_actions)
                score_mask = torch.zeros_like(active)
            elif supports_route_actions:
                step_kinds, step_actions, step_logits, _ = \
                    model.step_route_action(
                        state,
                        allow_halt=not (
                            force_nonhalt_first_step and route_step_idx == 0
                        ))
                state.apply_route_actions(step_kinds, step_actions)
                score_mask = active
            else:
                step_actions, step_logits, _ = model.step(state)
                step_kinds = torch.full(
                    (state.batch_size,), ROUTE_ACTION_EXTEND,
                    dtype=torch.long, device=state.device)
                step_kinds[step_actions[:, 0] < 0] = ROUTE_ACTION_HALT
                state.shortest_path_action(step_actions)
                score_mask = active
            route_step_idx += 1

            done_after = state.is_done()
            result = cost_obj(state)
            if diff_reward:
                step_rewards = (prev_cost - result.cost) * reward_scale
            else:
                step_rewards = torch.zeros_like(result.cost)
                just_done = done_after & active
                step_rewards[just_done] = -result.cost[just_done] * \
                    reward_scale
            step_rewards = step_rewards * active.to(step_rewards.dtype)

            prev_cost = torch.where(active, result.cost, prev_cost)
            last_cost = torch.where(active, result.cost, last_cost)

            just_finished = done_after & active
            if just_finished.any():
                episode_start_costs.append(
                    current_start_cost[just_finished].detach().cpu())
                episode_final_costs.append(
                    result.cost[just_finished].detach().cpu())

            _merge_step_action_stats(
                action_counts, step_actions, step_kinds, active)

            rewards.append(step_rewards.detach())
            logits.append(step_logits.detach())
            actions.append(step_actions.detach().clone())
            action_kinds.append(step_kinds.detach().clone())
            dones.append(done_after.detach().clone())
            active_masks.append(active.detach().clone())
            score_masks.append(score_mask.detach().clone())

        if state is None:
            final_value_estimates = torch.zeros_like(rewards[-1])
        else:
            unfinished = ~state.is_done()
            if unfinished.any():
                episode_start_costs.append(
                    current_start_cost[unfinished].detach().cpu())
                episode_final_costs.append(
                    last_cost[unfinished].detach().cpu())
            final_value_estimates = value_module.from_state(state)
            final_value_estimates = final_value_estimates * \
                unfinished.to(final_value_estimates.dtype)

    rollout = {
        "states": states,
        "rewards": torch.stack(rewards),
        "value_estimates": torch.stack(value_estimates),
        "logits": torch.stack(logits),
        "actions": torch.stack(actions),
        "action_kinds": torch.stack(action_kinds),
        "dones": torch.stack(dones),
        "active_masks": torch.stack(active_masks),
        "score_masks": torch.stack(score_masks),
        "final_value_estimates": final_value_estimates.detach(),
        "action_counts": action_counts,
    }
    if len(episode_start_costs) > 0:
        rollout["episode_start_costs"] = torch.cat(episode_start_costs)
        rollout["episode_final_costs"] = torch.cat(episode_final_costs)
    else:
        rollout["episode_start_costs"] = torch.empty(0)
        rollout["episode_final_costs"] = torch.empty(0)
    return rollout


def _update_lc_improvement_cfg_ppo_from_rollout(
        model, optimizer, value_module, rollout, returns, advantages,
        ppo_epochs, minibatch_size, clip_epsilon, entropy_weight, device):
    states = rollout["states"]
    rewards = rollout["rewards"]
    old_logits = rollout["logits"]
    actions = rollout["actions"]
    action_kinds = rollout["action_kinds"]
    score_masks = rollout.get("score_masks", rollout["active_masks"])
    batch_size = rewards.shape[1]
    n_states_per_minibatch = max(1, int(minibatch_size) // int(batch_size))
    supports_route_actions = getattr(model, "supports_trim_actions", False)

    ratio_sum = 0.0
    ratio_count = 0
    clipped_count = 0
    objectives = []

    train_orders = [
        torch.randperm(len(states), device=device)
        for _ in range(int(ppo_epochs))
    ]
    train_order = torch.cat(train_orders)

    for raw_idxs in torch.split(train_order, n_states_per_minibatch):
        for idxs in _split_state_indices_by_signature(states, raw_idxs):
            idx_list = idxs.detach().cpu().tolist()
            mb_states = [states[idx] for idx in idx_list]
            if len(mb_states) == 1:
                mb_states = mb_states[0]
            else:
                mb_states = RouteGenBatchState.batch_from_list(mb_states)
            mb_states = mb_states.to_device(device)

            mb_actions = actions[idxs].flatten(0, 1)
            mb_action_kinds = action_kinds[idxs].flatten(0, 1)
            mb_old_logits = old_logits[idxs].flatten(0, 1)
            mb_returns = returns[idxs].flatten(0, 1)
            mb_advantages = advantages[idxs].flatten(0, 1)
            mb_score = score_masks[idxs].flatten(0, 1)
            if not mb_score.any():
                continue

            active_idxs = torch.where(mb_score)[0]
            mb_states = mb_states.index_select(active_idxs)
            mb_actions = mb_actions[mb_score]
            mb_action_kinds = mb_action_kinds[mb_score]
            mb_old_logits = mb_old_logits[mb_score]
            mb_returns = mb_returns[mb_score]
            mb_advantages = mb_advantages[mb_score]

            value_module.from_state(mb_states)
            value_module.update(mb_returns)

            if supports_route_actions:
                _, _, new_logits, entropy = model.step_route_action(
                    mb_states, actions=mb_actions,
                    action_kinds=mb_action_kinds)
            else:
                _, new_logits, entropy = model.step(
                    mb_states, actions=mb_actions)

            if mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / \
                    (mb_advantages.std() + 1e-8)
            else:
                mb_advantages = mb_advantages - mb_advantages.mean()

            ratios = (new_logits - mb_old_logits).exp()
            clipped_ratios = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            clip_obj = torch.minimum(
                ratios * mb_advantages,
                clipped_ratios * mb_advantages)
            objective = clip_obj.mean() + entropy.mean() * entropy_weight

            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()

            objectives.append(float(objective.detach().item()))
            ratio_sum += float(ratios.detach().sum().item())
            ratio_count += int(ratios.numel())
            clipped = (ratios.detach() < 1 - clip_epsilon) | \
                (ratios.detach() > 1 + clip_epsilon)
            clipped_count += int(clipped.sum().item())

    stats = {
        "objective": float(np.mean(objectives)) if objectives else 0.0,
        "ratio_sum": ratio_sum,
        "ratio_count": ratio_count,
        "clipped_count": clipped_count,
    }
    return stats

@torch.no_grad()
def evaluate_lc_improvement(model, cost_obj, graphs, seed_routes, indices,
                            device, min_route_len, max_route_len,
                            batch_size=8, force_nonhalt_first_step=False,
                            max_route_edit_steps=None,
                            return_action_stats=False,
                            target_n_routes=None):
    model.eval()
    seed_costs = []
    final_costs = []
    route_change_masks = []
    route_outputs = []
    action_counts = _new_action_count_totals()
    eval_weights = cost_obj.get_weights(device)

    for batch_indices in tqdm(list(indices.split(batch_size)),
                              desc="eval", leave=False):
        graph_batch, route_batch = make_improvement_batch(
            graphs, seed_routes, batch_indices, device, training=False,
            target_n_routes=target_n_routes)
        rollout_output = rollout_lc_improvement(
            model, cost_obj, graph_batch, route_batch, min_route_len,
            max_route_len, greedy=True, cost_weights=eval_weights,
            return_actions=return_action_stats,
            force_nonhalt_first_step=force_nonhalt_first_step,
            max_route_edit_steps=max_route_edit_steps)
        state, seed_result, final_result, _, _ = rollout_output[:5]
        if return_action_stats:
            _, _, _, _, _, route_actions, route_action_kinds = rollout_output
            batch_action_stats = summarize_route_action_stats(
                route_actions, route_action_kinds)
            _merge_action_stats(action_counts, batch_action_stats)
        seed_costs.append(seed_result.cost.cpu())
        final_costs.append(final_result.cost.cpu())
        final_routes = get_batch_tensor_from_routes(state.routes).cpu()
        route_outputs.append(final_routes)
        route_change_masks.append(
            _route_change_mask(route_batch.cpu(), final_routes).cpu())

    seed_costs = torch.cat(seed_costs)
    final_costs = torch.cat(final_costs)
    route_change_mask = torch.cat(route_change_masks)
    result = {
        "target_n_routes": int(
            target_n_routes if target_n_routes is not None
            else seed_routes.shape[1]),
        "seed_cost": seed_costs.mean().item(),
        "final_cost": final_costs.mean().item(),
        "delta": (seed_costs - final_costs).mean().item(),
        "win_rate": (final_costs < seed_costs).float().mean().item(),
        "changed_route_rate": route_change_mask.float().mean().item(),
        "changed_graph_rate": route_change_mask.any(dim=1).float().mean().item(),
        "routes": route_outputs,
    }
    if return_action_stats:
        result["action_stats"] = _finalize_action_stats(action_counts)
    return result


def train_lc_improvement(model, cost_obj, graphs, seed_routes, device,
                         output_dir, run_name,
                         train_fraction=0.9, batch_size=8, n_epochs=20,
                         lr=1e-4, weight_decay=1e-4, entropy_weight=1e-3,
                         min_route_len=2, max_route_len=8,
                         warmup_batches=4, seed=0,
                         force_nonhalt_first_step=False,
                         max_route_edit_steps=None,
                         train_indices=None, val_indices=None,
                         best_model_path=None, target_n_routes=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    if (train_indices is None) != (val_indices is None):
        raise ValueError(
            "train_indices and val_indices must be provided together"
        )
    if train_indices is None:
        all_indices = torch.randperm(len(graphs))
        train_size = int(train_fraction * len(all_indices))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
    else:
        train_indices = torch.as_tensor(train_indices, dtype=torch.long)
        val_indices = torch.as_tensor(val_indices, dtype=torch.long)

    model.train()
    with torch.no_grad():
        warmup = list(train_indices.split(batch_size))[:warmup_batches]
        for batch_indices in tqdm(warmup, desc="feature norm"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                                   min_route_len, max_route_len, greedy=False,
                                   max_route_edit_steps=max_route_edit_steps)
    model.update_and_freeze_feature_norms()

    best_val_cost = float("inf")
    if best_model_path is None:
        best_model_path = output_dir / f"{run_name}.pt"
    else:
        best_model_path = Path(best_model_path)
    history = []
    last_val = None

    for epoch in range(n_epochs):
        model.train()
        epoch_indices = train_indices[torch.randperm(len(train_indices))]
        train_seed_costs = []
        train_final_costs = []
        train_route_change_rates = []
        train_action_counts = _new_action_count_totals()

        for batch_indices in tqdm(list(epoch_indices.split(batch_size)),
                                  desc=f"epoch {epoch + 1}/{n_epochs}"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            cost_weights = cost_obj.sample_variable_weights(
                graph_batch.num_graphs, device)
            with torch.no_grad():
                state, seed_result, final_result, _, _, route_actions, \
                    route_action_kinds = rollout_lc_improvement(
                        model, cost_obj, graph_batch, route_batch,
                        min_route_len, max_route_len, greedy=False,
                        cost_weights=cost_weights, return_actions=True,
                        force_nonhalt_first_step=force_nonhalt_first_step,
                        max_route_edit_steps=max_route_edit_steps,
                        sequential_empty_routes=False)
            batch_action_stats = summarize_route_action_stats(
                route_actions, route_action_kinds)
            _merge_action_stats(train_action_counts, batch_action_stats)

            improvement = seed_result.cost - final_result.cost
            advantages = improvement.detach() - improvement.detach().mean()
            if advantages.numel() > 1:
                advantages = advantages / (advantages.std() + 1e-8)

            optimizer.zero_grad()
            objective = _backward_on_sampled_actions(
                model, cost_obj, graph_batch, route_batch, min_route_len,
                max_route_len, cost_weights, route_actions,
                route_action_kinds, advantages, entropy_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_seed_costs.append(seed_result.cost.detach().cpu())
            train_final_costs.append(final_result.cost.detach().cpu())
            final_routes = get_batch_tensor_from_routes(state.routes).detach()
            changed = _route_change_mask(route_batch.detach(), final_routes)
            train_route_change_rates.append(changed.float().mean().cpu())

        train_seed = torch.cat(train_seed_costs).mean().item()
        train_final = torch.cat(train_final_costs).mean().item()
        train_changed = torch.stack(train_route_change_rates).mean().item()
        train_action_stats = _finalize_action_stats(train_action_counts)
        eval_due = epoch > 0 or epoch == n_epochs - 1
        if eval_due:
            last_val = evaluate_lc_improvement(
                model, cost_obj, graphs, seed_routes, val_indices, device,
                min_route_len, max_route_len, batch_size=batch_size,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                target_n_routes=target_n_routes)

            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                torch.save(model.state_dict(), best_model_path)

        val = last_val or {
            "seed_cost": float("nan"),
            "final_cost": float("nan"),
            "delta": float("nan"),
            "win_rate": float("nan"),
            "changed_route_rate": float("nan"),
            "changed_graph_rate": float("nan"),
        }

        row = {
            "epoch": epoch + 1,
            "target_n_routes": int(
                target_n_routes if target_n_routes is not None
                else seed_routes.shape[1]),
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_seed - train_final,
            "train_changed_route_rate": train_changed,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
            "val_changed_route_rate": val["changed_route_rate"],
            "val_changed_graph_rate": val["changed_graph_rate"],
        }
        row.update({
            f"train_action_{key}": value
            for key, value in train_action_stats.items()
        })
        history.append(row)
        print(
            f"epoch={row['epoch']:02d} "
            f"train_seed={row['train_seed_cost']:.4f} "
            f"train_final={row['train_final_cost']:.4f} "
            f"train_delta={row['train_delta']:.4f} "
            f"train_changed={row['train_changed_route_rate']:.2%} "
            f"val_seed={row['val_seed_cost']:.4f} "
            f"val_final={row['val_final_cost']:.4f} "
            f"val_delta={row['val_delta']:.4f} "
            f"win_rate={row['val_win_rate']:.2%} "
            f"val_changed={row['val_changed_route_rate']:.2%} "
            f"actions="
            f"ext:{row['train_action_extend_count']} "
            f"trim_s:{row['train_action_trim_start_count']} "
            f"trim_e:{row['train_action_trim_end_count']} "
            f"halt:{row['train_action_halt_count']} "
            f"avg_steps={row['train_action_avg_actions_per_route']:.2f}"
        )

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def train_lc_improvement_ppo(model, cost_obj, graphs, seed_routes, device,
                             output_dir, run_name,
                             train_fraction=0.9, batch_size=8, n_epochs=20,
                             lr=1e-4, weight_decay=1e-4,
                             entropy_weight=1e-3, min_route_len=2,
                             max_route_len=8, warmup_batches=4, seed=0,
                             force_nonhalt_first_step=False,
                             max_route_edit_steps=None,
                             train_indices=None, val_indices=None,
                             best_model_path=None, ppo_epochs=1,
                             clip_epsilon=0.2, target_n_routes=None):
    """Train LC improvement with a separate PPO-style clipped objective.

    This keeps the seeded-route improvement environment, but updates sampled
    actions with the same ratio-clipping idea used by PPO.  It is intentionally
    separate from both the construction PPO loop and the simpler improvement
    policy-gradient loop above.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    if (train_indices is None) != (val_indices is None):
        raise ValueError(
            "train_indices and val_indices must be provided together"
        )
    if train_indices is None:
        all_indices = torch.randperm(len(graphs))
        train_size = int(train_fraction * len(all_indices))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
    else:
        train_indices = torch.as_tensor(train_indices, dtype=torch.long)
        val_indices = torch.as_tensor(val_indices, dtype=torch.long)

    model.train()
    with torch.no_grad():
        warmup = list(train_indices.split(batch_size))[:warmup_batches]
        for batch_indices in tqdm(warmup, desc="feature norm"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                                   min_route_len, max_route_len, greedy=False,
                                   max_route_edit_steps=max_route_edit_steps)
    model.update_and_freeze_feature_norms()

    best_val_cost = float("inf")
    if best_model_path is None:
        best_model_path = output_dir / f"{run_name}.pt"
    else:
        best_model_path = Path(best_model_path)
    history = []
    last_val = None

    for epoch in range(n_epochs):
        model.train()
        epoch_indices = train_indices[torch.randperm(len(train_indices))]
        train_seed_costs = []
        train_final_costs = []
        train_route_change_rates = []
        train_action_counts = _new_action_count_totals()
        ppo_ratio_sum = 0.0
        ppo_ratio_count = 0
        ppo_clipped_count = 0
        ppo_objectives = []

        for batch_indices in tqdm(list(epoch_indices.split(batch_size)),
                                  desc=f"ppo epoch {epoch + 1}/{n_epochs}"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            cost_weights = cost_obj.sample_variable_weights(
                graph_batch.num_graphs, device)
            with torch.no_grad():
                state, seed_result, final_result, _, _, route_actions, \
                    route_action_kinds, old_step_logits, _ = \
                    rollout_lc_improvement(
                        model, cost_obj, graph_batch, route_batch,
                        min_route_len, max_route_len, greedy=False,
                        cost_weights=cost_weights, return_actions=True,
                        return_step_data=True,
                        force_nonhalt_first_step=force_nonhalt_first_step,
                        max_route_edit_steps=max_route_edit_steps,
                        sequential_empty_routes=False)

            if any(step_logits is None for step_logits in old_step_logits):
                raise RuntimeError(
                    "PPO improvement requires per-step old log-probs. "
                    "Use a route generator that records last_route_step_logits."
                )

            batch_action_stats = summarize_route_action_stats(
                route_actions, route_action_kinds)
            _merge_action_stats(train_action_counts, batch_action_stats)

            improvement = seed_result.cost - final_result.cost
            advantages = improvement.detach() - improvement.detach().mean()
            if advantages.numel() > 1:
                advantages = advantages / (advantages.std() + 1e-8)

            for _ in range(ppo_epochs):
                optimizer.zero_grad()
                objective, ppo_stats = _backward_on_sampled_actions_ppo(
                    model, cost_obj, graph_batch, route_batch, min_route_len,
                    max_route_len, cost_weights, route_actions,
                    route_action_kinds, old_step_logits, advantages,
                    entropy_weight, clip_epsilon)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                ppo_objectives.append(ppo_stats["objective"])
                ppo_ratio_sum += ppo_stats["ratio_sum"]
                ppo_ratio_count += ppo_stats["ratio_count"]
                ppo_clipped_count += ppo_stats["clipped_count"]

            train_seed_costs.append(seed_result.cost.detach().cpu())
            train_final_costs.append(final_result.cost.detach().cpu())
            final_routes = get_batch_tensor_from_routes(state.routes).detach()
            changed = _route_change_mask(route_batch.detach(), final_routes)
            train_route_change_rates.append(changed.float().mean().cpu())

        train_seed = torch.cat(train_seed_costs).mean().item()
        train_final = torch.cat(train_final_costs).mean().item()
        train_changed = torch.stack(train_route_change_rates).mean().item()
        train_action_stats = _finalize_action_stats(train_action_counts)
        eval_due = epoch > 0 or epoch == n_epochs - 1
        if eval_due:
            last_val = evaluate_lc_improvement(
                model, cost_obj, graphs, seed_routes, val_indices, device,
                min_route_len, max_route_len, batch_size=batch_size,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                target_n_routes=target_n_routes)

            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                torch.save(model.state_dict(), best_model_path)

        ppo_ratio_mean = ppo_ratio_sum / max(ppo_ratio_count, 1)
        ppo_clip_fraction = ppo_clipped_count / max(ppo_ratio_count, 1)
        ppo_objective = float(np.mean(ppo_objectives)) \
            if len(ppo_objectives) > 0 else 0.0
        val = last_val or {
            "seed_cost": float("nan"),
            "final_cost": float("nan"),
            "delta": float("nan"),
            "win_rate": float("nan"),
            "changed_route_rate": float("nan"),
            "changed_graph_rate": float("nan"),
        }

        row = {
            "epoch": epoch + 1,
            "algorithm": "ppo_improvement",
            "target_n_routes": int(
                target_n_routes if target_n_routes is not None
                else seed_routes.shape[1]),
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_seed - train_final,
            "train_changed_route_rate": train_changed,
            "train_ppo_ratio_mean": ppo_ratio_mean,
            "train_ppo_clip_fraction": ppo_clip_fraction,
            "train_ppo_objective": ppo_objective,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
            "val_changed_route_rate": val["changed_route_rate"],
            "val_changed_graph_rate": val["changed_graph_rate"],
        }
        row.update({
            f"train_action_{key}": value
            for key, value in train_action_stats.items()
        })
        history.append(row)
        print(
            f"ppo_epoch={row['epoch']:02d} "
            f"train_seed={row['train_seed_cost']:.4f} "
            f"train_final={row['train_final_cost']:.4f} "
            f"train_delta={row['train_delta']:.4f} "
            f"train_changed={row['train_changed_route_rate']:.2%} "
            f"val_seed={row['val_seed_cost']:.4f} "
            f"val_final={row['val_final_cost']:.4f} "
            f"val_delta={row['val_delta']:.4f} "
            f"win_rate={row['val_win_rate']:.2%} "
            f"val_changed={row['val_changed_route_rate']:.2%} "
            f"ratio={row['train_ppo_ratio_mean']:.3f} "
            f"clip={row['train_ppo_clip_fraction']:.2%} "
            f"actions="
            f"ext:{row['train_action_extend_count']} "
            f"trim_s:{row['train_action_trim_start_count']} "
            f"trim_e:{row['train_action_trim_end_count']} "
            f"halt:{row['train_action_halt_count']} "
            f"avg_steps={row['train_action_avg_actions_per_route']:.2f}"
        )

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def train_lc_improvement_cfg_ppo(
        model, cost_obj, graphs, seed_routes, device, cfg, output_dir, run_name,
        train_fraction=0.9, batch_size=None, n_iterations=None,
        val_period=None, horizon=None, ppo_epochs=None, minibatch_size=None,
        min_route_len=None, max_route_len=None, warmup_batches=4, seed=0,
        force_nonhalt_first_step=False, max_route_edit_steps=None,
        train_indices=None, val_indices=None, best_model_path=None,
        max_rollout_samples=8192, target_n_routes=None):
    """Train LC improvement with the construction PPO machinery adapted to edits.

    Unlike ``train_lc_improvement_ppo`` above, this function reads the PPO
    hyperparameters from the regular PPO config and uses the same core pieces:
    horizon rollouts, diff/non-diff rewards, reward scaling, discounting,
    GAE/simple returns, neural value baseline, PPO epochs, minibatches,
    entropy bonus, optimizer type/lr/decay, and clipped ratio updates.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_route_len is None:
        min_route_len = int(cfg.eval.min_route_len)
    if max_route_len is None:
        max_route_len = int(cfg.eval.max_route_len)
    if batch_size is None:
        batch_size = int(_get_cfg_value(cfg, "batch_size", 8))
    if n_iterations is None:
        n_iterations = int(cfg.ppo.n_iterations)
    if val_period is None:
        val_period = int(cfg.ppo.val_period)
    if horizon is None:
        horizon = int(cfg.ppo.horizon)
    if ppo_epochs is None:
        ppo_epochs = int(cfg.ppo.n_epochs)
    if minibatch_size is None:
        minibatch_size = int(cfg.ppo.minibatch_size)
    if max_route_edit_steps is None:
        max_route_edit_steps = _get_default_max_route_edit_steps(max_route_len)

    reward_scale = float(_get_cfg_value(cfg, "reward_scale", 1.0))
    diff_reward = bool(_get_cfg_value(cfg, "diff_reward", True))
    gamma = float(_get_cfg_value(cfg, "discount_rate", 1.0))
    entropy_weight = float(_get_cfg_value(cfg, "entropy_weight", 0.0))
    clip_epsilon = float(cfg.ppo.epsilon)
    use_gae = bool(cfg.ppo.use_gae)
    gae_lambda = float(cfg.ppo.gae_lambda)

    optimizer = _make_optimizer_from_cfg(model, cfg)

    # Reuse the original PPO value baseline.  The class relies on the module
    # global DEVICE, so set it before construction.
    from . import inductive_route_learning as il
    il.DEVICE = device
    value_module = il.NNBaseline(
        learning_rate=float(_get_cfg_value(cfg, "baseline_lr", 0.0005))
    )

    if (train_indices is None) != (val_indices is None):
        raise ValueError(
            "train_indices and val_indices must be provided together"
        )
    if train_indices is None:
        all_indices = torch.randperm(len(graphs))
        train_size = int(train_fraction * len(all_indices))
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]
    else:
        train_indices = torch.as_tensor(train_indices, dtype=torch.long)
        val_indices = torch.as_tensor(val_indices, dtype=torch.long)

    if len(train_indices) == 0:
        raise ValueError("No training graphs were selected")
    effective_batch_size = min(int(batch_size), len(train_indices))
    rollout_samples = effective_batch_size * int(horizon)
    if effective_batch_size > int(minibatch_size):
        raise ValueError(
            "LC improvement PPO keeps full RouteGenBatchState snapshots. "
            "Set BATCH_SIZE <= cfg.ppo.minibatch_size. "
            f"Got BATCH_SIZE={effective_batch_size}, "
            f"cfg.ppo.minibatch_size={minibatch_size}."
        )
    if rollout_samples > int(max_rollout_samples):
        raise ValueError(
            "LC improvement PPO rollout is too large for the notebook "
            "default memory budget. Lower BATCH_SIZE or PPO_HORIZON. "
            f"Got BATCH_SIZE * PPO_HORIZON = {effective_batch_size} * "
            f"{int(horizon)} = {rollout_samples}, limit is "
            f"{int(max_rollout_samples)}."
        )
    n_routes = int(
        target_n_routes if target_n_routes is not None
        else seed_routes.shape[1])
    if n_routes < int(seed_routes.shape[1]):
        raise ValueError(
            "target_n_routes must be at least the number of seed routes: "
            f"{n_routes} < {int(seed_routes.shape[1])}"
        )

    model.train()
    with torch.no_grad():
        warmup = list(train_indices.split(effective_batch_size))[
            :warmup_batches]
        for batch_indices in tqdm(warmup, desc="feature norm"):
            graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            rollout_lc_improvement(
                model, cost_obj, graph_batch, route_batch,
                min_route_len, max_route_len, greedy=False,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps)
    model.update_and_freeze_feature_norms()

    if best_model_path is None:
        best_model_path = output_dir / f"{run_name}.pt"
    else:
        best_model_path = Path(best_model_path)

    best_val_cost = float("inf")
    last_val = None
    history = []
    epoch_indices = train_indices[torch.randperm(len(train_indices))]
    index_cursor = 0
    route_cursor = 0

    def make_next_state():
        nonlocal epoch_indices, index_cursor, route_cursor

        if index_cursor + effective_batch_size > len(epoch_indices):
            epoch_indices = train_indices[torch.randperm(len(train_indices))]
            index_cursor = 0

        batch_indices = epoch_indices[
            index_cursor:index_cursor + effective_batch_size]
        index_cursor += effective_batch_size

        graph_batch, route_batch = make_improvement_batch(
            graphs, seed_routes, batch_indices, device, training=True,
            target_n_routes=target_n_routes)
        cost_weights = cost_obj.sample_variable_weights(
            graph_batch.num_graphs, device)

        route_idx = route_cursor
        route_cursor = (route_cursor + 1) % n_routes
        route_is_empty = (route_batch[:, route_idx] > -1).sum(dim=-1) == 0
        state = _make_route_context_state(
            cost_obj, graph_batch, route_batch, route_idx,
            min_route_len, max_route_len, cost_weights,
            invalid_directly_connected=bool(route_is_empty.all().item()))
        state = model.setup_planning(state)
        start_result = cost_obj(state)
        return state, start_result.cost.detach()

    pbar = tqdm(range(int(n_iterations)), desc="cfg ppo improvement")
    for iteration in pbar:
        model.train()
        rollout = _collect_lc_improvement_cfg_ppo_rollout(
            model, cost_obj, make_next_state, value_module, int(horizon),
            reward_scale, diff_reward,
            getattr(model, "supports_trim_actions", False), device,
            max_route_edit_steps=max_route_edit_steps,
            force_nonhalt_first_step=force_nonhalt_first_step)
        returns, advantages = _compute_ppo_returns_and_advantages(
            rollout["rewards"], rollout["value_estimates"],
            rollout["dones"], rollout["final_value_estimates"], gamma,
            use_gae, gae_lambda)
        ppo_stats = _update_lc_improvement_cfg_ppo_from_rollout(
            model, optimizer, value_module, rollout, returns, advantages,
            ppo_epochs, minibatch_size, clip_epsilon, entropy_weight, device)

        train_action_stats = _finalize_action_stats(
            rollout["action_counts"])
        active_rewards = rollout["rewards"][rollout["active_masks"]]
        active_returns = returns[rollout["active_masks"]]
        active_advantages = advantages[rollout["active_masks"]]
        start_costs = rollout["episode_start_costs"]
        final_costs = rollout["episode_final_costs"]
        if len(start_costs) > 0:
            train_seed = start_costs.mean().item()
            train_final = final_costs.mean().item()
            train_delta = (start_costs - final_costs).mean().item()
        else:
            train_seed = float("nan")
            train_final = float("nan")
            train_delta = float("nan")

        eval_due = (
            (iteration > 0 and (iteration + 1) % max(int(val_period), 1) == 0)
            or iteration == int(n_iterations) - 1
        )
        if eval_due:
            last_val = evaluate_lc_improvement(
                model, cost_obj, graphs, seed_routes, val_indices, device,
                min_route_len, max_route_len,
                batch_size=effective_batch_size,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                target_n_routes=target_n_routes)
            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                torch.save(model.state_dict(), best_model_path)

        ratio_mean = ppo_stats["ratio_sum"] / max(ppo_stats["ratio_count"], 1)
        clip_fraction = ppo_stats["clipped_count"] / \
            max(ppo_stats["ratio_count"], 1)
        val = last_val or {
            "seed_cost": float("nan"),
            "final_cost": float("nan"),
            "delta": float("nan"),
            "win_rate": float("nan"),
            "changed_route_rate": float("nan"),
            "changed_graph_rate": float("nan"),
        }
        row = {
            "iteration": iteration + 1,
            "epoch": iteration + 1,
            "algorithm": "cfg_ppo_improvement",
            "target_n_routes": n_routes,
            "diff_reward": diff_reward,
            "reward_scale": reward_scale,
            "discount_rate": gamma,
            "ppo_horizon": int(horizon),
            "ppo_epochs": int(ppo_epochs),
            "ppo_minibatch_size": int(minibatch_size),
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_delta,
            "train_reward_mean": active_rewards.mean().item()
                if active_rewards.numel() > 0 else 0.0,
            "train_return_mean": active_returns.mean().item()
                if active_returns.numel() > 0 else 0.0,
            "train_advantage_mean": active_advantages.mean().item()
                if active_advantages.numel() > 0 else 0.0,
            "train_active_sample_count": int(
                rollout["active_masks"].sum().item()),
            "train_ppo_ratio_mean": ratio_mean,
            "train_ppo_clip_fraction": clip_fraction,
            "train_ppo_objective": ppo_stats["objective"],
            "is_eval_iteration": eval_due,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
            "val_changed_route_rate": val["changed_route_rate"],
            "val_changed_graph_rate": val["changed_graph_rate"],
        }
        row.update({
            f"train_action_{key}": value
            for key, value in train_action_stats.items()
        })
        # Full PPO edits one active route slot at a time, so there is no cheap
        # per-iteration full-route-set change rate. Validation tracks it.
        row["train_changed_route_rate"] = float("nan")
        history.append(row)

        pbar.set_postfix({
            "reward": f"{row['train_reward_mean']:.3f}",
            "delta": f"{row['train_delta']:.3f}",
            "ratio": f"{row['train_ppo_ratio_mean']:.3f}",
            "clip": f"{row['train_ppo_clip_fraction']:.2%}",
        })
        print(
            f"cfg_ppo_iter={row['iteration']:03d} "
            f"reward={row['train_reward_mean']:.4f} "
            f"train_delta={row['train_delta']:.4f} "
            f"val_delta={row['val_delta']:.4f} "
            f"ratio={row['train_ppo_ratio_mean']:.3f} "
            f"clip={row['train_ppo_clip_fraction']:.2%} "
            f"actions="
            f"ext:{row['train_action_extend_count']} "
            f"trim_s:{row['train_action_trim_start_count']} "
            f"trim_e:{row['train_action_trim_end_count']} "
            f"halt:{row['train_action_halt_count']} "
            f"avg_steps={row['train_action_avg_actions_per_route']:.2f} "
            f"eval={row['is_eval_iteration']}"
        )

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }

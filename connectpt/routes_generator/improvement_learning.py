import pickle
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from .citygraph_dataset import (
    DEMAND_KEY,
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
from .models import FeatureNorm, get_mlp
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
    state.set_route_slot_context(route_batch, route_idx)
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


def _get_current_routes_from_state(state):
    current_routes = []
    for route in state.current_routes:
        current_routes.append(route[route > -1].clone())
    return current_routes


def _clone_route_list(route_list):
    return [route.clone() for route in route_list]


def _update_route_list_at_mask(route_list, new_routes, mask):
    mask = mask.detach().cpu().tolist()
    for batch_idx, should_update in enumerate(mask):
        if should_update:
            route_list[batch_idx] = new_routes[batch_idx].clone()


def _plan_lc_route_with_best_tracking(
        model, cost_obj, route_state, fallback_routes, context_route_counts,
        greedy=False, force_nonhalt_first_step=False,
        max_route_edit_steps=None, max_trim_actions_per_route=1):
    """Plan one route and return the best in-episode current-route version."""
    route_state = model.setup_planning(route_state)
    supports_route_actions = getattr(model, "supports_trim_actions", False)

    ended = torch.zeros((route_state.batch_size,), dtype=torch.bool,
                        device=route_state.device)
    all_logits = None
    all_entropy = None
    actions_log = []
    action_kinds_log = []
    step_logits_log = []
    step_entropies_log = []

    best_cost = cost_obj(route_state).cost.detach()
    best_routes = _get_planned_current_routes(
        route_state, fallback_routes, context_route_counts)

    step_idx = 0
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)
    trim_action_counts = torch.zeros(
        (route_state.batch_size,), dtype=torch.long,
        device=route_state.device)
    while not ended.all():
        was_ended = ended.clone()
        force_halt = (
            max_route_edit_steps is not None and
            step_idx >= max_route_edit_steps
        )
        if force_halt:
            step_kinds = torch.full(
                (route_state.batch_size,), ROUTE_ACTION_HALT,
                dtype=torch.long, device=route_state.device)
            action = torch.full(
                (route_state.batch_size, 2), -1, dtype=torch.long,
                device=route_state.device)
            template = best_cost.to(route_state.device)
            logits = torch.zeros_like(template)
            entropy = torch.zeros_like(template)
        elif supports_route_actions:
            encoding = model._encode_graph(route_state)
            allow_trim = _trim_actions_allowed(
                trim_action_counts, max_trim_actions_per_route)
            step_kinds, action, logits, entropy = model.step_route_action(
                route_state, greedy, precalc_data=encoding,
                allow_halt=not (
                    force_nonhalt_first_step and step_idx == 0
                ),
                allow_trim_start=allow_trim,
                allow_trim_end=allow_trim)
        else:
            action, logits, entropy = model.step(route_state, greedy)
            step_kinds = torch.full(
                (route_state.batch_size,), ROUTE_ACTION_EXTEND,
                dtype=torch.long, device=route_state.device)
            step_kinds[action[:, 0] < 0] = ROUTE_ACTION_HALT

        if all_logits is None:
            all_logits = torch.zeros_like(logits)
            all_entropy = torch.zeros_like(entropy)
        all_logits = all_logits + logits * (~was_ended)
        all_entropy = all_entropy + entropy * (~was_ended)

        just_ended = step_kinds == ROUTE_ACTION_HALT
        selected_trim = _is_trim_action(step_kinds) & ~was_ended
        trim_action_counts = trim_action_counts + selected_trim.long()
        ended = ended | just_ended
        step_kinds = step_kinds.clone()
        action = action.clone()
        step_kinds[ended] = ROUTE_ACTION_HALT
        action[ended] = -1

        if supports_route_actions:
            route_state.apply_route_actions(step_kinds, action)
        else:
            legacy_actions = action.clone()
            legacy_actions[step_kinds == ROUTE_ACTION_HALT] = -1
            route_state.shortest_path_action(legacy_actions)

        result = cost_obj(route_state)
        improved = (~was_ended) & (result.cost < best_cost)
        if improved.any():
            candidate_routes = _get_planned_current_routes(
                route_state, fallback_routes, context_route_counts)
            _update_route_list_at_mask(best_routes, candidate_routes, improved)
            best_cost = torch.where(improved, result.cost.detach(), best_cost)

        actions_log.append(action.detach().clone())
        action_kinds_log.append(step_kinds.detach().clone())
        step_logits = logits.detach().clone()
        step_entropies = entropy.detach().clone()
        step_logits[was_ended] = 0
        step_entropies[was_ended] = 0
        step_logits_log.append(step_logits)
        step_entropies_log.append(step_entropies)
        step_idx += 1

    actions = torch.stack(actions_log, dim=1)
    model.last_route_action_kinds = torch.stack(action_kinds_log, dim=1)
    model.last_route_step_logits = torch.stack(step_logits_log, dim=1)
    model.last_route_step_entropies = torch.stack(step_entropies_log, dim=1)
    return actions, all_logits, all_entropy, best_routes


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


def _normalize_max_trim_actions_per_route(value):
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(
            "max_trim_actions_per_route must be non-negative or None"
        )
    return value


def _trim_actions_allowed(trim_action_counts, max_trim_actions_per_route):
    if max_trim_actions_per_route is None:
        return torch.ones_like(trim_action_counts, dtype=torch.bool)
    return trim_action_counts < int(max_trim_actions_per_route)


def rollout_lc_improvement(model, cost_obj, graph_batch, route_batch,
                           min_route_len, max_route_len, greedy=False,
                           cost_weights=None, return_actions=False,
                           force_nonhalt_first_step=False,
                           max_route_edit_steps=None,
                           max_trim_actions_per_route=1,
                           return_step_data=False,
                           return_best_routes=False,
                           adjustment_target=None, adjustment_weight=None,
                           adjustment_use_current=False,
                           adjustment_gap=0.1, adjustment_mode='paper'):
    if cost_weights is None:
        cost_weights = cost_obj.sample_variable_weights(graph_batch.num_graphs,
                                                        graph_batch[STOP_KEY].x.device)
    if max_route_edit_steps is None:
        max_route_edit_steps = _get_default_max_route_edit_steps(max_route_len)
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)
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
    for route_idx in range(n_routes):
        # If the entire context (all other slots) has no real stops, the
        # network is empty and we must forbid trivial directly-connected
        # terminal pairs for the first extend.
        context_routes = working_route_batch.clone()
        context_routes[:, route_idx] = -1
        invalid_directly_connected = not bool(
            (context_routes >= 0).any().item())

        route_state = _make_route_context_state(
            cost_obj, graph_batch, working_route_batch, route_idx,
            min_route_len, max_route_len, cost_weights,
            invalid_directly_connected=invalid_directly_connected)
        # Gated adjustment conditioning at eval: a model built with
        # n_adjustment_cond_feats>0 expects the target/weight features, so set
        # them on every route state it sees (else 12-vs-14 dim mismatch).
        if adjustment_target is not None:
            # weight=None -> target-only conditioning (W excluded from features).
            # seed_routes=route_batch -> enables the live current-adj feature.
            route_state.set_adjustment_conditioning(
                adjustment_target, adjustment_weight,
                seed_routes=(route_batch if adjustment_use_current else None),
                gap=adjustment_gap, mode=adjustment_mode)
        context_route_counts = route_state.n_finished_routes.detach().clone()
        if return_best_routes:
            fallback_routes = _get_current_routes_from_state(route_state)
            actions, logits, entropy, planned_current_routes = \
                _plan_lc_route_with_best_tracking(
                    model, cost_obj, route_state, fallback_routes,
                    context_route_counts, greedy=greedy,
                    force_nonhalt_first_step=force_nonhalt_first_step,
                    max_route_edit_steps=max_route_edit_steps,
                    max_trim_actions_per_route=max_trim_actions_per_route)
        elif getattr(model, "supports_trim_actions", False):
            actions, logits, entropy = model.plan_new_route(
                route_state, greedy=greedy,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_steps=max_route_edit_steps,
                max_trim_actions=max_trim_actions_per_route)
        else:
            actions, logits, entropy = model.plan_new_route(
                route_state, greedy=greedy)
        if not return_best_routes:
            planned_current_routes = \
                _get_planned_current_routes(
                    route_state, working_route_batch[:, route_idx],
                    context_route_counts)
        route_logits.append(logits)
        route_entropies.append(entropy)
        routes_by_route.append(planned_current_routes)
        # Write the planned route back so subsequent slots see it as part of
        # their context, regardless of whether the seed slot was empty or
        # partial. This unifies the empty/partial code paths.
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


class D3POValueModule:
    """Multi-head critic for D3PO.

    When ``actor_model`` is provided and exposes ``get_critic_features``,
    the critic shares the actor's GNN encoder: it consumes pooled node
    embeddings (graph + current route) plus global state features. This
    gives the critic a rich, edit-aware state representation instead of a
    handful of graph-level summary scalars. The encoder is used in a
    stop-gradient manner — only the value head is trained here.

    When ``actor_model`` is None, it falls back to the legacy hand-crafted
    21-feature input (kept for backwards compatibility).
    """

    def __init__(self, learning_rate=0.0005, n_objectives=3, decay=0.01,
                 device=None, actor_model=None):
        self.learning_rate = learning_rate
        self.decay = decay
        self.n_objectives = int(n_objectives)
        self._curr_estimate = None
        self.loss_fn = torch.nn.MSELoss()
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.actor_model = actor_model
        self._shared_critic = actor_model is not None and \
            hasattr(actor_model, "get_critic_features")
        # The shared-critic input dim depends on the encoder embed size and
        # the global-feature count, both known only at first forward — so
        # the value head is built lazily. The legacy path has a fixed dim.
        self.model = None
        self.optim = None
        if not self._shared_critic:
            self._build_model(self.legacy_input_dim)

    @property
    def legacy_input_dim(self):
        return 1 + 1 + 2 + 2 + 4 + 11

    def _build_model(self, input_dim):
        self.model = torch.nn.Sequential(
            FeatureNorm(input_dim, 0.001),
            get_mlp(3, input_dim * 2, in_dim=input_dim,
                    out_dim=self.n_objectives, dropout=0.0)
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.decay)

    def inputs_from_data(self, graph_data, cost_weights):
        dev = graph_data[STOP_KEY].x.device
        input_data = torch.zeros(graph_data.num_graphs,
                                 self.legacy_input_dim,
                                 dtype=torch.float, device=dev)
        input_data[:, 0] = graph_data.demand.sum(dim=(1, 2))

        for bi, dl in enumerate(graph_data.to_data_list()):
            dmd_graph = dl[DEMAND_KEY]
            input_data[bi, 1] = dmd_graph.num_edges
            input_data[bi, 2] = dmd_graph.edge_attr[:, 0].mean()
            if dmd_graph.num_edges > 1:
                input_data[bi, 3] = dmd_graph.edge_attr[:, 0].std()
            else:
                input_data[bi, 3] = 0

            dmd_weighted_times = dmd_graph.edge_attr[:, 0] * \
                dmd_graph.edge_attr[:, 1]
            input_data[bi, 4] = dmd_weighted_times.mean()
            if dmd_graph.num_edges > 1:
                input_data[bi, 5] = dmd_weighted_times.std()
            else:
                input_data[bi, 5] = 0
            x_dim = dl[STOP_KEY].x.shape[1]
            input_data[bi, 6:6 + x_dim] = dl[STOP_KEY].x.mean(dim=0)

        for ii, cw in enumerate(cost_weights.values()):
            data_idx = -(1 + ii)
            if torch.is_tensor(cw):
                input_data[:, data_idx] = cw.to(dev)
            else:
                input_data[:, data_idx] = cw

        return input_data

    def _features_from_state(self, state):
        if self._shared_critic:
            return self.actor_model.get_critic_features(state)
        input_data = self.inputs_from_data(
            state.graph_data, state.cost_weights)
        glob_feats = state.get_global_state_features()
        input_data[..., -glob_feats.shape[-1]:] = glob_feats
        return input_data

    def from_state(self, state):
        input_data = self._features_from_state(state)
        if self.model is None:
            self._build_model(input_data.shape[-1])
        baseline = self.model(input_data)
        self._curr_estimate = baseline
        assert baseline.isfinite().all()
        return baseline.detach()

    def update(self, returns):
        """Run a critic update step.

        Returns a dict with per-objective MSE plus detached value/target
        snapshots so the trainer can log critic-quality metrics
        (per-objective MSE, explained variance, value-vs-return scatter)
        without re-running the model.
        """
        self.optim.zero_grad()
        returns = returns.to(self._curr_estimate.dtype)
        # Per-objective MSE computed before backward so we get an unbiased
        # snapshot of the predictor's current error per cost component.
        per_obj_mse = (self._curr_estimate.detach() - returns).pow(2) \
            .mean(dim=0)  # [n_objectives]
        loss = self.loss_fn(self._curr_estimate, returns)
        values_snapshot = self._curr_estimate.detach().clone()
        targets_snapshot = returns.detach().clone()
        loss.backward()
        self.optim.step()
        self._curr_estimate = None
        self.model[0].update()
        return {
            "loss": float(loss.detach().item()),
            "per_obj_mse": per_obj_mse,
            "values": values_snapshot,
            "targets": targets_snapshot,
        }


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


def _is_trim_action(action_kinds):
    return (
        (action_kinds == ROUTE_ACTION_TRIM_START) |
        (action_kinds == ROUTE_ACTION_TRIM_END)
    )


def _expand_mask_like(mask, value):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def _zero_trim_action_rewards(step_rewards, action_kinds, active):
    trim_action = _is_trim_action(action_kinds) & active
    trim_action = _expand_mask_like(trim_action, step_rewards)
    return torch.where(trim_action, torch.zeros_like(step_rewards),
                       step_rewards)


def _update_reward_baseline_cost(prev_cost, new_cost, action_kinds, active,
                                 zero_trim_reward=False):
    update_mask = active
    if zero_trim_reward:
        update_mask = update_mask & ~_is_trim_action(action_kinds)
    update_mask = _expand_mask_like(update_mask, new_cost)
    return torch.where(update_mask, new_cost, prev_cost)


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
            next_nonterminal = _expand_mask_like(
                next_nonterminal, next_values)
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
            next_nonterminal = _expand_mask_like(
                next_nonterminal, next_return)
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


def _state_is_on_device(state, device):
    target_device = torch.device(device)
    state_device = state.device
    if state_device == target_device:
        return True
    return state_device.type == target_device.type and target_device.index is None


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
        "context_node_covered_mask",
        "context_edge_covered_mask",
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
        max_route_edit_steps=None, force_nonhalt_first_step=False,
        edit_step_penalty=0.0, forced_halt_penalty=0.0,
        incumbent_reward=False, return_best_routes=False,
        zero_trim_reward=False, max_trim_actions_per_route=1,
        keep_rollout_on_device=False, adjustment_penalty_fn=None):
    states = []
    rewards = []
    value_estimates = []
    logits = []
    actions = []
    action_kinds = []
    dones = []
    active_masks = []
    score_masks = []
    trim_allowed_masks = []
    action_counts = _new_action_count_totals()
    episode_start_costs = []
    episode_final_costs = []
    # Per-objective component breakdowns at episode start vs end. Mirrors
    # the d3po collector so PPO history can populate ATT/RTT/connectivity
    # delta columns and the cell-16 component plot.
    episode_start_components = []
    episode_final_components = []

    state = None
    prev_cost = None
    current_start_cost = None
    current_start_components = None
    last_cost = None
    last_components = None
    best_cost = None
    best_current_routes = None
    context_route_counts = None
    route_step_idx = 0
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)
    trim_action_counts = None

    with torch.no_grad():
        for _ in range(horizon):
            if state is None or state.is_done().all():
                # Pass the just-finished state so make_next_state can extract
                # the finalized route for the previous slot and feed it back
                # into the working route batch as context for the next slot.
                state, current_start_cost = make_next_state(state)
                prev_cost = current_start_cost.clone()
                last_cost = current_start_cost.clone()
                best_cost = current_start_cost.clone()
                # Snapshot per-objective components at episode start.
                start_cho = cost_obj(state)
                current_start_components = cost_obj.get_cost_components(
                    state, result=start_cho).detach()
                last_components = current_start_components.clone()
                context_route_counts = state.n_finished_routes.detach().clone()
                best_current_routes = _get_current_routes_from_state(state)
                if return_best_routes:
                    state._lc_best_current_routes = \
                        _clone_route_list(best_current_routes)
                    state._lc_best_context_counts = \
                        context_route_counts.detach().clone()
                route_step_idx = 0
                trim_action_counts = torch.zeros(
                    (state.batch_size,), dtype=torch.long,
                    device=state.device)
                action_counts["route_plan_count"] += state.batch_size

            done_before = state.is_done()
            active = ~done_before
            allow_trim = _trim_actions_allowed(
                trim_action_counts, max_trim_actions_per_route)
            target_device = state.device if keep_rollout_on_device \
                else torch.device("cpu")
            state_for_buffer = state.snapshot_for_buffer(device=target_device)
            _clear_state_lazy_tensors(state_for_buffer)
            states.append(state_for_buffer)
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
                        ),
                        allow_trim_start=allow_trim,
                        allow_trim_end=allow_trim)
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
            selected_trim = _is_trim_action(step_kinds) & active
            trim_action_counts = trim_action_counts + selected_trim.long()

            done_after = state.is_done()
            result = cost_obj(state)
            # Per-component breakdown (demand/route/connectivity) is snapshot
            # from the *base* cost before any adjustment-degree shaping so the
            # history's per-component delta columns stay pure. The optional
            # adjustment penalty is then folded into result.cost only, so it
            # shapes the diff reward without polluting the component report.
            result_components = cost_obj.get_cost_components(
                state, result=result).detach()
            if adjustment_penalty_fn is not None:
                _adj_pen = adjustment_penalty_fn(state)
                if _adj_pen is not None:
                    result.cost = result.cost + _adj_pen.detach()
            if incumbent_reward:
                new_best_cost = torch.minimum(best_cost, result.cost)
                step_rewards = (best_cost - new_best_cost) * reward_scale
                improved = active & (result.cost < best_cost)
                if improved.any():
                    candidate_routes = _get_planned_current_routes(
                        state, best_current_routes, context_route_counts)
                    _update_route_list_at_mask(
                        best_current_routes, candidate_routes, improved)
                best_cost = torch.where(active, new_best_cost, best_cost)
            elif diff_reward:
                step_rewards = (prev_cost - result.cost) * reward_scale
            else:
                step_rewards = torch.zeros_like(result.cost)
                just_done = done_after & active
                step_rewards[just_done] = -result.cost[just_done] * \
                    reward_scale
            if return_best_routes and not incumbent_reward:
                improved = active & (result.cost < best_cost)
                if improved.any():
                    candidate_routes = _get_planned_current_routes(
                        state, best_current_routes, context_route_counts)
                    _update_route_list_at_mask(
                        best_current_routes, candidate_routes, improved)
                best_cost = torch.where(
                    improved, result.cost.detach(), best_cost)
            if zero_trim_reward:
                step_rewards = _zero_trim_action_rewards(
                    step_rewards, step_kinds, active)
            if edit_step_penalty > 0:
                edit_action = (step_kinds != ROUTE_ACTION_HALT) & active
                step_rewards = step_rewards - \
                    edit_action.to(step_rewards.dtype) * edit_step_penalty
            if forced_halt_penalty > 0 and force_halt:
                step_rewards = step_rewards - \
                    active.to(step_rewards.dtype) * forced_halt_penalty
            step_rewards = step_rewards * active.to(step_rewards.dtype)

            prev_cost = _update_reward_baseline_cost(
                prev_cost, result.cost, step_kinds, active,
                zero_trim_reward=zero_trim_reward)
            last_cost = torch.where(active, result.cost, last_cost)
            last_components = torch.where(
                active[:, None], result_components, last_components)
            if return_best_routes:
                state._lc_best_current_routes = \
                    _clone_route_list(best_current_routes)
                state._lc_best_context_counts = \
                    context_route_counts.detach().clone()

            just_finished = done_after & active
            if just_finished.any():
                episode_start_costs.append(
                    current_start_cost[just_finished].detach().cpu())
                output_cost = best_cost if return_best_routes else result.cost
                episode_final_costs.append(
                    output_cost[just_finished].detach().cpu())
                episode_start_components.append(
                    current_start_components[just_finished].detach().cpu())
                episode_final_components.append(
                    result_components[just_finished].detach().cpu())

            _merge_step_action_stats(
                action_counts, step_actions, step_kinds, active)

            rewards.append(step_rewards.detach())
            logits.append(step_logits.detach())
            actions.append(step_actions.detach().clone())
            action_kinds.append(step_kinds.detach().clone())
            dones.append(done_after.detach().clone())
            active_masks.append(active.detach().clone())
            score_masks.append(score_mask.detach().clone())
            trim_allowed_masks.append(allow_trim.detach().clone())

        if state is None:
            final_value_estimates = torch.zeros_like(rewards[-1])
        else:
            unfinished = ~state.is_done()
            if unfinished.any():
                episode_start_costs.append(
                    current_start_cost[unfinished].detach().cpu())
                output_cost = best_cost if return_best_routes else last_cost
                episode_final_costs.append(
                    output_cost[unfinished].detach().cpu())
                episode_start_components.append(
                    current_start_components[unfinished].detach().cpu())
                episode_final_components.append(
                    last_components[unfinished].detach().cpu())
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
        "trim_allowed_masks": torch.stack(trim_allowed_masks),
        "final_value_estimates": final_value_estimates.detach(),
        "action_counts": action_counts,
        "zero_trim_reward": bool(zero_trim_reward),
        "max_trim_actions_per_route": max_trim_actions_per_route,
        "keep_rollout_on_device": bool(keep_rollout_on_device),
    }
    if len(episode_start_costs) > 0:
        rollout["episode_start_costs"] = torch.cat(episode_start_costs)
        rollout["episode_final_costs"] = torch.cat(episode_final_costs)
        rollout["episode_start_components"] = torch.cat(
            episode_start_components)
        rollout["episode_final_components"] = torch.cat(
            episode_final_components)
    else:
        rollout["episode_start_costs"] = torch.empty(0)
        rollout["episode_final_costs"] = torch.empty(0)
        rollout["episode_start_components"] = torch.empty(0, 3)
        rollout["episode_final_components"] = torch.empty(0, 3)
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
    trim_allowed_masks = rollout.get("trim_allowed_masks")
    batch_size = rewards.shape[1]
    n_states_per_minibatch = max(1, int(minibatch_size) // int(batch_size))
    supports_route_actions = getattr(model, "supports_trim_actions", False)

    ratio_sum = 0.0
    ratio_count = 0
    clipped_count = 0
    objectives = []
    critic_mse_values = []
    critic_value_buffer = []
    critic_target_buffer = []
    last_critic_values = None
    last_critic_targets = None

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
            if not _state_is_on_device(mb_states, device):
                mb_states = mb_states.to_device(device)

            mb_actions = actions[idxs].flatten(0, 1)
            mb_action_kinds = action_kinds[idxs].flatten(0, 1)
            mb_old_logits = old_logits[idxs].flatten(0, 1)
            mb_returns = returns[idxs].flatten(0, 1)
            mb_advantages = advantages[idxs].flatten(0, 1)
            mb_score = score_masks[idxs].flatten(0, 1)
            mb_trim_allowed = None
            if trim_allowed_masks is not None:
                mb_trim_allowed = trim_allowed_masks[idxs].flatten(0, 1)
            if not mb_score.any():
                continue

            active_idxs = torch.where(mb_score)[0]
            mb_states = mb_states.index_select(active_idxs)
            mb_actions = mb_actions[mb_score]
            mb_action_kinds = mb_action_kinds[mb_score]
            mb_old_logits = mb_old_logits[mb_score]
            mb_returns = mb_returns[mb_score]
            mb_advantages = mb_advantages[mb_score]
            if mb_trim_allowed is not None:
                mb_trim_allowed = mb_trim_allowed[mb_score]

            value_module.from_state(mb_states)
            # old (collection-time) values for optional PPO value-clipping:
            # returns = advantages + value_estimates  ->  V_old = returns - adv.
            critic_step = value_module.update(
                mb_returns, old_values=(mb_returns - mb_advantages))
            if critic_step is not None:
                critic_mse_values.append(float(critic_step["loss"]))
                cv = critic_step["values"].detach().cpu()
                ct = critic_step["targets"].detach().cpu()
                critic_value_buffer.append(cv)
                critic_target_buffer.append(ct)
                last_critic_values = cv
                last_critic_targets = ct

            if supports_route_actions:
                _, _, new_logits, entropy = model.step_route_action(
                    mb_states, actions=mb_actions,
                    action_kinds=mb_action_kinds,
                    allow_trim_start=(
                        mb_trim_allowed if mb_trim_allowed is not None
                        else True
                    ),
                    allow_trim_end=(
                        mb_trim_allowed if mb_trim_allowed is not None
                        else True
                    ))
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

    if critic_value_buffer:
        all_values = torch.cat(critic_value_buffer)
        all_targets = torch.cat(critic_target_buffer)
        critic_mse_mean = float(np.mean(critic_mse_values))
        target_var = all_targets.float().var(unbiased=False).item()
        if target_var > 0:
            residual_var = (all_targets - all_values).float() \
                .var(unbiased=False).item()
            critic_explained_variance = 1.0 - residual_var / target_var
        else:
            critic_explained_variance = float("nan")
    else:
        critic_mse_mean = float("nan")
        critic_explained_variance = float("nan")

    stats = {
        "objective": float(np.mean(objectives)) if objectives else 0.0,
        "ratio_sum": ratio_sum,
        "ratio_count": ratio_count,
        "clipped_count": clipped_count,
        "critic_mse_mean": critic_mse_mean,
        "critic_explained_variance": critic_explained_variance,
        "critic_last_values": last_critic_values,
        "critic_last_targets": last_critic_targets,
    }
    return stats


def _normalize_preference_weights(weights):
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _preference_dict_from_tensor(weights):
    return {
        "demand_time_weight": weights[:, 0],
        "route_time_weight": weights[:, 1],
        "median_connectivity_weight": weights[:, 2],
    }


def _get_state_preferences(cost_obj, state):
    if hasattr(cost_obj, "get_preference_weights"):
        return cost_obj.get_preference_weights(state, normalize=True)
    return _normalize_preference_weights(state.get_cost_weights_tensor())


def _collect_lc_improvement_cfg_d3po_rollout(
        model, cost_obj, make_next_state, value_module, horizon,
        reward_scale, diff_reward, supports_route_actions, device,
        max_route_edit_steps=None, force_nonhalt_first_step=False,
        edit_step_penalty=0.0, forced_halt_penalty=0.0,
        zero_trim_reward=False, max_trim_actions_per_route=1,
        keep_rollout_on_device=False):
    states = []
    rewards = []
    value_estimates = []
    logits = []
    actions = []
    action_kinds = []
    dones = []
    active_masks = []
    score_masks = []
    trim_allowed_masks = []
    preferences = []
    action_counts = _new_action_count_totals()
    episode_start_costs = []
    episode_final_costs = []
    episode_start_components = []
    episode_final_components = []

    state = None
    prev_components = None
    current_start_cost = None
    current_start_components = None
    last_cost = None
    last_components = None
    route_step_idx = 0
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)
    trim_action_counts = None

    with torch.no_grad():
        for _ in range(horizon):
            if state is None or state.is_done().all():
                state, start_result = make_next_state(state)
                current_start_cost = start_result.cost.detach().clone()
                current_start_components = cost_obj.get_cost_components(
                    state, result=start_result).detach()
                prev_components = current_start_components.clone()
                last_cost = current_start_cost.clone()
                last_components = current_start_components.clone()
                route_step_idx = 0
                trim_action_counts = torch.zeros(
                    (state.batch_size,), dtype=torch.long,
                    device=state.device)
                action_counts["route_plan_count"] += state.batch_size

            done_before = state.is_done()
            active = ~done_before
            allow_trim = _trim_actions_allowed(
                trim_action_counts, max_trim_actions_per_route)
            target_device = state.device if keep_rollout_on_device \
                else torch.device("cpu")
            state_for_buffer = state.snapshot_for_buffer(device=target_device)
            _clear_state_lazy_tensors(state_for_buffer)
            states.append(state_for_buffer)
            value_estimates.append(value_module.from_state(state))
            preferences.append(_get_state_preferences(cost_obj, state).detach())

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
                        ),
                        allow_trim_start=allow_trim,
                        allow_trim_end=allow_trim)
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
            selected_trim = _is_trim_action(step_kinds) & active
            trim_action_counts = trim_action_counts + selected_trim.long()

            done_after = state.is_done()
            result = cost_obj(state)
            result_components = cost_obj.get_cost_components(
                state, result=result).detach()
            if diff_reward:
                step_rewards = \
                    (prev_components - result_components) * reward_scale
            else:
                step_rewards = torch.zeros_like(result_components)
                just_done = done_after & active
                step_rewards[just_done] = \
                    -result_components[just_done] * reward_scale
            if zero_trim_reward:
                step_rewards = _zero_trim_action_rewards(
                    step_rewards, step_kinds, active)
            if edit_step_penalty > 0:
                edit_action = (step_kinds != ROUTE_ACTION_HALT) & active
                step_rewards = step_rewards - \
                    edit_action.to(step_rewards.dtype)[:, None] * \
                    edit_step_penalty
            if forced_halt_penalty > 0 and force_halt:
                step_rewards = step_rewards - \
                    active.to(step_rewards.dtype)[:, None] * \
                    forced_halt_penalty
            step_rewards = step_rewards * active.to(step_rewards.dtype)[:, None]

            prev_components = _update_reward_baseline_cost(
                prev_components, result_components, step_kinds, active,
                zero_trim_reward=zero_trim_reward)
            last_cost = torch.where(active, result.cost, last_cost)
            last_components = torch.where(
                active[:, None], result_components, last_components)

            just_finished = done_after & active
            if just_finished.any():
                episode_start_costs.append(
                    current_start_cost[just_finished].detach().cpu())
                episode_final_costs.append(
                    result.cost[just_finished].detach().cpu())
                episode_start_components.append(
                    current_start_components[just_finished].detach().cpu())
                episode_final_components.append(
                    result_components[just_finished].detach().cpu())

            _merge_step_action_stats(
                action_counts, step_actions, step_kinds, active)

            rewards.append(step_rewards.detach())
            logits.append(step_logits.detach())
            actions.append(step_actions.detach().clone())
            action_kinds.append(step_kinds.detach().clone())
            dones.append(done_after.detach().clone())
            active_masks.append(active.detach().clone())
            score_masks.append(score_mask.detach().clone())
            trim_allowed_masks.append(allow_trim.detach().clone())

        if state is None:
            final_value_estimates = torch.zeros_like(rewards[-1])
        else:
            unfinished = ~state.is_done()
            if unfinished.any():
                episode_start_costs.append(
                    current_start_cost[unfinished].detach().cpu())
                episode_final_costs.append(
                    last_cost[unfinished].detach().cpu())
                episode_start_components.append(
                    current_start_components[unfinished].detach().cpu())
                episode_final_components.append(
                    last_components[unfinished].detach().cpu())
            final_value_estimates = value_module.from_state(state)
            final_value_estimates = final_value_estimates * \
                unfinished.to(final_value_estimates.dtype)[:, None]

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
        "trim_allowed_masks": torch.stack(trim_allowed_masks),
        "preferences": torch.stack(preferences),
        "final_value_estimates": final_value_estimates.detach(),
        "action_counts": action_counts,
        "zero_trim_reward": bool(zero_trim_reward),
        "max_trim_actions_per_route": max_trim_actions_per_route,
        "keep_rollout_on_device": bool(keep_rollout_on_device),
        "enabled_component_mask": tuple(
            getattr(cost_obj, "enabled_component_mask", (True, True, True))),
    }
    if len(episode_start_costs) > 0:
        rollout["episode_start_costs"] = torch.cat(episode_start_costs)
        rollout["episode_final_costs"] = torch.cat(episode_final_costs)
        rollout["episode_start_components"] = torch.cat(
            episode_start_components)
        rollout["episode_final_components"] = torch.cat(
            episode_final_components)
    else:
        rollout["episode_start_costs"] = torch.empty(0)
        rollout["episode_final_costs"] = torch.empty(0)
        rollout["episode_start_components"] = torch.empty(0, 3)
        rollout["episode_final_components"] = torch.empty(0, 3)
    return rollout


def _sample_neighbor_preferences(preferences, sigma, enabled_mask=None):
    """Sample a distractor preference vector by perturbing ``preferences``
    with Gaussian noise and re-projecting onto the simplex (paper Alg. 1
    line 21). ``sigma == 0`` yields the original preferences (no
    diversity pressure) — diversity should be disabled via
    ``diversity_weight=0`` instead.

    ``enabled_mask`` (an iterable of three bools) keeps the distractor on
    the same sub-simplex as ``preferences`` when a cost component is
    disabled, so the noise never reintroduces a dropped objective."""
    neighbor = preferences + torch.randn_like(preferences) * sigma
    neighbor = neighbor.clamp_min(1e-6)
    if enabled_mask is not None and not all(enabled_mask):
        mask_row = torch.tensor(
            [1.0 if on else 0.0 for on in enabled_mask],
            device=neighbor.device, dtype=neighbor.dtype)
        neighbor = neighbor * mask_row
    return _normalize_preference_weights(neighbor)


def _estimate_d3po_diversity_loss(
        model, states, actions, action_kinds, trim_allowed, new_logits,
        preferences, supports_route_actions, sigma, alpha,
        enabled_mask=None):
    neighbor_preferences = _sample_neighbor_preferences(
        preferences, sigma, enabled_mask=enabled_mask)
    target = alpha * (preferences - neighbor_preferences).abs().sum(dim=-1)

    # In-place cost_weights swap instead of states.clone() to avoid
    # a full RouteGenBatchState deepcopy on GPU. cost_weights tensors
    # do not require grad, so the prior new_logits autograd graph is
    # unaffected by rebinding their values here.
    saved_weights = {key: value.clone()
                     for key, value in states.cost_weights.items()}
    states.set_cost_weights(
        _preference_dict_from_tensor(neighbor_preferences))
    try:
        if supports_route_actions:
            _, _, neighbor_logits, _ = model.step_route_action(
                states, actions=actions, action_kinds=action_kinds,
                allow_trim_start=(
                    trim_allowed if trim_allowed is not None else True),
                allow_trim_end=(
                    trim_allowed if trim_allowed is not None else True))
        else:
            _, neighbor_logits, _ = model.step(states, actions=actions)
    finally:
        states.set_cost_weights(saved_weights)

    # Schulman's k3 estimator of D_KL(pi(.|omega) || pi(.|omega')) for the
    # sampled action: (r - 1) - log r, with r = pi(a|omega') / pi(a|omega).
    # Unlike the previous abs(log-ratio) heuristic, k3 is unbiased for the
    # KL, always >= 0 (matching the paper's D_KL >= 0), and lower variance.
    log_ratio = (neighbor_logits - new_logits).clamp(-10.0, 10.0)
    sampled_kl = log_ratio.exp() - 1.0 - log_ratio
    return (sampled_kl - target).pow(2).mean()


def _update_lc_improvement_cfg_d3po_from_rollout(
        model, optimizer, value_module, rollout, returns, advantages,
        d3po_epochs, minibatch_size, clip_epsilon, entropy_weight, device,
        diversity_weight=0.0, diversity_alpha=1.0,
        preference_noise_sigma=0.15,
        normalize_advantages_per_objective=True):
    states = rollout["states"]
    rewards = rollout["rewards"]
    old_logits = rollout["logits"]
    actions = rollout["actions"]
    action_kinds = rollout["action_kinds"]
    preferences = rollout["preferences"]
    score_masks = rollout.get("score_masks", rollout["active_masks"])
    trim_allowed_masks = rollout.get("trim_allowed_masks")
    enabled_component_mask = rollout.get(
        "enabled_component_mask", (True, True, True))
    batch_size = rewards.shape[1]
    n_states_per_minibatch = max(1, int(minibatch_size) // int(batch_size))
    supports_route_actions = getattr(model, "supports_trim_actions", False)

    ratio_sum = 0.0
    ratio_count = 0
    clipped_count = 0
    objectives = []
    diversity_losses = []
    critic_per_obj_mse_list = []
    critic_loss_list = []
    critic_value_buffer = []
    critic_target_buffer = []
    last_critic_values = None
    last_critic_targets = None

    return_sums = None
    return_counts = 0
    advantage_sums = None
    advantage_counts = 0

    train_orders = [
        torch.randperm(len(states), device=device)
        for _ in range(int(d3po_epochs))
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
            if not _state_is_on_device(mb_states, device):
                mb_states = mb_states.to_device(device)

            mb_actions = actions[idxs].flatten(0, 1)
            mb_action_kinds = action_kinds[idxs].flatten(0, 1)
            mb_old_logits = old_logits[idxs].flatten(0, 1)
            mb_returns = returns[idxs].flatten(0, 1)
            mb_advantages = advantages[idxs].flatten(0, 1)
            mb_preferences = preferences[idxs].flatten(0, 1)
            mb_score = score_masks[idxs].flatten(0, 1)
            mb_trim_allowed = None
            if trim_allowed_masks is not None:
                mb_trim_allowed = trim_allowed_masks[idxs].flatten(0, 1)
            if not mb_score.any():
                continue

            active_idxs = torch.where(mb_score)[0]
            mb_states = mb_states.index_select(active_idxs)
            mb_actions = mb_actions[mb_score]
            mb_action_kinds = mb_action_kinds[mb_score]
            mb_old_logits = mb_old_logits[mb_score]
            mb_returns = mb_returns[mb_score]
            mb_advantages = mb_advantages[mb_score]
            mb_preferences = mb_preferences[mb_score]
            if mb_trim_allowed is not None:
                mb_trim_allowed = mb_trim_allowed[mb_score]

            # Accumulate per-objective return / advantage stats (pre-normalize).
            r_cpu = mb_returns.detach().cpu()
            a_cpu = mb_advantages.detach().cpu()
            if return_sums is None:
                return_sums = r_cpu.sum(dim=0)
                advantage_sums = a_cpu.sum(dim=0)
            else:
                return_sums = return_sums + r_cpu.sum(dim=0)
                advantage_sums = advantage_sums + a_cpu.sum(dim=0)
            return_counts += r_cpu.shape[0]
            advantage_counts += a_cpu.shape[0]

            value_module.from_state(mb_states)
            critic_step = value_module.update(mb_returns)
            if critic_step is not None:
                critic_loss_list.append(float(critic_step["loss"]))
                critic_per_obj_mse_list.append(
                    critic_step["per_obj_mse"].detach().cpu())
                cv = critic_step["values"].detach().cpu()
                ct = critic_step["targets"].detach().cpu()
                critic_value_buffer.append(cv)
                critic_target_buffer.append(ct)
                last_critic_values = cv
                last_critic_targets = ct

            if supports_route_actions:
                _, _, new_logits, entropy = model.step_route_action(
                    mb_states, actions=mb_actions,
                    action_kinds=mb_action_kinds,
                    allow_trim_start=(
                        mb_trim_allowed if mb_trim_allowed is not None
                        else True
                    ),
                    allow_trim_end=(
                        mb_trim_allowed if mb_trim_allowed is not None
                        else True
                    ))
            else:
                _, new_logits, entropy = model.step(
                    mb_states, actions=mb_actions)

            if normalize_advantages_per_objective:
                if mb_advantages.shape[0] > 1:
                    mb_advantages = \
                        (mb_advantages - mb_advantages.mean(dim=0)) / \
                        (mb_advantages.std(dim=0) + 1e-8)
                else:
                    mb_advantages = mb_advantages - \
                        mb_advantages.mean(dim=0)

            ratios = (new_logits - mb_old_logits).exp()
            clipped_ratios = ratios.clamp(1 - clip_epsilon,
                                          1 + clip_epsilon)
            clip_obj = torch.minimum(
                ratios[:, None] * mb_advantages,
                clipped_ratios[:, None] * mb_advantages)
            weighted_clip_obj = (mb_preferences * clip_obj).sum(dim=-1)
            diversity_loss = torch.zeros((), device=device)
            if diversity_weight > 0:
                diversity_loss = _estimate_d3po_diversity_loss(
                    model, mb_states, mb_actions, mb_action_kinds,
                    mb_trim_allowed, new_logits, mb_preferences,
                    supports_route_actions, preference_noise_sigma,
                    diversity_alpha,
                    enabled_mask=enabled_component_mask)
            objective = weighted_clip_obj.mean() + \
                entropy.mean() * entropy_weight - \
                diversity_weight * diversity_loss

            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()

            objectives.append(float(objective.detach().item()))
            diversity_losses.append(float(diversity_loss.detach().item()))
            ratio_sum += float(ratios.detach().sum().item())
            ratio_count += int(ratios.numel())
            clipped = (ratios.detach() < 1 - clip_epsilon) | \
                (ratios.detach() > 1 + clip_epsilon)
            clipped_count += int(clipped.sum().item())

    # Aggregate per-objective critic stats across minibatches.
    if critic_per_obj_mse_list:
        per_obj_mse_stack = torch.stack(critic_per_obj_mse_list, dim=0)
        critic_mse_per_obj = per_obj_mse_stack.mean(dim=0)  # [n_obj]
        critic_mse_mean = float(critic_mse_per_obj.mean().item())
        all_values = torch.cat(critic_value_buffer, dim=0)
        all_targets = torch.cat(critic_target_buffer, dim=0)
        target_var = all_targets.float().var(dim=0, unbiased=False)
        residual_var = (all_targets - all_values).float() \
            .var(dim=0, unbiased=False)
        # Per-objective explained variance, NaN where target variance is 0.
        explained_var_per_obj = torch.where(
            target_var > 0,
            1.0 - residual_var / target_var.clamp_min(1e-12),
            torch.full_like(target_var, float("nan")),
        )
    else:
        critic_mse_per_obj = None
        critic_mse_mean = float("nan")
        explained_var_per_obj = None

    if return_sums is not None and return_counts > 0:
        return_mean_per_obj = (return_sums / return_counts).tolist()
        advantage_mean_per_obj = (advantage_sums / advantage_counts).tolist()
    else:
        return_mean_per_obj = None
        advantage_mean_per_obj = None

    stats = {
        "objective": float(np.mean(objectives)) if objectives else 0.0,
        "diversity_loss": float(np.mean(diversity_losses))
        if diversity_losses else 0.0,
        "ratio_sum": ratio_sum,
        "ratio_count": ratio_count,
        "clipped_count": clipped_count,
        "critic_mse_mean": critic_mse_mean,
        "critic_mse_per_obj": (
            critic_mse_per_obj.tolist()
            if critic_mse_per_obj is not None else None),
        "critic_explained_variance_per_obj": (
            explained_var_per_obj.tolist()
            if explained_var_per_obj is not None else None),
        "return_mean_per_obj": return_mean_per_obj,
        "advantage_mean_per_obj": advantage_mean_per_obj,
        "critic_last_values": last_critic_values,
        "critic_last_targets": last_critic_targets,
    }
    return stats

@torch.no_grad()
def evaluate_lc_improvement(model, cost_obj, graphs, seed_routes, indices,
                            device, min_route_len, max_route_len,
                            batch_size=8, force_nonhalt_first_step=False,
                            max_route_edit_steps=None,
                            max_trim_actions_per_route=1,
                            return_action_stats=False,
                            target_n_routes=None,
                            return_best_routes=False,
                            adjustment_target=None,
                            adjustment_weight=None,
                            adjustment_use_current=False,
                            adjustment_gap=0.1,
                            adjustment_mode='paper'):
    """Evaluate the improvement model on ``indices``.

    Returned dict includes seed/final scalar cost, win rate, route-change
    rates, and per-objective component breakdowns (demand / route /
    connectivity) used by the notebook for ATT/RTT/connectivity plots.
    """
    model.eval()
    seed_costs = []
    final_costs = []
    seed_components_list = []
    final_components_list = []
    route_change_masks = []
    route_outputs = []
    action_counts = _new_action_count_totals()
    eval_weights = cost_obj.get_weights(device)

    for batch_indices in tqdm(list(indices.split(batch_size)),
                              desc="eval", leave=False):
        graph_batch, route_batch = make_improvement_batch(
            graphs, seed_routes, batch_indices, device, training=False,
            target_n_routes=target_n_routes)

        # Build the seed state explicitly so we can pull its component
        # breakdown — rollout_lc_improvement constructs the same state
        # internally but only returns its cho, not the state object.
        seed_n_routes = route_batch.shape[1]
        seed_state = RouteGenBatchState(
            graph_batch, cost_obj, seed_n_routes,
            min_route_len, max_route_len,
            cost_weights=_clone_cost_weights(eval_weights))
        seed_state.add_new_routes(route_batch)
        if adjustment_target is not None:
            seed_state.set_adjustment_conditioning(
                adjustment_target, adjustment_weight,
                seed_routes=(route_batch if adjustment_use_current else None),
                gap=adjustment_gap, mode=adjustment_mode)
        seed_inline_result = cost_obj(seed_state)
        seed_components_list.append(
            cost_obj.get_cost_components(
                seed_state, result=seed_inline_result).detach().cpu())

        rollout_output = rollout_lc_improvement(
            model, cost_obj, graph_batch, route_batch, min_route_len,
            max_route_len, greedy=True, cost_weights=eval_weights,
            return_actions=return_action_stats,
            force_nonhalt_first_step=force_nonhalt_first_step,
            max_route_edit_steps=max_route_edit_steps,
            max_trim_actions_per_route=max_trim_actions_per_route,
            return_best_routes=return_best_routes,
            adjustment_target=adjustment_target,
            adjustment_weight=adjustment_weight,
            adjustment_use_current=adjustment_use_current,
            adjustment_gap=adjustment_gap,
            adjustment_mode=adjustment_mode)
        state, seed_result, final_result, _, _ = rollout_output[:5]
        if return_action_stats:
            _, _, _, _, _, route_actions, route_action_kinds = rollout_output
            batch_action_stats = summarize_route_action_stats(
                route_actions, route_action_kinds)
            _merge_action_stats(action_counts, batch_action_stats)
        seed_costs.append(seed_result.cost.cpu())
        final_costs.append(final_result.cost.cpu())
        final_components_list.append(
            cost_obj.get_cost_components(
                state, result=final_result).detach().cpu())
        final_routes = get_batch_tensor_from_routes(state.routes).cpu()
        route_outputs.append(final_routes)
        route_change_masks.append(
            _route_change_mask(route_batch.cpu(), final_routes).cpu())

    seed_costs = torch.cat(seed_costs)
    final_costs = torch.cat(final_costs)
    route_change_mask = torch.cat(route_change_masks)
    seed_components = torch.cat(seed_components_list, dim=0)
    final_components = torch.cat(final_components_list, dim=0)
    seed_component_means = seed_components.mean(dim=0)
    final_component_means = final_components.mean(dim=0)
    component_delta_means = seed_component_means - final_component_means

    result = {
        "target_n_routes": int(
            target_n_routes if target_n_routes is not None
            else seed_routes.shape[1]),
        "return_best_routes": bool(return_best_routes),
        "seed_cost": seed_costs.mean().item(),
        "final_cost": final_costs.mean().item(),
        "delta": (seed_costs - final_costs).mean().item(),
        "win_rate": (final_costs < seed_costs).float().mean().item(),
        "changed_route_rate": route_change_mask.float().mean().item(),
        "changed_graph_rate": route_change_mask.any(dim=1).float().mean().item(),
        "routes": route_outputs,
        # Per-objective breakdown: index 0 = demand, 1 = route, 2 = connectivity.
        "seed_component_demand": float(seed_component_means[0]),
        "seed_component_route": float(seed_component_means[1]),
        "seed_component_connectivity": float(seed_component_means[2]),
        "final_component_demand": float(final_component_means[0]),
        "final_component_route": float(final_component_means[1]),
        "final_component_connectivity": float(final_component_means[2]),
        "component_delta_demand": float(component_delta_means[0]),
        "component_delta_route": float(component_delta_means[1]),
        "component_delta_connectivity": float(component_delta_means[2]),
        # Which cost components were active for this run, so notebook
        # tables / plots can drop the disabled ones.
        "enabled_components": list(getattr(
            cost_obj, "enabled_component_names",
            ("demand", "route", "connectivity"))),
    }
    if return_action_stats:
        result["action_stats"] = _finalize_action_stats(action_counts)
    return result



def train_lc_improvement_cfg_ppo(
        model, cost_obj, graphs, seed_routes, device, cfg, output_dir, run_name,
        train_fraction=0.9, batch_size=None, n_iterations=None,
        val_period=None, horizon=None, ppo_epochs=None, minibatch_size=None,
        min_route_len=None, max_route_len=None, warmup_batches=4, seed=0,
        force_nonhalt_first_step=False, max_route_edit_steps=None,
        max_trim_actions_per_route=None, train_indices=None, val_indices=None,
        best_model_path=None,
        max_rollout_samples=8192, target_n_routes=None,
        curriculum_fn=None):
    """Train LC improvement with the construction PPO machinery adapted to edits.

    ``curriculum_fn(iteration) -> (indices, stage_label)`` optionally restricts
    which training graphs are sampled at each iteration (curriculum learning).
    ``indices`` is a subset of train graph indices; ``stage_label`` (str) is
    logged per-iteration as ``curriculum_stage`` so reward shifts can be
    attributed to the active stage. When None, all train_indices are used.

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
    if max_trim_actions_per_route is None:
        max_trim_actions_per_route = _get_cfg_value(
            cfg, "max_trim_actions_per_route", 1)
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)

    reward_scale = float(_get_cfg_value(cfg, "reward_scale", 1.0))
    diff_reward = bool(_get_cfg_value(cfg, "diff_reward", True))
    incumbent_reward = bool(_get_cfg_value(cfg, "incumbent_reward", False))
    return_best_routes = bool(_get_cfg_value(cfg, "return_best_routes", False))
    zero_trim_reward = bool(_get_cfg_value(cfg, "zero_trim_reward", False))
    keep_rollout_on_device = bool(
        _get_cfg_value(cfg, "keep_rollout_on_device", False))
    gamma = float(_get_cfg_value(cfg, "discount_rate", 1.0))
    edit_step_penalty = float(_get_cfg_value(cfg, "edit_step_penalty", 0.0))
    forced_halt_penalty = float(
        _get_cfg_value(cfg, "forced_halt_penalty", 0.0))
    entropy_weight = float(_get_cfg_value(cfg, "entropy_weight", 0.0))
    # Optionally disable force_nonhalt_first_step after a given iteration so the
    # agent is forced to edit early (anti-halt-collapse) then free to halt later.
    force_nonhalt_until = _get_cfg_value(
        cfg, "force_nonhalt_first_step_until_iter", None)
    force_nonhalt_until = None if force_nonhalt_until is None \
        else int(force_nonhalt_until)
    clip_epsilon = float(cfg.ppo.epsilon)
    use_gae = bool(cfg.ppo.use_gae)
    gae_lambda = float(cfg.ppo.gae_lambda)

    # Optional adjustment-degree reward shaping. When weight <= 0 (the default)
    # every code path below is bypassed and training is bit-for-bit unchanged.
    # When enabled, the diff reward also penalizes how far the full current
    # network's per-route adjustment degree (alignment dissimilarity vs the
    # seed routes) is from `adjustment_degree_target`, scaled by the weight.
    adjustment_degree_weight = float(
        _get_cfg_value(cfg, "adjustment_degree_weight", 0.0))
    adjustment_degree_target = float(
        _get_cfg_value(cfg, "adjustment_degree_target", 0.2))
    adjustment_degree_gap = float(
        _get_cfg_value(cfg, "adjustment_degree_gap", 0.1))
    adjustment_degree_mode = str(
        _get_cfg_value(cfg, "adjustment_degree_mode", "current"))
    # Penalty shape: 'target' = |adj - target| (pull toward target);
    # 'cap' = max(0, adj - target) (penalize only EXCEEDING target, no pull
    # below it). 'raw' = adj itself.
    adjustment_degree_objective = str(
        _get_cfg_value(cfg, "adjustment_degree_objective", "target"))
    # Conditioning: sample target/W per-graph each batch and feed them to the
    # agent via state.set_adjustment_conditioning (so it adapts). Requires the
    # model built with n_adjustment_cond_feats=2. target ~ U[t_min,t_max],
    # W ~ log-U[w_min,w_max]. When off, the fixed scalars above are used.
    adjustment_conditioning = bool(
        _get_cfg_value(cfg, "adjustment_conditioning", False))
    # Whether the penalty WEIGHT W is also conditioned (sampled per-graph and
    # fed into the model features). Default False -> only TARGET is conditioned;
    # W stays a fixed scalar (adjustment_degree_weight) applied as the penalty
    # weight and is NOT part of the model features.
    adjustment_condition_weight = bool(
        _get_cfg_value(cfg, "adjustment_condition_weight", False))
    # Whether the live "current adjustment degree vs seed" is fed as a feature
    # (closed-loop). Requires the model built with the extra cond feature.
    adjustment_condition_current = bool(
        _get_cfg_value(cfg, "adjustment_condition_current", False))
    adj_target_min = float(_get_cfg_value(cfg, "adjustment_target_min", 0.1))
    adj_target_max = float(_get_cfg_value(cfg, "adjustment_target_max", 0.4))
    adj_weight_min = float(_get_cfg_value(cfg, "adjustment_weight_min", 0.5))
    adj_weight_max = float(_get_cfg_value(cfg, "adjustment_weight_max", 8.0))
    use_adjustment_penalty = adjustment_conditioning or \
        adjustment_degree_weight > 0
    if use_adjustment_penalty:
        from .bee_colony import get_adjustment_degrees
    # When conditioning, the model is built with conditioning feats, so EVERY
    # forward (warmup feature-norm, validation) must feed conditioning too.
    # Use a fixed representative point: midpoint target, geometric-mean W.
    # If weight is NOT conditioned, pass weight=None at eval (it stays out of
    # the features; the fixed scalar weight is used for the penalty instead).
    if adjustment_conditioning:
        adj_eval_target = 0.5 * (adj_target_min + adj_target_max)
        adj_eval_weight = (math.sqrt(adj_weight_min * adj_weight_max)
                           if adjustment_condition_weight else None)
    else:
        adj_eval_target = None
        adj_eval_weight = None

    optimizer = _make_optimizer_from_cfg(model, cfg)

    # Reuse the original PPO value baseline.  The class relies on the module
    # global DEVICE, so set it before construction.
    from . import inductive_route_learning as il
    il.DEVICE = device
    use_shared_critic = bool(_get_cfg_value(cfg, "shared_critic", True))
    # Optional improved-critic mode (default OFF -> identical to before).
    _vc = _get_cfg_value(cfg, "critic_value_clip", None)
    value_module = il.NNBaseline(
        learning_rate=float(_get_cfg_value(cfg, "baseline_lr", 0.0005)),
        actor_model=model if use_shared_critic else None,
        normalize_returns=bool(
            _get_cfg_value(cfg, "critic_normalize_returns", False)),
        huber=bool(_get_cfg_value(cfg, "critic_huber", False)),
        huber_delta=float(_get_cfg_value(cfg, "critic_huber_delta", 1.0)),
        value_clip=None if _vc is None else float(_vc),
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
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
                return_best_routes=return_best_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current,
                adjustment_gap=adjustment_degree_gap,
                adjustment_mode=adjustment_degree_mode)
    model.update_and_freeze_feature_norms()

    if best_model_path is None:
        best_model_path = output_dir / f"{run_name}.pt"
    else:
        best_model_path = Path(best_model_path)

    best_val_cost = float("inf")
    last_val = None
    history = []
    # Curriculum: active subset of train indices for the current iteration.
    active_train_indices = train_indices
    cur_stage_label = "all"
    epoch_indices = active_train_indices[
        torch.randperm(len(active_train_indices))]
    index_cursor = 0
    route_cursor = 0
    cur_graph_batch = None
    cur_working_routes = None
    cur_cost_weights = None
    cur_seed_routes = None
    prev_route_idx_holder = [None]
    prev_context_counts_holder = [None]
    # Per-batch sampled adjustment target/weight (when conditioning); else None
    # and the fixed cfg scalars are used.
    cur_adj_target = None    # [batch] tensor or None
    cur_adj_weight = None    # [batch] tensor or None
    # Cached per-route penalties (shape [batch, n_routes]) for the current
    # episode. Only the slot being edited changes within an episode, so we
    # compute the full vector once per episode and refresh only the edited route
    # per step (avoids an n_routes-wide Needleman-Wunsch every step).
    episode_route_pen_holder = [None]

    def _pen_per_route(adj):
        """Per-route penalty [.,n_routes] from adjustment degrees, using the
        sampled per-graph target when conditioning, else the fixed scalar."""
        if adjustment_conditioning and cur_adj_target is not None:
            tgt = cur_adj_target.reshape(-1, 1)
        else:
            tgt = adjustment_degree_target
        if adjustment_degree_objective == "cap":
            return (adj - tgt).clamp(min=0.0)
        if adjustment_degree_objective == "cap_sq":
            # quadratic one-sided: gradient grows with overshoot (target-aware)
            return (adj - tgt).clamp(min=0.0) ** 2
        if adjustment_degree_objective == "target":
            return (adj - tgt).abs()
        return adj  # 'raw'

    def _net_weight():
        if adjustment_conditioning and cur_adj_weight is not None:
            return cur_adj_weight.reshape(-1)
        return adjustment_degree_weight

    def _init_episode_adjustment():
        """Compute the full per-route penalty vector once at episode start."""
        if not use_adjustment_penalty or cur_seed_routes is None \
                or cur_working_routes is None:
            return
        adj = get_adjustment_degrees(
            cur_working_routes, cur_seed_routes, cost_obj.symmetric_routes,
            gap=adjustment_degree_gap, mode=adjustment_degree_mode)
        episode_route_pen_holder[0] = _pen_per_route(adj)

    def adjustment_penalty_fn(state):
        """Per-network adjustment-degree penalty (vs seed routes) for shaping.

        Returns ``[batch]`` = weight * mean_route penalty for the full current
        network (frozen context slots + the live in-progress route), or None
        when disabled / not yet initialized. Reuses the episode-cached per-route
        penalties and recomputes only the edited route. With conditioning the
        per-graph sampled target/weight (also fed to the agent) are used.
        """
        if not use_adjustment_penalty:
            return None
        route_idx = prev_route_idx_holder[0]
        base_pen = episode_route_pen_holder[0]
        if route_idx is None or cur_seed_routes is None \
                or cur_working_routes is None or base_pen is None:
            return None
        cur_list = _get_current_routes_from_state(state)
        cur_tensor = get_batch_tensor_from_routes(
            [[cur_list[b]] for b in range(len(cur_list))],
            cur_working_routes.device,
            max_route_len=cur_working_routes.shape[-1])
        adj_cur = get_adjustment_degrees(
            cur_tensor,
            cur_seed_routes[:, route_idx:route_idx + 1, :],
            cost_obj.symmetric_routes,
            gap=adjustment_degree_gap, mode=adjustment_degree_mode)
        pen_cur = _pen_per_route(adj_cur)
        pens = base_pen.clone()
        pens[:, route_idx] = pen_cur[:, 0]
        return _net_weight() * pens.mean(dim=1)

    def make_next_state(prev_state=None):
        nonlocal epoch_indices, index_cursor, route_cursor
        nonlocal cur_graph_batch, cur_working_routes, cur_cost_weights
        nonlocal cur_seed_routes, cur_adj_target, cur_adj_weight

        # 1. Close out the previous slot: pull its finalized route out of
        # prev_state's _finished_routes and write it into working_routes so
        # the next slot sees it as part of the network context.
        prev_route_idx = prev_route_idx_holder[0]
        prev_context_counts = prev_context_counts_holder[0]
        if prev_state is not None and prev_route_idx is not None and \
                cur_working_routes is not None:
            if return_best_routes and \
                    hasattr(prev_state, "_lc_best_current_routes"):
                finalized = _clone_route_list(
                    prev_state._lc_best_current_routes)
            else:
                finalized = _get_planned_current_routes(
                    prev_state,
                    cur_working_routes[:, prev_route_idx],
                    prev_context_counts)
            planned_tensor = get_batch_tensor_from_routes(
                [[finalized[batch_idx]]
                 for batch_idx in range(len(finalized))],
                cur_working_routes.device,
                max_route_len=cur_working_routes.shape[-1])
            cur_working_routes[:, prev_route_idx] = planned_tensor[:, 0]

        # 2. If we are starting fresh (first call this collect) or we have
        # cycled through every slot of the current graph batch, fetch a new
        # batch of graphs and reset working_routes from their seeds.
        starting_fresh = prev_state is None or cur_working_routes is None
        cycled_through = route_cursor == 0
        if starting_fresh or cycled_through:
            if index_cursor + effective_batch_size > len(epoch_indices):
                epoch_indices = active_train_indices[
                    torch.randperm(len(active_train_indices))]
                index_cursor = 0

            batch_indices = epoch_indices[
                index_cursor:index_cursor + effective_batch_size]
            index_cursor += effective_batch_size

            cur_graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            cur_cost_weights = cost_obj.sample_variable_weights(
                cur_graph_batch.num_graphs, device)

            context_route_len = int(route_batch.shape[-1])
            if max_route_len is not None:
                if torch.is_tensor(max_route_len):
                    context_route_len = max(
                        context_route_len,
                        int(max_route_len.max().item()))
                else:
                    context_route_len = max(
                        context_route_len, int(max_route_len))
            cur_working_routes = torch.full(
                (route_batch.shape[0], route_batch.shape[1],
                 context_route_len),
                -1, dtype=route_batch.dtype, device=route_batch.device)
            cur_working_routes[..., :route_batch.shape[-1]] = route_batch
            # Freeze the seed network as the adjustment-degree reference for
            # this batch (only used when adjustment shaping is enabled).
            if use_adjustment_penalty:
                cur_seed_routes = cur_working_routes.clone()
            # Sample per-graph adjustment target for this batch when
            # conditioning: target ~ U[t_min,t_max]. The weight W is sampled
            # (and fed to the model) ONLY when adjustment_condition_weight;
            # otherwise W stays a fixed scalar (cur_adj_weight=None -> the
            # penalty uses adjustment_degree_weight and W is not a feature).
            if adjustment_conditioning:
                nbatch = cur_graph_batch.num_graphs
                cur_adj_target = (adj_target_min + (adj_target_max - adj_target_min)
                                  * torch.rand(nbatch, device=device))
                if adjustment_condition_weight:
                    log_lo, log_hi = math.log(adj_weight_min), math.log(adj_weight_max)
                    cur_adj_weight = torch.exp(
                        log_lo + (log_hi - log_lo)
                        * torch.rand(nbatch, device=device))
                else:
                    cur_adj_weight = None

            route_cursor = 0

        # 3. Build state for the current slot from the (possibly updated)
        # working_routes. Same code path for empty and partial slots —
        # min_route_len in the action mask handles the trim/halt difference.
        route_idx = route_cursor
        route_cursor = (route_cursor + 1) % n_routes

        context_routes = cur_working_routes.clone()
        context_routes[:, route_idx] = -1
        invalid_directly_connected = not bool(
            (context_routes >= 0).any().item())

        state = _make_route_context_state(
            cost_obj, cur_graph_batch, cur_working_routes, route_idx,
            min_route_len, max_route_len, cur_cost_weights,
            invalid_directly_connected=invalid_directly_connected)
        # Feed sampled per-graph adjustment target/weight to the agent (gated
        # conditioning): appended to get_global_state_features.
        if adjustment_conditioning and cur_adj_target is not None:
            state.set_adjustment_conditioning(
                cur_adj_target, cur_adj_weight,
                seed_routes=(cur_seed_routes
                             if adjustment_condition_current else None),
                gap=adjustment_degree_gap, mode=adjustment_degree_mode)
        state = model.setup_planning(state)

        prev_route_idx_holder[0] = route_idx
        prev_context_counts_holder[0] = state.n_finished_routes.detach().clone()
        # Refresh the per-route penalty cache for the new episode (context
        # slots may have been finalized since the last episode).
        _init_episode_adjustment()

        start_result = cost_obj(state)
        start_cost = start_result.cost.detach()
        # Keep the reward baseline consistent: the per-step result.cost in the
        # collector also has this penalty added, so prev_cost must include it.
        _adj_pen = adjustment_penalty_fn(state)
        if _adj_pen is not None:
            start_cost = start_cost + _adj_pen.detach()
        return state, start_cost

    pbar = tqdm(range(int(n_iterations)), desc="cfg ppo improvement")
    for iteration in pbar:
        model.train()
        # Per-iteration force-nonhalt: optionally only for the first
        # `force_nonhalt_first_step_until_iter` iterations (anti-halt-collapse).
        _fnh = force_nonhalt_first_step and (
            force_nonhalt_until is None or iteration < force_nonhalt_until)
        # Curriculum: switch the active training subset per schedule when the
        # stage changes (reshuffle from the new subset).
        if curriculum_fn is not None:
            _stage_idx, _stage_label = curriculum_fn(iteration)
            if _stage_label != cur_stage_label:
                active_train_indices = torch.as_tensor(
                    _stage_idx, dtype=torch.long)
                epoch_indices = active_train_indices[
                    torch.randperm(len(active_train_indices))]
                index_cursor = 0
                cur_stage_label = _stage_label
        rollout = _collect_lc_improvement_cfg_ppo_rollout(
            model, cost_obj, make_next_state, value_module, int(horizon),
            reward_scale, diff_reward,
            getattr(model, "supports_trim_actions", False), device,
            max_route_edit_steps=max_route_edit_steps,
            force_nonhalt_first_step=_fnh,
            edit_step_penalty=edit_step_penalty,
            forced_halt_penalty=forced_halt_penalty,
            incumbent_reward=incumbent_reward,
            return_best_routes=return_best_routes,
            zero_trim_reward=zero_trim_reward,
            max_trim_actions_per_route=max_trim_actions_per_route,
            keep_rollout_on_device=keep_rollout_on_device,
            adjustment_penalty_fn=adjustment_penalty_fn)
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
        start_components = rollout["episode_start_components"]
        final_components = rollout["episode_final_components"]
        if len(start_costs) > 0:
            train_seed = start_costs.mean().item()
            train_final = final_costs.mean().item()
            train_delta = (start_costs - final_costs).mean().item()
            component_delta = (start_components - final_components).mean(
                dim=0)
        else:
            train_seed = float("nan")
            train_final = float("nan")
            train_delta = float("nan")
            component_delta = torch.full((3,), float("nan"))

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
                max_trim_actions_per_route=max_trim_actions_per_route,
                target_n_routes=target_n_routes,
                return_best_routes=return_best_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current,
                adjustment_gap=adjustment_degree_gap,
                adjustment_mode=adjustment_degree_mode)
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
            "component_delta_demand": float("nan"),
            "component_delta_route": float("nan"),
            "component_delta_connectivity": float("nan"),
        }
        row = {
            "iteration": iteration + 1,
            "epoch": iteration + 1,
            "algorithm": "cfg_ppo_improvement",
            "target_n_routes": n_routes,
            "diff_reward": diff_reward,
            "incumbent_reward": incumbent_reward,
            "return_best_routes": return_best_routes,
            "zero_trim_reward": zero_trim_reward,
            "keep_rollout_on_device": keep_rollout_on_device,
            "reward_scale": reward_scale,
            "discount_rate": gamma,
            "edit_step_penalty": edit_step_penalty,
            "forced_halt_penalty": forced_halt_penalty,
            "max_trim_actions_per_route": max_trim_actions_per_route,
            "ppo_horizon": int(horizon),
            "ppo_epochs": int(ppo_epochs),
            "ppo_minibatch_size": int(minibatch_size),
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_delta,
            "train_component_demand_delta": float(component_delta[0]),
            "train_component_route_delta": float(component_delta[1]),
            "train_component_connectivity_delta": float(component_delta[2]),
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
            "train_critic_mse_mean": ppo_stats.get(
                "critic_mse_mean", float("nan")),
            "train_critic_explained_variance": ppo_stats.get(
                "critic_explained_variance", float("nan")),
            "curriculum_stage": cur_stage_label,
            "is_eval_iteration": eval_due,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
            "val_changed_route_rate": val["changed_route_rate"],
            "val_changed_graph_rate": val["changed_graph_rate"],
            "val_component_delta_demand": val.get(
                "component_delta_demand", float("nan")),
            "val_component_delta_route": val.get(
                "component_delta_route", float("nan")),
            "val_component_delta_connectivity": val.get(
                "component_delta_connectivity", float("nan")),
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

    # Capture last-iteration critic snapshot (values vs returns) for the
    # value-vs-target scatter in the notebook critic-analysis cell.
    critic_snapshot = None
    if "ppo_stats" in dir() and \
            ppo_stats.get("critic_last_values") is not None:
        critic_snapshot = {
            "values": ppo_stats["critic_last_values"].numpy(),
            "targets": ppo_stats["critic_last_targets"].numpy(),
        }

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "critic_snapshot": critic_snapshot,
        "enabled_components": list(getattr(
            cost_obj, "enabled_component_names",
            ("demand", "route", "connectivity"))),
    }


def train_lc_improvement_cfg_d3po(
        model, cost_obj, graphs, seed_routes, device, cfg, output_dir, run_name,
        train_fraction=0.9, batch_size=None, n_iterations=None,
        val_period=None, horizon=None, ppo_epochs=None, minibatch_size=None,
        min_route_len=None, max_route_len=None, warmup_batches=4, seed=0,
        force_nonhalt_first_step=False, max_route_edit_steps=None,
        max_trim_actions_per_route=None, train_indices=None, val_indices=None,
        best_model_path=None,
        max_rollout_samples=8192, target_n_routes=None):
    """Train LC improvement with D3PO as a PPO alternative.

    Reads shared PPO-style hyperparameters from ``cfg.ppo`` (n_iterations,
    val_period, n_epochs, minibatch_size, horizon, epsilon, use_gae,
    gae_lambda). D3PO-only knobs (n_objectives, diversity_*,
    preference_noise_sigma, normalize_advantages_per_objective) live in
    ``cfg.d3po``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = cfg.ppo
    d3po_cfg = cfg.d3po
    n_objectives = int(_get_cfg_value(d3po_cfg, "n_objectives", 3))
    if n_objectives != 3:
        raise ValueError(
            "LC D3PO currently expects exactly three cost objectives: "
            "demand, route time, and connectivity."
        )

    # D3PO path does not support adjustment conditioning; keep eval feeds off.
    adj_eval_target = None
    adj_eval_weight = None
    adjustment_condition_current = False

    if min_route_len is None:
        min_route_len = int(cfg.eval.min_route_len)
    if max_route_len is None:
        max_route_len = int(cfg.eval.max_route_len)
    if batch_size is None:
        batch_size = int(_get_cfg_value(cfg, "batch_size", 8))
    if n_iterations is None:
        n_iterations = int(_get_cfg_value(ppo_cfg, "n_iterations"))
    if val_period is None:
        val_period = int(_get_cfg_value(ppo_cfg, "val_period"))
    if horizon is None:
        horizon = int(_get_cfg_value(ppo_cfg, "horizon"))
    if ppo_epochs is None:
        d3po_epochs = int(_get_cfg_value(ppo_cfg, "n_epochs"))
    else:
        d3po_epochs = int(ppo_epochs)
    if minibatch_size is None:
        minibatch_size = int(_get_cfg_value(ppo_cfg, "minibatch_size"))
    if max_route_edit_steps is None:
        max_route_edit_steps = _get_default_max_route_edit_steps(max_route_len)
    if max_trim_actions_per_route is None:
        max_trim_actions_per_route = _get_cfg_value(
            cfg, "max_trim_actions_per_route", 1)
    max_trim_actions_per_route = _normalize_max_trim_actions_per_route(
        max_trim_actions_per_route)

    reward_scale = float(_get_cfg_value(cfg, "reward_scale", 1.0))
    diff_reward = bool(_get_cfg_value(cfg, "diff_reward", True))
    incumbent_reward = bool(_get_cfg_value(cfg, "incumbent_reward", False))
    return_best_routes = bool(_get_cfg_value(cfg, "return_best_routes", False))
    if incumbent_reward:
        raise ValueError(
            "D3PO v1 for LC improvement requires incumbent_reward=false. "
            "Use trainer=ppo for incumbent rewards."
        )
    if return_best_routes:
        raise ValueError(
            "D3PO v1 for LC improvement requires return_best_routes=false. "
            "Use trainer=ppo for best-route output rewards."
        )
    zero_trim_reward = bool(_get_cfg_value(cfg, "zero_trim_reward", False))
    keep_rollout_on_device = bool(
        _get_cfg_value(cfg, "keep_rollout_on_device", False))
    gamma = float(_get_cfg_value(cfg, "discount_rate", 1.0))
    edit_step_penalty = float(_get_cfg_value(cfg, "edit_step_penalty", 0.0))
    forced_halt_penalty = float(
        _get_cfg_value(cfg, "forced_halt_penalty", 0.0))
    entropy_weight = float(_get_cfg_value(cfg, "entropy_weight", 0.0))
    clip_epsilon = float(_get_cfg_value(ppo_cfg, "epsilon"))
    use_gae = bool(_get_cfg_value(ppo_cfg, "use_gae", True))
    gae_lambda = float(_get_cfg_value(ppo_cfg, "gae_lambda", 1.0))
    diversity_weight = float(
        _get_cfg_value(d3po_cfg, "diversity_weight", 0.0))
    diversity_alpha = float(
        _get_cfg_value(d3po_cfg, "diversity_alpha", 1.0))
    preference_noise_sigma = float(
        _get_cfg_value(d3po_cfg, "preference_noise_sigma", 0.15))
    normalize_advantages_per_objective = bool(_get_cfg_value(
        d3po_cfg, "normalize_advantages_per_objective", True))

    optimizer = _make_optimizer_from_cfg(model, cfg)
    use_shared_critic = bool(_get_cfg_value(cfg, "shared_critic", True))
    value_module = D3POValueModule(
        learning_rate=float(_get_cfg_value(cfg, "baseline_lr", 0.0005)),
        n_objectives=n_objectives,
        device=device,
        actor_model=model if use_shared_critic else None,
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
            "LC improvement D3PO keeps full RouteGenBatchState snapshots. "
            "Set BATCH_SIZE <= cfg.d3po.minibatch_size. "
            f"Got BATCH_SIZE={effective_batch_size}, "
            f"cfg.d3po.minibatch_size={minibatch_size}."
        )
    if rollout_samples > int(max_rollout_samples):
        raise ValueError(
            "LC improvement D3PO rollout is too large for the notebook "
            "default memory budget. Lower BATCH_SIZE or D3PO_HORIZON. "
            f"Got BATCH_SIZE * D3PO_HORIZON = {effective_batch_size} * "
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
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
                return_best_routes=return_best_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current)
    model.update_and_freeze_feature_norms()

    if best_model_path is None:
        best_model_path = output_dir / f"{run_name}.pt"
    else:
        best_model_path = Path(best_model_path)

    best_val_cost = float("inf")
    last_val = None
    history = []
    # Curriculum: active subset of train indices for the current iteration.
    active_train_indices = train_indices
    cur_stage_label = "all"
    epoch_indices = active_train_indices[
        torch.randperm(len(active_train_indices))]
    index_cursor = 0
    route_cursor = 0
    cur_graph_batch = None
    cur_working_routes = None
    cur_cost_weights = None
    prev_route_idx_holder = [None]
    prev_context_counts_holder = [None]

    def make_next_state(prev_state=None):
        nonlocal epoch_indices, index_cursor, route_cursor
        nonlocal cur_graph_batch, cur_working_routes, cur_cost_weights

        prev_route_idx = prev_route_idx_holder[0]
        prev_context_counts = prev_context_counts_holder[0]
        if prev_state is not None and prev_route_idx is not None and \
                cur_working_routes is not None:
            finalized = _get_planned_current_routes(
                prev_state,
                cur_working_routes[:, prev_route_idx],
                prev_context_counts)
            planned_tensor = get_batch_tensor_from_routes(
                [[finalized[batch_idx]]
                 for batch_idx in range(len(finalized))],
                cur_working_routes.device,
                max_route_len=cur_working_routes.shape[-1])
            cur_working_routes[:, prev_route_idx] = planned_tensor[:, 0]

        starting_fresh = prev_state is None or cur_working_routes is None
        cycled_through = route_cursor == 0
        if starting_fresh or cycled_through:
            if index_cursor + effective_batch_size > len(epoch_indices):
                epoch_indices = active_train_indices[
                    torch.randperm(len(active_train_indices))]
                index_cursor = 0

            batch_indices = epoch_indices[
                index_cursor:index_cursor + effective_batch_size]
            index_cursor += effective_batch_size

            cur_graph_batch, route_batch = make_improvement_batch(
                graphs, seed_routes, batch_indices, device, training=True,
                target_n_routes=target_n_routes)
            cur_cost_weights = cost_obj.sample_variable_weights(
                cur_graph_batch.num_graphs, device)

            context_route_len = int(route_batch.shape[-1])
            if max_route_len is not None:
                if torch.is_tensor(max_route_len):
                    context_route_len = max(
                        context_route_len,
                        int(max_route_len.max().item()))
                else:
                    context_route_len = max(
                        context_route_len, int(max_route_len))
            cur_working_routes = torch.full(
                (route_batch.shape[0], route_batch.shape[1],
                 context_route_len),
                -1, dtype=route_batch.dtype, device=route_batch.device)
            cur_working_routes[..., :route_batch.shape[-1]] = route_batch

            route_cursor = 0

        route_idx = route_cursor
        route_cursor = (route_cursor + 1) % n_routes

        context_routes = cur_working_routes.clone()
        context_routes[:, route_idx] = -1
        invalid_directly_connected = not bool(
            (context_routes >= 0).any().item())

        state = _make_route_context_state(
            cost_obj, cur_graph_batch, cur_working_routes, route_idx,
            min_route_len, max_route_len, cur_cost_weights,
            invalid_directly_connected=invalid_directly_connected)
        state = model.setup_planning(state)

        prev_route_idx_holder[0] = route_idx
        prev_context_counts_holder[0] = state.n_finished_routes.detach().clone()

        start_result = cost_obj(state)
        return state, start_result

    pbar = tqdm(range(int(n_iterations)), desc="cfg d3po improvement")
    for iteration in pbar:
        model.train()
        rollout = _collect_lc_improvement_cfg_d3po_rollout(
            model, cost_obj, make_next_state, value_module, int(horizon),
            reward_scale, diff_reward,
            getattr(model, "supports_trim_actions", False), device,
            max_route_edit_steps=max_route_edit_steps,
            force_nonhalt_first_step=force_nonhalt_first_step,
            edit_step_penalty=edit_step_penalty,
            forced_halt_penalty=forced_halt_penalty,
            zero_trim_reward=zero_trim_reward,
            max_trim_actions_per_route=max_trim_actions_per_route,
            keep_rollout_on_device=keep_rollout_on_device)
        returns, advantages = _compute_ppo_returns_and_advantages(
            rollout["rewards"], rollout["value_estimates"],
            rollout["dones"], rollout["final_value_estimates"], gamma,
            use_gae, gae_lambda)
        d3po_stats = _update_lc_improvement_cfg_d3po_from_rollout(
            model, optimizer, value_module, rollout, returns, advantages,
            d3po_epochs, minibatch_size, clip_epsilon, entropy_weight, device,
            diversity_weight=diversity_weight,
            diversity_alpha=diversity_alpha,
            preference_noise_sigma=preference_noise_sigma,
            normalize_advantages_per_objective=(
                normalize_advantages_per_objective))

        train_action_stats = _finalize_action_stats(
            rollout["action_counts"])
        active_rewards = rollout["rewards"][rollout["active_masks"]]
        active_preferences = rollout["preferences"][rollout["active_masks"]]
        active_returns = returns[rollout["active_masks"]]
        active_advantages = advantages[rollout["active_masks"]]
        scalar_rewards = (active_rewards * active_preferences).sum(dim=-1) \
            if active_rewards.numel() > 0 else torch.empty(0)
        scalar_returns = (active_returns * active_preferences).sum(dim=-1) \
            if active_returns.numel() > 0 else torch.empty(0)
        scalar_advantages = \
            (active_advantages * active_preferences).sum(dim=-1) \
            if active_advantages.numel() > 0 else torch.empty(0)
        start_costs = rollout["episode_start_costs"]
        final_costs = rollout["episode_final_costs"]
        start_components = rollout["episode_start_components"]
        final_components = rollout["episode_final_components"]
        if len(start_costs) > 0:
            train_seed = start_costs.mean().item()
            train_final = final_costs.mean().item()
            train_delta = (start_costs - final_costs).mean().item()
            component_delta = (start_components - final_components).mean(
                dim=0)
        else:
            train_seed = float("nan")
            train_final = float("nan")
            train_delta = float("nan")
            component_delta = torch.full((n_objectives,), float("nan"))

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
                max_trim_actions_per_route=max_trim_actions_per_route,
                target_n_routes=target_n_routes,
                return_best_routes=return_best_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current)
            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                torch.save(model.state_dict(), best_model_path)

        ratio_mean = d3po_stats["ratio_sum"] / \
            max(d3po_stats["ratio_count"], 1)
        clip_fraction = d3po_stats["clipped_count"] / \
            max(d3po_stats["ratio_count"], 1)
        val = last_val or {
            "seed_cost": float("nan"),
            "final_cost": float("nan"),
            "delta": float("nan"),
            "win_rate": float("nan"),
            "changed_route_rate": float("nan"),
            "changed_graph_rate": float("nan"),
            "component_delta_demand": float("nan"),
            "component_delta_route": float("nan"),
            "component_delta_connectivity": float("nan"),
        }
        critic_mse_per_obj = d3po_stats.get("critic_mse_per_obj") or [
            float("nan"), float("nan"), float("nan")]
        critic_ev_per_obj = d3po_stats.get(
            "critic_explained_variance_per_obj") or [
            float("nan"), float("nan"), float("nan")]
        return_per_obj = d3po_stats.get("return_mean_per_obj") or [
            float("nan"), float("nan"), float("nan")]
        adv_per_obj = d3po_stats.get("advantage_mean_per_obj") or [
            float("nan"), float("nan"), float("nan")]
        row = {
            "iteration": iteration + 1,
            "epoch": iteration + 1,
            "algorithm": "cfg_d3po_improvement",
            "target_n_routes": n_routes,
            "diff_reward": diff_reward,
            "incumbent_reward": incumbent_reward,
            "return_best_routes": return_best_routes,
            "zero_trim_reward": zero_trim_reward,
            "keep_rollout_on_device": keep_rollout_on_device,
            "reward_scale": reward_scale,
            "discount_rate": gamma,
            "edit_step_penalty": edit_step_penalty,
            "forced_halt_penalty": forced_halt_penalty,
            "max_trim_actions_per_route": max_trim_actions_per_route,
            "d3po_horizon": int(horizon),
            "d3po_epochs": int(d3po_epochs),
            "d3po_minibatch_size": int(minibatch_size),
            "d3po_diversity_weight": diversity_weight,
            "d3po_diversity_alpha": diversity_alpha,
            "d3po_preference_noise_sigma": preference_noise_sigma,
            "train_seed_cost": train_seed,
            "train_final_cost": train_final,
            "train_delta": train_delta,
            "train_component_demand_delta": float(component_delta[0]),
            "train_component_route_delta": float(component_delta[1]),
            "train_component_connectivity_delta": float(component_delta[2]),
            "train_reward_mean": scalar_rewards.mean().item()
                if scalar_rewards.numel() > 0 else 0.0,
            "train_return_mean": scalar_returns.mean().item()
                if scalar_returns.numel() > 0 else 0.0,
            "train_advantage_mean": scalar_advantages.mean().item()
                if scalar_advantages.numel() > 0 else 0.0,
            "train_return_mean_demand": float(return_per_obj[0]),
            "train_return_mean_route": float(return_per_obj[1]),
            "train_return_mean_connectivity": float(return_per_obj[2]),
            "train_advantage_mean_demand": float(adv_per_obj[0]),
            "train_advantage_mean_route": float(adv_per_obj[1]),
            "train_advantage_mean_connectivity": float(adv_per_obj[2]),
            "train_active_sample_count": int(
                rollout["active_masks"].sum().item()),
            "train_d3po_ratio_mean": ratio_mean,
            "train_d3po_clip_fraction": clip_fraction,
            "train_d3po_objective": d3po_stats["objective"],
            "train_d3po_diversity_loss": d3po_stats["diversity_loss"],
            "train_critic_mse_mean": d3po_stats.get(
                "critic_mse_mean", float("nan")),
            "train_critic_mse_demand": float(critic_mse_per_obj[0]),
            "train_critic_mse_route": float(critic_mse_per_obj[1]),
            "train_critic_mse_connectivity": float(critic_mse_per_obj[2]),
            "train_critic_explained_variance_demand": float(
                critic_ev_per_obj[0]),
            "train_critic_explained_variance_route": float(
                critic_ev_per_obj[1]),
            "train_critic_explained_variance_connectivity": float(
                critic_ev_per_obj[2]),
            "is_eval_iteration": eval_due,
            "val_seed_cost": val["seed_cost"],
            "val_final_cost": val["final_cost"],
            "val_delta": val["delta"],
            "val_win_rate": val["win_rate"],
            "val_changed_route_rate": val["changed_route_rate"],
            "val_changed_graph_rate": val["changed_graph_rate"],
            "val_component_delta_demand": val.get(
                "component_delta_demand", float("nan")),
            "val_component_delta_route": val.get(
                "component_delta_route", float("nan")),
            "val_component_delta_connectivity": val.get(
                "component_delta_connectivity", float("nan")),
        }
        row.update({
            f"train_action_{key}": value
            for key, value in train_action_stats.items()
        })
        row["train_changed_route_rate"] = float("nan")
        history.append(row)

        pbar.set_postfix({
            "reward": f"{row['train_reward_mean']:.3f}",
            "delta": f"{row['train_delta']:.3f}",
            "ratio": f"{row['train_d3po_ratio_mean']:.3f}",
            "clip": f"{row['train_d3po_clip_fraction']:.2%}",
        })
        print(
            f"cfg_d3po_iter={row['iteration']:03d} "
            f"reward={row['train_reward_mean']:.4f} "
            f"train_delta={row['train_delta']:.4f} "
            f"val_delta={row['val_delta']:.4f} "
            f"ratio={row['train_d3po_ratio_mean']:.3f} "
            f"clip={row['train_d3po_clip_fraction']:.2%} "
            f"div={row['train_d3po_diversity_loss']:.4f} "
            f"actions="
            f"ext:{row['train_action_extend_count']} "
            f"trim_s:{row['train_action_trim_start_count']} "
            f"trim_e:{row['train_action_trim_end_count']} "
            f"halt:{row['train_action_halt_count']} "
            f"avg_steps={row['train_action_avg_actions_per_route']:.2f} "
            f"eval={row['is_eval_iteration']}"
        )

    # Capture last-iteration critic snapshot (values vs returns per objective)
    # for the value-vs-target scatter in the notebook critic-analysis cell.
    critic_snapshot = None
    if "d3po_stats" in dir() and \
            d3po_stats.get("critic_last_values") is not None:
        critic_snapshot = {
            "values": d3po_stats["critic_last_values"].numpy(),
            "targets": d3po_stats["critic_last_targets"].numpy(),
        }

    return {
        "best_model_path": best_model_path,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "critic_snapshot": critic_snapshot,
        "enabled_components": list(getattr(
            cost_obj, "enabled_component_names",
            ("demand", "route", "connectivity"))),
    }


def train_lc_improvement_cfg(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if cfg is None and len(args) >= 6:
        cfg = args[5]
    trainer = _get_cfg_value(cfg, "trainer", "ppo")
    if trainer == "ppo":
        return train_lc_improvement_cfg_ppo(*args, **kwargs)
    if trainer == "d3po":
        return train_lc_improvement_cfg_d3po(*args, **kwargs)
    raise ValueError(
        f"Unknown LC improvement trainer {trainer!r}; expected 'ppo' or 'd3po'."
    )

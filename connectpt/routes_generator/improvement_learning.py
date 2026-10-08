from .policy.generate import generate
from .env.construction import normalize_nodes
from .policy.rollout import bind_env, rollout as policy_rollout
from .training.replay import evaluate_action
from .run.checkpoints import CheckpointStore
from connectpt.routes_generator.env.masks import ActionMask
from .env.features import global_state_features, critic_inputs, PREFERENCE_KEYS, observe
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .run.logger import flush, log as log_metrics
from .training.records import EditTrainRecord, EditValRecord
from .city.graph import STOP_KEY, DemandScaleTransform, SpaceScaleTransform
from .objectives.preferences import COST_WEIGHT_KEY_ORDER
from .env.actions import ROUTE_ACTION_EXTEND, ROUTE_ACTION_HALT, ROUTE_ACTION_TRIM_END, ROUTE_ACTION_TRIM_START
from .env.state import RouteGenBatchState
from .policy.models import FeatureNorm, get_mlp, DEFAULT_PREFERENCE_GOALS
from .network.routes import to_routes
from .city.graph import to_batch
from .objectives.penalties import edit_step_penalty as step_penalty, forced_halt_penalty as halt_penalty

ROUTE_ACTION_NAMES = {
    ROUTE_ACTION_EXTEND: "extend",
    ROUTE_ACTION_TRIM_START: "trim_start",
    ROUTE_ACTION_TRIM_END: "trim_end",
    ROUTE_ACTION_HALT: "halt",
}
ROUTE_ACTION_STAT_NAMES = tuple(ROUTE_ACTION_NAMES.values())


_OBJECTIVES = ("demand", "route", "connectivity")


def _edit_val_record(val):
    """The last validation result (or NaN before the first one)."""
    nan = float("nan")
    return EditValRecord(
        seed_cost=val["seed_cost"], final_cost=val["final_cost"], delta=val["delta"],
        win_rate=val["win_rate"], changed_route_rate=val["changed_route_rate"],
        changed_graph_rate=val["changed_graph_rate"],
        component_delta_demand=val.get("component_delta_demand", nan),
        component_delta_route=val.get("component_delta_route", nan),
        component_delta_connectivity=val.get("component_delta_connectivity", nan))


def _get_graph_batch_max_n_nodes(graph_batch):
    stop_data = graph_batch[STOP_KEY]
    if hasattr(stop_data, "ptr") and stop_data.ptr is not None:
        node_counts = stop_data.ptr[1:] - stop_data.ptr[:-1]
        return int(node_counts.max().item())
    return int(stop_data.num_nodes)


def _clone_cost_weights(cost_weights):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in cost_weights.items()
    }


def _make_route_context_state(cost_obj, graph_batch, route_batch, route_idx,
                              min_route_len, max_route_len, cost_weights,
                              invalid_directly_connected=False):
    route_batch = to_routes(
        route_batch,
        width=min(route_batch.shape[-1], _get_graph_batch_max_n_nodes(graph_batch)))
    n_routes = route_batch.shape[1]
    state = RouteGenBatchState(
        graph_batch, cost_obj.constants, n_routes, min_route_len, max_route_len,
        preferences=_clone_cost_weights(cost_weights))

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


def _assemble_routes_by_original_order(routes_by_route, device):
    batch_size = len(routes_by_route[0])
    batch_routes = []
    for batch_idx in range(batch_size):
        batch_routes.append([
            route_set[batch_idx] for route_set in routes_by_route
        ])
    return to_routes(batch_routes).to(device)


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
    route_batch = to_routes(
        route_batch,
        width=min(route_batch.shape[-1], _get_graph_batch_max_n_nodes(graph_batch)))

    n_routes = route_batch.shape[1]
    seed_state = RouteGenBatchState(
        graph_batch, cost_obj.constants, n_routes, min_route_len, max_route_len,
        preferences=_clone_cost_weights(cost_weights))
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
        is_edit = model.action_spec.role == "edit"
        env = bind_env(route_state, model, cost_obj, single_route=True,
            max_route_edit_steps=max_route_edit_steps if is_edit else None,
            max_trim_actions=max_trim_actions_per_route if is_edit else None,
            force_nonhalt_first_step=force_nonhalt_first_step if is_edit else False)
        trajectory = generate(env, model, "greedy" if greedy else "sample", return_trajectory=True)
        actions, logits, entropy = trajectory.actions, trajectory.total_logp, trajectory.total_entropy
        planned_current_routes = _get_planned_current_routes(
            route_state, working_route_batch[:, route_idx], context_route_counts)
        route_logits.append(logits)
        route_entropies.append(entropy)
        routes_by_route.append(planned_current_routes)
        # Write the planned route back so subsequent slots see it as part of
        # their context, regardless of whether the seed slot was empty or
        # partial. This unifies the empty/partial code paths.
        planned_tensor = to_routes(
            [[planned_current_routes[batch_idx]]
             for batch_idx in range(len(planned_current_routes))],
            width=working_route_batch.shape[-1]).to(route_batch.device)
        working_route_batch[:, route_idx] = planned_tensor[:, 0]
        if return_actions:
            route_actions.append(actions.detach().clone())
            route_action_kinds.append(trajectory.action_kinds.detach().clone() if is_edit else None)
            if return_step_data:
                route_step_logits.append(trajectory.logp.detach().clone() if is_edit else None)
                route_step_entropies.append(trajectory.entropy.detach().clone() if is_edit else None)

    final_routes = _assemble_routes_by_original_order(
        routes_by_route, graph_batch[STOP_KEY].x.device)
    state = RouteGenBatchState(
        graph_batch, cost_obj.constants, n_routes, min_route_len, max_route_len,
        preferences=_clone_cost_weights(cost_weights))
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

    # Feature specs (checkpoint-locked, declared like on the actor models):
    # order of the cost-weight slots inside the shared global state features,
    # and inside this critic's own legacy input tail (slot -1-ii = key ii).
    preference_goals = DEFAULT_PREFERENCE_GOALS
    weight_input_keys = COST_WEIGHT_KEY_ORDER

    def __init__(self, learning_rate=0.0005, n_objectives=3, decay=0.01,
                 device=None, actor_model=None, preference_goals=None,
                 holliday_features=False):
        self.learning_rate = learning_rate
        self._explicit_weight_features = preference_goals is not None
        if preference_goals is not None:
            self.preference_goals = tuple(preference_goals)
            self.weight_input_keys = tuple(k for k in COST_WEIGHT_KEY_ORDER
                                           if k in tuple(PREFERENCE_KEYS[goal] for goal in self.preference_goals))
        self.holliday_features = holliday_features
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
        return 10 + 9 + len(self.preference_goals) if self._explicit_weight_features else 21

    def _build_model(self, input_dim):
        self.model = torch.nn.Sequential(
            FeatureNorm(input_dim, 0.001, unbiased=self.holliday_features),
            get_mlp(3, input_dim * 2, in_dim=input_dim,
                    out_dim=self.n_objectives, dropout=0.0)
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.decay)

    def inputs_from_data(self, graph_data, cost_weights):
        return critic_inputs(graph_data, cost_weights,
                             input_dim=self.legacy_input_dim,
                             holliday_features=self.holliday_features,
                             weight_input_keys=self.weight_input_keys)

    def _features_from_state(self, state):
        if self._shared_critic:
            return self.actor_model.get_critic_features(observe(state, self.actor_model.feature_set, allow_below_min=self.actor_model.allow_trim_below_min if self.actor_model.action_spec.role == "edit" else False))
        input_data = self.inputs_from_data(
            state.graph_data, state.cost_weights)
        glob_feats = global_state_features(state,
            preference_goals=self.preference_goals)
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


def _update_reward_baseline_cost(prev_cost, new_cost, active):
    return torch.where(_expand_mask_like(active, new_cost), new_cost, prev_cost)


def _get_reward_delta_diagnostics(rewards, active_masks, start_costs,
                                  final_costs, reward_scale):
    """Compare collected step rewards with episode-level cost improvement.

    For plain differential rewards their sums telescope. A non-zero residual
    is still expected when reward shaping intentionally suppresses a trim
    reward, an edit penalty is enabled, or the horizon ends on a deferred trim.
    """
    reward_sum = float(rewards[active_masks].sum().item())
    scaled_delta_sum = float(
        ((start_costs - final_costs).sum() * float(reward_scale)).item())
    episode_count = int(len(start_costs))
    return {
        "reward_sum": reward_sum,
        "scaled_delta_sum": scaled_delta_sum,
        "residual": reward_sum - scaled_delta_sum,
        "reward_per_episode": reward_sum / max(episode_count, 1),
        "episode_count": episode_count,
    }


def _normalize_minibatch_advantages(advantages, *, per_objective=False):
    """Holliday's minibatch mean/sample-std, with a safe singleton fallback."""
    dim = 0 if per_objective else None
    centered = advantages - advantages.mean(dim=dim, keepdim=per_objective)
    count = advantages.shape[0] if per_objective else advantages.numel()
    if count < 2:
        return centered
    return centered / (advantages.std(dim=dim, keepdim=per_objective) + 1e-8)


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
        max_trim_actions_per_route=1, adjustment_penalty_fn=None):
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
                # Snapshot per-objective components at episode start.
                start_cho = cost_obj(state)
                current_start_components = cost_obj.get_cost_components(
                    state, result=start_cho).detach()
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

            force_halt = max_route_edit_steps is not None and route_step_idx >= max_route_edit_steps
            env = bind_env(state, model, cost_obj, normalize=False,
                max_route_edit_steps=0 if force_halt else None,
                mask=ActionMask(halt=not (force_nonhalt_first_step and route_step_idx == 0),
                                trim_start=allow_trim, trim_end=allow_trim))
            trajectory = policy_rollout(env, model, max_steps=1, critic=value_module)
            state_for_buffer = trajectory.states[0]
            _clear_state_lazy_tensors(state_for_buffer)
            states.append(state_for_buffer)
            value_estimates.append(trajectory.values[0])
            out = trajectory.outputs[0]
            step_kinds, step_actions, step_logits = out.action.kind, out.action.indices, out.logp
            score_mask = torch.zeros_like(active) if force_halt else active
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
                    result.add_penalty(_adj_pen.detach())
            if diff_reward:
                step_rewards = (prev_cost - result.cost) * reward_scale
            else:
                step_rewards = torch.zeros_like(result.cost)
                just_done = done_after & active
                step_rewards[just_done] = -result.cost[just_done] * \
                    reward_scale
            if edit_step_penalty > 0:
                step_rewards = step_rewards - step_penalty(step_kinds, active, edit_step_penalty).to(step_rewards.dtype)
            if forced_halt_penalty > 0 and force_halt:
                step_rewards = step_rewards - halt_penalty(active, force_halt, forced_halt_penalty).to(step_rewards.dtype)
            step_rewards = step_rewards * active.to(step_rewards.dtype)

            prev_cost = _update_reward_baseline_cost(prev_cost, result.cost, active)
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
        "max_trim_actions_per_route": max_trim_actions_per_route,
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

            out = evaluate_action(model, mb_states, mb_actions, mb_action_kinds if supports_route_actions else None,
                ActionMask(trim_start=mb_trim_allowed if mb_trim_allowed is not None else True,
                           trim_end=mb_trim_allowed if mb_trim_allowed is not None else True))
            new_logits, entropy = out.logp, out.entropy

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
    # Cost/reward math order: explicit canonical keys (matches
    # _preference_dict_from_tensor's unpacking).
    return _normalize_preference_weights(
        state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER))


def _collect_lc_improvement_cfg_d3po_rollout(
        model, cost_obj, make_next_state, value_module, horizon,
        reward_scale, diff_reward, supports_route_actions, device,
        max_route_edit_steps=None, force_nonhalt_first_step=False,
        edit_step_penalty=0.0, forced_halt_penalty=0.0,
        max_trim_actions_per_route=1):
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
            preferences.append(_get_state_preferences(cost_obj, state).detach())

            force_halt = max_route_edit_steps is not None and route_step_idx >= max_route_edit_steps
            env = bind_env(state, model, cost_obj, normalize=False,
                max_route_edit_steps=0 if force_halt else None,
                mask=ActionMask(halt=not (force_nonhalt_first_step and route_step_idx == 0),
                                trim_start=allow_trim, trim_end=allow_trim))
            trajectory = policy_rollout(env, model, max_steps=1, critic=value_module)
            state_for_buffer = trajectory.states[0]
            _clear_state_lazy_tensors(state_for_buffer)
            states.append(state_for_buffer)
            value_estimates.append(trajectory.values[0])
            out = trajectory.outputs[0]
            step_kinds, step_actions, step_logits = out.action.kind, out.action.indices, out.logp
            score_mask = torch.zeros_like(active) if force_halt else active
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
            if edit_step_penalty > 0:
                step_rewards = step_rewards - step_penalty(step_kinds, active, edit_step_penalty).to(step_rewards.dtype)[:, None]
            if forced_halt_penalty > 0 and force_halt:
                step_rewards = step_rewards - halt_penalty(active, force_halt, forced_halt_penalty).to(step_rewards.dtype)[:, None]
            step_rewards = step_rewards * active.to(step_rewards.dtype)[:, None]

            prev_components = _update_reward_baseline_cost(
                prev_components, result_components, active)
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
        "max_trim_actions_per_route": max_trim_actions_per_route,
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


def _categorical_kl(log_probs, neighbor_log_probs):
    """Exact KL over the common valid action set, without 0 * -inf NaNs."""
    supported = torch.isfinite(log_probs)
    safe_log_probs = torch.where(supported, log_probs, 0.)
    safe_neighbor = torch.where(supported, neighbor_log_probs, 0.)
    return (log_probs.exp() * (safe_log_probs - safe_neighbor)).sum(-1)


def _estimate_d3po_diversity_loss(
        model, states, actions, action_kinds, trim_allowed, new_logits,
        preferences, supports_route_actions, sigma, alpha,
        enabled_mask=None, action_log_probs=None):
    neighbor_preferences = _sample_neighbor_preferences(
        preferences, sigma, enabled_mask=enabled_mask)
    target = alpha * (preferences - neighbor_preferences).abs().sum(dim=-1)

    mask = ActionMask(trim_start=trim_allowed if trim_allowed is not None else True,
                      trim_end=trim_allowed if trim_allowed is not None else True)
    def replay():
        return evaluate_action(model, states, actions,
            action_kinds if supports_route_actions else None, mask).log_probs
    if action_log_probs is None:
        action_log_probs = replay()

    # In-place cost_weights swap instead of states.clone() to avoid
    # a full RouteGenBatchState deepcopy on GPU. cost_weights tensors
    # do not require grad, so the prior new_logits autograd graph is
    # unaffected by rebinding their values here.
    saved_weights = {key: value.clone()
                     for key, value in states.cost_weights.items()}
    states.set_cost_weights(
        _preference_dict_from_tensor(neighbor_preferences))
    try:
        neighbor_log_probs = replay()
    finally:
        states.set_cost_weights(saved_weights)

    # Paper Eq. 2 squares the full state-wise KL, not a sampled estimator.
    # Squaring a one-action estimate adds its variance to the loss and can
    # produce enormous spikes from a single rare replayed action.
    kl = _categorical_kl(action_log_probs, neighbor_log_probs)
    return (kl - target).pow(2).mean()


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
    # Construction can learn only active objective heads while diversity still
    # needs the full named cost-weight vector for neighborhood sampling.
    diversity_preferences = rollout.get("diversity_preferences", preferences)
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
    entropy_values = []
    initial_ratio_error = None
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
            mb_diversity_preferences = diversity_preferences[idxs].flatten(0, 1)
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
            mb_diversity_preferences = mb_diversity_preferences[mb_score]
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

            out = evaluate_action(model, mb_states, mb_actions, mb_action_kinds if supports_route_actions else None,
                ActionMask(trim_start=mb_trim_allowed if mb_trim_allowed is not None else True,
                           trim_end=mb_trim_allowed if mb_trim_allowed is not None else True))
            new_logits, entropy = out.logp, out.entropy
            step_output = (out.log_probs,)

            if normalize_advantages_per_objective:
                mb_advantages = _normalize_minibatch_advantages(
                    mb_advantages, per_objective=True)

            ratios = (new_logits - mb_old_logits).exp()
            if initial_ratio_error is None:
                initial_ratio_error = float((ratios.detach()-1).abs().max())
                if rollout.get('check_initial_replay', False) and initial_ratio_error > 2e-4:
                    raise RuntimeError(f'Initial D3PO replay ratio differs from 1: {initial_ratio_error}')
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
                    mb_trim_allowed, new_logits, mb_diversity_preferences,
                    supports_route_actions, preference_noise_sigma,
                    diversity_alpha,
                    enabled_mask=enabled_component_mask,
                    action_log_probs=step_output[-1])
            objective = weighted_clip_obj.mean() + \
                entropy.mean() * entropy_weight - \
                diversity_weight * diversity_loss

            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()

            objectives.append(float(objective.detach().item()))
            entropy_values.append(float(entropy.detach().mean()))
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
        "entropy": float(np.mean(entropy_values)) if entropy_values else float('nan'),
        "initial_ratio_error": initial_ratio_error,
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
        graph_batch = to_batch([graphs[int(i)] for i in batch_indices], device, edit_features=True)
        route_batch = to_routes(
            seed_routes[torch.as_tensor(batch_indices, dtype=torch.long)],
            n_routes=target_n_routes,
            width=min(seed_routes.shape[-1], _get_graph_batch_max_n_nodes(graph_batch)),
        ).to(device)

        # Build the seed state explicitly so we can pull its component
        # breakdown — rollout_lc_improvement constructs the same state
        # internally but only returns its cho, not the state object.
        seed_n_routes = route_batch.shape[1]
        seed_state = RouteGenBatchState(
            graph_batch, cost_obj.constants, seed_n_routes,
            min_route_len, max_route_len,
            preferences=_clone_cost_weights(eval_weights))
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
        final_routes = to_routes(state.routes).cpu()
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
        curriculum_fn=None, val_curriculum_fn=None, logger_list=()):
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
    normalize_advantages = bool(
        _get_cfg_value(cfg, "normalize_advantages", True))

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
        from .objectives.penalties import adjustment_degree, adjustment_penalty
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
            graph_batch = to_batch(
                [graphs[int(i)] for i in batch_indices],
                device,
                transforms=(SpaceScaleTransform(0.95, 1.05), DemandScaleTransform(0.9, 1.1)),
                edit_features=True,
            )
            route_batch = to_routes(
                seed_routes[torch.as_tensor(batch_indices, dtype=torch.long)],
                n_routes=target_n_routes,
                width=min(seed_routes.shape[-1], _get_graph_batch_max_n_nodes(graph_batch)),
            ).to(device)
            rollout_lc_improvement(
                model, cost_obj, graph_batch, route_batch,
                min_route_len, max_route_len, greedy=False,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
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
    # Cached per-route adjustment DEGREES (shape [batch, n_routes]) for the
    # current episode. Only the slot being edited changes within an episode, so
    # we compute the full vector once per episode and refresh only the edited
    # route per step (avoids an n_routes-wide Needleman-Wunsch every step).
    episode_route_adj_holder = [None]

    def _net_weight():
        if adjustment_conditioning and cur_adj_weight is not None:
            return cur_adj_weight.reshape(-1)
        return adjustment_degree_weight

    def _net_target():
        if adjustment_conditioning and cur_adj_target is not None:
            return cur_adj_target.reshape(-1)
        return adjustment_degree_target

    def _init_episode_adjustment():
        """Compute the full per-route degree vector once at episode start."""
        if not use_adjustment_penalty or cur_seed_routes is None \
                or cur_working_routes is None:
            return
        episode_route_adj_holder[0] = adjustment_degree(
            cur_working_routes, cur_seed_routes, cost_obj.symmetric_routes,
            gap=adjustment_degree_gap, mode=adjustment_degree_mode)

    def adjustment_penalty_fn(state):
        """Network-level adjustment-degree penalty (vs seed routes) for shaping.

        Returns ``[batch]`` = weight * penalty(mean_i adj_i) -- the penalty of
        the NETWORK-mean degree, exactly matching the eval/BCO form in
        MyCostModule._adjustment_penalty. (Penalizing each route's deviation
        separately and averaging the penalties is a different, much more
        restrictive objective: it forbids the concentrated rewrites -- e.g.
        fully replacing one duplicated route -- that the network-budget
        semantics deliberately allow.) Reuses the episode-cached per-route
        degrees and recomputes only the edited route. With conditioning the
        per-graph sampled target/weight (also fed to the agent) are used.
        """
        if not use_adjustment_penalty:
            return None
        route_idx = prev_route_idx_holder[0]
        base_adj = episode_route_adj_holder[0]
        if route_idx is None or cur_seed_routes is None \
                or cur_working_routes is None or base_adj is None:
            return None
        cur_list = _get_current_routes_from_state(state)
        cur_tensor = to_routes(
            [[cur_list[b]] for b in range(len(cur_list))],
            width=cur_working_routes.shape[-1]).to(cur_working_routes.device)
        adj_cur = adjustment_degree(
            cur_tensor,
            cur_seed_routes[:, route_idx:route_idx + 1, :],
            cost_obj.symmetric_routes,
            gap=adjustment_degree_gap, mode=adjustment_degree_mode)
        degrees = base_adj.clone()
        degrees[:, route_idx] = adj_cur[:, 0]
        return _net_weight() * adjustment_penalty(
            degrees.mean(dim=-1), adjustment_degree_objective, _net_target())

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
            finalized = _get_planned_current_routes(
                prev_state,
                cur_working_routes[:, prev_route_idx],
                prev_context_counts)
            planned_tensor = to_routes(
                [[finalized[batch_idx]]
                 for batch_idx in range(len(finalized))],
                width=cur_working_routes.shape[-1]).to(cur_working_routes.device)
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

            cur_graph_batch = to_batch(
                [graphs[int(i)] for i in batch_indices],
                device,
                transforms=(SpaceScaleTransform(0.95, 1.05), DemandScaleTransform(0.9, 1.1)),
                edit_features=True,
            )
            route_batch = to_routes(
                seed_routes[torch.as_tensor(batch_indices, dtype=torch.long)],
                n_routes=target_n_routes,
                width=min(seed_routes.shape[-1], _get_graph_batch_max_n_nodes(cur_graph_batch)),
            ).to(device)
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
        normalize_nodes(state, model.node_norm, model.feature_set)

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
            max_trim_actions_per_route=max_trim_actions_per_route,
            adjustment_penalty_fn=adjustment_penalty_fn)
        returns, advantages = _compute_ppo_returns_and_advantages(
            rollout["rewards"], rollout["value_estimates"],
            rollout["dones"], rollout["final_value_estimates"], gamma,
            use_gae, gae_lambda)
        if normalize_advantages:
            # Standardize over the active steps of this update. Makes the
            # policy gradient invariant to the reward scale, which otherwise
            # varies a lot across batches (sampled cost weights alpha, graph
            # difficulty) and drowns the small per-edit improvement signal.
            _act = rollout["active_masks"]
            if _act.any():
                _a = advantages[_act]
                advantages = (advantages - _a.mean()) / (_a.std() + 1e-8)
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
        reward_delta_diag = _get_reward_delta_diagnostics(
            rollout["rewards"], rollout["active_masks"],
            start_costs, final_costs, reward_scale)
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
            # Curriculum-aware validation: when a val_curriculum_fn is given,
            # validate ONLY on the tiers currently active in training, so the
            # validation signal (and best-checkpoint selection) is comparable to
            # what the agent is being trained on. At the final stage all tiers
            # are active, so this reduces to the full validation set.
            active_val_indices = (val_curriculum_fn(iteration)
                                  if val_curriculum_fn is not None
                                  else val_indices)
            last_val = evaluate_lc_improvement(
                model, cost_obj, graphs, seed_routes, active_val_indices, device,
                min_route_len, max_route_len,
                batch_size=effective_batch_size,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
                target_n_routes=target_n_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current,
                adjustment_gap=adjustment_degree_gap,
                adjustment_mode=adjustment_degree_mode)
            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                CheckpointStore.save_model_weights(model, best_model_path)

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
        train = EditTrainRecord(
            seed_cost=train_seed, final_cost=train_final, delta=train_delta,
            component_demand_delta=float(component_delta[0]),
            component_route_delta=float(component_delta[1]),
            component_connectivity_delta=float(component_delta[2]),
            reward_mean=active_rewards.mean().item()
                if active_rewards.numel() > 0 else 0.0,
            reward_sum=reward_delta_diag["reward_sum"],
            reward_per_episode=reward_delta_diag["reward_per_episode"],
            scaled_delta_sum=reward_delta_diag["scaled_delta_sum"],
            reward_delta_residual=reward_delta_diag["residual"],
            episode_count=reward_delta_diag["episode_count"],
            return_mean=active_returns.mean().item()
                if active_returns.numel() > 0 else 0.0,
            advantage_mean=active_advantages.mean().item()
                if active_advantages.numel() > 0 else 0.0,
            active_sample_count=int(rollout["active_masks"].sum().item()),
            ppo_ratio_mean=ratio_mean, ppo_clip_fraction=clip_fraction,
            ppo_objective=ppo_stats["objective"],
            critic_mse_mean=ppo_stats.get("critic_mse_mean", float("nan")),
            critic_explained_variance=ppo_stats.get(
                "critic_explained_variance", float("nan")),
            is_eval_iteration=eval_due, action=dict(train_action_stats),
            # Full PPO edits one active route slot at a time, so there is no
            # cheap per-iteration full-route-set change rate. Validation has it.
            changed_route_rate=float("nan"))
        val_record = _edit_val_record(val)

        # Unified, comparable progress metrics: the per-episode reward is the
        # mean cost reduction over an episode (seed - final), computed the SAME
        # way for train and validation, so train_ep_rew, val_ep_rew and their
        # gap are directly comparable.
        _tr_ep = train.delta
        _va_ep = val_record.delta
        _gap = (_tr_ep - _va_ep
                if _tr_ep == _tr_ep and _va_ep == _va_ep else float("nan"))
        train.val_gap = _gap
        log_metrics(logger_list, train, iteration + 1)
        log_metrics(logger_list, val_record, iteration + 1)
        flush(logger_list)
        pbar.set_postfix({
            "stage": cur_stage_label,
            "tr_rew": f"{_tr_ep:+.3f}",
            "val_rew": f"{_va_ep:+.3f}",
            "gap": f"{_gap:+.3f}",
        })
        print(
            f"iter={iteration + 1:03d} stage={cur_stage_label:<10} "
            f"train_ep_rew={_tr_ep:+.4f} val_ep_rew={_va_ep:+.4f} "
            f"train_val_gap={_gap:+.4f}"
            + ("  [eval]" if eval_due else "")
        )
        # Do not keep the previous rollout reachable while collecting the
        # next one. This matters especially when rollout states stay on GPU.
        del rollout, returns, advantages
        del active_rewards, active_returns, active_advantages
        del start_costs, final_costs, start_components, final_components
        del component_delta, reward_delta_diag

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
        max_rollout_samples=8192, target_n_routes=None,
        curriculum_fn=None, val_curriculum_fn=None, logger_list=()):
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
            graph_batch = to_batch(
                [graphs[int(i)] for i in batch_indices],
                device,
                transforms=(SpaceScaleTransform(0.95, 1.05), DemandScaleTransform(0.9, 1.1)),
                edit_features=True,
            )
            route_batch = to_routes(
                seed_routes[torch.as_tensor(batch_indices, dtype=torch.long)],
                n_routes=target_n_routes,
                width=min(seed_routes.shape[-1], _get_graph_batch_max_n_nodes(graph_batch)),
            ).to(device)
            rollout_lc_improvement(
                model, cost_obj, graph_batch, route_batch,
                min_route_len, max_route_len, greedy=False,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
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
            planned_tensor = to_routes(
                [[finalized[batch_idx]]
                 for batch_idx in range(len(finalized))],
                width=cur_working_routes.shape[-1]).to(cur_working_routes.device)
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

            cur_graph_batch = to_batch(
                [graphs[int(i)] for i in batch_indices],
                device,
                transforms=(SpaceScaleTransform(0.95, 1.05), DemandScaleTransform(0.9, 1.1)),
                edit_features=True,
            )
            route_batch = to_routes(
                seed_routes[torch.as_tensor(batch_indices, dtype=torch.long)],
                n_routes=target_n_routes,
                width=min(seed_routes.shape[-1], _get_graph_batch_max_n_nodes(cur_graph_batch)),
            ).to(device)
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
        normalize_nodes(state, model.node_norm, model.feature_set)

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
            max_trim_actions_per_route=max_trim_actions_per_route)
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
            # Curriculum-aware validation: when a val_curriculum_fn is given,
            # validate ONLY on the tiers currently active in training, so the
            # validation signal (and best-checkpoint selection) is comparable to
            # what the agent is being trained on. At the final stage all tiers
            # are active, so this reduces to the full validation set.
            active_val_indices = (val_curriculum_fn(iteration)
                                  if val_curriculum_fn is not None
                                  else val_indices)
            last_val = evaluate_lc_improvement(
                model, cost_obj, graphs, seed_routes, active_val_indices, device,
                min_route_len, max_route_len,
                batch_size=effective_batch_size,
                force_nonhalt_first_step=force_nonhalt_first_step,
                max_route_edit_steps=max_route_edit_steps,
                max_trim_actions_per_route=max_trim_actions_per_route,
                target_n_routes=target_n_routes,
                adjustment_target=adj_eval_target,
                adjustment_weight=adj_eval_weight,
                adjustment_use_current=adjustment_condition_current)
            if last_val["final_cost"] < best_val_cost:
                best_val_cost = last_val["final_cost"]
                CheckpointStore.save_model_weights(model, best_model_path)

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
        train = EditTrainRecord(
            seed_cost=train_seed, final_cost=train_final, delta=train_delta,
            component_demand_delta=float(component_delta[0]),
            component_route_delta=float(component_delta[1]),
            component_connectivity_delta=float(component_delta[2]),
            reward_mean=scalar_rewards.mean().item()
                if scalar_rewards.numel() > 0 else 0.0,
            return_mean=scalar_returns.mean().item()
                if scalar_returns.numel() > 0 else 0.0,
            advantage_mean=scalar_advantages.mean().item()
                if scalar_advantages.numel() > 0 else 0.0,
            return_mean_by_objective=dict(zip(_OBJECTIVES, map(float, return_per_obj))),
            advantage_mean_by_objective=dict(zip(_OBJECTIVES, map(float, adv_per_obj))),
            active_sample_count=int(rollout["active_masks"].sum().item()),
            d3po_ratio_mean=ratio_mean, d3po_clip_fraction=clip_fraction,
            d3po_objective=d3po_stats["objective"],
            d3po_diversity_loss=d3po_stats["diversity_loss"],
            critic_mse_mean=d3po_stats.get("critic_mse_mean", float("nan")),
            critic_mse_by_objective=dict(zip(_OBJECTIVES, map(float, critic_mse_per_obj))),
            critic_explained_variance_by_objective=dict(
                zip(_OBJECTIVES, map(float, critic_ev_per_obj))),
            is_eval_iteration=eval_due, action=dict(train_action_stats),
            changed_route_rate=float("nan"))
        val_record = _edit_val_record(val)
        log_metrics(logger_list, train, iteration + 1)
        log_metrics(logger_list, val_record, iteration + 1)
        flush(logger_list)

        pbar.set_postfix({
            "reward": f"{train.reward_mean:.3f}",
            "delta": f"{train.delta:.3f}",
            "ratio": f"{train.d3po_ratio_mean:.3f}",
            "clip": f"{train.d3po_clip_fraction:.2%}",
        })
        action = train.action
        print(
            f"cfg_d3po_iter={iteration + 1:03d} "
            f"reward={train.reward_mean:.4f} "
            f"train_delta={train.delta:.4f} "
            f"val_delta={val_record.delta:.4f} "
            f"ratio={train.d3po_ratio_mean:.3f} "
            f"clip={train.d3po_clip_fraction:.2%} "
            f"div={train.d3po_diversity_loss:.4f} "
            f"actions="
            f"ext:{action['extend_count']} "
            f"trim_s:{action['trim_start_count']} "
            f"trim_e:{action['trim_end_count']} "
            f"halt:{action['halt_count']} "
            f"avg_steps={action['avg_actions_per_route']:.2f} "
            f"eval={eval_due}"
        )
        # Do not keep the previous rollout reachable while collecting the
        # next one. This matters especially when rollout states stay on GPU.
        del rollout, returns, advantages
        del active_rewards, active_preferences, active_returns
        del active_advantages, scalar_rewards, scalar_returns
        del scalar_advantages, start_costs, final_costs
        del start_components, final_components, component_delta

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

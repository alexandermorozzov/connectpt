"""Replay minibatches through the collected action masks."""
from dataclasses import dataclass, fields
import numpy as np
import torch

from ..env.actions import RouteAction
from ..env.features import observe
from ..env.masks import ActionMask
from ..env.state import RouteGenBatchState
from .algorithms.base import normalize_advantages


@dataclass
class TrainingBatch:
    states: list
    actions: torch.Tensor
    kinds: torch.Tensor
    logp: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    score: torch.Tensor
    masks: list[ActionMask]
    preferences: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    objective_rewards: torch.Tensor
    penalty_rewards: torch.Tensor


@dataclass
class UpdateStats:
    objective: float
    policy_objective: float
    entropy: float
    value_loss: float
    clip_fraction: float
    approx_kl: float
    initial_ratio_error: float
    diversity_loss: float
    diversity_kl: float
    diversity_target: float
    explained_variance: float
    explained_variance_by_objective: list
    values: torch.Tensor
    targets: torch.Tensor
    value_loss_by_objective: list


def shape_groups(states, indices):
    groups = {}
    for index in indices.detach().cpu().tolist():
        state = states[index]
        state.extra_data.shortest_path_sequences = torch.zeros(
            (state.batch_size, 0, 0, 0), dtype=torch.long, device=state.device
        )
        signature = tuple(
            (key, tuple(state.extra_data[key].shape[1:]))
            for key in sorted(state.extra_data.keys())
            if torch.is_tensor(state.extra_data[key])
        )
        groups.setdefault(signature, []).append(index)
    return [torch.tensor(group, device=indices.device) for group in groups.values()]


def replay_mask(masks, indices, batch_size, score, device):
    values = {}
    for field in fields(ActionMask):
        parts = []
        for index in indices.detach().cpu().tolist():
            value = getattr(masks[index], field.name)
            value = torch.as_tensor(value, device=device, dtype=torch.bool)
            parts.append(value.expand(batch_size))
        values[field.name] = torch.cat(parts)[score]
    return ActionMask(**values)


def update(policy, algorithm, critic, optimizer, batch):
    p = algorithm.params
    device = next(policy.parameters()).device
    batch_size = batch.rewards.shape[1]
    states_per_minibatch = max(1, p.minibatch_size // batch_size)
    order = torch.cat([torch.randperm(len(batch.states), device=device) for _ in range(p.n_epochs)])
    stats, value_buffer, target_buffer, per_goal_mse = [], [], [], []
    initial_error = None
    for raw_indices in torch.split(order, states_per_minibatch):
        for indices in shape_groups(batch.states, raw_indices):
            selected = [batch.states[index] for index in indices.detach().cpu().tolist()]
            state = selected[0] if len(selected) == 1 else RouteGenBatchState.batch_from_list(selected)
            state = state.to_device(device)
            score = batch.score[indices].flatten()
            if not score.any():
                continue
            if not score.all():
                state = state.index_select(torch.where(score)[0])
            action = RouteAction(batch.kinds[indices].flatten()[score], batch.actions[indices].flatten(0, 1)[score])
            old_logp = batch.logp[indices].flatten()[score]
            returns = batch.returns[indices].flatten(0, 1)[score]
            advantages = batch.advantages[indices].flatten(0, 1)[score]
            preferences = batch.preferences[indices].flatten(0, 1)[score]
            mask = replay_mask(batch.masks, indices, batch_size, score, device)
            critic.from_state(state)
            value_stats = critic.update(returns, old_values=returns - advantages)
            value_buffer.append(value_stats["values"])
            target_buffer.append(value_stats["targets"])
            per_goal_mse.append(value_stats["per_obj_mse"].detach())
            obs = observe(
                state,
                policy.feature_set,
                allow_below_min=policy.allow_trim_below_min if policy.action_spec.role == "edit" else False,
            )
            output = policy.evaluate(obs, mask, action)
            if p.normalize_advantages:
                advantages = normalize_advantages(advantages, per_objective=advantages.ndim > 1)
            log_ratio = output.logp - old_logp
            ratios = log_ratio.exp()
            if initial_error is None:
                initial_error = float((ratios.detach() - 1).abs().max())
                if initial_error > 2e-4:
                    raise RuntimeError(f"Initial replay ratio differs from 1: {initial_error}")
            clipped = ratios.clamp(1 - p.epsilon, 1 + p.epsilon)
            surrogate = algorithm.surrogate(ratios, clipped, advantages, preferences)
            diversity = algorithm.diversity(policy, state, output, mask)
            objective = (
                surrogate.mean() + output.entropy.mean() * p.entropy_weight - p.diversity_weight * diversity.loss
            )
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), p.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            stats.append(
                [
                    float(objective.detach()),
                    float(surrogate.detach().mean()),
                    float(output.entropy.detach().mean()),
                    value_stats["loss"],
                    float(((ratios.detach() - 1).abs() > p.epsilon).float().mean()),
                    float(((ratios - 1) - log_ratio).detach().mean()),
                    float(diversity.loss.detach()),
                    float(diversity.kl.detach()),
                    float(diversity.target.detach()),
                ]
            )
    if not stats:
        empty = batch.rewards.new_empty(0)
        return UpdateStats(*([float("nan")] * 6), 0.0, 0.0, 0.0, 0.0, float("nan"), [], empty, empty, [])
    means = np.mean(stats, axis=0).tolist()
    values, targets = torch.cat(value_buffer), torch.cat(target_buffer)
    variance = targets.var(unbiased=False)
    explained = float(1 - (targets - values).var(unbiased=False) / variance) if variance > 0 else float("nan")
    per_goal = []
    if targets.ndim > 1:
        means[3] = float(torch.stack(per_goal_mse).mean(dim=0).mean())
        variance = targets.float().var(dim=0, unbiased=False)
        residual = (targets - values).float().var(dim=0, unbiased=False)
        per_goal = torch.where(
            variance > 0, 1 - residual / variance.clamp_min(1e-12), torch.full_like(variance, float("nan"))
        ).tolist()
    goal_loss = torch.stack(per_goal_mse).mean(dim=0).tolist() if targets.ndim > 1 else []
    return UpdateStats(*means[:6], initial_error, *means[6:9], explained, per_goal, values, targets, goal_loss)

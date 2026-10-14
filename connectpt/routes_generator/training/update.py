"""Replay minibatches through the collected action masks."""
from dataclasses import dataclass, fields, field
import numpy as np
import torch

from ..env.actions import RouteAction
from ..env.masks import ActionMask, CandidateMasks
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
    diagnostics: dict = field(default_factory=dict)


def gradient_diagnostics(policy_objective, diversity_loss, weight, parameters):
    """Separate ascent gradients before clipping; retain the training graph.

    autograd.grad leaves parameter .grad and RNG untouched. Zero norms make
    ratios/cosines undefined (NaN), rather than suggesting agreement.
    """
    def gradients(objective):
        if not objective.requires_grad:
            return [torch.zeros_like(p) for p in parameters]
        return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(
            parameters, torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True))]

    pg, dg = gradients(policy_objective), gradients(-diversity_loss)
    pn = sum(g.double().square().sum() for g in pg).sqrt()
    dn = sum(g.double().square().sum() for g in dg).sqrt()
    dot = sum((a.double() * b.double()).sum() for a, b in zip(pg, dg))
    return {
        'policy_gradient_norm': float(pn),
        'diversity_gradient_norm': float(dn),
        'weighted_diversity_gradient_norm': float(dn * weight),
        'diversity_gradient_ratio': float(dn * weight / pn) if pn > 0 else float('nan'),
        'gradient_cosine': float(dot / (pn * dn)) if pn > 0 and dn > 0 and weight > 0 else float('nan'),
    }


def summarize_diagnostics(gradients, samples, *, diversity_reduction='state_mean'):
    """Mean sampled minibatch gradients; state-weighted distribution statistics."""
    result = {key: float(np.nanmean([row[key] for row in gradients]))
              if any(np.isfinite(row[key]) for row in gradients) else float('nan')
              for key in gradients[0]} if gradients else {}
    result['gradient_samples'] = len(gradients)
    if samples:
        kl, target, counts = [torch.cat(items) for items in zip(*samples)]
        free = counts > 1
        result.update(diversity_free_fraction=float(free.double().mean()),
                      diversity_state_count=kl.numel(), diversity_free_state_count=int(free.sum()))
        if diversity_reduction == 'state_mean':
            # With batch_mean, singleton errors can cancel other states' errors;
            # they are not an independently unavoidable contribution to loss.
            result['diversity_singleton_loss_floor'] = float((target.square() * ~free).mean())
        for name, values in [('diversity_kl_free', kl[free]), ('diversity_target_free', target[free]),
                             ('diversity_abs_error_free', (kl-target).abs()[free])]:
            result[name + '_mean'] = float(values.mean()) if values.numel() else float('nan')
            for q, suffix in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
                result[name + '_' + suffix] = float(values.double().quantile(q)) if values.numel() else float('nan')
    return result


def shape_groups(states, indices):
    groups = {}
    for index in indices.detach().cpu().tolist():
        state = states[index]
        signature = tuple(
            (key, tuple(state.extra_data[key].shape[1:]))
            for key in sorted(state.extra_data.keys())
            if key != 'shortest_path_sequences' and torch.is_tensor(state.extra_data[key])
        )
        groups.setdefault(signature, []).append(index)
    return [torch.tensor(group, device=indices.device) for group in groups.values()]


def prepare_replay(states, indices, device):
    """Own the selected snapshots and their lazy caches before any forward."""
    selected = [states[index].snapshot_for_buffer(device) for index in indices.detach().cpu().tolist()]
    return RouteGenBatchState.batch_from_list(selected)


def replay_mask(masks, indices, batch_size, score, device):
    selected = indices.detach().cpu().tolist()
    values = {}
    for field in fields(ActionMask):
        if field.name == 'candidates':
            continue
        parts = []
        for index in selected:
            value = getattr(masks[index], field.name)
            value = torch.as_tensor(value, device=device, dtype=torch.bool)
            parts.append(value.expand(batch_size))
        values[field.name] = torch.cat(parts)[score]
    candidates = CandidateMasks.concatenate([masks[i].require_candidates() for i in selected], device)
    values['candidates'] = candidates.index_select(torch.where(score)[0])
    return ActionMask(**values)


def update(env, policy, algorithm, critic, optimizer, batch):
    p = algorithm.params
    diagnostic_interval = p.gradient_diagnostics_interval
    if diagnostic_interval < 0:
        raise ValueError('gradient_diagnostics_interval must be nonnegative')
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    device = next(policy.parameters()).device
    batch_size = batch.rewards.shape[1]
    states_per_minibatch = max(1, p.minibatch_size // batch_size)
    order = torch.cat([torch.randperm(len(batch.states), device=device) for _ in range(p.n_epochs)])
    stats, value_buffer, target_buffer, per_goal_mse = [], [], [], []
    initial_error = None
    gradient_samples, diversity_samples = [], []
    for raw_indices in torch.split(order, states_per_minibatch):
        for indices in shape_groups(batch.states, raw_indices):
            state = prepare_replay(batch.states, indices, device)
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
            critic.prepare_update(state)
            value_stats = critic.update(returns, old_values=returns - advantages)
            value_buffer.append(value_stats["values"])
            target_buffer.append(value_stats["targets"])
            per_goal_mse.append(value_stats["per_obj_mse"].detach())
            obs = env.observation(state)
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
            diversity = algorithm.diversity(policy, env, state, output, mask, diagnostics=diagnostic_interval > 0)
            if diagnostic_interval:
                if diversity.kl_by_state is not None:
                    diversity_samples.append((diversity.kl_by_state, diversity.target_by_state, diversity.action_counts))
                if len(stats) % diagnostic_interval == 0:
                    gradient_samples.append(gradient_diagnostics(surrogate.mean(), diversity.loss,
                                                               p.diversity_weight, parameters))
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
    diagnostics = summarize_diagnostics(gradient_samples, diversity_samples,
                                        diversity_reduction=p.diversity_reduction) if diagnostic_interval else {}
    return UpdateStats(*means[:6], initial_error, *means[6:9], explained, per_goal, values, targets, goal_loss, diagnostics)

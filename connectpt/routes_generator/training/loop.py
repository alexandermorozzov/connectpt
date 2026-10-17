"""One training cycle for any environment and algorithm."""
from dataclasses import dataclass
import time
from pathlib import Path

import torch

from ..objectives.const import ACTION_PENALTIES
from ..env.rewards import potential_reward
from ..policy.rollout import rollout
from ..policy.predict import predict
from ..common.logging import flush
from ..policy.checkpoints import save_model_weights
from .logging import log_values, log_iteration
from .update import TrainingBatch, update
from .validate import ValidationTracker


@dataclass(frozen=True)
class Budget:
    n_iterations: int
    horizon: int
    val_period: int
    validate_before_update: bool
    max_seconds: float | None
    warmup_batches: int
    max_rollout_samples: int | None
    calibrate_before_validation: bool
    action_seed: int | None
    force_nonhalt_until: int | None
    output_dir: Path
    run_name: str


@dataclass
class TrainingResult:
    best_policy: object
    best_cost: float
    iterations: int
    critic_snapshot: dict


def collect(env, policy, algorithm, critic, data, horizon):
    device = next(policy.parameters()).device
    data.start(env)
    states, outputs, masks, rewards, dones, scores, preferences = [], [], [], [], [], [], []
    objective_rewards, penalty_rewards = [], []
    terminal = {
        key: [] for key in ("cost", "ATT", "RTT", "feasible_fraction", "unserved_fraction", "alpha")
    }
    completed_rewards, completed_lengths = [], []
    with torch.no_grad():
        for _ in range(horizon):
            data.prepare_step(env)
            active = ~env.is_done().clone()
            _, mask = env.observe()
            masks.append(mask)
            previous = algorithm.potential(env.result)
            trajectory = rollout(env, policy, max_steps=1)
            objective_rewards.append(trajectory.rewards[0][0])
            penalty_rewards.append(trajectory.rewards[0][1])
            out = trajectory.outputs[0]
            states.append(trajectory.states[0])
            outputs.append(out)
            done = trajectory.dones[0]
            current = algorithm.potential(env.result)
            reward = potential_reward(previous, current, active, done,
                                      diff=env.params.diff_reward, scale=env.params.reward_scale)
            action_penalty = trajectory.rewards[0][1][:, len(env.result.penalty_names):]
            for index, name in enumerate(ACTION_PENALTIES):
                weight = env.cost.penalty_weights[name]
                if weight:
                    reward = reward + action_penalty[:, index].reshape((-1,) + (1,) * (reward.ndim - 1)) * weight
            reward = reward * (active[:, None] if reward.ndim > 1 else active)
            rewards.append(reward)
            dones.append(done)
            scores.append(active & ~torch.full_like(active, trajectory.infos[0]["forced_halt"]))
            pref = env.result.preferences
            preferences.append(pref / pref.sum(-1, keepdim=True).clamp_min(1e-8))
            scalar_reward = reward if reward.ndim == 1 else (reward * preferences[-1]).sum(-1)
            ended = done & active
            episode_reward, episode_length = data.record_step(scalar_reward,ended,active)
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            if ended.any():
                result = env.result
                terminal["cost"].append(result.cost[ended])
                terminal["ATT"].append(result.mean_demand_time[ended] / 60)
                terminal["RTT"].append(result.total_route_time[ended] / 60)
                terminal["feasible_fraction"].append((~result.are_constraints_violated())[ended].float())
                terminal["unserved_fraction"].append((result.unserved_demand / result.total_demand)[ended])
                terminal["alpha"].append(env.state.cost_weights["demand_time_weight"][ended])
            data.after_step(env, done)
        # Values are computed after collection; rollout has no critic dependency.
        values = torch.stack([critic.predict(state.to_device(device)) for state in states])
        bootstrap = critic.predict(env.state)
        mask = data.bootstrap_mask(env)
        bootstrap = algorithm.bootstrap(bootstrap, mask)
        rewards, dones = torch.stack(rewards), torch.stack(dones)
        returns, advantages = algorithm.returns(rewards, values, dones, bootstrap)
    batch = TrainingBatch(
        states,
        torch.stack([out.action.indices for out in outputs]),
        torch.stack([out.action.kind for out in outputs]),
        torch.stack([out.logp for out in outputs]).float(),
        rewards,
        dones,
        torch.stack(scores),
        masks,
        torch.stack(preferences),
        values,
        returns,
        advantages,
        torch.stack(objective_rewards),
        torch.stack(penalty_rewards),
    )
    return batch, terminal, torch.cat(completed_rewards), torch.cat(completed_lengths)


def train(env, policy, algorithm, critic, optimizer, logger_list, data, budget, *, validation):
    if budget.n_iterations < 1 or budget.horizon < 1 or budget.val_period < 1:
        raise ValueError("training budgets must be positive")
    if budget.max_rollout_samples is not None and data.batch_size * budget.horizon > budget.max_rollout_samples:
        raise ValueError("rollout exceeds max_rollout_samples")
    if data.batch_size > algorithm.params.minibatch_size:
        raise ValueError("graph batch size exceeds minibatch_size")
    if budget.action_seed is not None:
        torch.manual_seed(budget.action_seed)
    policy.train()
    with torch.no_grad():
        for graphs, seeds in data.warmup(budget.warmup_batches):
            env.reset(graphs, env.cost.get_weights(), seeds)
            predict(policy, graphs, env.state.cost_weights, env.cost, type(env), env.params,
                    "sample", seed_networks=seeds, mask=env.mask_override)
    if budget.warmup_batches:
        policy.update_and_freeze_feature_norms()
    policy.eval()
    data.initialize()
    checkpoint_dir = budget.output_dir / f"{budget.run_name}_checkpoints"
    tracker = ValidationTracker(validation, on_best=lambda best:
                                save_model_weights(best, checkpoint_dir / 'best.pt'))
    started = time.monotonic()
    iterations = 0
    for iteration in range(budget.n_iterations):
        if iteration and budget.max_seconds is not None and time.monotonic() - started >= budget.max_seconds:
            break
        progress = iteration / budget.n_iterations
        if budget.validate_before_update and iteration % budget.val_period == 0:
            if iteration == 0 and budget.calibrate_before_validation:
                validation.run(policy, (), iteration)
                policy.update_and_freeze_feature_norms()
            tracker.run(policy, logger_list, iteration,
                        freeze=iteration == 0 and not budget.calibrate_before_validation)
        env.prepare_iteration(iteration, budget.force_nonhalt_until)
        data.get(progress)
        batch, terminal, episode_rewards, episode_lengths = collect(
            env, policy, algorithm, critic, data, budget.horizon
        )
        log_values(algorithm, batch, logger_list, iteration)
        stats = update(env, policy, algorithm, critic, optimizer, batch)
        log_iteration(env, algorithm, batch, stats, logger_list, data, budget, iteration,
                      started, episode_rewards, episode_lengths, terminal,
                      learning_rate=optimizer.param_groups[0]['lr'])
        if iteration % budget.val_period == 0:
            save_model_weights(policy, checkpoint_dir / f"iter{iteration}.pt")
        iterations = iteration + 1
        if not budget.validate_before_update and (
            (iteration > 0 and iterations % budget.val_period == 0) or iterations == budget.n_iterations
        ):
            tracker.run(policy, logger_list, iterations)
        flush(logger_list)
    # Preserve completed training before the potentially expensive final evaluation.
    save_model_weights(policy, checkpoint_dir / 'last.pt')
    snapshot = {
        "state_dict": critic.model.state_dict(),
        "values": stats.values.detach().cpu(),
        "targets": stats.targets.detach().cpu(),
    }
    torch.save(snapshot, checkpoint_dir / 'critic_snapshot.pt')
    tracker.finish(policy, logger_list, iterations)
    return TrainingResult(tracker.best_policy, tracker.best_cost, iterations, snapshot)

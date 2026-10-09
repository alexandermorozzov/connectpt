"""One training cycle for any environment and algorithm."""
from dataclasses import dataclass, replace
import copy
import time
from pathlib import Path

import torch

from ..objectives.const import ACTION_PENALTIES
from ..policy.rollout import rollout
from ..policy.generate import generate
from ..common.logging import log, flush
from ..policy.checkpoints import save_model_weights
from .records import (
    ValueRecord,
    DiagnosticRecord,
    TrainRecord,
    GoalTrainRecord,
    RolloutRecord,
    TimeRecord,
)
from .update import TrainingBatch, update


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


def episode_statistics(rewards, dones, reward_sums, lengths):
    completed_rewards, completed_lengths = [], []
    for step_rewards, step_dones in zip(rewards, dones):
        reward_sums.add_(step_rewards)
        lengths.add_(1)
        completed_rewards.append(reward_sums[step_dones].clone())
        completed_lengths.append(lengths[step_dones].clone())
        reward_sums[step_dones] = 0
        lengths[step_dones] = 0
    return torch.cat(completed_rewards), torch.cat(completed_lengths)


def collect(env, policy, algorithm, critic, data, horizon):
    device = next(policy.parameters()).device
    data.start(env)
    states, outputs, masks, rewards, dones, scores, preferences = [], [], [], [], [], [], []
    objective_rewards, penalty_rewards = [], []
    terminal = {
        key: [] for key in ("cost", "att_minutes", "rtt_minutes", "feasible_fraction", "unserved_fraction", "alpha")
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
            if env.params.diff_reward:
                reward = (previous - current) * env.params.reward_scale
            else:
                ended = done & active
                reward = -current * env.params.reward_scale * (ended[:, None] if current.ndim > 1 else ended)
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
            data.episode_lengths -= (~active).long()
            episode_reward, episode_length = episode_statistics(
                scalar_reward[None], ended[None], data.episode_rewards, data.episode_lengths
            )
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            if ended.any():
                result = env.result
                terminal["cost"].append(result.cost[ended])
                terminal["att_minutes"].append(result.mean_demand_time[ended] / 60)
                terminal["rtt_minutes"].append(result.total_route_time[ended] / 60)
                terminal["feasible_fraction"].append((~result.are_constraints_violated())[ended].float())
                terminal["unserved_fraction"].append((result.unserved_demand / result.total_demand)[ended])
                terminal["alpha"].append(env.state.cost_weights["demand_time_weight"][ended])
            data.after_step(env, done)
        # Values are computed after collection; rollout has no critic dependency.
        values = torch.stack([critic.from_state(state.to_device(device)) for state in states])
        bootstrap = critic.from_state(env.state)
        mask = data.bootstrap_mask(env)
        bootstrap = bootstrap * (mask[:, None] if bootstrap.ndim > 1 else mask)
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


def train(env, policy, algorithm, critic, optimizer, logger_list, data, budget):
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
            generate(env, policy, "sample", 1, seed_networks=seeds)
    if budget.warmup_batches:
        policy.update_and_freeze_feature_norms()
    policy.eval()
    data.initialize()
    best_cost, best_policy = float("inf"), None
    started = time.monotonic()
    checkpoint_dir = budget.output_dir / f"{budget.run_name}_checkpoints"
    iterations = 0
    last_validation_step = None
    val_cost = None
    for iteration in range(budget.n_iterations):
        if iteration and budget.max_seconds is not None and time.monotonic() - started >= budget.max_seconds:
            break
        progress = iteration / budget.n_iterations
        if budget.validate_before_update and iteration % budget.val_period == 0:
            if iteration == 0 and budget.calibrate_before_validation:
                data.validation.run(policy, (), iteration, progress)
                policy.update_and_freeze_feature_norms()
            val_cost = data.validation.run(policy, logger_list, iteration, progress)
            last_validation_step = iteration
            if iteration == 0 and not budget.calibrate_before_validation:
                policy.update_and_freeze_feature_norms()
            data.validation.monitor(policy, logger_list, iteration)
            if val_cost < best_cost:
                best_cost, best_policy = val_cost, copy.deepcopy(policy)
        force_first = env.params.force_nonhalt_first_step
        if budget.force_nonhalt_until is not None:
            force_first = force_first and iteration < budget.force_nonhalt_until
        env.params = replace(env.params, force_nonhalt_first_step=force_first)
        data.get(progress)
        batch, terminal, episode_rewards, episode_lengths = collect(
            env, policy, algorithm, critic, data, budget.horizon
        )
        scalar_rewards = batch.rewards if batch.rewards.ndim == 2 else (batch.rewards * batch.preferences).sum(-1)
        scalar_values = batch.values if batch.values.ndim == 2 else (batch.values * batch.preferences).sum(-1)
        names = tuple(algorithm.cost.objectives)
        log(
            logger_list,
            ValueRecord(
                float(scalar_values.mean()),
                {name: float(batch.values[..., i].mean()) for i, name in enumerate(names)}
                if batch.values.ndim > 2
                else {},
            ),
            iteration,
        )
        stats = update(env, policy, algorithm, critic, optimizer, batch)
        log(
            logger_list,
            TrainRecord(
                -stats.policy_objective,
                stats.value_loss,
                -stats.entropy,
                stats.clip_fraction,
                stats.approx_kl,
                stats.explained_variance,
                stats.diversity_loss,
                optimizer.param_groups[0]["lr"],
                algorithm.params.epsilon,
                (iteration + 1) * algorithm.params.n_epochs,
                -stats.objective,
                stats.diversity_kl,
                stats.diversity_target,
            ),
            iteration,
        )
        log(
            logger_list,
            GoalTrainRecord(
                dict(zip(names, stats.value_loss_by_objective)),
                dict(zip(names, stats.explained_variance_by_objective)),
                {name: float(batch.advantages[..., i].std()) for i, name in enumerate(names)}
                if batch.advantages.ndim > 2
                else {},
            ),
            iteration,
        )
        log(
            logger_list,
            RolloutRecord(
                float(episode_rewards.mean()) if episode_rewards.numel() else float("nan"),
                float(episode_lengths.float().mean()) if episode_lengths.numel() else float("nan"),
                {name: float(batch.objective_rewards[..., i].mean()) for i, name in enumerate(names)},
                {
                    name: float(batch.penalty_rewards[..., i].mean())
                    for i, name in enumerate((*env.result.penalty_names, *ACTION_PENALTIES))
                },
            ),
            iteration,
        )
        elapsed = time.monotonic() - started
        log(
            logger_list,
            TimeRecord(
                (iteration + 1) * budget.horizon * data.batch_size / elapsed,
                iteration + 1,
                elapsed,
                (iteration + 1) * budget.horizon * data.batch_size,
            ),
            iteration,
        )
        record = DiagnosticRecord(
            initial_ratio_error=stats.initial_ratio_error,
            graph_instances_loaded=data.graph_instances_loaded,
            graph_batches_loaded=data.graph_batches_loaded,
            completed_networks=int(batch.dones.sum()),
            return_target_mean=float(batch.returns.mean()),
            step_reward=float(scalar_rewards.mean()),
        )
        for key, values in terminal.items():
            if key != "alpha":
                setattr(record, key, float(torch.cat(values).mean()) if values else float("nan"))
        alpha = torch.cat(terminal["alpha"]) if terminal["alpha"] else None
        record.completed_alpha_mean = float(alpha.mean()) if alpha is not None else float("nan")
        for name, condition in (
            ("alpha_0", lambda a: a == 0),
            ("alpha_1", lambda a: a == 1),
            ("alpha_intermediate", lambda a: (a > 0) & (a < 1)),
        ):
            setattr(
                record,
                f"completed_{name}_fraction",
                float(condition(alpha).float().mean()) if alpha is not None else float("nan"),
            )
        log(logger_list, record, iteration)
        if iteration % budget.val_period == 0:
            save_model_weights(policy, checkpoint_dir / f"iter{iteration}.pt")
        iterations = iteration + 1
        if not budget.validate_before_update and (
            (iteration > 0 and iterations % budget.val_period == 0) or iterations == budget.n_iterations
        ):
            val_cost = data.validation.run(policy, logger_list, iterations, progress)
            last_validation_step = iterations
            data.validation.monitor(policy, logger_list, iterations)
            if val_cost < best_cost:
                best_cost, best_policy = val_cost, copy.deepcopy(policy)
        flush(logger_list)
    final_cost = (
        val_cost if last_validation_step == iterations else data.validation.run(policy, logger_list, iterations, 1.0)
    )
    data.validation.monitor(policy, logger_list, iterations, final=True)
    if final_cost < best_cost:
        best_cost, best_policy = final_cost, copy.deepcopy(policy)
    snapshot = {
        "state_dict": critic.model.state_dict(),
        "values": stats.values.detach().cpu(),
        "targets": stats.targets.detach().cpu(),
    }
    return TrainingResult(best_policy, best_cost, iterations, snapshot)

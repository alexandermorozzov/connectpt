"""Training record projections, with value logs before the policy update."""
import time

import torch

from ..common.logging import log
from ..objectives.const import ACTION_PENALTIES
from .records import (
    ValueRecord, DiagnosticRecord, TrainRecord, GoalTrainRecord, RolloutRecord, TimeRecord,
)


def log_values(algorithm, batch, logger_list, iteration):
    scalar_values = batch.values if algorithm.scalar_values else (batch.values * batch.preferences).sum(-1)
    names = tuple(algorithm.cost.objectives)
    log(
        logger_list,
        ValueRecord(
            float(scalar_values.mean()),
            {name: float(batch.values[..., i].mean()) for i, name in enumerate(names)}
            if not algorithm.scalar_values
            else {},
        ),
        iteration,
    )


def log_iteration(env, algorithm, batch, stats, logger_list, data, budget, iteration,
                  started, episode_rewards, episode_lengths, terminal, *, learning_rate):
    scalar_rewards = batch.rewards if algorithm.scalar_values else (batch.rewards * batch.preferences).sum(-1)
    names = tuple(algorithm.cost.objectives)
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
            learning_rate,
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

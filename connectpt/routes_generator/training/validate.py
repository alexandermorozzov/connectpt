"""Validation and preference-grid evaluation above the shared policy evaluator."""
from __future__ import annotations

import time
import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..common.logging import flush, log
from ..policy.predict import evaluate_policy
from ..env.base import EnvParams,EnvironmentBinding
from ..objectives.pareto import canonical_network, summarize_evaluation
from ..objectives.metrics import evaluate_network, monitor_metrics
from .records import ParetoRecord, ValidationRecord
from .split import evaluation_indices


@dataclass(frozen=True)
class EvaluationParams:
    validation_mode: str
    graph_count: int | None
    test_graph_count: int | None
    pareto_enabled: bool
    pareto_period: int
    pareto_preference_count: int
    pareto_stochastic_samples: int
    hv_reference: list
    eu_preference_count: int
    preference_count: int
    stochastic_samples: int
    seed: int
    final_test: bool


@dataclass(frozen=True)
class EvaluationProfile:
    evaluation: EvaluationParams
    env: EnvParams
    batch_size: int
    symmetric_routes: bool
    env_factory: object


class Validation:
    def __init__(self, data, cost, env, profile, device):
        self.data, self.cost, self.env = data, cost, env
        self.profile, self.device = profile, device
        self.last_monitor_step = None

    def batches(self):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, policy, logger_list, step):
        collected, results = [], []
        for graphs, seeds, route_count in self.batches():
            self.env.set_route_count(route_count)
            result, routes = evaluate_policy(
                self.env,
                policy,
                graphs.to(self.device),
                self.cost.get_weights(),
                self.profile.evaluation.validation_mode,
                1,
                seed_networks=seeds,
            )
            metrics = evaluate_network(result)
            collected.append(metrics)
            results.append(result)
        metrics = {key: torch.cat([item[key] for item in collected]) for key in collected[0]}
        log(
            logger_list,
            ValidationRecord(
                cost=float(metrics["cost"].mean()),
                objective={
                    name: float(torch.cat([r.goals[:, i] for r in results]).mean())
                    for i, name in enumerate(self.cost.objectives)
                },
                penalty={
                    name: float(torch.cat([r.penalties[:, i] for r in results]).mean())
                    for i, name in enumerate(results[0].penalty_names)
                },
                monitor={name: float(value.mean()) for name, value in
                         monitor_metrics(metrics, self.cost.monitor).items()},
                feasible_fraction=float(torch.cat([~r.are_constraints_violated() for r in results]).float().mean()),
            ),
            step,
        )
        return float(metrics["cost"].mean())

    def monitor_batches(self):
        raise NotImplementedError

    def monitor(self, policy, logger_list, step, *, batches, final=False):
        params = self.profile.evaluation
        if params.pareto_enabled and step != self.last_monitor_step and (final or step % params.pareto_period == 0):
            for graphs, seeds in batches:
                evaluate_pareto(policy, graphs, self.profile, self.cost, self.device,
                                logger_list, step, seed_networks=seeds)
            self.last_monitor_step = step


class ConstructionValidation(Validation):
    def batches(self):
        for count in self.data.n_routes:
            for graphs in self.data.validation_loader:
                yield graphs, None, count

    def monitor_batches(self):
        yield self.data.validation_loader.dataset, None


class EditValidation(Validation):
    def batches(self):
        for chunk in self.data.monitor_indices.split(self.data.batch_size):
            graphs, seeds = self.data.module.make_batch(chunk)
            yield graphs, seeds, seeds.shape[1]

    def monitor_batches(self):
        graphs, seeds = self.data.module.make_batch(self.data.monitor_indices)
        yield graphs.to_data_list(), seeds


class ValidationTracker:
    """One validation/monitor/best sequence before, after and at training end."""

    def __init__(self, validation):
        self.validation = validation
        self.best_cost, self.best_policy = float('inf'), None
        self.last_step, self.last_cost = None, None

    def _best(self, policy, cost):
        if cost < self.best_cost:
            self.best_cost, self.best_policy = cost, copy.deepcopy(policy)

    def run(self, policy, logger_list, step, *, freeze=False, final=False):
        self.last_cost = self.validation.run(policy, logger_list, step)
        self.last_step = step
        if freeze:
            policy.update_and_freeze_feature_norms()
        self.validation.monitor(policy, logger_list, step,
                                batches=self.validation.monitor_batches(), final=final)
        self._best(policy, self.last_cost)

    def finish(self, policy, logger_list, step):
        if self.last_step != step:
            self.run(policy, logger_list, step, final=True)
        else:
            self.validation.monitor(policy, logger_list, step,
                                    batches=self.validation.monitor_batches(), final=True)
            self._best(policy, self.last_cost)


@dataclass(frozen=True)
class FinalEvaluation:
    """Prepared held-out graph subset, references and a fixed last/best protocol."""

    data: object
    seed_networks: object
    split_name: str
    profile: EvaluationProfile
    protocol: dict

    def run(self, policies, cost, device, logger_list, step):
        for label, policy in policies:
            for mode in ('greedy', 'stochastic'):
                rows = evaluate_preferences(policy, self.data, self.profile, cost, device,
                                            mode=mode, seed_networks=self.seed_networks)
                prefix = f'eval/{self.split_name}/{label}/{mode}'
                log_evaluation_metrics(rows, logger_list, step, prefix)
                if self.profile.evaluation.pareto_enabled:
                    log_pareto_metrics(rows, self.profile, logger_list, step, prefix)


def prepare_final_evaluation(data, profile, protocol):
    graphs, seeds, sources = data.final_evaluation()
    indices = evaluation_indices(torch.arange(len(graphs)), profile.evaluation.test_graph_count,
                                 profile.evaluation.seed).tolist()
    selected = [graphs[index] for index in indices]
    selected_seeds = None if seeds is None else seeds[indices]
    prepared = copy.deepcopy(protocol)
    prepared.update(
        split='independent test' if data.final_split == 'test' else 'held-out validation',
        indices=[sources[index] for index in indices], graph_count=len(selected),
        outcomes_per_checkpoint={
            mode: len(selected) * profile.evaluation.preference_count * samples
            for mode, samples in prepared['samples_per_preference'].items()
        },
    )
    return FinalEvaluation(selected, selected_seeds, data.final_split, profile, prepared)


def evaluate_preferences(
    model,
    data,
    profile,
    cost,
    device,
    *,
    mode="greedy",
    preference_count=None,
    stochastic_samples=None,
    seed_networks=None,
):
    """Return per-network results on a fixed grid; preserve the torch RNG.

    The seed networks stay aligned with data order. Cost includes all configured
    penalties, including adjustment against these seeds. Logging and Pareto
    aggregation consume the returned values without scoring the network again.
    """
    rows = []
    count = int(preference_count if preference_count is not None else profile.evaluation.preference_count)
    samples = (
        1
        if mode == "greedy"
        else int(stochastic_samples if stochastic_samples is not None else profile.evaluation.stochastic_samples)
    )
    if mode not in ("greedy", "stochastic") or count < 2 or samples < 1 or not len(data):
        raise ValueError("Evaluation needs data, >=2 preferences and >=1 sample in a supported mode")
    for pref_idx, preference in enumerate(np.linspace(0, 1, count)):
        weights = {"demand_time_weight": 0.0, "route_time_weight": 0.0, "median_connectivity_weight": 0.0}
        from ..env.features import PREFERENCE_KEYS

        if len(cost.objectives) != 2:
            raise ValueError("alpha evaluation requires exactly two active goals")
        weights[PREFERENCE_KEYS[cost.objectives[0]]] = float(preference)
        weights[PREFERENCE_KEYS[cost.objectives[1]]] = float(1 - preference)
        fixed_cost = cost.with_params(**weights)
        for sample in range(samples):
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(profile.evaluation.seed) + pref_idx * 1000 + sample)
                for start in range(0, len(data), int(profile.batch_size)):
                    chunk = [data[i] for i in range(start, min(start + int(profile.batch_size), len(data)))]
                    loader = DataLoader(chunk, batch_size=int(profile.batch_size))
                    seeds = None if seed_networks is None else seed_networks[start : start + len(chunk)].to(device)
                    env = EnvironmentBinding(profile.env_factory, profile.env).create(model, fixed_cost)
                    result, routes = evaluate_policy(
                        env,
                        model,
                        loader,
                        fixed_cost.get_weights(),
                        "greedy" if mode == "greedy" else "sample",
                        1,
                        seed_networks=seeds,
                    )
                    metrics = evaluate_network(result)
                    pure = result.goals
                    penalty_cost = torch.zeros_like(result.cost)
                    for j, weight in enumerate(result.penalty_weights):
                        penalty_cost = penalty_cost + result.penalties[:, j] * weight
                    penalized = pure + penalty_cost[:, None]
                    feasible = ~result.are_constraints_violated()
                    for i in range(len(chunk)):
                        route_list = result.batch_routes[i].cpu().tolist()
                        rows.append(
                            dict(
                                graph=start + i,
                                preference=float(preference),
                                sample=sample,
                                ATT=float(result.mean_demand_time[i] / 60),
                                RTT=float(result.total_route_time[i] / 60),
                                objective_0=float(pure[i, 0]),
                                objective_1=float(pure[i, 1]),
                                penalized_0=float(penalized[i, 0]),
                                penalized_1=float(penalized[i, 1]),
                                scalar_cost=float(result.cost[i]),
                                objective={name: float(result.goals[i, j]) for j, name in enumerate(cost.objectives)},
                                penalty={name: float(result.penalties[i, j]) for j, name in enumerate(result.penalty_names)},
                                monitor={name: float(value[i]) for name, value in
                                         monitor_metrics(metrics, cost.monitor).items()},
                                feasible=bool(feasible[i]),
                                unserved_fraction=float(result.unserved_demand[i] / result.total_demand[i]),
                                network_signature=canonical_network(route_list, profile.symmetric_routes),
                            )
                        )
    return rows


def log_evaluation_metrics(rows, logger_list, step, prefix):
    """Log the evaluated cost/goals/penalties, independently of Pareto monitoring."""
    log(
        logger_list,
        ValidationRecord(
            cost=float(np.mean([row["scalar_cost"] for row in rows])),
            **{
                field: {name: float(np.mean([row[field][name] for row in rows])) for name in rows[0][field]}
                for field in ("objective", "penalty", "monitor")
            },
            feasible_fraction=float(np.mean([row["feasible"] for row in rows])),
        ),
        step,
        prefix=prefix,
    )


def log_pareto_metrics(rows, profile, logger_list, step, prefix):
    report = summarize_evaluation(rows, profile.evaluation.hv_reference, int(profile.evaluation.eu_preference_count))
    summary = {k: float("nan") if v is None else v for k, v in report["summary"].items()}
    reference_0, reference_1 = profile.evaluation.hv_reference
    log(
        logger_list,
        ParetoRecord(
            **{
                dict(
                    hv_feasible="hv", eu_conditioned_penalized="eu_conditioned", eu_archive_penalized="eu_archive"
                ).get(k, k): v
                for k, v in summary.items()
            },
            reference_0=reference_0,
            reference_1=reference_1,
        ),
        step,
        prefix=prefix,
    )
    return report


def evaluate_pareto(model, data, profile, cost, device, logger_list, step, *, seed_networks=None):
    """Fixed held-out cities/preferences/action seeds, isolated from training RNG."""
    for mode in ("greedy", "stochastic"):
        started = time.monotonic()
        rows = evaluate_preferences(
            model,
            data,
            profile,
            cost,
            device,
            mode=mode,
            preference_count=int(profile.evaluation.pareto_preference_count),
            stochastic_samples=int(profile.evaluation.pareto_stochastic_samples),
            seed_networks=seed_networks,
        )
        log_pareto_metrics(rows, profile, logger_list, step, f"pareto/{mode}")
        log(logger_list, ParetoRecord(evaluation_seconds=time.monotonic() - started), step, prefix=f"pareto/{mode}")
    flush(logger_list)

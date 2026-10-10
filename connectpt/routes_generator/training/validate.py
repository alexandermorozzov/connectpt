"""Validation and preference-grid evaluation above the shared policy evaluator."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ..common.logging import flush, log
from ..policy.predict import evaluate_policy
from ..policy.rollout import bind_env
from ..env.state import RouteGenBatchState
from ..env.base import EnvParams,EnvironmentBinding
from ..objectives.pareto import canonical_network, summarize_evaluation
from ..objectives.metrics import evaluate_network
from .records import ParetoRecord, ValidationRecord


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

    def batches(self, progress):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, policy, logger_list, step, progress):
        self.progress = progress
        collected, results = [], []
        for graphs, seeds, route_count in self.batches(progress):
            self.env.set_route_count(route_count)
            self.env.reset(graphs.to(self.device), self.cost.get_weights(), seeds, normalize=False)
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
                monitor={
                    name: float(metrics[key].mean())
                    for name, key in (
                        ("ATT", "ATT"),
                        ("RTT", "RTT"),
                        ("WMC", "median_connectivity_weighted"),
                        ("d0", "$d_0$"),
                        ("d1", "$d_1$"),
                        ("d2", "$d_2$"),
                        ("d_un", "$d_{un}$"),
                        ("n_disconnected", "# disconnected node pairs"),
                    )
                    if name in self.cost.monitor
                },
                feasible_fraction=float(torch.cat([~r.are_constraints_violated() for r in results]).float().mean()),
            ),
            step,
        )
        return float(metrics["cost"].mean())

    def monitor(self, policy, logger_list, step, *, final=False):
        raise NotImplementedError


class ConstructionValidation(Validation):
    def batches(self, progress):
        for count in self.data.n_routes:
            for graphs in self.data.validation_loader:
                yield graphs, None, count

    def monitor(self, policy, logger_list, step, *, final=False):
        params = self.profile.evaluation
        if params.pareto_enabled and step != self.last_monitor_step and (final or step % params.pareto_period == 0):
            evaluate_pareto(
                policy,
                self.data.validation_loader.dataset,
                self.profile,
                self.cost,
                self.device,
                logger_list,
                step,
            )
            self.last_monitor_step = step


class EditValidation(Validation):
    def batches(self, progress):
        for chunk in self.data.monitor_indices.split(self.data.batch_size):
            graphs, seeds = self.data.module.make_batch(chunk)
            yield graphs, seeds, seeds.shape[1]

    def monitor(self, policy, logger_list, step, *, final=False):
        params = self.profile.evaluation
        if params.pareto_enabled and step != self.last_monitor_step and (final or step % params.pareto_period == 0):
            graphs, seeds = self.data.module.make_batch(self.data.monitor_indices)
            evaluate_pareto(
                policy,
                graphs.to_data_list(),
                self.profile,
                self.cost,
                self.device,
                logger_list,
                step,
                seed_networks=seeds,
            )
            self.last_monitor_step = step


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
                    template = RouteGenBatchState(
                        Batch.from_data_list(chunk).to(device),
                        fixed_cost.constants,
                        int(profile.env.n_routes),
                        int(profile.env.min_route_len),
                        int(profile.env.max_route_len),
                        preferences=fixed_cost.get_weights(),
                    )
                    env = bind_env(template, model, fixed_cost, normalize=False,
                                   binding=EnvironmentBinding(profile.env_factory,profile.env))
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
                                att_minutes=float(result.mean_demand_time[i] / 60),
                                rtt_minutes=float(result.total_route_time[i] / 60),
                                objective_0=float(pure[i, 0]),
                                objective_1=float(pure[i, 1]),
                                penalized_0=float(penalized[i, 0]),
                                penalized_1=float(penalized[i, 1]),
                                scalar_cost=float(result.cost[i]),
                                objective={name: float(result.goals[i, j]) for j, name in enumerate(cost.objectives)},
                                penalty={name: float(result.penalties[i, j]) for j, name in enumerate(result.penalty_names)},
                                monitor={
                                    name: float(metrics[key][i])
                                    for name, key in (
                                        ("ATT", "ATT"), ("RTT", "RTT"), ("WMC", "median_connectivity_weighted"),
                                        ("d0", "$d_0$"), ("d1", "$d_1$"), ("d2", "$d_2$"),
                                        ("d_un", "$d_{un}$"), ("n_disconnected", "# disconnected node pairs"),
                                    )
                                    if name in cost.monitor
                                },
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

"""Validation and preference-grid evaluation above the shared policy evaluator."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ..run.artifacts import ArtifactStore
from ..run.logger import flush, log
from ..policy.generate import evaluate_policy
from ..policy.rollout import bind_env
from ..env.state import RouteGenBatchState
from ..env.construction import EnvParams
from ..objectives.pareto import canonical_network, summarize_evaluation
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


class Validation:
    def __init__(self, data, cost, env, profile, output, device):
        self.data, self.cost, self.env = data, cost, env
        self.profile, self.output, self.device = profile, output, device
        self.last_monitor_step = None

    def batches(self, progress):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, policy, logger_list, step, progress):
        self.progress = progress
        collected, results = [], []
        for graphs, seeds, route_count in self.batches(progress):
            from dataclasses import replace

            self.env.params = replace(self.env.params, n_routes=route_count)
            self.env.reset(graphs.to(self.device), self.cost.get_weights(), seeds, normalize=False)
            metrics, routes = evaluate_policy(
                self.env,
                policy,
                graphs.to(self.device),
                self.cost.get_weights(),
                self.profile.evaluation.validation_mode,
                1,
                seed_networks=seeds,
            )
            collected.append(metrics)
            results.append(self.cost(self.env.state))
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
                    for i, name in enumerate(("constraints", "adjustment"))
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
    def __init__(self, data, cost, env, profile, output, device):
        super().__init__(data, cost, env, profile, output, device)
        if profile.evaluation.pareto_enabled:
            ArtifactStore(output).save_json(
                dict(
                    indices=data.validation_indices,
                    seed=profile.evaluation.seed,
                    reference=profile.evaluation.hv_reference,
                    split="held-out validation; independent test reserved for final evaluation",
                ),
                "pareto/validation_protocol.json",
            )

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
                self.output,
                self.device,
                logger_list,
                step,
            )
            self.last_monitor_step = step


class EditValidation(Validation):
    def batches(self, progress):
        _, indices, _ = self.data.curriculum.get(progress)
        for chunk in indices.split(self.data.batch_size):
            graphs, seeds = self.data.module.make_batch(chunk)
            yield graphs, seeds, seeds.shape[1]

    def monitor(self, policy, logger_list, step, *, final=False):
        params = self.profile.evaluation
        if params.pareto_enabled and step != self.last_monitor_step and (final or step % params.pareto_period == 0):
            _, indices, _ = self.data.curriculum.get(self.progress)
            graphs, seeds = self.data.module.make_batch(indices)
            evaluate_pareto(
                policy,
                graphs.to_data_list(),
                self.profile,
                self.cost,
                self.output,
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
    output,
    device,
    *,
    checkpoint="last",
    mode="greedy",
    preference_count=None,
    stochastic_samples=None,
    save_artifacts=True,
    seed_networks=None,
):
    artifacts = ArtifactStore(output)
    rows, networks = [], []
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
                    env = bind_env(template, model, fixed_cost, normalize=False)
                    env.params = profile.env
                    _, routes = evaluate_policy(
                        env,
                        model,
                        loader,
                        fixed_cost.get_weights(),
                        "greedy" if mode == "greedy" else "sample",
                        1,
                        seed_networks=seeds,
                    )
                    state = RouteGenBatchState(
                        Batch.from_data_list(chunk).to(device),
                        fixed_cost.constants,
                        int(profile.env.n_routes),
                        int(profile.env.min_route_len),
                        int(profile.env.max_route_len),
                        preferences=fixed_cost.get_weights(),
                    )
                    state.add_new_routes(routes.to(device))
                    if seeds is not None:
                        state.set_adjustment_seed(seeds)
                    result = fixed_cost(state)
                    from ..objectives.preferences import GOAL_NAMES

                    columns = [GOAL_NAMES.index(name) for name in fixed_cost.objectives]
                    pure = fixed_cost.get_cost_components(state, result, include_constraint=False)[:, columns]
                    penalized = fixed_cost.get_cost_components(state, result)[:, columns]
                    feasible = ~result.are_constraints_violated()
                    for i in range(len(chunk)):
                        route_list = [r.cpu().tolist() for r in state.routes[i]]
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
                                feasible=bool(feasible[i]),
                                unserved_fraction=float(result.unserved_demand[i] / result.total_demand[i]),
                                network_signature=canonical_network(route_list, profile.symmetric_routes),
                            )
                        )
                        if save_artifacts:
                            networks.append(
                                dict(graph=start + i, preference=float(preference), sample=sample, routes=route_list)
                            )
    suffix = "" if checkpoint == "last" and mode == "greedy" else f"_{checkpoint}_{mode}"
    if save_artifacts:
        artifacts.save_json(
            {
                "rows": rows,
                "checkpoint": checkpoint,
                "mode": mode,
                "samples_per_preference": samples,
                "hv_reference": list(profile.evaluation.hv_reference),
                "objectives": list(cost.objectives),
                "eu_preference_count": int(profile.evaluation.eu_preference_count),
                "protocol": dict(
                    graph_count=len(data),
                    preference_count=count,
                    seed=int(profile.evaluation.seed),
                    n_routes=int(profile.env.n_routes),
                    min_route_len=int(profile.env.min_route_len),
                    max_route_len=int(profile.env.max_route_len),
                ),
                "mean_cost": float(np.mean([r["scalar_cost"] for r in rows])),
            },
            f"test_evaluation{suffix}.json",
        )
        artifacts.save_json(networks, f"test_routes{suffix}.json")
    return rows


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


def evaluate_pareto(model, data, profile, cost, output, device, logger_list, step, *, seed_networks=None):
    """Fixed held-out cities/preferences/action seeds, isolated from training RNG."""
    artifacts = ArtifactStore(output)
    for mode in ("greedy", "stochastic"):
        started = time.monotonic()
        rows = evaluate_preferences(
            model,
            data,
            profile,
            cost,
            output,
            device,
            mode=mode,
            preference_count=int(profile.evaluation.pareto_preference_count),
            stochastic_samples=int(profile.evaluation.pareto_stochastic_samples),
            save_artifacts=False,
            seed_networks=seed_networks,
        )
        report = log_pareto_metrics(rows, profile, logger_list, step, f"pareto/{mode}")
        report["evaluation_seconds"] = time.monotonic() - started
        log(logger_list, ParetoRecord(evaluation_seconds=report["evaluation_seconds"]), step, prefix=f"pareto/{mode}")
        example = [r for r in rows if r["graph"] == 0 and r["sample"] == 0]
        report["example_front"] = example
        artifacts.save_json(report, f"pareto/iter{step:05d}_{mode}.json")
    flush(logger_list)

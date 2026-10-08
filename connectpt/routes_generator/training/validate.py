"""Construction test evaluation and artifact helpers; training lives in setup_and_train."""
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
    log_pareto_front: bool
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

    def batches(self, progress):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, policy, logger_list, step, progress):
        collected = []
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
        metrics = {key: torch.cat([item[key] for item in collected]) for key in collected[0]}
        log(
            logger_list,
            ValidationRecord(
                cost=metrics["cost"].mean(),
                att=metrics["ATT"].mean(),
                rtt=metrics["RTT"].mean(),
                d_un=metrics["$d_{un}$"].mean(),
                disconnected_pairs=metrics["# disconnected node pairs"].mean(),
                wmc=metrics["median_connectivity_weighted"].mean(),
            ),
            step,
        )
        return float(metrics["cost"].mean())

    def monitor(self, policy, logger_list, step, *, final=False):
        pass


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
        if params.pareto_enabled and (final or step % params.pareto_period == 0):
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


class EditValidation(Validation):
    def batches(self, progress):
        _, indices, _ = self.data.curriculum.get(progress)
        for chunk in indices.split(self.data.batch_size):
            graphs, seeds = self.data.module.make_batch(chunk)
            yield graphs, seeds, seeds.shape[1]


def evaluate_test(
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
                    pure = fixed_cost.get_cost_components(state, result, include_constraint=False)[:, :2]
                    penalized = fixed_cost.get_cost_components(state, result)[:, :2]
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
                                att_norm=float(pure[i, 0]),
                                rtt_norm=float(pure[i, 1]),
                                penalized_att=float(penalized[i, 0]),
                                penalized_rtt=float(penalized[i, 1]),
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
    reference_att, reference_rtt = profile.evaluation.hv_reference
    log(
        logger_list,
        ParetoRecord(**summary, reference_att=reference_att, reference_rtt=reference_rtt),
        step,
        prefix=prefix,
    )
    return report


def evaluate_pareto(model, data, profile, cost, output, device, logger_list, step):
    """Fixed held-out cities/preferences/action seeds, isolated from training RNG."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    artifacts = ArtifactStore(output)
    for mode in ("greedy", "stochastic"):
        started = time.monotonic()
        rows = evaluate_test(
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
        )
        report = log_pareto_metrics(rows, profile, logger_list, step, f"pareto/{mode}")
        report["evaluation_seconds"] = time.monotonic() - started
        log(logger_list, ParetoRecord(evaluation_seconds=report["evaluation_seconds"]), step, prefix=f"pareto/{mode}")
        example = [r for r in rows if r["graph"] == 0 and r["sample"] == 0]
        report["example_front"] = example
        artifacts.save_json(report, f"pareto/iter{step:05d}_{mode}.json")
        if profile.evaluation.log_pareto_front:
            from ..objectives.pareto import nondominated

            fig = Figure(figsize=(5, 4))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            points = np.array([[r["att_norm"], r["rtt_norm"]] for r in example])
            feasible = np.array([r["feasible"] for r in example])
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=[r["preference"] for r in example],
                vmin=0,
                vmax=1,
                cmap="viridis",
                label="Generated networks",
            )
            front = nondominated(points[feasible])
            if len(front):
                front = front[np.argsort(front[:, 0])]
                ax.plot(front[:, 0], front[:, 1], "k.-", label="Feasible Pareto front")
            ax.set(
                xlabel="Normalized ATT", ylabel="Normalized RTT", title=f"Validation city 0, {mode}, iteration {step}"
            )
            ax.legend(loc="best")
            fig.colorbar(scatter, ax=ax, label="Passenger weight alpha")
            fig.tight_layout()
            for logger in logger_list:
                logger.figure(f"pareto/{mode}/front_city_0", fig, step)
    flush(logger_list)

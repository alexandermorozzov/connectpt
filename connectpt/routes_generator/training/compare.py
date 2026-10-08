"""Paired two-objective evaluation on shared construction test instances."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from itertools import combinations, permutations
from omegaconf import OmegaConf

from ..run.artifacts import ArtifactStore
from ..objectives.pareto import hypervolume_2d, coverage, front_metrics


def _run_summary(run):
    """Status, seed and commit from ``run.json``; counters from the last logged
    values in ``history.csv``; the budget from ``config.yaml``."""
    info = json.loads((run / "run.json").read_text())
    config = OmegaConf.load(run / "config.yaml")
    history = pd.read_csv(run / "history.csv", float_precision="round_trip")
    steps = history.dropna(subset=["time/total_timesteps"])
    return dict(
        status=info["status"],
        seed=info["seed"],
        commit=info["commit"],
        iterations=int(steps["step"].iloc[-1]) + 1,
        training_seconds=float(steps["elapsed seconds"].iloc[-1]),
        total_active_steps=int(steps["time/total_timesteps"].iloc[-1]),
        max_seconds=config.train_loop.max_seconds,
        preference_reset_policy="episode" if config.train_loop.resample_preferences_on_terminal else "batch",
    )


def compare_runs(root, names=("ppo", "d3po"), *, checkpoint="last", mode="greedy", budget_mode="transitions"):
    root = Path(root)
    evaluations, statuses, metadata = {}, {}, {}
    for name in names:
        run = root / "runs" / name
        suffix = "" if checkpoint == "last" and mode == "greedy" else f"_{checkpoint}_{mode}"
        metadata[name] = json.loads((run / f"test_evaluation{suffix}.json").read_text())
        evaluations[name] = metadata[name]["rows"]
        statuses[name] = _run_summary(run)
    keys = [{(r["graph"], r["preference"], r.get("sample", 0)) for r in rows} for rows in evaluations.values()]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError("Evaluation graph/preference grids differ")
    protocols = [m.get("protocol") for m in metadata.values()]
    if any(v != protocols[0] for v in protocols):
        raise ValueError("Evaluation protocol differs")
    if any(s["status"] not in ("iterations", "time_budget") for s in statuses.values()):
        raise ValueError("All runs must complete before comparison")
    if budget_mode == "transitions":
        if len({s["total_active_steps"] for s in statuses.values()}) != 1:
            raise ValueError("Training interaction budgets differ")
    elif budget_mode == "wall_time":
        limits = {s["max_seconds"] for s in statuses.values()}
        if len(limits) != 1 or None in limits or next(iter(limits)) <= 0:
            raise ValueError("Training time budgets differ or are missing")
        if any(s["status"] != "time_budget" for s in statuses.values()):
            raise ValueError("All runs must reach the training time budget")
    else:
        raise ValueError(f"Unknown budget mode: {budget_mode}")
    # Same seed and code -> the same initial actor (was a state-dict hash).
    if len({(s["seed"], s["commit"]) for s in statuses.values()}) != 1:
        raise ValueError("Training seeds or commits differ")
    reset_policies = {s["preference_reset_policy"] for s in statuses.values()}
    if len(reset_policies) != 1:
        raise ValueError("Preference reset policies differ")
    # New runs store fixed bounds. Keep pooled bounds only for legacy artifacts.
    pooled = np.array([[r["penalized_att"], r["penalized_rtt"]] for rows in evaluations.values() for r in rows])
    reference = (pooled.max(0) * 1.1 + 0.01).tolist()
    pure_pool = np.array([[r["att_norm"], r["rtt_norm"]] for rows in evaluations.values() for r in rows])
    pure_reference = (pure_pool.max(0) * 1.1 + 0.01).tolist()
    references = [m.get("hv_reference") for m in metadata.values()]
    fixed_reference = any(r is not None for r in references)
    if fixed_reference:
        if any(r != references[0] for r in references):
            raise ValueError("HV references differ")
        reference = pure_reference = references[0]
    query_counts = {m.get("eu_preference_count", 101) for m in metadata.values()}
    if len(query_counts) != 1:
        raise ValueError("EU query preference counts differ")
    query_count = query_counts.pop()
    per_graph = []
    for graph, sample in sorted({(r["graph"], r.get("sample", 0)) for r in evaluations[names[0]]}):
        points, feasible_points, metrics = {}, {}, {}
        for name in names:
            rows = [r for r in evaluations[name] if r["graph"] == graph and r.get("sample", 0) == sample]
            points[name] = np.array([[r["penalized_att"], r["penalized_rtt"]] for r in rows])
            feasible_points[name] = np.array([[r["att_norm"], r["rtt_norm"]] for r in rows if r["feasible"]]).reshape(
                -1, 2
            )
            feasible = feasible_points[name]
            metrics[name] = front_metrics(rows, reference, query_count)
            if not fixed_reference:
                metrics[name]["hv_feasible"] = hypervolume_2d(feasible, pure_reference)
        per_graph.append(
            dict(
                graph=graph,
                sample=sample,
                metrics=metrics,
                coverage_penalized={
                    f"{a}_dominates_{b}": coverage(points[a], points[b]) for a, b in permutations(names, 2)
                },
                coverage_feasible={
                    f"{a}_dominates_{b}": coverage(feasible_points[a], feasible_points[b])
                    for a, b in permutations(names, 2)
                },
            )
        )
    summary = {}
    for name in names:
        fields = per_graph[0]["metrics"][name]
        summary[name] = {
            field: float(
                np.mean([g["metrics"][name][field] for g in per_graph if g["metrics"][name][field] is not None])
            )
            if any(g["metrics"][name][field] is not None for g in per_graph)
            else None
            for field in fields
        }
        summary[name].update(
            iterations=statuses[name]["iterations"],
            training_seconds=statuses[name]["training_seconds"],
            active_steps=statuses[name]["total_active_steps"],
            graphs_with_feasible_points=len(
                {g["graph"] for g in per_graph if g["metrics"][name]["pareto_points_feasible"] > 0}
            ),
        )
    paired = {}
    for a, b in combinations(names, 2):
        paired[f"{b}_minus_{a}"] = {}
        for field in ("hv_penalized", "eu_conditioned_penalized", "hv_feasible", "feasible_fraction"):
            # Samples are repeated measurements, not independent test graphs.
            differences = np.array(
                [
                    np.mean([g["metrics"][b][field] - g["metrics"][a][field] for g in per_graph if g["graph"] == graph])
                    for graph in sorted({g["graph"] for g in per_graph})
                ]
            )
            rng = np.random.default_rng(1503)
            bootstrap = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(1)
            paired[f"{b}_minus_{a}"][field] = dict(
                difference=float(differences.mean()),
                ci95_graph_bootstrap=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            )
    report = dict(
        summary=summary,
        per_graph=per_graph,
        paired_differences=paired,
        reference_penalized=reference,
        reference_feasible=pure_reference,
        direction="minimize costs; maximize HV and EU (EU is negative cost)",
        hv_reference_rule="Fixed normalized reference from evaluation config; shared across iterations/runs"
        if fixed_reference
        else "Legacy: 1.1 * pooled coordinate maximum + 0.01",
        eu_conditioned_rule="Mean utility of the network generated with the requested preference",
        eu_archive_rule=f"Best linear utility among evaluated preferences, {query_count} uniform query weights",
        coverage_rule="Weak domination of unique points, computed within each graph",
        checkpoint=checkpoint,
        evaluation_mode=mode,
        budget="equal environment transitions" if budget_mode == "transitions" else "equal training wall time",
        time_limit_seconds=statuses[names[0]]["max_seconds"] if budget_mode == "wall_time" else None,
        stochastic_rule="Mean metrics of independent preference-grid draws; no pooling or best-of-N",
        n_training_seeds=1,
    )
    report["preference_reset_policy"] = next(iter(reset_policies))
    suffix = "" if checkpoint == "last" and mode == "greedy" else f"_{checkpoint}_{mode}"
    ArtifactStore(root).save_json(report, f"comparison{suffix}.json")
    return report

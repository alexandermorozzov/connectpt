"""Run NEA/RSL and classical baselines on all MACSA/Mandl scenarios.

The default full grid is 4 scenarios x 7 methods x 10 seeds x 11 alpha values.
Every row, including GA/HH/SA, is scored against the scenario's original route
network and therefore contains ``adj_vs_seed``.  Completed method/seed/scenario
tasks are resumed from their manifests.

Examples:
    python scripts/run_macsa_methods_alpha_seed_sweep.py --profile smoke
    python scripts/run_macsa_methods_alpha_seed_sweep.py --profile full
    python scripts/run_macsa_methods_alpha_seed_sweep.py --profile smoke \
        --scenarios mandl_8 --seeds 0 --methods NEA GA HH SA
    python scripts/run_macsa_methods_alpha_seed_sweep.py --profile full --gpus 0 1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _preselect_single_gpu(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", nargs="+", type=int)
    args, _unknown = parser.parse_known_args(argv)
    if args.gpus is not None and len(args.gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])


_preselect_single_gpu(sys.argv[1:])

import torch
from omegaconf import open_dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from _paper_cli import setup_logging  # noqa: E402

from connectpt.routes_generator.baselines import (  # noqa: E402
    build_ga_cfg,
    build_hh_cfg,
    build_sa_cfg,
    run_ga,
    run_hh,
    run_sa,
)
from connectpt.routes_generator.core import (  # noqa: E402
    ExperimentRunFactory,
    build_experiment,
    load_experiment,
    load_suite,
    seed_everything,
)
from connectpt.routes_generator.data import MACSADataSource, as_route_tensor  # noqa: E402
from connectpt.routes_generator.evaluation import (  # noqa: E402
    full_metric_row,
    select_metrics,
)
from connectpt.routes_generator.paper_experiments.paper_runs import (  # noqa: E402
    paper_dir,
)


CONFIG = "macsa_methods_alpha_seed_sweep"
DEFAULT_ALPHAS = [round(index / 10, 1) for index in range(11)]
DEFAULT_SEEDS = list(range(10))
BASELINE_METHODS = {"GA", "HH", "SA"}


def _slug(value: object) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_" for char in str(value)
    ).strip("_")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _method_specs(cfg) -> dict[str, dict[str, object]]:
    return {
        str(method.label): {
            key: method.get(key)
            for key in ("kind", "bee_sets", "models", "n_bees")
        }
        for method in cfg.methods
    }


def _profile_budgets(cfg, profile: str, args=None) -> dict[str, int]:
    raw = cfg.budgets[profile]
    budgets = {
        "bco_iterations": int(raw.bco_iterations),
        "sa_iterations": int(raw.sa_iterations),
        "hh_iterations": int(raw.hh_iterations),
        "hh_max_repair_iters": int(raw.hh_max_repair_iters),
        "ga_iterations": int(raw.ga_iterations),
        "ga_population_size": int(raw.ga_population_size),
    }
    if args is not None:
        for key in budgets:
            override = getattr(args, key, None)
            if override is not None:
                budgets[key] = int(override)
    return budgets


def _method_budget(method: str, budgets: dict[str, int]) -> dict[str, int]:
    if method not in BASELINE_METHODS:
        iterations = budgets["bco_iterations"]
        return {"n_iterations": iterations, "nominal_eval_budget": iterations}
    if method == "GA":
        iterations = budgets["ga_iterations"]
        population = budgets["ga_population_size"]
        return {
            "n_iterations": iterations,
            "population_size": population,
            "nominal_eval_budget": 2 * population * iterations,
        }
    iterations = budgets[f"{method.lower()}_iterations"]
    out = {"n_iterations": iterations, "nominal_eval_budget": iterations}
    if method == "HH":
        out["max_repair_iters"] = budgets["hh_max_repair_iters"]
    return out


def _experiment_tasks(scenarios, methods, seeds):
    return [
        (str(scenario), str(method), int(seed))
        for method in methods
        for seed in seeds
        for scenario in scenarios
    ]


def _shard_tasks(tasks, worker_index: int, worker_count: int):
    return list(tasks[worker_index::worker_count])


def _aggregate_paths(result_root: Path, worker_tag: str | None = None):
    suffix = f"_{_slug(worker_tag)}" if worker_tag else ""
    return (
        result_root / f"metrics{suffix}_partial.csv",
        result_root / f"metrics{suffix}.csv",
        result_root / f"manifest{suffix}.json",
        result_root / f"run{suffix}.log",
    )


def _task_manifest(
    *, scenario: str, method: str, seed: int, alphas, profile: str,
    method_budget: dict[str, int], spec: dict,
) -> dict:
    return {
        "config": CONFIG,
        "scenario": scenario,
        "method": method,
        "seed": int(seed),
        "alphas": [float(alpha) for alpha in alphas],
        "profile": profile,
        "method_budget": {key: int(value) for key, value in method_budget.items()},
        "spec": {
            key: int(spec[key])
            for key in ("n_routes", "min_route_len", "max_route_len")
        },
        "n_points": len(alphas),
    }


def _load_completed(result_dir: Path, expected: dict) -> pd.DataFrame | None:
    manifest_path = result_dir / "manifest.json"
    metrics_path = result_dir / "metrics.csv"
    all_routes_path = result_dir / "all_routes.pt"
    if not (manifest_path.exists() and metrics_path.exists() and all_routes_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pd.read_csv(metrics_path)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError):
        return None
    if not manifest.get("complete", False):
        return None
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if len(table) != expected["n_points"] or "adj_vs_seed" not in table:
        return None
    if pd.to_numeric(table["adj_vs_seed"], errors="coerce").isna().any():
        return None
    route_files = [REPO_ROOT / path for path in manifest.get("route_files", [])]
    if len(route_files) != expected["n_points"]:
        return None
    if not all(path.is_file() for path in route_files):
        return None
    return table


def _route_items(artifact):
    items = [(key, value) for key, value in artifact.routes.items() if key != "Initial"]
    if artifact.table is None or len(items) != len(artifact.table):
        n_rows = 0 if artifact.table is None else len(artifact.table)
        raise RuntimeError(
            f"Route/table size mismatch for {artifact.run_name}: "
            f"{len(items)} route sets vs {n_rows} rows"
        )
    return [
        (dict(row), routes)
        for row, (_key, routes) in zip(artifact.table.to_dict("records"), items)
    ]


def _save_task(
    points,
    *,
    instance,
    result_dir: Path,
    expected_manifest: dict,
    method_budget: dict[str, int],
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = result_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    all_routes = {
        "Initial": as_route_tensor(instance.init_routes).detach().cpu().clone()
    }
    rows: list[dict] = []
    route_files: list[str] = []

    for row, routes in points:
        alpha = float(row["alpha"])
        route_tensor = as_route_tensor(routes).detach().cpu().clone()
        alpha_name = f"alpha_{alpha:.1f}"
        route_path = routes_dir / f"{alpha_name}.pt"
        _atomic_torch_save(
            {
                "format_version": 1,
                "scenario": expected_manifest["scenario"],
                "method": expected_manifest["method"],
                "seed": expected_manifest["seed"],
                "alpha": alpha,
                "routes": route_tensor,
            },
            route_path,
        )
        all_routes[alpha_name] = route_tensor
        route_files.append(_relative(route_path))

        augmented = dict(row)
        augmented.update(
            scenario=expected_manifest["scenario"],
            city=f"MACSA/{expected_manifest['scenario']}",
            method=expected_manifest["method"],
            seed=expected_manifest["seed"],
            profile=expected_manifest["profile"],
            n_iterations=method_budget["n_iterations"],
            nominal_eval_budget=method_budget["nominal_eval_budget"],
            route_file=_relative(route_path),
        )
        rows.append(augmented)

    expected_alphas = expected_manifest["alphas"]
    actual_alphas = [float(row["alpha"]) for row in rows]
    if actual_alphas != expected_alphas:
        raise RuntimeError(
            f"unexpected alpha order for {expected_manifest['method']}: "
            f"{actual_alphas}, expected {expected_alphas}"
        )

    table = pd.DataFrame(rows)
    if "adj_vs_seed" not in table:
        raise RuntimeError(
            f"{expected_manifest['method']} did not produce adj_vs_seed"
        )
    adjustment = pd.to_numeric(table["adj_vs_seed"], errors="coerce")
    if adjustment.isna().any():
        raise RuntimeError(
            f"{expected_manifest['method']} produced non-finite adj_vs_seed"
        )

    first = [
        name for name in (
            "scenario", "city", "method", "seed", "alpha", "profile",
            "n_iterations", "nominal_eval_budget", "duration_s", "route_file",
        ) if name in table.columns
    ]
    table = table[first + [name for name in table.columns if name not in first]]
    metrics_path = result_dir / "metrics.csv"
    _atomic_csv(table, metrics_path)

    all_routes_path = result_dir / "all_routes.pt"
    _atomic_torch_save(
        {
            "format_version": 1,
            "scenario": expected_manifest["scenario"],
            "method": expected_manifest["method"],
            "seed": expected_manifest["seed"],
            "alphas": expected_alphas,
            "routes": all_routes,
            "coords": instance.coords,
            "street_adj": instance.street_adj,
            "spec": dict(instance.spec),
        },
        all_routes_path,
    )
    manifest = {
        **expected_manifest,
        "complete": True,
        "metrics_file": _relative(metrics_path),
        "all_routes_file": _relative(all_routes_path),
        "route_files": route_files,
    }
    _atomic_json(manifest, result_dir / "manifest.json")
    return table


def _run_bco_task(
    *, scenario: str, method: str, seed: int, alphas, method_spec,
    method_budget: dict[str, int], cpu: bool, namespace: str,
):
    cfg = build_experiment(
        CONFIG,
        bee_sets=str(method_spec["bee_sets"]),
        models=str(method_spec["models"]),
        n_bees=int(method_spec["n_bees"]),
        label=method,
        alpha=alphas,
        adj_weight=0.0,
        n_iterations=method_budget["n_iterations"],
        seed=seed,
        cpu=cpu,
    )
    run_name = "/".join(
        [namespace, _slug(scenario), _slug(method), f"seed_{seed}"]
    )
    with open_dict(cfg):
        cfg.data.source = "macsa"
        cfg.data.scenario = scenario
        cfg.run.name = run_name
        cfg.paths.output_dir = f"artifacts/runs/{run_name}"
    started = time.perf_counter()
    artifact = ExperimentRunFactory.from_cfg(cfg).run()
    duration = time.perf_counter() - started
    points = _route_items(artifact)
    per_point_duration = duration / max(len(points), 1)
    for row, _routes in points:
        row.setdefault("duration_s", per_point_duration)
    return artifact.instance, points


def _set_baseline_alpha(cfg, alpha: float, *, seed: int, cpu: bool):
    with open_dict(cfg):
        cfg.experiment.seed = int(seed)
        cfg.experiment.cpu = bool(cpu)
        cfg.experiment.logdir = None
        kwargs = cfg.experiment.cost_function.kwargs
        kwargs.demand_time_weight = 0.0
        kwargs.route_time_weight = float(alpha)
        kwargs.median_connectivity_weight = float(1.0 - alpha)
        kwargs.connectivity_mode = "median_weighted"
    return cfg


def _build_baseline_cfg(method: str, run_name: str, spec: dict,
                        method_budget: dict[str, int]):
    common = (
        run_name,
        spec["n_routes"],
        spec["min_route_len"],
        spec["max_route_len"],
    )
    if method == "SA":
        return build_sa_cfg(
            *common, n_iterations=method_budget["n_iterations"])
    if method == "GA":
        return build_ga_cfg(
            *common,
            n_iterations=method_budget["n_iterations"],
            population_size=method_budget["population_size"],
        )
    if method == "HH":
        return build_hh_cfg(
            *common,
            n_iterations=method_budget["n_iterations"],
            max_repair_iters=method_budget["max_repair_iters"],
        )
    raise ValueError(f"unknown baseline method {method!r}")


def _call_baseline(method: str, cfg, instance, run_name_scope: str):
    kwargs = {
        "tensors": instance.tensors,
        "run_name_scope": run_name_scope,
        "adjustment_degree_weight": 0.0,
    }
    if method == "SA":
        return run_sa(cfg, instance.init_routes, **kwargs)
    if method == "GA":
        return run_ga(cfg, instance.init_routes, **kwargs)
    if method == "HH":
        return run_hh(cfg, instance.init_routes, **kwargs)
    raise ValueError(f"unknown baseline method {method!r}")


def _run_baseline_task(
    *, scenario: str, method: str, seed: int, alphas,
    method_budget: dict[str, int], cpu: bool, namespace: str, metrics,
):
    instance = MACSADataSource(scenario).load()
    points = []
    for alpha in alphas:
        seed_everything(seed)
        run_name = "/".join(
            [namespace, _slug(scenario), _slug(method), f"seed_{seed}",
             f"alpha_{float(alpha):.1f}"]
        )
        cfg = _build_baseline_cfg(method, run_name, instance.spec, method_budget)
        cfg = _set_baseline_alpha(cfg, float(alpha), seed=seed, cpu=cpu)
        started = time.perf_counter()
        _run_name, raw_metrics, _unserved, routes = _call_baseline(
            method, cfg, instance, f"{namespace}_")
        duration = time.perf_counter() - started
        row = select_metrics(
            full_metric_row(raw_metrics, routes, instance.init_routes), metrics)
        row.update(alpha=float(alpha), duration_s=duration)
        points.append((row, routes))
    return instance, points


def _release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sort_requested_order(table: pd.DataFrame, *, scenarios, methods, seeds, alphas):
    orders = {
        "scenario": {value: index for index, value in enumerate(scenarios)},
        "method": {value: index for index, value in enumerate(methods)},
        "seed": {int(value): index for index, value in enumerate(seeds)},
        "alpha": {
            round(float(value), 12): index for index, value in enumerate(alphas)
        },
    }
    out = table.copy()
    out["_scenario_order"] = out["scenario"].map(orders["scenario"])
    out["_method_order"] = out["method"].map(orders["method"])
    out["_seed_order"] = out["seed"].astype(int).map(orders["seed"])
    out["_alpha_order"] = out["alpha"].astype(float).round(12).map(orders["alpha"])
    order_columns = [
        "_scenario_order", "_method_order", "_seed_order", "_alpha_order"
    ]
    if out[order_columns].isna().any().any():
        raise RuntimeError("metrics contain an unexpected task key")
    return out.sort_values(order_columns, kind="stable").drop(
        columns=order_columns).reset_index(drop=True)


def _worker_command(args, *, gpu: int, worker_index: int,
                    worker_count: int, worker_tag: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile", args.profile,
        "--scenarios", *args.scenarios,
        "--methods", *args.methods,
        "--seeds", *(str(seed) for seed in args.seeds),
        "--alphas", *(str(alpha) for alpha in args.alphas),
        "--suite", args.suite,
        "--suite-smoke", args.suite_smoke,
        "--gpus", str(gpu),
        "--worker-index", str(worker_index),
        "--worker-count", str(worker_count),
        "--worker-tag", worker_tag,
    ]
    for key in (
        "bco_iterations", "sa_iterations", "hh_iterations",
        "hh_max_repair_iters", "ga_iterations", "ga_population_size",
    ):
        value = getattr(args, key)
        if value is not None:
            command.extend([f"--{key.replace('_', '-')}", str(value)])
    if args.no_resume:
        command.append("--no-resume")
    return command


def _merge_worker_tables(worker_csvs, final_csv: Path, *, args) -> pd.DataFrame:
    tables = [pd.read_csv(path) for path in worker_csvs]
    combined = _sort_requested_order(
        pd.concat(tables, ignore_index=True),
        scenarios=args.scenarios,
        methods=args.methods,
        seeds=args.seeds,
        alphas=args.alphas,
    )
    expected_rows = (
        len(args.scenarios) * len(args.methods) * len(args.seeds) * len(args.alphas)
    )
    if len(combined) != expected_rows:
        raise RuntimeError(
            f"GPU workers produced {len(combined)} rows; expected {expected_rows}"
        )
    keys = ["scenario", "method", "seed", "alpha"]
    if combined.duplicated(keys).any():
        raise RuntimeError(f"GPU worker metrics contain duplicate {keys} rows")
    if "adj_vs_seed" not in combined or combined["adj_vs_seed"].isna().any():
        raise RuntimeError("combined metrics are missing adj_vs_seed")
    _atomic_csv(combined, final_csv)
    return combined


def _aggregate_manifest(args, budgets, combined, final_csv, n_tasks):
    return {
        "complete": True,
        "config": CONFIG,
        "profile": args.profile,
        "scenarios": args.scenarios,
        "methods": args.methods,
        "seeds": args.seeds,
        "alphas": args.alphas,
        "budgets": budgets,
        "n_tasks": n_tasks,
        "n_points": len(combined),
        "metrics_file": _relative(final_csv),
        "adj_vs_seed_complete": bool(
            "adj_vs_seed" in combined and not combined["adj_vs_seed"].isna().any()
        ),
    }


def _run_multi_gpu(args, *, result_root: Path, budgets, log) -> None:
    tasks = _experiment_tasks(args.scenarios, args.methods, args.seeds)
    worker_count = min(len(args.gpus), len(tasks))
    workers: list[tuple[int, Path, subprocess.Popen]] = []

    def stop_workers(signum, _frame):
        for _gpu, _csv, process in workers:
            if process.poll() is None:
                process.terminate()
        raise SystemExit(128 + signum)

    old_sigterm = signal.signal(signal.SIGTERM, stop_workers)
    old_sigint = signal.signal(signal.SIGINT, stop_workers)
    try:
        for worker_index, gpu in enumerate(args.gpus[:worker_count]):
            worker_tag = f"gpu{gpu}_part{worker_index + 1}"
            _partial, worker_csv, _manifest, worker_log = _aggregate_paths(
                result_root, worker_tag)
            assigned = _shard_tasks(tasks, worker_index, worker_count)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = _worker_command(
                args,
                gpu=gpu,
                worker_index=worker_index,
                worker_count=worker_count,
                worker_tag=worker_tag,
            )
            log.info(
                "starting GPU worker | physical_gpu=%s tasks=%s log=%s",
                gpu, len(assigned), worker_log,
            )
            process = subprocess.Popen(command, env=env)
            workers.append((gpu, worker_csv, process))

        failures = []
        for gpu, _worker_csv, process in workers:
            return_code = process.wait()
            if return_code:
                failures.append((gpu, return_code))
        if failures:
            raise RuntimeError(f"one or more GPU workers failed: {failures}")
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        for _gpu, _csv, process in workers:
            if process.poll() is None:
                process.terminate()

    partial_csv, final_csv, manifest_path, _log_path = _aggregate_paths(result_root)
    combined = _merge_worker_tables(
        [csv_path for _gpu, csv_path, _process in workers], final_csv, args=args)
    _atomic_json(
        _aggregate_manifest(args, budgets, combined, final_csv, len(tasks)),
        manifest_path,
    )
    if partial_csv.exists():
        partial_csv.unlink()
    log.info("multi-GPU sweep done: %s rows -> %s", len(combined), final_csv)


def main() -> None:
    experiment_cfg = load_experiment(CONFIG)
    method_specs = _method_specs(experiment_cfg)
    default_scenarios = [str(value) for value in experiment_cfg.scenarios]
    default_methods = list(method_specs)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--scenarios", nargs="+", choices=default_scenarios,
        default=default_scenarios,
    )
    parser.add_argument(
        "--methods", nargs="+", choices=default_methods, default=default_methods,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--bco-iterations", type=int, default=None)
    parser.add_argument("--sa-iterations", type=int, default=None)
    parser.add_argument("--hh-iterations", type=int, default=None)
    parser.add_argument("--hh-max-repair-iters", type=int, default=None)
    parser.add_argument("--ga-iterations", type=int, default=None)
    parser.add_argument("--ga-population-size", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--suite", default="suite_rerun")
    parser.add_argument("--suite-smoke", default="suite_rerun_smoke")
    parser.add_argument("--worker-index", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-tag", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        parser.error("every alpha must be between 0 and 1")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("alphas must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    budget_args = (
        args.bco_iterations, args.sa_iterations, args.hh_iterations,
        args.hh_max_repair_iters, args.ga_iterations, args.ga_population_size,
    )
    if any(value is not None and value <= 0 for value in budget_args):
        parser.error("all explicit budgets must be positive")
    if args.gpus is not None:
        if any(gpu < 0 for gpu in args.gpus):
            parser.error("GPU indices must be non-negative")
        if len(set(args.gpus)) != len(args.gpus):
            parser.error("GPU indices must be unique")
    if args.cpu and args.gpus is not None:
        parser.error("--cpu and --gpus cannot be used together")
    worker_values = (args.worker_index, args.worker_count, args.worker_tag)
    if any(value is not None for value in worker_values) and not all(
            value is not None for value in worker_values):
        parser.error("internal worker arguments must be provided together")
    if args.worker_index is not None:
        if args.gpus is None or len(args.gpus) != 1:
            parser.error("a worker requires exactly one physical GPU")
        if args.worker_count <= 0:
            parser.error("worker count must be positive")
        if not 0 <= args.worker_index < args.worker_count:
            parser.error("worker index must be in [0, worker count)")

    if args.gpus is not None and len(args.gpus) == 1:
        if not torch.cuda.is_available():
            parser.error(
                f"physical GPU {args.gpus[0]} is unavailable to this worker"
            )

    budgets = _profile_budgets(experiment_cfg, args.profile, args)
    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    output_dir = paper_dir(suite) or (REPO_ROOT / "artifacts" / "reruns")
    prefix = str(suite.output_prefix or "")
    namespace = f"{prefix}{CONFIG}"
    result_root = output_dir / namespace
    result_root.mkdir(parents=True, exist_ok=True)
    partial_csv, final_csv, manifest_path, log_path = _aggregate_paths(
        result_root, args.worker_tag)
    log = setup_logging(log_path)

    if args.gpus is not None and len(args.gpus) > 1:
        _run_multi_gpu(args, result_root=result_root, budgets=budgets, log=log)
        return

    all_tasks = _experiment_tasks(args.scenarios, args.methods, args.seeds)
    tasks = (
        _shard_tasks(all_tasks, args.worker_index, args.worker_count)
        if args.worker_index is not None else all_tasks
    )
    total_points = len(tasks) * len(args.alphas)
    metrics = [str(value) for value in experiment_cfg.metrics]
    log.info(
        "MACSA methods sweep | profile=%s scenarios=%s methods=%s seeds=%s "
        "alphas=%s tasks=%s/%s points=%s budgets=%s resume=%s",
        args.profile, args.scenarios, args.methods, args.seeds, args.alphas,
        len(tasks), len(all_tasks), total_points, budgets, not args.no_resume,
    )

    tables = []
    for task_index, (scenario, method, seed) in enumerate(tasks, start=1):
        method_budget = _method_budget(method, budgets)
        instance = MACSADataSource(scenario).load()
        expected_manifest = _task_manifest(
            scenario=scenario,
            method=method,
            seed=seed,
            alphas=args.alphas,
            profile=args.profile,
            method_budget=method_budget,
            spec=instance.spec,
        )
        result_dir = result_root / _slug(scenario) / _slug(method) / f"seed_{seed}"
        cached = None if args.no_resume else _load_completed(
            result_dir, expected_manifest)
        if cached is not None:
            tables.append(cached)
            log.info(
                "resume skip | scenario=%s method=%s seed=%s (%s/%s)",
                scenario, method, seed, task_index, len(tasks),
            )
            _atomic_csv(pd.concat(tables, ignore_index=True), partial_csv)
            continue

        log.info(
            "starting | scenario=%s method=%s seed=%s budget=%s (%s/%s)",
            scenario, method, seed, method_budget, task_index, len(tasks),
        )
        if method in BASELINE_METHODS:
            instance, points = _run_baseline_task(
                scenario=scenario,
                method=method,
                seed=seed,
                alphas=args.alphas,
                method_budget=method_budget,
                cpu=args.cpu,
                namespace=namespace,
                metrics=metrics,
            )
        else:
            instance, points = _run_bco_task(
                scenario=scenario,
                method=method,
                seed=seed,
                alphas=args.alphas,
                method_spec=method_specs[method],
                method_budget=method_budget,
                cpu=args.cpu,
                namespace=namespace,
            )
        table = _save_task(
            points,
            instance=instance,
            result_dir=result_dir,
            expected_manifest=expected_manifest,
            method_budget=method_budget,
        )
        tables.append(table)
        _atomic_csv(pd.concat(tables, ignore_index=True), partial_csv)
        log.info(
            "finished | scenario=%s method=%s seed=%s (%s/%s) -> %s",
            scenario, method, seed, task_index, len(tasks), result_dir,
        )
        _release_accelerator_cache()

    combined = _sort_requested_order(
        pd.concat(tables, ignore_index=True),
        scenarios=args.scenarios,
        methods=args.methods,
        seeds=args.seeds,
        alphas=args.alphas,
    )
    expected_rows = len(tasks) * len(args.alphas)
    if len(combined) != expected_rows:
        raise RuntimeError(f"produced {len(combined)} rows; expected {expected_rows}")
    if "adj_vs_seed" not in combined or combined["adj_vs_seed"].isna().any():
        raise RuntimeError("final metrics are missing adj_vs_seed")
    _atomic_csv(combined, final_csv)
    if partial_csv.exists():
        partial_csv.unlink()
    _atomic_json(
        _aggregate_manifest(args, budgets, combined, final_csv, len(tasks)),
        manifest_path,
    )
    log.info("MACSA methods sweep done: %s rows -> %s", len(combined), final_csv)


if __name__ == "__main__":
    main()

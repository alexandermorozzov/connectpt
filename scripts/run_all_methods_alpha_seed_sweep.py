"""Run all requested EA/NEA methods on every benchmark graph.

Full grid:

* graphs: Mandl, Mumford0, Mumford1, Mumford2, Mumford3;
* methods: EA, NEA (GNN + type 2), NEA-Edit,
  NEA-Combined (GNN + trim/extend), RSL-EA;
* alpha: 0.0, 0.1, ..., 1.0;
* seeds: 0, 1, ..., 9;
* adjustment penalty: disabled.

The full iteration budgets are 200/250/300/350/400 for Mandl/Mumford0/.../3.
Every graph/method/seed experiment gets its own metrics table, one route dump
per alpha, a combined route dump, TensorBoard histories, and a completion
manifest. Completed experiments are skipped on restart unless ``--no-resume``
is supplied.

Examples:

    python scripts/run_all_methods_alpha_seed_sweep.py --profile smoke
    python scripts/run_all_methods_alpha_seed_sweep.py --profile full
    python scripts/run_all_methods_alpha_seed_sweep.py --profile full \
        --cities Mandl Mumford0 --seeds 0 1
    python scripts/run_all_methods_alpha_seed_sweep.py --profile full \
        --methods EA NEA-Combined
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Must be set before importing torch. Unsupported MPS operators then fall back
# to CPU instead of aborting a long experiment.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from omegaconf import open_dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from _paper_cli import setup_logging  # noqa: E402

from connectpt.routes_generator.core import (  # noqa: E402
    ExperimentRunFactory,
    build_experiment,
    load_experiment,
    load_suite,
)
from connectpt.routes_generator.data import as_route_tensor  # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (  # noqa: E402
    paper_dir,
)


CONFIG = "all_methods_alpha_seed_sweep"
DEFAULT_CITIES = ["Mandl", "Mumford0", "Mumford1", "Mumford2", "Mumford3"]
DEFAULT_ALPHAS = [round(index / 10, 1) for index in range(11)]
DEFAULT_SEEDS = list(range(10))
SMOKE_ITERATIONS = 2


def _slug(value: object) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_" for char in str(value)
    ).strip("_")


def _method_specs(cfg) -> dict[str, dict[str, object]]:
    return {
        str(method.label): {
            "bee_sets": str(method.bee_sets),
            "models": str(method.models),
            "n_bees": int(method.n_bees),
        }
        for method in cfg.methods
    }


def _iteration_budgets(cfg) -> dict[str, int]:
    return {str(city): int(value) for city, value in cfg.iteration_budgets.items()}


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


def _route_items(artifact):
    """Yield final route sets in the same alpha order as the result table."""
    items = [(key, value) for key, value in artifact.routes.items() if key != "Initial"]
    if artifact.table is None or len(items) != len(artifact.table):
        n_rows = 0 if artifact.table is None else len(artifact.table)
        raise RuntimeError(
            f"Route/table size mismatch for {artifact.run_name}: "
            f"{len(items)} route sets vs {n_rows} rows"
        )
    return zip(artifact.table.to_dict("records"), items)


def _save_experiment_artifacts(
    artifact,
    *,
    result_dir: Path,
    city: str,
    method: str,
    seed: int,
    alphas: list[float],
    n_iterations: int,
) -> pd.DataFrame:
    """Persist every final network and return the augmented metrics table."""
    if artifact.instance is None:
        raise RuntimeError(f"{artifact.run_name} did not retain its benchmark instance")

    result_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = result_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    all_routes = {"Initial": as_route_tensor(artifact.instance.init_routes).clone()}
    route_files: list[str] = []
    rows: list[dict] = []

    for row, (route_key, routes) in _route_items(artifact):
        alpha = float(row["alpha"])
        route_tensor = as_route_tensor(routes).clone()
        alpha_name = f"alpha_{alpha:.1f}"
        route_path = routes_dir / f"{alpha_name}.pt"
        route_payload = {
            "format_version": 1,
            "city": city,
            "method": method,
            "seed": int(seed),
            "alpha": alpha,
            "n_iterations": int(n_iterations),
            "routes": route_tensor,
            "source_key": route_key,
        }
        _atomic_torch_save(route_payload, route_path)
        all_routes[alpha_name] = route_tensor
        route_files.append(_relative(route_path))

        augmented = dict(row)
        augmented.update(
            city=city,
            method=method,
            seed=int(seed),
            n_iterations=int(n_iterations),
            route_file=_relative(route_path),
        )
        rows.append(augmented)

    actual_alphas = [float(row["alpha"]) for row in rows]
    if actual_alphas != [float(alpha) for alpha in alphas]:
        raise RuntimeError(
            f"Unexpected alpha order for {artifact.run_name}: {actual_alphas}"
        )

    all_routes_path = result_dir / "all_routes.pt"
    _atomic_torch_save(
        {
            "format_version": 1,
            "city": city,
            "method": method,
            "seed": int(seed),
            "alphas": [float(alpha) for alpha in alphas],
            "n_iterations": int(n_iterations),
            "routes": all_routes,
            "coords": (
                artifact.instance.coords.detach().cpu()
                if hasattr(artifact.instance.coords, "detach")
                else artifact.instance.coords
            ),
            "street_adj": (
                artifact.instance.street_adj.detach().cpu()
                if hasattr(artifact.instance.street_adj, "detach")
                else artifact.instance.street_adj
            ),
            "spec": dict(artifact.instance.spec),
        },
        all_routes_path,
    )

    table = pd.DataFrame(rows)
    first = [
        name for name in
        ("city", "method", "seed", "alpha", "n_iterations", "route_file")
        if name in table.columns
    ]
    table = table[first + [name for name in table.columns if name not in first]]
    metrics_path = result_dir / "metrics.csv"
    _atomic_csv(table, metrics_path)

    _atomic_json(
        {
            "complete": True,
            "config": CONFIG,
            "city": city,
            "method": method,
            "seed": int(seed),
            "alphas": [float(alpha) for alpha in alphas],
            "n_iterations": int(n_iterations),
            "adj_weight": 0.0,
            "n_points": len(table),
            "metrics_file": _relative(metrics_path),
            "all_routes_file": _relative(all_routes_path),
            "route_files": route_files,
            "tensorboard_dir": _relative(Path(artifact.output_dir) / "tb"),
        },
        result_dir / "manifest.json",
    )
    return table


def _load_completed(
    result_dir: Path,
    *,
    city: str,
    method: str,
    seed: int,
    alphas: list[float],
    n_iterations: int,
) -> pd.DataFrame | None:
    manifest_path = result_dir / "manifest.json"
    metrics_path = result_dir / "metrics.csv"
    all_routes_path = result_dir / "all_routes.pt"
    if not (manifest_path.exists() and metrics_path.exists() and all_routes_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = {
        "complete": True,
        "config": CONFIG,
        "city": city,
        "method": method,
        "seed": int(seed),
        "alphas": [float(alpha) for alpha in alphas],
        "n_iterations": int(n_iterations),
        "adj_weight": 0.0,
        "n_points": len(alphas),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    route_files = [REPO_ROOT / path for path in manifest.get("route_files", [])]
    if len(route_files) != len(alphas) or not all(path.exists() for path in route_files):
        return None
    table = pd.read_csv(metrics_path)
    return table if len(table) == len(alphas) else None


def _release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Do not call torch.mps.empty_cache() between experiments.  On the PyTorch
    # 2.11 MPS runtime used for these reruns it may segfault after an EA sweep;
    # let the allocator reclaim cached unified memory on its own.


def main() -> None:
    experiment_cfg = load_experiment(CONFIG)
    methods = _method_specs(experiment_cfg)
    budgets = _iteration_budgets(experiment_cfg)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["smoke", "full"], default="smoke",
        help="smoke = 2 iterations per point; full = graph-specific budgets",
    )
    parser.add_argument(
        "--cities", nargs="+", choices=list(budgets), default=DEFAULT_CITIES,
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(methods), default=list(methods),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="reproducible seeds (default: 0 ... 9)",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
        help="RTT/WMC alpha grid (default: 0.0 ... 1.0, step 0.1)",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=None,
        help="override every graph budget (primarily for diagnostics)",
    )
    parser.add_argument("--suite", default="suite_rerun")
    parser.add_argument("--suite-smoke", default="suite_rerun_smoke")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="recompute experiments even when their completion manifest exists",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="force CPU instead of automatic CUDA/MPS selection",
    )
    args = parser.parse_args()

    if any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        parser.error("every alpha must be between 0 and 1")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("alphas must be unique")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.n_iterations is not None and args.n_iterations <= 0:
        parser.error("--n-iterations must be positive")

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    output_dir = paper_dir(suite) or (REPO_ROOT / "artifacts" / "reruns")
    prefix = str(suite.output_prefix or "")
    namespace = f"{prefix}{CONFIG}"
    result_root = output_dir / namespace
    result_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(result_root / "run.log")

    total_experiments = len(args.cities) * len(args.methods) * len(args.seeds)
    total_points = total_experiments * len(args.alphas)
    log.info(
        "All-method benchmark sweep | profile=%s | cities=%s | methods=%s | "
        "seeds=%s | alphas=%s | experiments=%s | points=%s | resume=%s",
        args.profile, args.cities, args.methods, args.seeds, args.alphas,
        total_experiments, total_points, not args.no_resume,
    )

    completed_tables: list[pd.DataFrame] = []
    completed_count = 0
    partial_csv = result_root / "metrics_partial.csv"
    final_csv = result_root / "metrics.csv"

    for city in args.cities:
        graph_iterations = (
            args.n_iterations
            if args.n_iterations is not None
            else (SMOKE_ITERATIONS if args.profile == "smoke" else budgets[city])
        )
        for method in args.methods:
            method_spec = methods[method]
            for seed in args.seeds:
                result_dir = result_root / _slug(city) / _slug(method) / f"seed_{seed}"
                cached = None if args.no_resume else _load_completed(
                    result_dir,
                    city=city,
                    method=method,
                    seed=seed,
                    alphas=args.alphas,
                    n_iterations=graph_iterations,
                )
                if cached is not None:
                    completed_tables.append(cached)
                    completed_count += 1
                    log.info(
                        "resume skip | city=%s method=%s seed=%s (%s/%s)",
                        city, method, seed, completed_count, total_experiments,
                    )
                    _atomic_csv(pd.concat(completed_tables, ignore_index=True), partial_csv)
                    continue

                log.info(
                    "starting | city=%s method=%s seed=%s iterations=%s (%s/%s)",
                    city, method, seed, graph_iterations,
                    completed_count + 1, total_experiments,
                )
                cfg = build_experiment(
                    CONFIG,
                    city=city,
                    bee_sets=str(method_spec["bee_sets"]),
                    models=str(method_spec["models"]),
                    n_bees=int(method_spec["n_bees"]),
                    label=method,
                    alpha=args.alphas,
                    adj_weight=0.0,
                    n_iterations=graph_iterations,
                    seed=seed,
                    cpu=args.cpu,
                )
                run_name = "/".join(
                    [namespace, _slug(city), _slug(method), f"seed_{seed}"]
                )
                with open_dict(cfg):
                    cfg.run.name = run_name
                    cfg.paths.output_dir = f"artifacts/runs/{run_name}"

                artifact = ExperimentRunFactory.from_cfg(cfg).run()
                table = _save_experiment_artifacts(
                    artifact,
                    result_dir=result_dir,
                    city=city,
                    method=method,
                    seed=seed,
                    alphas=args.alphas,
                    n_iterations=graph_iterations,
                )
                completed_tables.append(table)
                completed_count += 1
                combined = pd.concat(completed_tables, ignore_index=True)
                _atomic_csv(combined, partial_csv)
                log.info(
                    "finished | city=%s method=%s seed=%s (%s/%s) -> %s",
                    city, method, seed, completed_count, total_experiments, result_dir,
                )
                del artifact, cfg, table
                _release_accelerator_cache()

    combined = pd.concat(completed_tables, ignore_index=True)
    _atomic_csv(combined, final_csv)
    if partial_csv.exists():
        partial_csv.unlink()
    _atomic_json(
        {
            "complete": True,
            "config": CONFIG,
            "profile": args.profile,
            "cities": args.cities,
            "methods": args.methods,
            "seeds": args.seeds,
            "alphas": args.alphas,
            "iteration_budgets": {
                city: (
                    args.n_iterations
                    if args.n_iterations is not None
                    else (SMOKE_ITERATIONS if args.profile == "smoke" else budgets[city])
                )
                for city in args.cities
            },
            "n_experiments": total_experiments,
            "n_points": len(combined),
            "metrics_file": _relative(final_csv),
        },
        result_root / "manifest.json",
    )
    log.info("All-method benchmark sweep done: %s rows -> %s", len(combined), final_csv)


if __name__ == "__main__":
    main()

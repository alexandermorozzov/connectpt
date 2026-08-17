"""Table-5-style baseline optimizer sweep with LC initial routes.

Runs simulated annealing, genetic algorithm, and hyper-heuristic on benchmark
cities using the same LC+realistic seeded initial network path as the seeded
NBCO experiments. Adjustment penalty is off, and the alpha grid defaults to the
Table 5 reduced grid: 0, 0.5, 1.

Examples:
    python scripts/run_table5_baselines.py --profile full --suite suite_rerun
    python scripts/run_table5_baselines.py --profile full --suite suite_rerun --cities Mumford2 Mumford3
    python scripts/run_table5_baselines.py --profile full --budget-mode scaled
    python scripts/run_table5_baselines.py --profile smoke --suite-smoke suite_rerun_smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import setup_logging  # noqa: E402

from connectpt.routes_generator.baselines import (  # noqa: E402
    build_ga_cfg,
    build_hh_cfg,
    build_sa_cfg,
    run_ga,
    run_hh,
    run_sa,
    safe_run_name,
)
from connectpt.routes_generator.core import load_suite  # noqa: E402
from connectpt.routes_generator.core.paths import CFG_DIR  # noqa: E402
from connectpt.routes_generator.data import BenchmarkDataSource  # noqa: E402
from connectpt.routes_generator.evaluation import (  # noqa: E402
    full_metric_row,
    score_fixed_routes,
    select_metrics,
)
from connectpt.routes_generator.reports import (  # noqa: E402
    plot_pareto,
    save_paper_routes,
    save_paper_table,
)
from connectpt.routes_generator.paper_experiments.paper_runs import paper_dir  # noqa: E402


DEFAULT_CITIES = ["Mandl", "Mumford0", "Mumford1", "Mumford2", "Mumford3"]
DEFAULT_ALPHAS = [0.0, 0.5, 1.0]
DEFAULT_METHODS = ["sa", "ga", "hh"]
METRICS = ["WMC", "RTT", "cost", "adj_vs_seed", "d0", "d1", "d2", "d_un"]
STEM = "table5_baselines_lcinit"
ADJ_TARGET = 0.2
BCO_MATCHED_EVAL_BUDGET = 20_000
PAPER_EA_EVAL_BUDGET = 40_000
EVAL_MATCHED_GA_POP_SIZE = 10


def _slug(text: str) -> str:
    return safe_run_name(str(text).lower())


def _load_budgets():
    return OmegaConf.load(CFG_DIR / "search" / "budgets.yaml")


def _budget(budgets, profile: str, key: str, city: str):
    if profile == "full":
        by_city = budgets.get("by_city", {}).get(key, {})
        if city in by_city:
            return by_city[city]
        return budgets.full.default[key]
    return budgets.smoke.default[key]


def _eval_budget_for_mode(budget_mode: str) -> int | None:
    if budget_mode == "eval20k":
        return BCO_MATCHED_EVAL_BUDGET
    if budget_mode == "paper40k":
        return PAPER_EA_EVAL_BUDGET
    if budget_mode == "scaled":
        return None
    raise ValueError(f"unknown budget mode {budget_mode!r}")


def _method_budget(budgets, profile: str, method: str, city: str,
                   budget_mode: str) -> dict[str, int | None]:
    if profile != "full":
        if method == "ga":
            return {
                "n_iterations": int(_budget(budgets, profile, "ga", city)),
                "population_size": int(_budget(budgets, profile, "ga_pop", city)),
                "max_repair_iters": None,
            }
        key = "sa" if method == "sa" else "hh"
        return {
            "n_iterations": int(_budget(budgets, profile, key, city)),
            "population_size": None,
            "max_repair_iters": None,
        }

    eval_budget = _eval_budget_for_mode(budget_mode)
    if eval_budget is None:
        if method == "ga":
            return {
                "n_iterations": int(_budget(budgets, profile, "ga", city)),
                "population_size": int(_budget(budgets, profile, "ga_pop", city)),
                "max_repair_iters": None,
            }
        key = "sa" if method == "sa" else "hh"
        return {
            "n_iterations": int(_budget(budgets, profile, key, city)),
            "population_size": None,
            "max_repair_iters": _hh_max_repair_iters_scaled(budgets, city)
            if method == "hh" else None,
        }

    if method == "ga":
        pop_size = EVAL_MATCHED_GA_POP_SIZE
        return {
            "n_iterations": max(1, int(eval_budget // (2 * pop_size))),
            "population_size": pop_size,
            "max_repair_iters": None,
        }
    return {
        "n_iterations": int(eval_budget),
        "population_size": None,
        "max_repair_iters": int(eval_budget) if method == "hh" else None,
    }


def _hh_max_repair_iters_scaled(budgets, city: str):
    by_city = budgets.get("by_city", {}).get("hh_max_repair_iters", {})
    return by_city.get(city, budgets.get("hh_max_repair_iters_default", None))


def _nominal_eval_budget(method: str, cfg) -> int:
    if method == "sa":
        return int(cfg.alg_args.n_iterations)
    if method == "hh":
        return int(cfg.n_iterations)
    if method == "ga":
        return 2 * int(cfg.population_size) * int(cfg.n_iterations)
    raise ValueError(f"unknown method {method!r}")


def _iteration_budget(method: str, cfg) -> int:
    return int(cfg.alg_args.n_iterations) if method == "sa" else int(cfg.n_iterations)


def _set_alpha(cfg, alpha: float):
    kwargs = cfg.experiment.cost_function.kwargs
    kwargs.demand_time_weight = 0.0
    kwargs.route_time_weight = float(alpha)
    kwargs.median_connectivity_weight = float(1.0 - alpha)
    kwargs.connectivity_mode = "median_weighted"
    return cfg


def _build_method_cfg(method: str, city: str, spec: dict, alpha: float, budgets,
                      profile: str, budget_mode: str):
    run_name = f"{STEM}_{city}_{method}_a{alpha:g}"
    method_budget = _method_budget(budgets, profile, method, city, budget_mode)
    if method == "sa":
        cfg = build_sa_cfg(
            run_name,
            spec["n_routes"],
            spec["min_route_len"],
            spec["max_route_len"],
            n_iterations=int(method_budget["n_iterations"]),
        )
    elif method == "ga":
        cfg = build_ga_cfg(
            run_name,
            spec["n_routes"],
            spec["min_route_len"],
            spec["max_route_len"],
            n_iterations=int(method_budget["n_iterations"]),
            population_size=int(method_budget["population_size"]),
        )
    elif method == "hh":
        cfg = build_hh_cfg(
            run_name,
            spec["n_routes"],
            spec["min_route_len"],
            spec["max_route_len"],
            n_iterations=int(method_budget["n_iterations"]),
            max_repair_iters=method_budget["max_repair_iters"],
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {DEFAULT_METHODS}")
    return _set_alpha(cfg, alpha)


def _run_method(method: str, cfg, init_routes, tensors):
    if method == "sa":
        return run_sa(
            cfg,
            init_routes,
            tensors=tensors,
            run_name_scope=f"{STEM}_",
            adjustment_degree_weight=0.0,
        )
    if method == "ga":
        return run_ga(
            cfg,
            init_routes,
            tensors=tensors,
            run_name_scope=f"{STEM}_",
            adjustment_degree_weight=0.0,
        )
    if method == "hh":
        return run_hh(
            cfg,
            init_routes,
            tensors=tensors,
            run_name_scope=f"{STEM}_",
            adjustment_degree_weight=0.0,
        )
    raise ValueError(f"unknown method {method!r}")


def _partial_paths(out_dir: Path, prefix: str, city: str):
    partial_dir = out_dir / f"{prefix}{STEM}_{city.lower()}_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = partial_dir / "partial_routes"
    routes_dir.mkdir(exist_ok=True)
    return partial_dir / "partial.csv", routes_dir


def _append_partial(path: Path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_city(city: str, *, profile: str, methods: list[str], alphas: list[float],
             out_dir: Path, prefix: str, seed: int, budget_mode: str,
             log) -> pd.DataFrame:
    budgets = _load_budgets()
    inst = BenchmarkDataSource(city=city, seed=seed).load()
    stem = f"{STEM}_{city.lower()}"
    partial_csv, partial_routes_dir = _partial_paths(out_dir, prefix, city)
    routes = {"Initial": inst.init_routes}
    rows = []

    log.info("city=%s | spec=%s | methods=%s | alphas=%s | budget_mode=%s",
             city, inst.spec, methods, alphas, budget_mode)
    for alpha in alphas:
        metrics, scored_init = score_fixed_routes(
            inst.init_routes,
            inst.tensors,
            inst.spec,
            alpha=float(alpha),
        )
        row = select_metrics(full_metric_row(metrics, scored_init, inst.init_routes), METRICS)
        row.update(
            method="Initial",
            alpha=float(alpha),
            adj_target=ADJ_TARGET,
            adj_weight=0.0,
            n_iterations=0,
            nominal_eval_budget=0,
            budget_mode=budget_mode,
            duration_s=0.0,
            run_name=f"{STEM}_{city}_initial_a{alpha:g}",
        )
        rows.append(row)
        _append_partial(partial_csv, row)
        log.info("initial | city=%s alpha=%s row=%s", city, alpha, row)

    for method in methods:
        for alpha in alphas:
            cfg = _build_method_cfg(
                method, city, inst.spec, alpha, budgets, profile, budget_mode)
            budget = _iteration_budget(method, cfg)
            nominal_eval_budget = _nominal_eval_budget(method, cfg)
            log.info(
                "--- city=%s method=%s alpha=%s budget=%s nominal_eval_budget=%s "
                "budget_mode=%s ---",
                city, method.upper(), alpha, budget, nominal_eval_budget,
                budget_mode)
            started = time.time()
            run_name, metrics, _unserved, out_routes = _run_method(
                method, cfg, inst.init_routes, inst.tensors)
            duration_s = time.time() - started

            row = select_metrics(full_metric_row(metrics, out_routes, inst.init_routes), METRICS)
            row.update(
                method=method.upper(),
                alpha=float(alpha),
                adj_target=ADJ_TARGET,
                adj_weight=0.0,
                n_iterations=budget,
                nominal_eval_budget=nominal_eval_budget,
                budget_mode=budget_mode,
                duration_s=round(duration_s, 1),
                run_name=run_name,
            )
            rows.append(row)
            key = f"{method.upper()} a={alpha:g}"
            routes[key] = out_routes
            _append_partial(partial_csv, row)
            torch.save(out_routes, partial_routes_dir / f"{_slug(key)}.pt")
            log.info("done | city=%s method=%s alpha=%s duration=%.1fs row=%s",
                     city, method.upper(), alpha, duration_s, row)

    table = pd.DataFrame(rows)
    save_paper_table(table.round(4), stem, prefix=prefix, out_dir=out_dir)
    save_paper_routes(
        stem,
        routes,
        inst.coords,
        inst.street_adj,
        meta={
            "stem": stem,
            "city": city,
            "init": "LC+realistic",
            "profile": profile,
            "budget_mode": budget_mode,
            "adjustment": "off",
        },
        prefix=prefix,
        out_dir=out_dir,
    )
    fig = plot_pareto(
        table,
        title=f"{city}: SA / GA / HH, LC init",
    )
    fig_path = out_dir / f"{prefix}{stem}_pareto.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info("city=%s outputs: table/routes/pareto -> %s", city, out_dir)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--suite", default="suite_rerun")
    parser.add_argument("--suite-smoke", default="suite_rerun_smoke")
    parser.add_argument("--cities", nargs="*", default=DEFAULT_CITIES)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS)
    parser.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--budget-mode",
        choices=["eval20k", "paper40k", "scaled"],
        default="eval20k",
        help=(
            "full-profile budget scheme: eval20k matches Table-5 BCO "
            "200*10*5*2 candidate evaluations; paper40k matches the "
            "Holliday EA count B*IT*E=40000; scaled uses cfg/search/budgets.yaml"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/reruns")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / f"{prefix}{STEM}_run.log")
    log.info(
        "Table 5 baselines | profile=%s | cities=%s | methods=%s | alphas=%s "
        "| budget_mode=%s",
        args.profile, args.cities, args.methods, args.alphas, args.budget_mode)

    for city in args.cities:
        run_city(
            city,
            profile=args.profile,
            methods=[m.lower() for m in args.methods],
            alphas=[float(a) for a in args.alphas],
            out_dir=out_dir,
            prefix=prefix,
            seed=int(args.seed),
            budget_mode=str(args.budget_mode),
            log=log,
        )
    log.info("Table 5 baselines done.")


if __name__ == "__main__":
    main()

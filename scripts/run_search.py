"""Run a bee-colony search from composable config presets.

    python scripts/run_search.py
    python scripts/run_search.py --methods nea_edit --instances mandl --iters 2
    python scripts/run_search.py --dry-run --methods ea

The defaults are intentionally a small single run. Broader method, instance,
alpha and seed sweeps are selected explicitly on the command line.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from connectpt.routes_generator.run.factory import ExperimentRunFactory
from connectpt.routes_generator.run.config import build_experiment
from connectpt.routes_generator.run.context import LOGS_DIR, RESULTS_DIR, resolve_under_root
from connectpt.routes_generator.run.network import NetworkRun
from connectpt.routes_generator.objectives.metrics import select_metrics
from connectpt.routes_generator.run.artifacts import save_routes

CONFIG = "search"
METHODS = ["ea", "nea", "nea_edit", "nea_combined"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=["nea_combined"], choices=METHODS)
    parser.add_argument("--instances", nargs="*", default=["mandl"])
    parser.add_argument("--alpha", nargs="*", type=float, default=[0.5])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--adj-weight", type=float, default=None)
    parser.add_argument("--adj-target", nargs="*", type=float, default=[None])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="compose and validate the selected run without search")
    parser.add_argument("--no-initial", action="store_true")
    args = parser.parse_args()

    out_dir = resolve_under_root(args.out_dir) if args.out_dir else RESULTS_DIR / args.config
    log_dir = resolve_under_root(args.log_dir) if args.log_dir else LOGS_DIR
    log_path = (
        resolve_under_root(args.log) if args.log else (log_dir / f"{args.config}_{datetime.now():%Y%m%d_%H%M%S}.log")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("connectpt.cli")

    common = {}
    for key, value in (
        ("n_iterations", args.iters),
        ("adj_weight", args.adj_weight),
        ("cpu", args.cpu or None),
        ("runs_dir", args.runs_dir),
        ("weights_dir", args.weights_dir),
    ):
        if value is not None:
            common[key] = value

    log.info(
        "methods=%s instances=%s alpha=%s adj_target=%s seeds=%s | out=%s",
        args.methods,
        args.instances,
        args.alpha,
        args.adj_target,
        args.seeds,
        out_dir,
    )

    all_rows = []
    for name in args.instances:
        rows, seen_initial = [], set()
        csv_path = out_dir / f"{args.config}_{name}.csv"
        for method in args.methods:
            for seed in args.seeds:
                routes = {}
                for alpha in args.alpha:
                    for adj_target in args.adj_target:
                        log.info("%s | %s | seed=%s | alpha=%s adj_target=%s", method, name, seed, alpha, adj_target)
                        cfg = build_experiment(
                            args.config,
                            method=method,
                            instance=name,
                            seed=seed,
                            alpha=alpha,
                            adj_target=adj_target,
                            **common,
                        )
                        artifact = ExperimentRunFactory.from_cfg(cfg).run(dry_run=args.dry_run)
                        if args.dry_run:
                            log.info("dry-run OK: %s", artifact.plan)
                            continue
                        rows.append(artifact.table.assign(instance=name, seed=seed))
                        routes.update(artifact.routes)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        pd.concat(rows, ignore_index=True).round(4).to_csv(csv_path, index=False)
                if args.dry_run:
                    continue

                save_routes(
                    out_dir / f"{args.config}_{name}_{method}_seed{seed}_routes.pt",
                    routes,
                    coords=artifact.instance.coords,
                    street_adj=artifact.instance.street_adj,
                    meta={"instance": name, "method": method, "seed": seed},
                )

                if not args.no_initial and name not in seen_initial:
                    seen_initial.add(name)
                    scorer = NetworkRun(cfg)
                    scorer.setup()
                    for alpha in args.alpha:
                        for adj_target in args.adj_target:
                            row = select_metrics(
                                scorer.table(
                                    artifact.instance,
                                    alpha=alpha,
                                    adj_target=adj_target,
                                    adj_weight=cfg.sweep.adj_weight,
                                ),
                                cfg.metrics,
                            )
                            row.update(method="Initial", alpha=alpha, adj_target=adj_target, instance=name, seed=0)
                            rows.append(pd.DataFrame([row]))
                    pd.concat(rows, ignore_index=True).round(4).to_csv(csv_path, index=False)

        if rows:
            all_rows.extend(rows)

    if all_rows:
        table = pd.concat(all_rows, ignore_index=True).round(4)
        log.info("rows=%s", len(table))
        log.info("\n%s", table.to_string())


if __name__ == "__main__":
    main()

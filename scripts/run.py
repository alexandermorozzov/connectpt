"""Run training or search from a composed configuration.

    python scripts/run.py --config-name runs/train_edit
    python scripts/run.py --config-name runs/train_construction training/algorithm=d3po
    python scripts/run.py --config-name runs/search --methods ea nea --instances mandl

Search flags select a grid of methods, instances, alpha and seeds. Positional
key=value arguments override configuration values for either run type.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import open_dict

from connectpt.routes_generator.run.config import build_experiment, load_experiment
from connectpt.routes_generator.run.const import LOGS_DIR, RESULTS_DIR
from connectpt.routes_generator.run.context import resolve_under_root
from connectpt.routes_generator.run.factory import make_run


METHODS = ["ea", "nea", "nea_edit", "nea_combined"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", "--config", required=True, help="config under cfg/runs/")
    parser.add_argument("--dry-run", action="store_true", help="check configuration and inputs without execution")
    parser.add_argument("--weights-dir", help="root of model checkpoints")
    parser.add_argument("--runs-dir", help="root of run folders")
    parser.add_argument("--log-dir", help="folder for logs")
    parser.add_argument("--log", help="explicit log file")
    parser.add_argument("--cpu", action="store_true", help="use CPU")
    search = parser.add_argument_group("search")
    search.add_argument("--methods", nargs="+", choices=METHODS)
    search.add_argument("--instances", nargs="+")
    search.add_argument("--alpha", nargs="+", type=float)
    search.add_argument("--seeds", nargs="+", type=int)
    search.add_argument("--iters", type=int)
    search.add_argument("--adj-weight", type=float)
    search.add_argument("--adj-target", nargs="+", type=float)
    search.add_argument("--out-dir", help="folder for search tables and routes")
    search.add_argument("--no-initial", action="store_true", help="omit initial-network rows")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. run.seed=3")

    # Keep key=value overrides out of variable-length search flag values.
    single_values = {flag for action in parser._actions if action.nargs is None for flag in action.option_strings}
    tokens = list(sys.argv[1:] if argv is None else argv)
    cli, overrides = [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in single_values and index + 1 < len(tokens):
            cli.extend(tokens[index:index + 2])
            index += 2
            continue
        if not token.startswith("--") and ("=" in token or token.startswith("~")):
            overrides.append(token)
        else:
            cli.append(token)
        index += 1
    args = parser.parse_intermixed_args(cli)
    args.overrides = overrides + args.overrides
    return parser, args


def run_training(args, cfg):
    with open_dict(cfg):
        if args.weights_dir is not None:
            cfg.paths.weights_dir = args.weights_dir
        if args.runs_dir is not None:
            cfg.paths.output_dir = str(Path(args.runs_dir) / cfg.run.name)
        if args.cpu:
            cfg.run.cpu = True
    log = logging.getLogger("connectpt.cli")
    artifact = make_run(cfg).run(dry_run=args.dry_run)
    if args.dry_run:
        log.info("dry-run OK: %s", artifact.metadata)
    else:
        log.info("done: %s -> %s", cfg.run.type, artifact.output_dir)


def run_search(args, base_cfg):
    import pandas as pd
    from connectpt.routes_generator.run.evaluation import evaluate_initial, export_evaluation
    from connectpt.routes_generator.objectives.metrics import select_metrics
    from connectpt.routes_generator.network.io import save_route_artifact

    args.methods = args.methods or [None]
    args.instances = args.instances or [str(base_cfg.instance.name)]
    args.alpha = args.alpha or [float(base_cfg.sweep.alpha)]
    args.seeds = args.seeds or [int(base_cfg.run.seed)]
    args.adj_target = args.adj_target or [base_cfg.sweep.adj_target]
    config_slug = args.config_name.removeprefix("runs/")
    out_dir = resolve_under_root(args.out_dir) if args.out_dir else RESULTS_DIR / config_slug
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
        csv_path = out_dir / f"{config_slug}_{name}.csv"
        for method in args.methods:
            for seed in args.seeds:
                routes = {}
                for alpha in args.alpha:
                    for adj_target in args.adj_target:
                        log.info("%s | %s | seed=%s | alpha=%s adj_target=%s", method, name, seed, alpha, adj_target)
                        cfg = build_experiment(
                            args.config_name,
                            overrides=args.overrides,
                            method=method,
                            instance=name,
                            seed=seed,
                            alpha=alpha,
                            adj_target=adj_target,
                            **common,
                        )
                        run = make_run(cfg)
                        artifact = run.run(dry_run=args.dry_run)
                        if args.dry_run:
                            log.info("dry-run OK: %s", artifact.plan)
                            continue
                        rows.append(artifact.table.assign(instance=name, seed=seed))
                        routes.update(artifact.routes)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        pd.concat(rows, ignore_index=True).round(4).to_csv(csv_path, index=False)
                if args.dry_run:
                    continue

                method_name = method or cfg.run.label.lower().replace("-", "_")
                save_route_artifact(
                    out_dir / f"{config_slug}_{name}_{method_name}_seed{seed}_routes.pt",
                    routes,
                    coords=artifact.instance.coords,
                    street_adj=artifact.instance.street_adj,
                    meta={"instance": name, "method": method_name, "seed": seed},
                )

                if not args.no_initial and name not in seen_initial:
                    seen_initial.add(name)
                    for alpha in args.alpha:
                        for adj_target in args.adj_target:
                            initial_cost = run.cost_obj.with_params(
                                route_time_weight=alpha, median_connectivity_weight=1-alpha,
                                adjustment_degree_target=run.cost_obj.adjustment_degree_target if adj_target is None else adj_target,
                                adjustment_degree_weight=run.cost_obj.adjustment_degree_weight if cfg.sweep.adj_weight is None else cfg.sweep.adj_weight,
                            )
                            row = select_metrics(
                                export_evaluation(evaluate_initial(artifact.instance, initial_cost, device=run.context.device), index=0),
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


def main(argv=None):
    parser, args = parse_args(argv)
    cfg = load_experiment(args.config_name, overrides=args.overrides)
    search_options = (args.methods, args.instances, args.alpha, args.seeds, args.iters,
                      args.adj_weight, args.adj_target, args.out_dir)
    if cfg.run.type != "search" and (args.no_initial or any(value is not None for value in search_options)):
        parser.error("search flags require a search configuration")

    if args.log:
        log_path = resolve_under_root(args.log)
    else:
        log_dir = resolve_under_root(args.log_dir) if args.log_dir else LOGS_DIR
        stem = args.config_name.replace("/", "_")
        log_path = log_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("connectpt.cli")
    log.info("config=%s | dry_run=%s | overrides=%s", args.config_name, args.dry_run, args.overrides or "-")
    try:
        if cfg.run.type == "search":
            run_search(args, cfg)
        else:
            run_training(args, cfg)
    except Exception:
        log.exception("run failed")
        raise
    if not args.dry_run:
        log.info("TensorBoard: use the existing server (--logdir artifacts)")


if __name__ == "__main__":
    main()

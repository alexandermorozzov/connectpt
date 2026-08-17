"""Run the learned edit-operator ablation without adjustment control.

The experiment compares the paper methods EA, RSL-EA, and NEA-Edit at
``alpha = 0.0, 0.1, ..., 1.0`` for 400 BCO iterations with seed 0.  All methods
start from the same seeded benchmark network, use the classical full-route
replacement operator, and differ only in their local edit operator.

By default the full profile runs every city listed in ``suite.cities`` (Mandl
and Mumford0-3 in the paper suite).  Pass ``--cities`` to run any subset:

    python scripts/run_edit_operator_ablation.py --profile full
    python scripts/run_edit_operator_ablation.py --profile full --cities Mumford1
    python scripts/run_edit_operator_ablation.py --profile full --cities Mandl Mumford0
    python scripts/run_edit_operator_ablation.py --profile smoke

Each city produces a metrics CSV, route/history dumps, an RTT-WMC trade-off
figure, and per-iteration TensorBoard summaries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import save_figures, setup_logging  # noqa: E402

from connectpt.routes_generator.core import load_suite  # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (  # noqa: E402
    paper_dir,
    run_batch,
)

CONFIG = "edit_operator_ablation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke = 2 BCO iterations; full = the 400-iteration paper budget",
    )
    parser.add_argument(
        "--suite",
        default="suite",
        help="full-profile suite (use suite_rerun for artifacts/reruns)",
    )
    parser.add_argument(
        "--suite-smoke",
        default="suite_smoke",
        help="smoke-profile suite name",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="benchmark cities to run (default: every city in suite.cities)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="override sweep.n_iterations (the full default is 400)",
    )
    args = parser.parse_args()

    suite_name = args.suite_smoke if args.profile == "smoke" else args.suite
    suite = load_suite(suite_name)
    cfg = load_suite(CONFIG)
    cities = args.cities or list(suite.cities)
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")
    base_stem = str(cfg.output.paper_stem)
    log_stem = (
        f"{base_stem}_{str(cities[0]).lower()}"
        if len(cities) == 1
        else f"{base_stem}_all_benchmarks"
    )

    log = setup_logging(out_dir / f"{prefix}{log_stem}_run.log")
    log.info(
        "Edit-operator ablation | profile=%s | config=%s | cities=%s | out=%s",
        args.profile,
        CONFIG,
        cities,
        out_dir,
    )

    params = (
        {} if args.n_iterations is None
        else {"n_iterations": args.n_iterations}
    )
    for city in cities:
        city = str(city)
        stem = f"{base_stem}_{city.lower()}"
        log.info("--- Edit-operator ablation city=%s ---", city)
        run = run_batch(CONFIG, suite, city=city, **params)
        figures = save_figures(run.figures, out_dir, stem, prefix=prefix)
        log.info("city=%s: CSV -> %s/%s%s.csv", city, out_dir, prefix, stem)
        log.info("city=%s: figures=%s", city, [str(path) for path in figures])
        if run.table is not None:
            log.info("city=%s rows:\n%s", city, run.table.to_string())

    log.info(
        "Edit-operator ablation done. TensorBoard: tensorboard --logdir %s",
        Path("artifacts/runs") / CONFIG,
    )


if __name__ == "__main__":
    main()

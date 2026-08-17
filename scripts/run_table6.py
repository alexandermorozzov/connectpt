"""Table 6 BCO comparison runner.

Runs four ten-bee configurations on the requested benchmark city/cities:
GNN + trim/extend, GNN + type2, classic BCO, and classical type-1 +
trim/extend (New NBCO). The full profile uses 400 BCO iterations. Each per-city
CSV also contains Initial rows and the transfer metrics d0, d1, d2, d_un plus
total cost.

    python scripts/run_table6.py --profile full --suite suite_rerun
    python scripts/run_table6.py --profile smoke
    python scripts/run_table6.py --profile full --cities Mandl Mumford0

Watch it online: tensorboard --logdir artifacts/runs/table6_bco_comparison
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

CONFIG = "table6_bco_comparison"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke = 1-2 BCO iterations; full = the 400-iteration budget",
    )
    parser.add_argument(
        "--suite",
        default="suite",
        help="full-profile suite name (use suite_rerun for artifacts/reruns)",
    )
    parser.add_argument("--suite-smoke", default="suite_smoke")
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="benchmark cities to run (default: the config city)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="override sweep.n_iterations (the YAML default is 400)",
    )
    args = parser.parse_args()

    suite_name = args.suite_smoke if args.profile == "smoke" else args.suite
    suite = load_suite(suite_name)
    cfg = load_suite(CONFIG)
    base_stem = str(cfg.output.paper_stem)
    cities = args.cities or [str(cfg.data.city)]
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")
    log_stem = (
        f"{base_stem}_{cities[0].lower()}"
        if len(cities) == 1
        else f"{base_stem}_all_benchmarks"
    )

    log = setup_logging(out_dir / f"{prefix}{log_stem}_run.log")
    log.info(
        "Table 6 | profile=%s | config=%s | cities=%s | out=%s",
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
        stem = f"{base_stem}_{city.lower()}"
        log.info("--- Table 6 city=%s ---", city)
        run = run_batch(CONFIG, suite, city=city, **params)
        figures = save_figures(run.figures, out_dir, stem, prefix=prefix)
        log.info("city=%s: CSV -> %s/%s%s.csv", city, out_dir, prefix, stem)
        log.info("city=%s: figures=%s", city, [str(p) for p in figures])
        if run.table is not None:
            log.info("city=%s rows:\n%s", city, run.table.to_string())

    log.info(
        "Table 6 done. TensorBoard: tensorboard --logdir %s",
        Path("artifacts/runs") / CONFIG,
    )


if __name__ == "__main__":
    main()

"""Table 3 rerun -- CLI, no Jupyter.

Re-runs the NeuralBCO vs Improved NeuralBCO alpha sweep (config
``experiments/table3_nbco_vs_our``, paper \\label{tab:nbco-vs-our-only-with-init})
across EVERY benchmark city (Mandl, Mumford0-3) at adjustment target 0.3,
alpha {0, 0.5, 1}, 200 BCO iterations, on the SAME library path the notebook uses
(``paper_runs.run_batch`` once per city). The per-city metrics table + route /
history dumps are persisted by the library; each method x alpha point streams to
TensorBoard per BCO iteration and saves incrementally (partial CSV + route dump)
so a crash never loses completed points. This script only loops the cities and
adds file+console logging.

    python scripts/run_table3.py --profile full --suite suite_rerun
    python scripts/run_table3.py --profile smoke                 # fast dry check
    python scripts/run_table3.py --profile full --cities Mandl Mumford0

Watch it online:  tensorboard --logdir artifacts/runs/table3_nbco_vs_our
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import save_figures, setup_logging   # noqa: E402

from connectpt.routes_generator.core import load_suite   # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (       # noqa: E402
    paper_dir, run_batch)

CONFIG = "table3_nbco_vs_our"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                        help="smoke = 1-2 BCO iters (fast check); full = paper budget")
    parser.add_argument("--suite", default="suite",
                        help="full-profile suite name (use suite_rerun to write artifacts/reruns)")
    parser.add_argument("--suite-smoke", default="suite_smoke",
                        help="smoke-profile suite name")
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="override sweep.n_iterations (else the YAML value, 200)")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="override the city list (default: suite.cities)")
    args = parser.parse_args()

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")
    cities = args.cities or list(suite.cities)

    log = setup_logging(out_dir / f"{prefix}{CONFIG}_run.log")
    log.info("Table 3 | profile=%s | config=%s | cities=%s | out=%s",
             args.profile, CONFIG, cities, out_dir)

    params = {} if args.n_iterations is None else {"n_iterations": args.n_iterations}
    for city in cities:
        log.info("--- Table 3 city=%s ---", city)
        run = run_batch(CONFIG, suite, city=city, **params)
        figs = save_figures(run.figures, out_dir, f"{CONFIG}_{city.lower()}", prefix=prefix)
        log.info("city=%s: table + route/history dumps -> %s (prefix %r); figures: %s",
                 city, out_dir, prefix, [str(p) for p in figs])
        if run.table is not None:
            log.info("city=%s rows:\n%s", city, run.table.to_string())
    log.info("Table 3 done. TensorBoard: tensorboard --logdir %s",
             Path("artifacts/runs") / CONFIG)


if __name__ == "__main__":
    main()

"""Table 4 / Figure 4 rerun -- CLI, no Jupyter.

Re-runs the Improved NeuralBCO adjustment-target sweep on Mumford0 (config
``experiments/table4_fig4_our_pareto``, paper \\label{tab:e2c_rtt_median_wmc_mumford0}
/ \\label{fig:e2-mumford0-ourpareto-topdown}): the 2D alpha x adjustment-target
grid, 200 BCO iterations, on the SAME library path the notebook uses
(``paper_runs.run_experiment``). The metrics table + route / history dumps are
persisted by the library; each grid point streams to TensorBoard per BCO
iteration and saves incrementally so a crash never loses completed points. This
script only adds logging + writes the Pareto figure to disk.

    python scripts/run_table4.py --profile full --suite suite_rerun
    python scripts/run_table4.py --profile smoke                 # fast dry check

Watch it online:  tensorboard --logdir artifacts/runs/table4_fig4_our_pareto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import save_figures, setup_logging   # noqa: E402

from connectpt.routes_generator.core import load_experiment, load_suite   # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (       # noqa: E402
    paper_dir, run_experiment)

CONFIG = "table4_fig4_our_pareto"


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
    args = parser.parse_args()

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    stem = load_experiment(CONFIG).output.paper_stem
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")

    log = setup_logging(out_dir / f"{prefix}{stem}_run.log")
    log.info("Table 4 / Figure 4 | profile=%s | config=%s | stem=%s | out=%s",
             args.profile, CONFIG, stem, out_dir)

    params = {} if args.n_iterations is None else {"n_iterations": args.n_iterations}
    run = run_experiment(CONFIG, suite,
                         title="Adjustment-target sweep on Mumford0 (Improved NBCO)",
                         **params)

    figs = save_figures(run.figures, out_dir, stem, prefix=prefix)
    log.info("Table 4 (metrics) + route/history dumps -> %s (stem %r, prefix %r)",
             out_dir, stem, prefix)
    log.info("Figure 4 (Pareto) saved: %s", [str(p) for p in figs])
    log.info("TensorBoard: tensorboard --logdir %s", Path("artifacts/runs") / CONFIG)
    log.info("Table 4 rows:\n%s", run.table.to_string() if run.table is not None else "(none)")


if __name__ == "__main__":
    main()

"""Figure 5 / Table 5 runner -- CLI, no Jupyter.

Re-runs the five operator combinations on Mumford1 (adjustment penalty off) and
produces Table 5 + the Figure 5 Pareto front, on the SAME library path the
notebook uses (``paper_runs.run_batch`` over the ``methods:`` list in
``experiments/table5_fig5_5model``). The combined metrics table + route dumps are
persisted by the library; each method x alpha point is streamed to TensorBoard
per BCO iteration and saved incrementally so a crash never loses completed
points. This script only adds logging + writes the Pareto figure to disk.

Windows (PowerShell):
    .venv\\Scripts\\python.exe scripts\\run_figure5_pareto.py --profile full
    .venv\\Scripts\\python.exe scripts\\run_figure5_pareto.py --profile smoke

Watch it online:  tensorboard --logdir artifacts/runs/table5_fig5_5model
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

CONFIG = "table5_fig5_5model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                        help="smoke = 1-2 BCO iters (fast check); full = paper budget")
    parser.add_argument("--suite", default="suite",
                        help="full-profile suite name (use suite_rerun to write artifacts/reruns)")
    parser.add_argument("--suite-smoke", default="suite_smoke",
                        help="smoke-profile suite name")
    args = parser.parse_args()

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    stem = load_suite(CONFIG).output.paper_stem
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")

    log = setup_logging(out_dir / f"{prefix}{stem}_run.log")
    log.info("Figure 5 / Table 5 | profile=%s | config=%s | stem=%s", args.profile,
             CONFIG, stem)

    run = run_batch(CONFIG, suite)

    figs = save_figures(run.figures, out_dir, stem, prefix=prefix)
    log.info("Table 5 (combined metrics) -> %s (stem %r, prefix %r)",
             out_dir, stem, prefix)
    log.info("Figure 5 saved: %s", [str(p) for p in figs])
    log.info("TensorBoard: tensorboard --logdir %s",
             Path("artifacts/runs") / CONFIG)
    log.info("Table 5 rows:\n%s", run.table.to_string() if run.table is not None else "(none)")


if __name__ == "__main__":
    main()

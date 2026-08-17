"""EKB (Ekaterinburg) case study runner -- CLI, no Jupyter.

Runs NEA-Edit or NEA-Combined on the real EKB network across the alpha grid
{0, 0.5, 1.0} at adjustment target 0.2 (config
``experiments/ekb_case_study``), on the SAME library path the notebook uses
(``paper_runs.run_experiment``): the sweep table (all metrics) + the route
dumps are persisted by the library, and each grid point is streamed to
TensorBoard per BCO iteration and saved incrementally (partial CSV + route
dump) so a crash never loses completed alphas. This script only adds logging +
writes the geo (routes-on-basemap) figure to disk.

    python scripts/run_ekb_case_study.py --method nea-edit --profile full
    python scripts/run_ekb_case_study.py --method nea-combined --profile smoke
    python scripts/run_ekb_case_study.py --method nea-edit --profile full \
        --n-iterations 100

Watch it online:  tensorboard --logdir artifacts/runs/ekb_case_study/tb
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_cli import save_figures, setup_logging   # noqa: E402

from connectpt.routes_generator.core import load_experiment, load_suite   # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (       # noqa: E402
    paper_dir, run_experiment)

CONFIG = "ekb_case_study"


def _method_key(label: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(label).lower()).strip("-")


def _method_specs() -> dict[str, dict[str, object]]:
    cfg = load_experiment(CONFIG)
    return {
        _method_key(method.label): {
            "bee_sets": str(method.bee_sets),
            "models": str(method.models),
            "n_bees": int(method.n_bees),
            "label": str(method.label),
            "stem": f"ekb_{_method_key(method.label).replace('-', '_')}",
        }
        for method in cfg.methods
    }


def main() -> None:
    methods = _method_specs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                        help="smoke = 1-2 BCO iters (fast check); full = paper budget")
    parser.add_argument("--suite", default="suite",
                        help="full-profile suite name (use suite_rerun to write artifacts/reruns)")
    parser.add_argument("--suite-smoke", default="suite_smoke",
                        help="smoke-profile suite name")
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="override sweep.n_iterations (else the YAML value)")
    parser.add_argument("--method", choices=methods, default="nea-combined",
                        help="method to run (default: nea-combined)")
    args = parser.parse_args()

    suite = load_suite(args.suite_smoke if args.profile == "smoke" else args.suite)
    method = methods[args.method]
    stem = method["stem"]
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite) or Path("artifacts/paper_results")

    log = setup_logging(out_dir / f"{prefix}{stem}_run.log")
    log.info("EKB case study | profile=%s | method=%s | config=%s | stem=%s",
             args.profile, args.method, CONFIG, stem)

    params = {} if args.n_iterations is None else {"n_iterations": args.n_iterations}
    run = run_experiment(
        CONFIG,
        suite,
        kind="gis",
        title=f"EKB case study ({method['label']})",
        bee_sets=method["bee_sets"],
        models=method["models"],
        n_bees=method["n_bees"],
        label=method["label"],
        overrides=[f"output.paper_stem={method['stem']}"],
        **params,
    )

    figs = save_figures(run.figures, out_dir, stem, prefix=prefix)
    art = run.artifact
    log.info("metrics table + route dumps -> %s (stem %r, prefix %r)",
             out_dir, stem, prefix)
    log.info("figures saved: %s", [str(p) for p in figs])
    log.info("TensorBoard: tensorboard --logdir %s", Path(art.output_dir) / "tb")
    log.info("sweep rows:\n%s", run.table.to_string() if run.table is not None else "(none)")


if __name__ == "__main__":
    main()

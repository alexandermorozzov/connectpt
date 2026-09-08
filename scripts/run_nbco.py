"""Run any declarative NBCO experiment from its config -- CLI, no Jupyter.

ONE runner for every experiment: the config name says what runs, the output
flags say where everything lands. Batch configs (a ``batch:`` block = several
methods) and single-sweep configs are dispatched automatically, so there is
nothing per-experiment to maintain here.

    python scripts/run_nbco.py table3_nbco_vs_our --cities Mandl Mumford0
    python scripts/run_nbco.py table5_fig5_5model --out-dir artifacts/results/table5_rerun
    python scripts/run_nbco.py ekb_case_study -p bee_sets=classic_trim_extend
    python scripts/run_nbco.py table4_fig4_our_pareto -p n_iterations=200

Configs live in ``connectpt/routes_generator/cfg/experiments/<name>.yaml``. The
library persists the metrics table + route/history dumps and streams
TensorBoard; this script only adds file+console logging, the city loop and
writing the rendered figures to disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from connectpt.routes_generator.core import load_experiment   # noqa: E402
from connectpt.routes_generator.core.paths import (   # noqa: E402
    LOGS_DIR, resolve_under_root,
)
from connectpt.routes_generator.paper_experiments.paper_runs import (   # noqa: E402
    results_dir, run_batch, run_experiment,
)


def setup_logging(log_path: Path) -> logging.Logger:
    """Log INFO+ to both ``log_path`` and stdout, so a detached run is followable
    through the file and, live, through the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [file_handler, console]
    return logging.getLogger("connectpt.cli")


def log_path_for(args) -> Path:
    """``--log`` wins; otherwise ``<log-dir>/<config>_<timestamp>.log``."""
    if args.log:
        return resolve_under_root(args.log)
    log_dir = resolve_under_root(args.log_dir) if args.log_dir else LOGS_DIR
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{args.config}_{stamp}.log"


def save_figures(figures: dict, out_dir: Path, stem: str) -> list:
    """Write each rendered figure to ``<out_dir>/<stem>_<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fig in (figures or {}).items():
        path = out_dir / f"{stem}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
    return saved


def _coerce(text: str):
    """``-p`` values arrive as strings; give them their obvious Python type."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if "," in text:
        return [_coerce(part.strip()) for part in text.split(",")]
    return text


def _params(items) -> dict:
    """``-p n_iterations=200 -p alpha=0.5`` -> ``build_experiment`` kwargs."""
    params = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        params[key.strip()] = _coerce(value.strip())
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",
                        help="experiment config under cfg/experiments/ (e.g. table3_nbco_vs_our)")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="run once per city (default: whatever the config itself targets)")
    parser.add_argument("--out-dir", default=None,
                        help="where tables / route dumps / figures land "
                             "(default: artifacts/results/<config>)")
    parser.add_argument("--runs-dir", default=None,
                        help="root of the per-run scratch dir: partial CSV, route dumps, "
                             "TensorBoard (default: artifacts/runs)")
    parser.add_argument("--weights-dir", default=None,
                        help="root the config's checkpoint paths resolve against "
                             "(default: artifacts/model_weights)")
    parser.add_argument("--log-dir", default=None,
                        help="folder for the run log (default: artifacts/logs)")
    parser.add_argument("--log", default=None,
                        help="explicit log file path (overrides --log-dir)")
    parser.add_argument("--kind", default=None,
                        help="report kind override passed to render_report")
    parser.add_argument("-p", "--param", action="append", metavar="KEY=VALUE",
                        help="config knob forwarded to build_experiment "
                             "(n_iterations, alpha, adj_target, route_len, ...); repeatable")
    args = parser.parse_args()

    params = _params(args.param)
    if args.runs_dir is not None:
        params["runs_dir"] = args.runs_dir
    if args.weights_dir is not None:
        params["weights_dir"] = args.weights_dir

    out_dir = results_dir(args.config, args.out_dir)
    log = setup_logging(log_path_for(args))

    # A `batch:` block means several methods in one config -> run_batch; anything
    # else is a single (possibly 2D) sweep -> run_experiment.
    cfg = load_experiment(args.config)
    is_batch = "batch" in cfg
    runner = run_batch if is_batch else run_experiment

    cities = list(args.cities) if args.cities else [None]
    log.info("config=%s | mode=%s | cities=%s | params=%s | out=%s",
             args.config, "batch" if is_batch else "single",
             cities if args.cities else "(from config)", params or "-", out_dir)

    for city in cities:
        kwargs = dict(params)
        if city is not None:
            kwargs["city"] = city
            log.info("--- %s city=%s ---", args.config, city)
        run = runner(args.config, out_dir=out_dir, kind=args.kind, **kwargs)

        stem = args.config if city is None else f"{args.config}_{city.lower()}"
        figures = save_figures(run.figures, out_dir, stem)
        log.info("table + route/history dumps -> %s; figures: %s",
                 out_dir, [str(p) for p in figures])
        if run.table is not None:
            log.info("rows:\n%s", run.table.to_string())

    log.info("done. TensorBoard: tensorboard --logdir %s",
             Path(args.runs_dir or "artifacts/runs"))


if __name__ == "__main__":
    main()

"""Run any declarative NBCO experiment from its config -- CLI, no Jupyter.

ONE runner for every experiment of the paper: the config name says what runs,
the suite says with which budget and where the results land. Batch configs (a
``batch:`` block = several methods) and single-sweep configs are dispatched
automatically, so there is nothing per-experiment to maintain here.

    python scripts/run_nbco.py table3_nbco_vs_our --suite suite_rerun
    python scripts/run_nbco.py table5_fig5_5model --cities Mumford1
    python scripts/run_nbco.py ekb_case_study --suite suite_smoke
    python scripts/run_nbco.py table4_fig4_our_pareto -p n_iterations=200
    python scripts/run_nbco.py macsa_alpha_sweep_iter1 --no-smoke

Configs live in ``connectpt/routes_generator/cfg/experiments/<name>.yaml``;
suites in the same folder (``suite``, ``suite_rerun``, ``suite_smoke``,
``suite_rerun_smoke``). The library persists the metrics table + route/history
dumps and streams TensorBoard; this script only adds file+console logging, the
city loop and writing the rendered figures to disk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from connectpt.routes_generator.core import load_suite   # noqa: E402
from connectpt.routes_generator.paper_experiments.paper_runs import (   # noqa: E402
    paper_dir, run_batch, run_experiment,
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


def save_figures(figures: dict, out_dir: Path, stem: str, prefix: str = "") -> list:
    """Write each rendered figure to ``<prefix><stem>_<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fig in (figures or {}).items():
        path = out_dir / f"{prefix}{stem}_{name}.png"
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
    parser.add_argument("--suite", default="suite",
                        help="suite config: suite (paper budget) | suite_rerun (-> artifacts/reruns) "
                             "| suite_smoke | suite_rerun_smoke (default: suite)")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="run once per city (default: whatever the config itself targets)")
    parser.add_argument("--log", default=None,
                        help="log file (default: <suite paper dir>/<prefix><config>_run.log)")
    parser.add_argument("--kind", default=None,
                        help="report kind override passed to render_report")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None,
                        help="force/forbid the smoke budget (default: whatever the suite says)")
    parser.add_argument("-p", "--param", action="append", metavar="KEY=VALUE",
                        help="config knob forwarded to build_experiment "
                             "(n_iterations, alpha, adj_target, route_len, ...); repeatable")
    args = parser.parse_args()

    params = _params(args.param)
    suite = load_suite(args.suite)
    prefix = str(suite.output_prefix or "")

    out_dir = paper_dir(suite)
    if out_dir is None:
        raise SystemExit(f"suite {args.suite!r} sets no paper_output_dir -- "
                         f"nowhere to write results")

    log_path = Path(args.log) if args.log else out_dir / f"{prefix}{args.config}_run.log"
    log = setup_logging(log_path)

    # A `batch:` block means several methods in one config -> run_batch; anything
    # else is a single (possibly 2D) sweep -> run_experiment.
    cfg = load_suite(args.config)
    is_batch = "batch" in cfg
    runner = run_batch if is_batch else run_experiment

    cities = list(args.cities) if args.cities else [None]
    log.info("config=%s | mode=%s | suite=%s | cities=%s | params=%s | out=%s",
             args.config, "batch" if is_batch else "single", args.suite,
             cities if args.cities else "(from config)", params or "-", out_dir)

    for city in cities:
        kwargs = dict(params)
        if city is not None:
            kwargs["city"] = city
            log.info("--- %s city=%s ---", args.config, city)
        run = runner(args.config, suite, kind=args.kind, smoke=args.smoke, **kwargs)

        stem = args.config if city is None else f"{args.config}_{city.lower()}"
        figures = save_figures(run.figures, out_dir, stem, prefix=prefix)
        log.info("table + route/history dumps -> %s (prefix %r); figures: %s",
                 out_dir, prefix, [str(p) for p in figures])
        if run.table is not None:
            log.info("rows:\n%s", run.table.to_string())

    log.info("done. TensorBoard: tensorboard --logdir %s",
             Path("artifacts/runs") / args.config)


if __name__ == "__main__":
    main()

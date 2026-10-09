"""Run one config from ``cfg/runs/``: training.

    python scripts/run.py --config-name runs/train                    # edit training
    python scripts/run.py --config-name runs/train_construction algorithm=d3po
    python scripts/run.py --config-name runs/train --dry-run

``--dry-run`` builds every object and checks the config, then stops. Positional
arguments are Hydra overrides, e.g. ``run.seed=3``. Search has its own script,
``run_search.py``: it loops over methods, instances, alpha and seeds.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from connectpt.routes_generator.run.const import CFG_DIR, LOGS_DIR
from connectpt.routes_generator.run.context import resolve_under_root
from connectpt.routes_generator.run.factory import make_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, help="config under cfg/, e.g. runs/train")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the objects and check the config, do not run")
    parser.add_argument("--weights-dir", default=None,
                        help="root the checkpoint paths resolve against "
                             "(default: artifacts/model_weights)")
    parser.add_argument("--runs-dir", default=None,
                        help="root of the run folder <runs-dir>/<run.name> (default: artifacts/runs)")
    parser.add_argument("--log-dir", default=None,
                        help="folder for the run log (default: artifacts/logs)")
    parser.add_argument("--log", default=None,
                        help="explicit log file path (overrides --log-dir)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. run.seed=3")
    args = parser.parse_intermixed_args()

    if args.log:
        log_path = resolve_under_root(args.log)
    else:
        log_dir = resolve_under_root(args.log_dir) if args.log_dir else LOGS_DIR
        stem = args.config_name.replace("/", "_")
        log_path = log_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("connectpt.cli")
    log.info("config=%s | dry_run=%s | overrides=%s",
             args.config_name, args.dry_run, args.overrides or "-")

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    with open_dict(cfg):
        if args.weights_dir is not None:
            cfg.paths.weights_dir = args.weights_dir
        if args.runs_dir is not None:
            cfg.paths.output_dir = str(Path(args.runs_dir) / cfg.run.name)
    log.info("run outputs -> %s", cfg.paths.output_dir)

    try:
        artifact = make_run(cfg).run(dry_run=args.dry_run)
    except Exception:
        log.exception("run failed")
        raise
    if args.dry_run:
        log.info("dry-run OK: %s", artifact.metadata)
    else:
        log.info("done: %s -> %s", cfg.run.type, artifact.output_dir)
        log.info("TensorBoard: use the existing server (--logdir artifacts)")


if __name__ == "__main__":
    main()

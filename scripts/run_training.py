"""Train an edit / trim model from a composed config -- CLI, no Jupyter.

The config name selects what is trained; everything else (dataset, objective,
model, PPO budget) lives in that YAML, so a new training run is a new config,
not a new flag. The output flags say where the checkpoint and the log land.

    python scripts/run_training.py                                  # cfg/training/edit
    python scripts/run_training.py --config-name training/edit_scratch
    python scripts/run_training.py --weights-dir artifacts/model_weights_v2
    python scripts/run_training.py --dry-run                        # build only, no PPO loop
    python scripts/run_training.py run.seed=3                       # Hydra-style override

``--dry-run`` builds the data/model/cost/trainer and validates the wiring
WITHOUT loading the generated dataset or entering the long PPO loop -- the fast
check that the config still composes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from connectpt.routes_generator.core.paths import (CFG_DIR, LOGS_DIR, RUNS_DIR,
                                                   resolve_under_root)
from connectpt.routes_generator.training import EditTrainingRun


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="training/edit",
                        help="config under cfg/ (default: training/edit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build + validate the wiring, skip dataset load and PPO loop")
    parser.add_argument("--weights-dir", default=None,
                        help="root the checkpoint paths resolve against "
                             "(default: artifacts/model_weights)")
    parser.add_argument("--runs-dir", default=None,
                        help="root of the per-run scratch dir: TensorBoard, run outputs "
                             "(default: artifacts/runs)")
    parser.add_argument("--log-dir", default=None,
                        help="folder for the run log (default: artifacts/logs)")
    parser.add_argument("--log", default=None,
                        help="explicit log file path (overrides --log-dir)")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra-style cfg overrides, e.g. run.seed=3")
    args = parser.parse_args()

    stem = args.config_name.replace("/", "_")
    if args.log:
        log_path = resolve_under_root(args.log)
    else:
        log_dir = resolve_under_root(args.log_dir) if args.log_dir else LOGS_DIR
        log_path = log_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = setup_logging(log_path)
    log.info("config=%s | dry_run=%s | overrides=%s",
             args.config_name, args.dry_run, args.overrides or "-")

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    with open_dict(cfg):
        if args.weights_dir is not None:
            cfg.paths.weights_dir = args.weights_dir
        if args.runs_dir is not None:
            runs_root = Path(args.runs_dir)
            cfg.paths.output_dir = str(runs_root / cfg.run.name)
    log.info("weights -> %s | run outputs -> %s",
             cfg.paths.get("weights_dir"), cfg.paths.output_dir)

    artifact = EditTrainingRun(cfg).run(dry_run=args.dry_run)
    if args.dry_run:
        log.info("dry-run OK: %s", artifact.metadata)
    else:
        log.info("training done: checkpoint -> %s", artifact.checkpoint_path)
        log.info("TensorBoard: tensorboard --logdir %s",
                 Path(args.runs_dir) if args.runs_dir else RUNS_DIR)


if __name__ == "__main__":
    main()

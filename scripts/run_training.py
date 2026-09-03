"""Train an edit / trim model from a composed config -- CLI, no Jupyter.

The config name selects what is trained; everything else (dataset, objective,
model, PPO budget) lives in that YAML, so a new training run is a new config,
not a new flag.

    python scripts/run_training.py                                  # cfg/training/edit
    python scripts/run_training.py --config-name training/edit_scratch
    python scripts/run_training.py --config-name training/edit_scratch_smoke
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
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.paths import CFG_DIR
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
    parser.add_argument("--log", default=None,
                        help="log file (default: artifacts/cli_logs/<config>_train.log)")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra-style cfg overrides, e.g. run.seed=3")
    args = parser.parse_args()

    stem = args.config_name.replace("/", "_")
    log_path = Path(args.log) if args.log else Path("artifacts/cli_logs") / f"{stem}_train.log"
    log = setup_logging(log_path)
    log.info("config=%s | dry_run=%s | overrides=%s",
             args.config_name, args.dry_run, args.overrides or "-")

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    artifact = EditTrainingRun(cfg).run(dry_run=args.dry_run)
    if args.dry_run:
        log.info("dry-run OK: %s", artifact.metadata)
    else:
        log.info("training done: checkpoint -> %s", artifact.checkpoint_path)
        log.info("TensorBoard: tensorboard --logdir %s", Path("artifacts/runs"))


if __name__ == "__main__":
    main()

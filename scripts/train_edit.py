"""Train the edit / trim model from a composed config.

    python scripts/train_edit.py                       # full training
    python scripts/train_edit.py --dry-run             # build data/model/cost/trainer only
    python scripts/train_edit.py --config-name train/edit_adj_conditioned
    python scripts/train_edit.py run.seed=3            # CLI override (quick tweak)

``--dry-run`` builds the model/cost/trainer and validates the wiring WITHOUT
loading the generated dataset or running the long PPO loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.training import EditTrainingRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train/edit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra-style cfg overrides")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    artifact = EditTrainingRun(cfg).run(dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run OK: {artifact.metadata}")
    else:
        print(f"training done: checkpoint -> {artifact.checkpoint_path}")


if __name__ == "__main__":
    main()

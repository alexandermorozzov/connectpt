"""Evaluate a checkpoint into structured artifacts (CSV/JSON, no plotting).

    python scripts/run_evaluation.py --dry-run
    python scripts/run_evaluation.py --config-name evaluation/edit_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.evaluation import ModelEvaluationRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="evaluation/edit_eval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    artifact = ModelEvaluationRun(cfg).run(dry_run=args.dry_run)
    print(f"{'dry-run OK' if args.dry_run else 'evaluation done'}: {artifact.metadata}")


if __name__ == "__main__":
    main()

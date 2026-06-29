"""Run a batch of ready experiment configs.

    python scripts/run_experiment_batch.py --config-name experiments/bee_type_comparison/batch --dry-run

The batch config lists ready experiment config names; this script composes the
batch, then ExperimentBatch composes + dispatches each listed run. No experiment
parameters are built in Python -- every run is a complete config on disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core import ExperimentBatch

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="experiments/bee_type_comparison/batch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # compose the batch config (context closed before ExperimentBatch opens its
    # own per-run compose contexts -- avoids nested initialize_config_dir).
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    batch = ExperimentBatch(cfg, cfg_dir=CFG_DIR).run(dry_run=args.dry_run)
    print(f"batch '{batch.name}': {len(batch.artifacts)} runs")
    for art in batch.artifacts:
        tag = art.plan.get("counts") if hasattr(art, "plan") else art.metadata.get("model_class")
        print(f"  - {art.run_name}: {tag}")


if __name__ == "__main__":
    main()

"""Run flexible bee-colony search from a composed config.

    python scripts/run_search.py --dry-run
    python scripts/run_search.py --config-name search/bco_flexible_bees
    python scripts/run_search.py run.seed=3 n_iterations=100   # CLI overrides

``--dry-run`` loads the benchmark config + cost, builds and strict-loads the
construction/edit models, builds the policies + bee operators and the translated
plan -- WITHOUT running the BCO loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import BeeColonySearchRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="search/bco_flexible_bees")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra-style cfg overrides")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    artifact = BeeColonySearchRun(cfg).run(dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run OK: models={artifact.metadata['models_loaded']} "
              f"policies={artifact.metadata['policies']}")
        print(f"plan: {artifact.plan}")
    else:
        print(f"search done -> {artifact.output_dir}")


if __name__ == "__main__":
    main()

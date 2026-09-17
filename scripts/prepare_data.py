"""Generate the training dataset for the edit policy.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --config-name runs/prepare_dataset --force

Builds the copy-tier dataset under ``datasets/<dataset.dataset_dirname>`` described
by the training config. Run once before ``scripts/run_training.py``; an existing
complete dataset is reused.
"""
from __future__ import annotations

import argparse

from connectpt.routes_generator.core import ExperimentRunFactory
from connectpt.routes_generator.training.dataset_build import load_train_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="runs/prepare_dataset")
    parser.add_argument("--force", action="store_true", help="regenerate even if present")
    parser.add_argument("overrides", nargs="*", help="Hydra-style cfg overrides")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.force:
        overrides.append("dataset_build.force_regen=true")
    cfg = load_train_config(args.config_name, overrides=overrides)
    artifact = ExperimentRunFactory.from_cfg(cfg).run()
    ds = artifact.metadata["dataset"]

    print(f"graphs={ds.n_graphs} nodes={ds.raw_n_nodes} routes={ds.target_n_routes} "
          f"len={ds.min_route_len}..{ds.max_route_len}")
    print(f"dataset dir: {ds.new_dataset_dir}")
    seconds = artifact.metadata["seconds_per_graph"]
    if seconds is None:
        print("dataset already complete -- nothing to do")
    else:
        print(f"done: {seconds:.2f} s/graph")


if __name__ == "__main__":
    main()

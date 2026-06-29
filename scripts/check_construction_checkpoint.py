"""Strict-load smoke check for the construction route model.

Run after every refactor commit to prove the old construction checkpoint still
loads straight into ``PathCombiningRouteGenerator`` via ``strict=True``::

    python scripts/check_construction_checkpoint.py
    # -> construction checkpoint strict load: OK

If the checkpoint is absent locally the check is skipped (not failed) with an
explicit message; point it elsewhere with ``--checkpoint path/to/ckpt.pt``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from connectpt.routes_generator.core.checkpoints import CheckpointStore
from connectpt.routes_generator.model_factory import RouteModelFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "model_weights"
    / "inductive_random_graphs_weighted_connectivity.pt"
)
MODEL_CONFIG = "bestsofar_feb2023"  # PathCombiningRouteGenerator


def build_construction_model():
    return RouteModelFactory.build_construction_model_by_name(MODEL_CONFIG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"construction checkpoint check skipped: file not found ({args.checkpoint})")
        return

    model = build_construction_model()
    CheckpointStore.load_model_weights(model, args.checkpoint, strict=True)
    print("construction checkpoint strict load: OK")


if __name__ == "__main__":
    main()

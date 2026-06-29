"""Strict-load smoke check for the edit / trim route model.

Run after every refactor commit to prove the old edit checkpoint still loads
straight into ``TrimPathCombiningRouteGenerator`` via ``strict=True``::

    python scripts/check_edit_checkpoint.py
    # -> edit checkpoint strict load: OK

The model is built through the existing ``build_model_from_cfg`` path (the same
one the notebook uses); a later commit swaps this for ``RouteModelFactory``.
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
    / "improvement"
    / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)
MODEL_CONFIG = "bestsofar_feb2023_trim"  # TrimPathCombiningRouteGenerator, in_edge_dim=18


def build_edit_model():
    return RouteModelFactory.build_edit_model_by_name(MODEL_CONFIG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"edit checkpoint check skipped: file not found ({args.checkpoint})")
        return

    model = build_edit_model()
    CheckpointStore.load_model_weights(model, args.checkpoint, strict=True)
    print("edit checkpoint strict load: OK")


if __name__ == "__main__":
    main()

"""Canonical repo paths for route-generation experiments.

The single, library-level source of truth for the filesystem layout:

* inputs  -- ``datasets/`` at the repo root (raw graphs, benchmark files,
  MACSA scenarios, pretrained weights);
* outputs -- ``artifacts/`` at the repo root (trained weights, route dumps,
  logs, paper tables).

``eval_lib.context`` mirrors these values today; a later commit makes it
delegate here. ``core`` must not import ``eval_lib`` (the arrow points the
other way), so the paths are computed independently from this file's location.
"""
from __future__ import annotations

from pathlib import Path

# connectpt/routes_generator/core/paths.py -> repo root is 3 levels up.
ROOT_DIR = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT_DIR / "connectpt" / "routes_generator" / "cfg"

# --- inputs ---
DATASETS_DIR = ROOT_DIR / "datasets"
BENCHMARK_DIR = DATASETS_DIR / "benchmark"
MACSA_DATA_DIR = DATASETS_DIR / "MACSA_data"

# --- outputs / checkpoints ---
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_WEIGHTS_DIR = ARTIFACTS_DIR / "model_weights"
EDIT_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "improvement"
MODEL_OUTPUTS_DIR = ARTIFACTS_DIR / "lc_improvement_outputs"
OUTPUT_ROUTES_DIR = ARTIFACTS_DIR / "output_routes"

# Pretrained construction (neural-BCO bee) checkpoint that ships with the repo.
CONSTRUCTION_MODEL_WEIGHTS_PATH = (
    MODEL_WEIGHTS_DIR / "inductive_random_graphs_weighted_connectivity.pt"
)
# Preserved edit/improvement checkpoint (the one the experiments load).
EDIT_MODEL_WEIGHTS_PATH = (
    EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)


def ensure_output_dirs() -> None:
    """Create the artifact write targets (inputs under datasets/ must exist)."""
    for d in (MODEL_WEIGHTS_DIR, EDIT_MODEL_WEIGHTS_DIR, MODEL_OUTPUTS_DIR,
              OUTPUT_ROUTES_DIR):
        d.mkdir(parents=True, exist_ok=True)

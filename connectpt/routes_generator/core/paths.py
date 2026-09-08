"""Canonical repo paths for route-generation experiments.

The single, library-level source of truth for the filesystem layout:

* inputs  -- ``datasets/`` at the repo root (raw graphs, benchmark files,
  MACSA scenarios, pretrained weights);
* outputs -- ``artifacts/`` at the repo root (trained weights, route dumps,
  logs, paper tables).

Paths are computed from this file's location, so they hold no matter which
directory the process runs from (the notebooks run from
``examples/route_generator``).
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
# Default output roots. Each is overridable per run: `--out-dir` / `--runs-dir`
# / `--log-dir` on the CLI, `paths.*` in the config.
RESULTS_DIR = ARTIFACTS_DIR / "results"      # tables, route dumps, figures
RUNS_DIR = ARTIFACTS_DIR / "runs"            # per-run scratch: partial CSV, TensorBoard
LOGS_DIR = ARTIFACTS_DIR / "logs"            # run logs
MODEL_WEIGHTS_DIR = ARTIFACTS_DIR / "model_weights"
EDIT_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "improvement"
MODEL_OUTPUTS_DIR = ARTIFACTS_DIR / "lc_improvement_outputs"

# Pretrained construction (neural-BCO bee) checkpoint that ships with the repo.
CONSTRUCTION_MODEL_WEIGHTS_PATH = (
    MODEL_WEIGHTS_DIR / "inductive_random_graphs_weighted_connectivity.pt"
)
# Preserved edit/improvement checkpoint (the one the experiments load).
EDIT_MODEL_WEIGHTS_PATH = (
    EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)


def resolve_under_root(path) -> Path:
    """Resolve a config path against the repo root when it is relative.

    Config output paths (e.g. ``artifacts/runs/...``) are written relative to
    the repo root, so they must NOT be resolved against the process CWD -- the
    notebook runs from ``examples/route_generator``. Absolute paths pass through
    unchanged. Checkpoints resolve against the weights root instead, see
    :func:`resolve_weights_path`.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def resolve_weights_path(path, weights_dir=None) -> Path:
    """Resolve a checkpoint path against the weights root.

    Checkpoint paths in config (``cfg/search/models/*``, ``paths.checkpoint_path``)
    are written RELATIVE to the weights root -- ``paths.weights_dir`` when the cfg
    sets one, else :data:`MODEL_WEIGHTS_DIR`. Absolute paths pass through
    unchanged, so a checkpoint outside the repo is named by its full path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = resolve_under_root(weights_dir) if weights_dir else MODEL_WEIGHTS_DIR
    return root / p


def ensure_output_dirs() -> None:
    """Create the artifact write targets (inputs under datasets/ must exist)."""
    for d in (MODEL_WEIGHTS_DIR, EDIT_MODEL_WEIGHTS_DIR, MODEL_OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

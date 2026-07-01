"""Repo paths for the route-evaluation / training notebooks.

Single source of truth for every filesystem path the notebooks touch:

* inputs  -- ``datasets/`` at the repo root (raw graphs, LC/NX results,
  benchmark text files, MACSA scenarios, pretrained model weights);
* outputs -- ``artifacts/`` at the repo root (trained edit-model weights,
  dumped route sets, training logs).

The helper modules import these names instead of relying on the notebook
namespace. Experiment parameters that the user tunes per run live in the
algorithm YAMLs (``connectpt/routes_generator/cfg``); the unified objective is
read from ``cfg/objective/*.yaml`` via
``connectpt.routes_generator.objectives.load_unified_objective``.
"""
import warnings
from pathlib import Path

# The canonical repo paths now live in the library at
# connectpt.routes_generator.core.paths; this module re-exports the shared ones
# and is kept only as a compatibility surface for the notebook / eval_lib
# helpers. New code should import from core.paths instead.
from connectpt.routes_generator.core import paths as _core_paths

warnings.warn(
    "eval_lib.context is deprecated; use connectpt.routes_generator.core.paths",
    DeprecationWarning,
    stacklevel=2,
)


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "connectpt").exists() and (candidate / "examples").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root from current working directory")


# Shared paths sourced from core.paths (single source of truth).
ROOT_DIR = _core_paths.ROOT_DIR
CFG_DIR = _core_paths.CFG_DIR
DATASETS_DIR = _core_paths.DATASETS_DIR
BENCHMARK_DIR = _core_paths.BENCHMARK_DIR
MACSA_DATA_DIR = _core_paths.MACSA_DATA_DIR
ARTIFACTS_DIR = _core_paths.ARTIFACTS_DIR
MODEL_WEIGHTS_DIR = _core_paths.MODEL_WEIGHTS_DIR
EDIT_MODEL_WEIGHTS_DIR = _core_paths.EDIT_MODEL_WEIGHTS_DIR
MODEL_OUTPUTS_DIR = _core_paths.MODEL_OUTPUTS_DIR
OUTPUT_ROUTES_DIR = _core_paths.OUTPUT_ROUTES_DIR
# pretrained neural-BCO bee model (input checkpoint, ships with the repo)
MODEL_WEIGHTS_PATH = _core_paths.CONSTRUCTION_MODEL_WEIGHTS_PATH
# trained LC edit / improvement model weights -- produced by the training
# notebook, kept under model_weights/improvement/ alongside the other weights.
EDIT_MODEL_WEIGHTS_PATH = (
    EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_rttconn_adj_w10_t02_finetune100.pt"
)

# Make sure the artifact output dirs exist (inputs under datasets/ must already
# be present; only the write targets are created here).
for _artifact_dir in (MODEL_WEIGHTS_DIR, EDIT_MODEL_WEIGHTS_DIR,
                      MODEL_OUTPUTS_DIR, OUTPUT_ROUTES_DIR):
    _artifact_dir.mkdir(parents=True, exist_ok=True)

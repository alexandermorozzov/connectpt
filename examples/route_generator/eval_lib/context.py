"""Repo paths for the route-evaluation / training notebooks.

Single source of truth for every filesystem path the notebooks touch:

* inputs  -- ``datasets/`` at the repo root (raw graphs, LC/NX results,
  benchmark text files, MACSA scenarios, pretrained model weights);
* outputs -- ``artifacts/`` at the repo root (trained edit-model weights,
  dumped route sets, training logs).

The helper modules import these names instead of relying on the notebook
namespace. Experiment parameters that the user tunes per run live in the
algorithm YAMLs (``connectpt/routes_generator/cfg``) and in
:mod:`eval_lib.params`.
"""
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "connectpt").exists() and (candidate / "examples").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root from current working directory")


ROOT_DIR = find_repo_root()
CFG_DIR = ROOT_DIR / "connectpt" / "routes_generator" / "cfg"

# === input data / datasets (repo-root datasets/) ===
# datasets/ holds only input data (graphs / demand / benchmark text files /
# LC-NX results); model checkpoints live under artifacts/ (see below).
DATASETS_DIR = ROOT_DIR / "datasets"
BENCHMARK_DIR = DATASETS_DIR / "benchmark"
MACSA_DATA_DIR = DATASETS_DIR / "MACSA_data"

# === run artifacts + model checkpoints (repo-root artifacts/) ===
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_WEIGHTS_DIR = ARTIFACTS_DIR / "model_weights"
EDIT_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "improvement"
MODEL_OUTPUTS_DIR = ARTIFACTS_DIR / "lc_improvement_outputs"
OUTPUT_ROUTES_DIR = ARTIFACTS_DIR / "output_routes"
# pretrained neural-BCO bee model (input checkpoint, ships with the repo)
MODEL_WEIGHTS_PATH = (
    MODEL_WEIGHTS_DIR / "inductive_random_graphs_weighted_connectivity.pt"
)
# trained LC edit / improvement model weights -- produced by the training
# notebook, kept under model_weights/improvement/ alongside the other weights.
EDIT_MODEL_WEIGHTS_PATH = (
    EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_realistic_scratch_rttwmc_adj_v1.pt"
)

# Make sure the artifact output dirs exist (inputs under datasets/ must already
# be present; only the write targets are created here).
for _artifact_dir in (MODEL_WEIGHTS_DIR, EDIT_MODEL_WEIGHTS_DIR,
                      MODEL_OUTPUTS_DIR, OUTPUT_ROUTES_DIR):
    _artifact_dir.mkdir(parents=True, exist_ok=True)

"""Constants owned by this package."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CFG_DIR = ROOT_DIR / "connectpt" / "routes_generator" / "cfg"

DATASETS_DIR = ROOT_DIR / "datasets"

INSTANCES_DIR = DATASETS_DIR / "instances"
INSTANCE_CONFIG_DIR = CFG_DIR / "instance"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RESULTS_DIR = ARTIFACTS_DIR / "results"      # tables, route dumps, figures

LOGS_DIR = ARTIFACTS_DIR / "logs"            # run logs

MODEL_WEIGHTS_DIR = ARTIFACTS_DIR / "model_weights"

EDIT_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "improvement"

MODEL_OUTPUTS_DIR = ARTIFACTS_DIR / "lc_improvement_outputs"

EDIT_MODEL_WEIGHTS_PATH = (
    EDIT_MODEL_WEIGHTS_DIR / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)

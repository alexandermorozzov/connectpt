"""The paper workflow glue runs each stage's dry-run."""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "paper_workflow", REPO_ROOT / "scripts" / "run_paper_combined_workflow.py")
_wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wf)
run_workflow = _wf.run_workflow

WEIGHTS = REPO_ROOT / "artifacts" / "model_weights"
_HAVE_CKPTS = (
    (WEIGHTS / "improvement" / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt").exists()
    and (WEIGHTS / "inductive_random_graphs_weighted_connectivity.pt").exists()
)


def test_workflow_train_and_report_dry_run():
    # train builds a fresh model (no checkpoint needed); report needs no load
    out = run_workflow(train=True, report=True, dry_run=True)
    assert out["train"].metadata["dry_run"] is True
    assert out["report"].metadata["dry_run"] is True


@pytest.mark.skipif(not _HAVE_CKPTS, reason="model checkpoints not present")
def test_workflow_full_pipeline_dry_run():
    out = run_workflow(train=True, search=True, evaluate=True, report=True, dry_run=True)
    assert set(out) == {"train", "search", "evaluate", "report"}
    assert out["search"].metadata["models_loaded"] == ["construction", "edit"]

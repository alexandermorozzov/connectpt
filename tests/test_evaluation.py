"""Evaluation: pure metric helper, result roundtrip, run dry-run."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.artifacts import ArtifactStore
from connectpt.routes_generator.evaluation import (
    EvaluationResult, MetricComputer, ModelEvaluationRun,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
EDIT_CKPT = (
    REPO_ROOT / "artifacts" / "model_weights" / "improvement"
    / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)


def test_metric_computer_row_from_components():
    comps = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    row = MetricComputer.row_from_components(comps, ("demand", "route", "connectivity"),
                                            total=torch.tensor([6.0, 12.0]))
    assert row["demand_cost"] == 2.0
    assert row["route_cost"] == 3.0
    assert row["connectivity_cost"] == 4.0
    assert row["cost"] == 9.0


def test_evaluation_result_roundtrip(tmp_path):
    import pandas as pd
    store = ArtifactStore(tmp_path)
    res = EvaluationResult(
        per_instance=pd.DataFrame({"instance": [0, 1], "cost": [1.0, 2.0]}),
        summary=pd.DataFrame([{"mean_cost": 1.5}]),
        metadata={"run": "x"},
    )
    res.save(store, "eval_x")
    loaded = EvaluationResult.load(store, "eval_x")
    assert loaded.metadata == {"run": "x"}
    assert list(loaded.per_instance["cost"]) == [1.0, 2.0]
    assert loaded.summary.iloc[0]["mean_cost"] == 1.5


@pytest.mark.skipif(not EDIT_CKPT.exists(), reason="edit checkpoint not present")
def test_model_evaluation_run_dry_run(tmp_path):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="evaluation/edit_eval",
                      overrides=[f"paths.output_dir={tmp_path.as_posix()}"])
    artifact = ModelEvaluationRun(cfg).run(dry_run=True)
    assert artifact.metadata["dry_run"] is True
    assert artifact.metadata["model_class"] == "TrimPathCombiningRouteGenerator"

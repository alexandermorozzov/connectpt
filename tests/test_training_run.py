"""EditTrainingRun wires data/model/cost/trainer; dry-run skips dataset + loop."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.training import EditTrainingRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
EDIT_CKPT = (
    REPO_ROOT / "artifacts" / "model_weights" / "improvement"
    / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
)


def _cfg(overrides=None):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        return compose(config_name="training/edit", overrides=overrides or [])


def test_dry_run_builds_pipeline(tmp_path):
    cfg = _cfg([f"paths.output_dir={tmp_path.as_posix()}"])
    run = EditTrainingRun(cfg)
    artifact = run.run(dry_run=True)

    assert artifact.metadata["dry_run"] is True
    assert artifact.metadata["model_class"] == "TrimPathCombiningRouteGenerator"
    # everything got constructed
    assert type(run.model).__name__ == "TrimPathCombiningRouteGenerator"
    assert run.cost_obj is not None
    assert run.trainer is not None
    # cost was configured from the training-form unified objective (cap)
    assert run.cost_obj.adjustment_degree_objective == "cap"
    assert run.cost_obj.connectivity_mode == "median_weighted"


@pytest.mark.skipif(not EDIT_CKPT.exists(), reason="edit checkpoint not present")
def test_dry_run_warm_start_loads_checkpoint(tmp_path):
    cfg = _cfg([
        f"paths.output_dir={tmp_path.as_posix()}",
        f"++paths.init_checkpoint_path={EDIT_CKPT.as_posix()}",
    ])
    # strict warm-start must succeed (loads straight into the model class)
    EditTrainingRun(cfg).run(dry_run=True)

"""BeeColonySearchRun dry-run loads models/policies/plan; search ⊥ training."""
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import BeeColonySearchRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
WEIGHTS = REPO_ROOT / "artifacts" / "model_weights"
EDIT_CKPT = WEIGHTS / "improvement" / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
CONSTRUCTION_CKPT = WEIGHTS / "inductive_random_graphs_weighted_connectivity.pt"

_HAVE_CKPTS = EDIT_CKPT.exists() and CONSTRUCTION_CKPT.exists()


def test_search_does_not_import_training():
    import connectpt.routes_generator.search  # noqa: F401
    assert not [m for m in sys.modules if "routes_generator.training" in m]


@pytest.mark.skipif(not _HAVE_CKPTS, reason="model checkpoints not present")
def test_search_dry_run(tmp_path):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="search/bco_flexible_bees",
                      overrides=[f"paths.output_dir={tmp_path.as_posix()}"])
    artifact = BeeColonySearchRun(cfg).run(dry_run=True)

    assert artifact.metadata["dry_run"] is True
    assert artifact.metadata["models_loaded"] == ["construction", "edit"]
    assert sorted(artifact.metadata["policies"]) == ["construction", "edit"]
    # the flexible specs translate to per-slot bee groups, keyed by bee name;
    # the two edit-extend specs merge into one slot-5 group (names joined).
    counts = artifact.plan["counts"]
    assert counts["neural_on_construction"] == 4
    assert counts["neural_on_edit_extend_only+neural_on_edit_full"] == 6
    assert counts["neural_on_edit_trim_only"] == 2
    assert counts["compound_trim_then_extend"] == 2
    assert artifact.plan["needs_construction"] and artifact.plan["needs_edit"]

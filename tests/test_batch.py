"""ExperimentBatch iterates ready configs and dispatches by run.type."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core import ExperimentBatch, ExperimentRunFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
WEIGHTS = REPO_ROOT / "artifacts" / "model_weights"
_HAVE_CKPTS = (
    (WEIGHTS / "improvement" / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt").exists()
    and (WEIGHTS / "inductive_random_graphs_weighted_connectivity.pt").exists()
)


def test_run_factory_dispatches_by_type():
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="experiments/bee_type_comparison/00_classic_bco")
    run = ExperimentRunFactory.from_cfg(cfg)
    assert type(run).__name__ == "BeeColonySearchRun"


def test_run_factory_unknown_type_raises():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"run": {"type": "nope"}})
    with pytest.raises(ValueError, match="Unknown run.type"):
        ExperimentRunFactory.from_cfg(cfg)


@pytest.mark.skipif(not _HAVE_CKPTS, reason="model checkpoints not present")
def test_batch_dry_run_all_experiments():
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="experiments/bee_type_comparison/batch")
    batch = ExperimentBatch(cfg, cfg_dir=CFG_DIR).run(dry_run=True)
    assert batch.name == "bee_type_comparison"
    assert len(batch.artifacts) == 7
    # first is classic BCO (no neural models), last is compound (slot 7).
    # counts are keyed by bee name (the plan group's name).
    assert batch.artifacts[0].plan["counts"]["random_mutation"] == 20
    assert batch.artifacts[-1].plan["counts"]["compound_trim_then_construct"] == 20

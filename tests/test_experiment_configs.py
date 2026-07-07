"""bee_type_comparison experiment configs compose + translate to the right plan.

Config-first: each experiment is a standalone composable config (search base +
model-availability group + bee-set group). These assert the composition yields
the expected per-type bee counts + model loading, without running the BCO loop.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import BeeColonySearchRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
WEIGHTS = REPO_ROOT / "artifacts" / "model_weights"
_HAVE_CKPTS = (
    (WEIGHTS / "improvement" / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt").exists()
    and (WEIGHTS / "inductive_random_graphs_weighted_connectivity.pt").exists()
)

# experiment -> (expected non-zero counts keyed by bee name, models loaded)
EXPECTED = {
    "00_classic_bco": ({"random_mutation": 20}, []),
    "01_nbco_construction": ({"neural_construction": 20}, ["construction"]),
    "02_nbco_edit_extend": ({"neural_edit_extend": 20}, ["edit"]),
    "03_nbco_edit_trim": ({"neural_edit_trim": 20}, ["edit"]),
    "04_nbco_edit_full": ({"neural_edit_full": 20}, ["edit"]),
    "05_nbco_construction_plus_edit": ({"neural_construction": 10,
                                        "neural_edit_full": 10},
                                       ["construction", "edit"]),
    "06_nbco_compound_trim_then_construct": ({"compound_trim_then_construct": 20},
                                             ["construction", "edit"]),
}


def _dry(name, tmp_path):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=f"experiments/bee_type_comparison/{name}",
                      overrides=[f"paths.output_dir={tmp_path.as_posix()}"])
    return BeeColonySearchRun(cfg).run(dry_run=True)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_experiment_composes_and_plans(name, tmp_path):
    expected_counts, expected_models = EXPECTED[name]
    if expected_models and not _HAVE_CKPTS:
        pytest.skip("model checkpoints not present")
    art = _dry(name, tmp_path)
    counts = art.plan["counts"]
    for key, val in expected_counts.items():
        assert counts[key] == val, (name, key, counts)
    # all other types are zero
    assert sum(counts.values()) == sum(expected_counts.values())
    assert art.metadata["models_loaded"] == expected_models

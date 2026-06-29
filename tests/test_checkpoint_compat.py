"""Regression guard: old checkpoints must keep loading into the model classes.

These tests pin the core refactor invariant -- the moment a change renames a
``state_dict`` key (or wraps the model in a new ``nn.Module``), the
key-compatibility assertion fails here instead of silently breaking
``strict=True`` loading of shipped checkpoints.

If a checkpoint file is not present locally the corresponding test is skipped,
so the suite stays green on machines without the (large) weight files.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.checkpoints import CheckpointStore
from connectpt.routes_generator.utils import build_model_from_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
WEIGHTS_DIR = REPO_ROOT / "artifacts" / "model_weights"

EDIT_CHECKPOINT = WEIGHTS_DIR / "improvement" / "improvement_lc_redundancy_rttwmc_v1_PRESERVED.pt"
CONSTRUCTION_CHECKPOINT = WEIGHTS_DIR / "inductive_random_graphs_weighted_connectivity.pt"


def _build(model_config):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="ppo_50nodes.yaml", overrides=[f"model={model_config}"])
    return build_model_from_cfg(cfg.model, cfg.experiment)


def test_edit_model_class():
    model = _build("bestsofar_feb2023_trim")
    assert type(model).__name__ == "TrimPathCombiningRouteGenerator"


def test_construction_model_class():
    model = _build("bestsofar_feb2023")
    assert type(model).__name__ == "PathCombiningRouteGenerator"


@pytest.mark.skipif(not EDIT_CHECKPOINT.exists(), reason="edit checkpoint not present")
def test_edit_checkpoint_strict_load():
    model = _build("bestsofar_feb2023_trim")
    CheckpointStore.assert_state_dict_compatible(model, EDIT_CHECKPOINT)


@pytest.mark.skipif(
    not CONSTRUCTION_CHECKPOINT.exists(), reason="construction checkpoint not present"
)
def test_construction_checkpoint_strict_load():
    model = _build("bestsofar_feb2023")
    CheckpointStore.assert_state_dict_compatible(model, CONSTRUCTION_CHECKPOINT)

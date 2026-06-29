"""core/ re-export shims expose the same objects as their source modules."""
from pathlib import Path

import connectpt.routes_generator.core as core
import connectpt.routes_generator.transit_time_estimator as tte
from connectpt.routes_generator.core import paths as core_paths


def test_actions_match_source():
    assert core.ROUTE_ACTION_EXTEND == tte.ROUTE_ACTION_EXTEND == 0
    assert core.ROUTE_ACTION_TRIM_START == tte.ROUTE_ACTION_TRIM_START == 1
    assert core.ROUTE_ACTION_TRIM_END == tte.ROUTE_ACTION_TRIM_END == 2
    assert core.ROUTE_ACTION_HALT == tte.ROUTE_ACTION_HALT == 3
    assert core.ACTION_NAME_TO_KIND["extend"] == 0
    assert core.ACTION_KIND_TO_NAME[3] == "halt"


def test_route_state_is_same_class():
    assert core.RouteGenBatchState is tte.RouteGenBatchState


def test_paths_resolve_to_repo_root():
    root = Path(__file__).resolve().parents[1]
    assert core_paths.ROOT_DIR == root
    assert core_paths.CFG_DIR == root / "connectpt" / "routes_generator" / "cfg"
    assert core_paths.ARTIFACTS_DIR == root / "artifacts"
    # the construction checkpoint path matches the real shipped file location
    assert core_paths.CONSTRUCTION_MODEL_WEIGHTS_PATH.name == (
        "inductive_random_graphs_weighted_connectivity.pt"
    )


def test_runtime_device_and_seed():
    dev = core.resolve_device(cpu=True)
    assert dev.type == "cpu"
    core.seed_everything(0)  # must not raise


def test_artifact_store_roundtrip(tmp_path):
    store = core.ArtifactStore(tmp_path)
    store.save_json({"a": 1, "b": "x"}, "meta")
    assert store.load_json("meta") == {"a": 1, "b": "x"}

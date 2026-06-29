"""Real bee_colony execution on Mandl (no checkpoints needed for classic_bco).

End-to-end smoke test: BeeColonySearchRun actually runs the existing bee_colony
through test_method on the Mandl benchmark and persists artifacts. Kept tiny
(1 iteration) so it stays a fast regression guard rather than a benchmark.
"""
import math
from pathlib import Path

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import BeeColonySearchRun

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "connectpt" / "routes_generator" / "cfg"


def test_classic_bco_runs_on_mandl(tmp_path):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(
            config_name="experiments/bee_type_comparison/00_classic_bco",
            overrides=[f"paths.output_dir={tmp_path.as_posix()}",
                       "search.n_iterations=1"],
        )
    artifact = BeeColonySearchRun(cfg).run(dry_run=False)

    # produced a finite cost + routes, and persisted the artifacts
    assert math.isfinite(artifact.result["mean_cost"])
    assert artifact.result["routes"] is not None
    assert (tmp_path / "bee_type_comparison_00_classic_bco_search.json").exists()
    assert (tmp_path / "bee_type_comparison_00_classic_bco_routes.pt").exists()

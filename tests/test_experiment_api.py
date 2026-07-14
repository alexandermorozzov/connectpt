"""Top-level experiment orchestration API (M010 stage 6).

The narrow public surface the notebook uses: name an experiment (load_experiment
/ load_suite), run it (ExperimentRunFactory / ExperimentBatch), render it
(render_report -- one duck-typed renderer for table + figure). This test drives
that surface without running BCO (compose + render off a table + batch table
aggregation).
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
from types import SimpleNamespace

from connectpt.routes_generator import (
    ExperimentRunFactory, ExperimentBatch, build_experiment, load_suite,
    render_report)
from connectpt.routes_generator.core.paths import (
    CONSTRUCTION_MODEL_WEIGHTS_PATH)
from connectpt.routes_generator.search import BeeColonySearchRun


def test_build_experiment_and_suite_compose():
    # method picked in code (bee_sets + models groups); city is a runtime param.
    cfg = build_experiment("table3_nbco_vs_our", bee_sets="our_nbco",
                           models="construction_and_edit_seeded",
                           city="Mumford0", label="Our NBCO")
    assert cfg.run.type == "bee_colony_search"
    assert cfg.data.city == "Mumford0"
    assert cfg.run.label == "Our NBCO"
    assert isinstance(ExperimentRunFactory.from_cfg(cfg), BeeColonySearchRun)

    # methods-based artifact: one file, N methods (no per-method leaf files).
    batch = load_suite("table3_nbco_vs_our")
    assert batch.batch.name == "table3_nbco_vs_our"
    assert len(batch.methods) == 2


def test_render_report_infers_pareto_from_table():
    art = SimpleNamespace(
        run_name="e1/x", metadata={}, routes={}, instance=None,
        table=pd.DataFrame([{"RTT": 1.0, "WMC": 2.0, "alpha": 0.0},
                            {"RTT": 1.5, "WMC": 1.2, "alpha": 1.0}]))
    rep = render_report(art)
    assert len(rep.table) == 2
    assert "pareto" in rep.figures


def test_render_report_explicit_kind_and_single_row_default():
    # a single-row table without a hint is NOT a Pareto front -> no pareto fig
    art = SimpleNamespace(run_name="r", metadata={}, routes={}, instance=None,
                          table=pd.DataFrame([{"RTT": 1.0, "WMC": 2.0}]))
    rep = render_report(art)
    assert rep.figures == {}


@pytest.mark.skipif(not CONSTRUCTION_MODEL_WEIGHTS_PATH.exists(),
                    reason="construction weights not present")
def test_batch_dry_run_aggregates_no_table():
    # methods-based batch dry-run: one run per method (both group choices compose
    # + load), no sweep table produced -> None. base_name lets each method
    # re-compose the artifact config with its bee_sets/models override.
    batch = ExperimentBatch(load_suite("table3_nbco_vs_our"),
                            base_name="table3_nbco_vs_our").run(
                                dry_run=True, city="Mumford0")
    assert batch.name == "table3_nbco_vs_our"
    assert len(batch.artifacts) == 2
    assert batch.table is None


def test_batch_combine_tables_tags_runs():
    from connectpt.routes_generator.core.batches import _combine_tables
    a1 = SimpleNamespace(run_name="m/a", table=pd.DataFrame([{"RTT": 1.0}]))
    a2 = SimpleNamespace(run_name="m/b", table=pd.DataFrame([{"RTT": 2.0}]))
    dry = SimpleNamespace(run_name="dry", table=None)
    combined = _combine_tables([a1, a2, dry])
    assert list(combined["run"]) == ["m/a", "m/b"]
    assert list(combined["RTT"]) == [1.0, 2.0]

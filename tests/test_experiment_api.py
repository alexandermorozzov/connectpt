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
    # collapsed leaf (one per method); the city is a runtime parameter.
    cfg = build_experiment("e1/our_nbco", city="Mumford0")   # auto experiments/ prefix
    assert cfg.run.type == "bee_colony_search"
    assert cfg.data.city == "Mumford0"
    assert isinstance(ExperimentRunFactory.from_cfg(cfg), BeeColonySearchRun)

    batch = load_suite("e1/batch")
    assert batch.batch.name == "e1"
    assert len(batch.batch.runs) == 2


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
    # e1 batch dry-run: runs validate wiring but produce no sweep table -> None.
    batch = ExperimentBatch(load_suite("e1/batch")).run(dry_run=True, city="Mumford0")
    assert batch.name == "e1"
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

"""Reports build tables from artifacts and never import training/ or search/."""
import sys

import pandas as pd

from connectpt.routes_generator.reports import (
    make_search_plan_table, make_comparison_table,
)


def test_reports_do_not_import_training_or_search():
    # fresh import of reports must not pull in the training/search apps
    for mod in [m for m in list(sys.modules)
                if "routes_generator.training" in m or "routes_generator.search" in m]:
        del sys.modules[mod]
    import importlib
    import connectpt.routes_generator.reports as reports
    importlib.reload(reports)
    leaked = [m for m in sys.modules
              if "routes_generator.training" in m or "routes_generator.search" in m]
    assert not leaked, leaked


def test_make_search_plan_table():
    plan = {"counts": {"n_type4": 4, "n_type5": 6, "n_type6": 2, "n_type7": 2,
                       "n_type1": 0}}
    df = make_search_plan_table(plan)
    # zero-count types dropped; labels attached
    assert set(df["bee_type"]) == {"n_type4", "n_type5", "n_type6", "n_type7"}
    assert df.loc[df.bee_type == "n_type4", "count"].item() == 4
    assert "neural construction" in df.loc[df.bee_type == "n_type4", "label"].item()


def test_search_comparison_table_from_artifacts(tmp_path):
    from connectpt.routes_generator.core.artifacts import ArtifactStore
    from connectpt.routes_generator.reports import (
        load_search_summary, make_search_comparison_table,
    )
    store = ArtifactStore(tmp_path)
    store.save_json(
        {"mean_cost": 2.0, "metrics": {"RTT": 1.0},
         "plan": {"counts": {"n_type1": 20}}},
        "bee_type_comparison_00_classic_bco_search")
    store.save_json(
        {"mean_cost": 1.5, "metrics": {"RTT": 0.8},
         "plan": {"counts": {"n_type5": 20}}},
        "bee_type_comparison_04_nbco_edit_full_search")

    s0 = load_search_summary(tmp_path, "bee_type_comparison/00_classic_bco")
    s4 = load_search_summary(tmp_path, "bee_type_comparison/04_nbco_edit_full")
    df = make_search_comparison_table({"classic": s0, "edit_full": s4})
    # sorted by mean_cost ascending -> edit_full (1.5) first
    assert list(df["run"]) == ["edit_full", "classic"]
    assert df.iloc[0]["mean_cost"] == 1.5
    assert "n_type5=20" in df.iloc[0]["bee_mix"]


def test_make_comparison_table():
    a = pd.DataFrame([{"cost": 1.0}])
    b = pd.DataFrame([{"cost": 2.0}])
    out = make_comparison_table({"ours": a, "baseline": b})
    assert list(out["method"]) == ["ours", "baseline"]
    assert list(out["cost"]) == [1.0, 2.0]

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
    plan = {"counts": {"neural_construction": 4, "neural_edit_full": 6,
                       "neural_edit_trim": 2, "compound": 2,
                       "random_mutation": 0}}
    df = make_search_plan_table(plan)
    # zero-count bees dropped; bee name is the label
    assert set(df["bee"]) == {"neural_construction", "neural_edit_full",
                              "neural_edit_trim", "compound"}
    assert df.loc[df.bee == "neural_construction", "count"].item() == 4


def test_search_comparison_table_from_artifacts(tmp_path):
    from connectpt.routes_generator.core.artifacts import ArtifactStore
    from connectpt.routes_generator.reports import (
        load_search_summary, make_search_comparison_table,
    )
    store = ArtifactStore(tmp_path)
    store.save_json(
        {"mean_cost": 2.0, "metrics": {"RTT": 1.0},
         "plan": {"counts": {"random_mutation": 20}}},
        "bee_type_comparison_00_classic_bco_search")
    store.save_json(
        {"mean_cost": 1.5, "metrics": {"RTT": 0.8},
         "plan": {"counts": {"neural_edit_full": 20}}},
        "bee_type_comparison_04_nbco_edit_full_search")

    s0 = load_search_summary(tmp_path, "bee_type_comparison/00_classic_bco")
    s4 = load_search_summary(tmp_path, "bee_type_comparison/04_nbco_edit_full")
    df = make_search_comparison_table({"classic": s0, "edit_full": s4})
    # sorted by mean_cost ascending -> edit_full (1.5) first
    assert list(df["run"]) == ["edit_full", "classic"]
    assert df.iloc[0]["mean_cost"] == 1.5
    assert "neural_edit_full=20" in df.iloc[0]["bee_mix"]


def test_make_comparison_table():
    a = pd.DataFrame([{"cost": 1.0}])
    b = pd.DataFrame([{"cost": 2.0}])
    out = make_comparison_table({"ours": a, "baseline": b})
    assert list(out["method"]) == ["ours", "baseline"]
    assert list(out["cost"]) == [1.0, 2.0]

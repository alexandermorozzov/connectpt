import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

import eval_lib.baselines as baselines  # noqa: E402
import eval_lib.helpers as helpers  # noqa: E402


MEAN_WEIGHTED = "mean_weighted"


def _cost_kwargs(cfg):
    return cfg.experiment.cost_function.kwargs


def test_eval_lib_builders_propagate_mean_weighted():
    lc_cfg = helpers.build_lc_cfg(
        "conn_mode_lc", 2, 2, 5, connectivity_mode=MEAN_WEIGHTED
    )
    bco_cfg = helpers.build_bco_cfg(
        "conn_mode_bco",
        2,
        2,
        5,
        n_bees=2,
        n_type1_bees=1,
        n_type2_bees=1,
        connectivity_mode=MEAN_WEIGHTED,
    )
    sa_cfg = baselines.build_sa_cfg(
        "conn_mode_sa",
        2,
        2,
        5,
        n_iterations=1,
        connectivity_mode=MEAN_WEIGHTED,
    )
    ga_cfg = baselines.build_ga_cfg(
        "conn_mode_ga",
        2,
        2,
        5,
        n_iterations=1,
        population_size=2,
        connectivity_mode=MEAN_WEIGHTED,
    )
    hh_cfg = baselines.build_hh_cfg(
        "conn_mode_hh",
        2,
        2,
        5,
        n_iterations=1,
        connectivity_mode=MEAN_WEIGHTED,
    )
    nsgaii_cfg = baselines.build_nsgaii_cfg(
        "conn_mode_nsga",
        2,
        2,
        5,
        n_iterations=1,
        pop_size=2,
        connectivity_mode=MEAN_WEIGHTED,
    )

    cfgs = [lc_cfg, bco_cfg, sa_cfg, ga_cfg, hh_cfg, nsgaii_cfg]
    assert all(
        _cost_kwargs(cfg).connectivity_mode == MEAN_WEIGHTED for cfg in cfgs
    )


def test_eval_lib_builders_use_runtime_disabled_components(monkeypatch):
    monkeypatch.setattr(helpers, "DISABLED_COST_COMPONENTS", ["demand"])

    cfg = helpers.build_bco_cfg(
        "runtime_disabled_components",
        2,
        2,
        5,
        n_bees=2,
        n_type1_bees=1,
        n_type2_bees=1,
        demand_time_weight=0.0,
        route_time_weight=0.25,
        median_connectivity_weight=0.75,
        connectivity_mode=MEAN_WEIGHTED,
    )

    assert list(_cost_kwargs(cfg).disabled_components) == ["demand"]


def test_run_nsgaii_applies_runtime_connectivity_mode(monkeypatch):
    cfg = OmegaConf.create(
        {
            "eval": {
                "dataset": {"type": "tensor"},
                "n_routes": 2,
                "min_route_len": 2,
                "max_route_len": 5,
            },
            "n_iterations": 1,
            "pop_size": 2,
            "p_crossover": 0.0,
            "p_mutation": 0.0,
            "mutator_p_t": 0.0,
            "gen_batch_size": 2,
            "use_cost_based_heuristics": False,
        }
    )
    cost_obj = SimpleNamespace(
        use_weighted_connectivity=False, connectivity_mode="legacy"
    )
    seen = {}

    monkeypatch.setattr(
        baselines.lrnu,
        "process_standard_experiment_cfg",
        lambda *args, **kwargs: (
            torch.device("cpu"), "nsga_test", None, cost_obj, None
        ),
    )
    monkeypatch.setattr(
        baselines, "make_tensor_dataloader",
        lambda dataset_cfg, tensors: iter([SimpleNamespace()]),
    )

    class FakeState:
        def __init__(self, data, cost_obj, n_routes, min_route_len, max_route_len):
            seen["state_cost_obj"] = cost_obj

    class FakeNSGAII:
        def __init__(self, cost_obj, *args, **kwargs):
            seen["optimizer_cost_obj"] = cost_obj

        def run(self, state, init_mode, sum_writer=None, seed_routes=None):
            return {"pareto_pop": []}

    monkeypatch.setattr(baselines, "RouteGenBatchState", FakeState)
    monkeypatch.setattr(baselines, "NSGAII", FakeNSGAII)

    _, output = baselines.run_nsgaii(
        cfg,
        tensors={"dummy": object()},
        use_weighted_connectivity=True,
        connectivity_mode=MEAN_WEIGHTED,
    )

    assert output == {"pareto_pop": []}
    assert cost_obj.use_weighted_connectivity is True
    assert cost_obj.connectivity_mode == MEAN_WEIGHTED
    assert seen["state_cost_obj"] is cost_obj
    assert seen["optimizer_cost_obj"] is cost_obj


def test_params_is_single_source_of_unified_objective():
    import eval_lib.params as params
    import eval_lib.paper as paper

    assert params.CONNECTIVITY_MODE == "median_weighted"
    assert params.DISABLED_COST_COMPONENTS == ["demand"]
    assert params.UNIFIED_COST_WEIGHTS == {
        "demand_time_weight": 0.0,
        "route_time_weight": 0.5,
        "median_connectivity_weight": 0.5,
    }
    assert params.ADJ_WEIGHT == 10.0
    assert params.ADJ_TARGET == 0.2
    assert params.ADJ_OBJECTIVE == "target"
    # UNIFIED_ADJ is built from the params constants.
    assert paper.UNIFIED_ADJ == {
        "adjustment_degree_weight": params.ADJ_WEIGHT,
        "adjustment_degree_target": params.ADJ_TARGET,
        "adjustment_degree_objective": params.ADJ_OBJECTIVE,
        "adjustment_degree_gap": params.ADJ_GAP,
        "adjustment_degree_mode": params.ADJ_MODE,
    }


def test_paper_combined_sets_connectivity_mode_everywhere():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # Both the PART 1 (training) and PART 2 (experiments) config cells import
    # the unified objective from eval_lib.params; no local re-definitions.
    assert text.count("from eval_lib.params import") == 2
    assert 'CONNECTIVITY_MODE = "' not in text
    assert "DISABLED_COST_COMPONENTS = [" not in text
    assert "ADJ_WEIGHT = " not in text
    assert "ADJ_TARGET = " not in text
    assert "ADJ_OBJECTIVE = " not in text
    assert "UNIFIED_COST_WEIGHTS = dict" not in text
    assert "RUN_NSGAII_BASELINES = False" in text
    assert "if RUN_NSGAII_BASELINES:" in text
    # E1 legacy (ATT+RTT baselines / main_df) was removed; only the unified
    # E1u table (unified_df) remains.
    assert "main_df" not in text
    assert "BASE_WEIGHTS" not in text
    assert (
        'unified_df = unified_df[unified_df["method"] != "NSGA-II"].reset_index(drop=True)'
        in text
    )
    assert "connectivity_mode=CONNECTIVITY_MODE" in text
    assert (
        'f"++experiment.cost_function.kwargs.connectivity_mode='
        '{CONNECTIVITY_MODE}"'
    ) in text
    assert text.count("run_nsgaii(build_nsgaii_cfg") == 1
    assert text.count("connectivity_mode=CONNECTIVITY_MODE),") >= 1
    assert "connectivity_mode=CONNECTIVITY_MODE, **UNIFIED_ADJ)" in text


def test_paper_combined_streams_csv_rows_with_duration():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # append_paper_row / paper_row(_row) now live in eval_lib.paper; the
    # notebook imports them from there.
    assert "append_paper_row" in text
    assert "paper_row as _row" in text
    assert "def _e2_row(ctx, series_col, label, alpha, target, routes, metrics, duration_s=None):" in text

    assert "append_paper_row(row, _e2_our_table, ndigits=4)" in text
    assert "append_paper_row(row, _e2_abl_table, ndigits=4)" in text
    assert "append_paper_row(row, table_name, ndigits=3)" in text
    assert "append_paper_row(row, comparison_table_name, ndigits=3)" in text

    assert "r, m, dt = _run_rttwmc" in text
    assert text.count('r, dt = run_one(f"Initial (LC+{EXP_INIT_TIER})"') == 1


def test_paper_combined_uses_two_sided_adj_objective():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # adj penalty everywhere comes from the single params source: training
    # passes ADJ_OBJECTIVE, the BCO/eval sweeps spread UNIFIED_ADJ (which the
    # params test pins to the two-sided "target" objective). No literals left.
    assert 'adjustment_degree_objective="cap"' not in text
    assert 'adjustment_degree_objective="target"' not in text
    assert "adj_objective=ADJ_OBJECTIVE" in text or "objective=ADJ_OBJECTIVE" in text
    assert text.count("**UNIFIED_ADJ") >= 3
    assert text.count("**dict(UNIFIED_ADJ") == 2  # our-model + E2 (variable target)

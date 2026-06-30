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
    assert params.ADJ_OBJECTIVE == "target"        # search / BCO acceptance
    assert params.ADJ_TRAIN_OBJECTIVE == "cap"     # PPO reward shaping
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

    # The objective is sourced from eval_lib.params (single source); the
    # notebook imports the shared names and never re-defines them locally
    # (line-anchored so experiment-local prefixed constants like
    # M0_NBCO_ADJ_TARGET stay allowed).
    import re
    assert "from eval_lib.params import" in text
    for name in ("CONNECTIVITY_MODE", "DISABLED_COST_COMPONENTS", "ADJ_WEIGHT",
                 "ADJ_TARGET", "ADJ_OBJECTIVE", "UNIFIED_COST_WEIGHTS"):
        assert not re.search(rf"^{name} *=", text, re.M), \
            f"{name} is re-defined in the notebook (params.py is the source)"
    assert "RUN_NSGAII_BASELINES = False" in text
    assert "main_df" not in text
    assert "BASE_WEIGHTS" not in text

    # Config-driven training (refactor CF33): the notebook loads the named train
    # config via load_train_config and runs EditTrainingRun(cfg). The inline
    # compose + build_edit_run + ~25-entry override list are gone; the
    # connectivity_mode override lives in cfg/train/edit.yaml -> objective YAML.
    assert "load_train_config(" in text
    assert "EditTrainingRun(train_cfg)" in text
    assert "ppo_50nodes.yaml" not in text
    assert 'f"++experiment.cost_function.kwargs.connectivity_mode=' not in text

    # PART 2 (experiments) still threads the unified connectivity mode through
    # every method's cost config.
    assert "connectivity_mode=CONNECTIVITY_MODE" in text


def test_paper_combined_streams_csv_rows_with_duration():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # Results IO lives in eval_lib.paper; the notebook streams rows through it
    # rather than building tables inline. Anchor on the stable helper names, not
    # on the volatile per-experiment call sites / signatures.
    assert "append_paper_row" in text
    assert "paper_row as _row" in text
    assert "save_paper_table" in text
    assert "reset_paper_table" in text


def test_paper_combined_uses_two_sided_adj_objective():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # adj penalty everywhere comes from the single params/objective source: no
    # hardcoded objective literals leak into the notebook.
    assert 'adjustment_degree_objective="cap"' not in text
    assert 'adjustment_degree_objective="target"' not in text
    # PART 1 training shapes with the one-sided cap objective, now sourced from
    # cfg/train/edit.yaml (was the notebook's ADJ_TRAIN_OBJECTIVE constant before
    # the config-driven training refactor).
    edit_yaml = (REPO_ROOT / "connectpt" / "routes_generator" / "cfg" / "train"
                 / "edit.yaml").read_text(encoding="utf-8")
    assert "adjustment_degree_objective: cap" in edit_yaml
    # PART 2 search spreads the unified adj kwargs (UNIFIED_ADJ, usually via
    # dict(UNIFIED_ADJ, adjustment_degree_target=...) overrides).
    assert text.count("UNIFIED_ADJ") >= 3

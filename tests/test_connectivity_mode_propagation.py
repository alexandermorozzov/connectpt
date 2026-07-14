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

# baselines moved to the library; patch the module that owns run_nsgaii so the
# monkeypatched names (make_tensor_dataloader / RouteGenBatchState / NSGAII)
# resolve in the same namespace run_nsgaii uses.
import connectpt.routes_generator.baselines as baselines  # noqa: E402
# The LC eval builder moved to the library; expose it under the historical
# ``helpers`` name so the propagation test reads unchanged. (The flat BCO cfg
# builder was removed with the compat orchestration in M010.)
from types import SimpleNamespace as _NS  # noqa: E402
from connectpt.routes_generator import lc_eval as _lc_eval  # noqa: E402
from connectpt.routes_generator.objectives import load_unified_objective as _luo  # noqa: E402
helpers = _NS(build_lc_cfg=_lc_eval.build_lc_cfg,
              DISABLED_COST_COMPONENTS=list(_luo().disabled_components))


MEAN_WEIGHTED = "mean_weighted"


def _cost_kwargs(cfg):
    return cfg.experiment.cost_function.kwargs


def test_eval_lib_builders_propagate_mean_weighted():
    lc_cfg = helpers.build_lc_cfg(
        "conn_mode_lc", 2, 2, 5, connectivity_mode=MEAN_WEIGHTED
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

    cfgs = [lc_cfg, sa_cfg, ga_cfg, hh_cfg, nsgaii_cfg]
    assert all(
        _cost_kwargs(cfg).connectivity_mode == MEAN_WEIGHTED for cfg in cfgs
    )


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


def test_accessor_is_single_source_of_unified_objective():
    from connectpt.routes_generator.objectives import load_unified_objective
    from connectpt.routes_generator.paper_experiments import macsa_run

    o = load_unified_objective()
    assert o.connectivity_mode == "median_weighted"
    assert list(o.disabled_components) == ["demand"]
    assert o.weights == {
        "demand_time_weight": 0.0,
        "route_time_weight": 0.5,
        "median_connectivity_weight": 0.5,
    }
    assert o.adj_weight == 10.0
    assert o.adj_target == 0.2
    assert o.adj_objective == "target"        # search / BCO acceptance
    assert o.adj_train_objective == "cap"     # PPO reward shaping
    # MACSA's fixed-network scoring point is read off the accessor (M013):
    # the atom itself takes weight/gap/mode from the objective via CostFactory.
    assert macsa_run.EVAL_ADJ_TARGET == o.adj_target
    assert macsa_run.EVAL_ADJ_OBJECTIVE == o.adj_objective


def test_paper_combined_sets_connectivity_mode_everywhere():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # The objective is sourced from the single YAML source via the library
    # factory (load_unified_objective reads cfg/objective/rtt_wmc_no_demand.yaml);
    # the notebook binds the shared names off that accessor object -- it never
    # hardcodes objective literal values.
    import re
    assert "from eval_lib.params import" not in text  # params.py is deleted
    # Objective names may be bound from the accessor object (= _OBJ.<field>) but
    # never from a hardcoded literal.
    for name, literal in (("CONNECTIVITY_MODE", '"median_weighted"'),
                          ("ADJ_WEIGHT", "10.0"), ("ADJ_TARGET", "0.2")):
        assert not re.search(rf"^{name} *= *{re.escape(literal)}", text, re.M), \
            f"{name} is bound to a hardcoded objective literal in the notebook"
    # Experiment selection is now config-driven via the suite profile
    # (cfg/experiments/suite*.yaml), loaded once as the lowercase ``suite`` handle
    # (M012 -- no UPPERCASE constants / RunContext in the notebook). The notebook
    # drives the experiment switches from suite.run, and the full-run default keeps
    # the (heaviest) NSGA-II baseline off.
    assert "suite = load_suite(" in text
    assert "suite.run." in text
    assert "SUITE" not in text          # M012: no UPPERCASE handle
    assert "CTX" not in text            # M012: RunContext removed
    suite_yaml = (REPO_ROOT / "connectpt" / "routes_generator" / "cfg"
                  / "experiments" / "suite.yaml").read_text(encoding="utf-8")
    assert "nsgaii_baselines: false" in suite_yaml
    assert "main_df" not in text
    assert "BASE_WEIGHTS" not in text

    # Config-driven training (refactor CF33): the notebook loads the named train
    # config via load_train_config and runs EditTrainingRun(cfg). The inline
    # compose + build_edit_run + ~25-entry override list are gone; the
    # connectivity_mode override lives in cfg/training/edit.yaml -> objective YAML.
    assert "load_train_config(" in text
    assert "EditTrainingRun(train_cfg)" in text
    assert "ppo_50nodes.yaml" not in text
    assert 'f"++experiment.cost_function.kwargs.connectivity_mode=' not in text

    # PART 2 (experiments) threads the unified connectivity mode config-first:
    # search runs build the cost via CostFactory from the objective YAML, and
    # fixed-network scoring goes through the evaluation atom (M013) -- the
    # notebook no longer passes a connectivity mode by hand.
    assert "connectivity_mode=CONNECTIVITY_MODE" not in text
    scoring_py = (REPO_ROOT / "connectpt" / "routes_generator"
                  / "evaluation" / "route_scoring.py").read_text(encoding="utf-8")
    assert "def score_fixed_routes" in scoring_py
    assert "build_unified" in scoring_py


def test_paper_combined_streams_csv_rows_with_duration():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    # Results IO lives entirely in the library (M012): the notebook never calls
    # save_paper_table / paper_row directly -- it runs config-first wrappers
    # (run_experiment / run_batch / run_macsa_table_b) that persist internally
    # with the prefix + folder read from the suite. Anchor on the wrappers here
    # and on the sink helpers in the library.
    assert "save_paper_table" not in text          # no direct sink calls in cells
    assert "run_macsa_table_b(" in text
    assert ("run_experiment(" in text) or ("run_batch(" in text)
    paper_io = (REPO_ROOT / "connectpt" / "routes_generator" / "reports"
                / "paper_io.py").read_text(encoding="utf-8")
    assert "paper_row" in paper_io
    assert "save_paper_table" in paper_io
    assert "append_paper_row" in paper_io
    assert "reset_paper_table" in paper_io


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
    # cfg/training/edit.yaml (was the notebook's ADJ_TRAIN_OBJECTIVE constant before
    # the config-driven training refactor).
    edit_yaml = (REPO_ROOT / "connectpt" / "routes_generator" / "cfg" / "training"
                 / "edit.yaml").read_text(encoding="utf-8")
    assert "adjustment_degree_objective: cap" in edit_yaml
    # PART 2 search spreads the unified adj kwargs. The threading lives in the
    # library, not inline in the notebook: fixed-network scoring takes the
    # penalty (weight/gap/mode) from the objective via CostFactory inside the
    # evaluation atom (M013), and MACSA reads its eval point off the accessor.
    assert "UNIFIED_ADJ" not in text  # adj threading lives in the library now
    scoring_py = (REPO_ROOT / "connectpt" / "routes_generator"
                  / "evaluation" / "route_scoring.py").read_text(encoding="utf-8")
    assert "adjustment_degree_target" in scoring_py
    macsa_py = (REPO_ROOT / "connectpt" / "routes_generator"
                / "paper_experiments" / "macsa_run.py").read_text(encoding="utf-8")
    assert "load_unified_objective" in macsa_py

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


WEIGHTED_MEDIAN = "weighted_median"


def _cost_kwargs(cfg):
    return cfg.experiment.cost_function.kwargs


def test_eval_lib_builders_propagate_weighted_median():
    lc_cfg = helpers.build_lc_cfg(
        "conn_mode_lc", 2, 2, 5, connectivity_mode=WEIGHTED_MEDIAN
    )
    bco_cfg = helpers.build_bco_cfg(
        "conn_mode_bco",
        2,
        2,
        5,
        n_bees=2,
        n_type1_bees=1,
        n_type2_bees=1,
        connectivity_mode=WEIGHTED_MEDIAN,
    )
    sa_cfg = baselines.build_sa_cfg(
        "conn_mode_sa",
        2,
        2,
        5,
        n_iterations=1,
        connectivity_mode=WEIGHTED_MEDIAN,
    )
    ga_cfg = baselines.build_ga_cfg(
        "conn_mode_ga",
        2,
        2,
        5,
        n_iterations=1,
        population_size=2,
        connectivity_mode=WEIGHTED_MEDIAN,
    )
    hh_cfg = baselines.build_hh_cfg(
        "conn_mode_hh",
        2,
        2,
        5,
        n_iterations=1,
        connectivity_mode=WEIGHTED_MEDIAN,
    )
    nsgaii_cfg = baselines.build_nsgaii_cfg(
        "conn_mode_nsga",
        2,
        2,
        5,
        n_iterations=1,
        pop_size=2,
        connectivity_mode=WEIGHTED_MEDIAN,
    )

    cfgs = [lc_cfg, bco_cfg, sa_cfg, ga_cfg, hh_cfg, nsgaii_cfg]
    assert all(
        _cost_kwargs(cfg).connectivity_mode == WEIGHTED_MEDIAN for cfg in cfgs
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
        connectivity_mode=WEIGHTED_MEDIAN,
    )

    assert output == {"pareto_pop": []}
    assert cost_obj.use_weighted_connectivity is True
    assert cost_obj.connectivity_mode == WEIGHTED_MEDIAN
    assert seen["state_cost_obj"] is cost_obj
    assert seen["optimizer_cost_obj"] is cost_obj


def test_paper_combined_sets_connectivity_mode_everywhere():
    notebook = json.loads(
        (ROUTE_EXAMPLES / "paper_combined.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert 'CONNECTIVITY_MODE = "weighted_median"' in text
    assert "connectivity_mode=CONNECTIVITY_MODE" in text
    assert (
        'f"++experiment.cost_function.kwargs.connectivity_mode='
        '{CONNECTIVITY_MODE}"'
    ) in text
    assert text.count("run_nsgaii(build_nsgaii_cfg") == 2
    assert text.count("connectivity_mode=CONNECTIVITY_MODE),") >= 2
    assert "connectivity_mode=CONNECTIVITY_MODE, **UNIFIED_ADJ)" in text

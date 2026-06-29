"""Unified experiment runner: YAML sweep drives alpha (no Python loop)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))


def test_sweep_applies_alpha_from_yaml():
    """run_experiment must sweep the YAML alpha grid and set the per-alpha cost
    weights -- the unification that removes the notebook's hand-written loop."""
    from eval_lib.experiment_runner import run_experiment, load_experiment_spec

    spec = load_experiment_spec("ekb_sweep")

    seen_weights = []

    def mock_method(cfg, init, tensors, run_name_scope):
        kw = cfg.experiment.cost_function.kwargs
        seen_weights.append((float(kw.route_time_weight),
                             float(kw.median_connectivity_weight)))
        # mimic run_bco's return: (run_name, metrics, unserved, routes, counts)
        return ("run", {"m": 1}, None, init, {})

    def mock_metrics(m, routes, init):
        return {"RTT": 1.0, "cost": 2.0}

    result = run_experiment(spec, method_fn=mock_method, metrics_fn=mock_metrics)

    # the YAML alpha grid [0, 0.5, 1.0] drove the weights, in order
    assert seen_weights == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    assert list(result.table["alpha"]) == [0.0, 0.5, 1.0]
    assert list(result.table["adj_target"]) == [0.3, 0.3, 0.3]
    assert result.name == "ekb_sweep"
    # init + one route set per alpha
    assert "Initial" in result.routes
    assert any("a=0.5" in k for k in result.routes)


def test_ekb_data_source_loads():
    """EKB data source loads real tensors + seed routes + spec (no eval_lib
    notebook globals needed)."""
    from eval_lib.data_sources import EKBDataSource
    inst = EKBDataSource().load()
    assert inst.label == "EKB"
    assert inst.spec["n_routes"] == 67
    assert inst.init_routes is not None
    assert inst.coords is not None and inst.street_adj is not None


def test_spec_yaml_shapes():
    from eval_lib.experiment_runner import load_experiment_spec
    for name, source in [("ekb_sweep", "ekb"), ("e1_mandl", "benchmark"),
                         ("macsa_sweep", "macsa")]:
        spec = load_experiment_spec(name)
        assert spec.data.source == source
        assert list(spec.sweep.alpha) == [0.0, 0.5, 1.0]
        assert spec.metrics  # non-empty


def test_render_report_pareto_from_table():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from types import SimpleNamespace
    from eval_lib.experiment_report import render_report

    res = SimpleNamespace(
        table=pd.DataFrame([{"RTT": 1.0, "WMC": 2.0, "alpha": 0.0},
                            {"RTT": 1.5, "WMC": 1.2, "alpha": 1.0}]),
        routes={}, instance=None, spec={"report": {"routes_plot": "pareto"}})
    rep = render_report(res)
    assert len(rep.table) == 2
    assert "pareto" in rep.figures


def test_multi_method_sweep_iterates_methods_x_alpha():
    """methods x alpha grid -> one row per (method, alpha); the comparison the
    experiment cells did by hand."""
    from eval_lib.experiment_runner import run_experiment, load_experiment_spec
    spec = load_experiment_spec("e1_mandl")

    calls = []
    def mock_method(cfg, init, tensors, run_name_scope):
        calls.append(run_name_scope)
        return ("run", {"m": 1}, None, init, {})
    def mock_metrics(m, routes, init):
        return {"cost": 1.0}

    result = run_experiment(spec, method_fn=mock_method, metrics_fn=mock_metrics)
    # 2 methods x 3 alphas = 6 runs/rows
    assert len(result.table) == 6
    assert set(result.table["method"]) == {"neural BCO", "Our NBCO (GNN rebuild + trim/extend)"}
    assert list(result.table["alpha"]).count(0.0) == 2  # both methods at alpha 0

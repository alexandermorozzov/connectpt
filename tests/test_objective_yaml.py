"""The objective YAML is the single source of truth for the cost objective.

``cfg/objective/rtt_wmc_no_demand.yaml`` holds the unified objective; both
``load_unified_objective`` (the Python accessor) and ``CostFactory`` read from
it. These tests pin the YAML values and prove the two readers agree.
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

from connectpt.routes_generator.objectives import CostFactory


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

OBJECTIVE_YAML = (
    REPO_ROOT / "connectpt" / "routes_generator" / "cfg" / "objective" / "rtt_wmc_no_demand.yaml"
)


def test_objective_yaml_values():
    obj = OmegaConf.load(OBJECTIVE_YAML)
    assert obj.connectivity_mode == "median_weighted"
    assert list(obj.disabled_components) == ["demand"]
    assert float(obj.weights.demand_time_weight) == 0.0
    assert float(obj.weights.route_time_weight) == 0.5
    assert float(obj.weights.median_connectivity_weight) == 0.5
    assert float(obj.adjustment.weight) == 10.0
    assert float(obj.adjustment.target) == 0.2
    assert obj.adjustment.objective == "target"
    assert obj.adjustment.train_objective == "cap"
    assert float(obj.adjustment.gap) == 0.1
    assert obj.adjustment.mode == "paper"


def test_accessor_reads_objective_yaml():
    from connectpt.routes_generator.objectives import load_unified_objective

    obj = OmegaConf.load(OBJECTIVE_YAML)
    o = load_unified_objective()
    assert o.connectivity_mode == obj.connectivity_mode
    assert list(o.disabled_components) == list(obj.disabled_components)
    assert o.weights == {
        "demand_time_weight": float(obj.weights.demand_time_weight),
        "route_time_weight": float(obj.weights.route_time_weight),
        "median_connectivity_weight": float(obj.weights.median_connectivity_weight),
    }
    assert o.adj_weight == float(obj.adjustment.weight)
    assert o.adj_target == float(obj.adjustment.target)
    assert o.adj_objective == obj.adjustment.objective
    assert o.adj_train_objective == obj.adjustment.train_objective
    assert o.adj_gap == float(obj.adjustment.gap)
    assert o.adj_mode == obj.adjustment.mode
    # adj_kwargs is what search/eval runs spread as the two-sided adj penalty.
    assert o.adj_kwargs == {
        "adjustment_degree_weight": o.adj_weight,
        "adjustment_degree_target": o.adj_target,
        "adjustment_degree_objective": o.adj_objective,
        "adjustment_degree_gap": o.adj_gap,
        "adjustment_degree_mode": o.adj_mode,
    }


def test_cost_factory_applies_objective():
    obj = OmegaConf.load(OBJECTIVE_YAML)
    cost_cfg = OmegaConf.create(
        {"type": "mine", "kwargs": {"use_weighted_connectivity": True}}
    )
    cost = CostFactory.build(cost_cfg)

    CostFactory.apply_objective(cost, "rtt_wmc_no_demand")
    assert cost.route_time_weight == 0.5
    assert cost.median_connectivity_weight == 0.5
    assert cost.demand_time_weight == 0.0
    assert cost.connectivity_mode == "median_weighted"
    assert cost.adjustment_degree_weight == 10.0
    assert cost.adjustment_degree_objective == obj.adjustment.objective  # eval -> "target"

    CostFactory.apply_objective(cost, "rtt_wmc_no_demand", for_training=True)
    assert cost.adjustment_degree_objective == obj.adjustment.train_objective  # "cap"

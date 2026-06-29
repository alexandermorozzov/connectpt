"""BeeSpec / policy adapters / operators: one neural bee, many policies.

The model is irrelevant to action validation (it depends only on the policy
role), so these use a stub model object -- no heavy GNN build needed.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import (
    BeeSpec, parse_bee_specs, ConstructionSearchPolicy, EditSearchPolicy,
    NeuralRouteActionBee, get_route_selector, get_acceptance, BeeColonyPlan,
)

CFG_DIR = Path(__file__).resolve().parents[1] / "connectpt" / "routes_generator" / "cfg"


class _StubModel:
    pass


def _bee(policy, actions):
    return NeuralRouteActionBee(
        policy=policy, allowed_actions=actions,
        route_selector=get_route_selector("uniform"),
        acceptance_policy=get_acceptance("greedy"),
    )


def test_same_operator_works_across_policies_and_actions():
    constr = ConstructionSearchPolicy(_StubModel(), name="construction")
    edit = EditSearchPolicy(_StubModel(), name="edit")

    # one NeuralRouteActionBee class, three valid (policy, actions) combos
    _bee(constr, ["extend", "halt"])
    _bee(edit, ["extend", "halt"])
    _bee(edit, ["trim_start", "trim_end"])
    _bee(edit, ["extend", "trim_start", "trim_end", "halt"])


def test_invalid_action_combo_raises_clear_error():
    constr = ConstructionSearchPolicy(_StubModel(), name="construction")
    with pytest.raises(ValueError, match="does not support"):
        _bee(constr, ["trim_start"])


def test_bco_flexible_bees_yaml_parses_and_plans():
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="search/bco_flexible_bees")

    specs = parse_bee_specs(cfg.bees)
    assert [s.name for s in specs][0] == "neural_on_construction"

    policies = {
        "construction": ConstructionSearchPolicy(_StubModel(), name="construction"),
        "edit": EditSearchPolicy(_StubModel(), name="edit"),
    }
    plan = BeeColonyPlan.from_specs(specs, policies)

    # construction extend -> type4; edit extend -> type5; edit trim-only -> type6;
    # edit full -> type5; compound -> type7
    assert plan.counts["n_type4"] == 4   # neural_on_construction
    assert plan.counts["n_type5"] == 6   # extend_only(4) + full(2)
    assert plan.counts["n_type6"] == 2   # trim_only
    assert plan.counts["n_type7"] == 2   # compound
    assert plan.needs_construction and plan.needs_edit
    assert len(plan.bees) == len(specs)


def test_beespec_dataclass_defaults():
    s = BeeSpec(name="x", count=1, operator="neural_route_action", policy="edit",
               allowed_actions=["extend"])
    assert s.route_selection == "uniform" and s.acceptance == "greedy"

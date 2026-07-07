"""BeeSpec / policy adapters / operators: one neural bee, many policies.

The model is irrelevant to action validation (it depends only on the policy
role), so these use a stub model object -- no heavy GNN build needed.
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.search import (
    BeeSpec, parse_bee_specs, ConstructionSearchPolicy, EditSearchPolicy,
    NeuralRouteActionBee, get_route_selector, get_acceptance,
)
from connectpt.routes_generator.search.executable_plan import ExecutablePlan

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
    plan = ExecutablePlan.from_specs(specs, policies)
    # groups are the canonical slots in order (slot i -> groups[i-1]).
    counts = [g.count for g in plan.groups]

    # construction extend -> type4; edit extend -> type5; edit trim-only -> type6;
    # edit full -> type5; compound -> type7
    assert counts[3] == 4   # slot 4: neural_on_construction
    assert counts[4] == 6   # slot 5: extend_only(4) + full(2)
    assert counts[5] == 2   # slot 6: trim_only
    assert counts[6] == 2   # slot 7: compound
    assert plan.needs_construction and plan.needs_edit
    assert plan.total_bees == sum(s.count for s in specs)


def test_beespec_dataclass_defaults():
    s = BeeSpec(name="x", count=1, operator="neural_route_action", policy="edit",
               allowed_actions=["extend"])
    assert s.route_selection == "uniform" and s.acceptance == "greedy"

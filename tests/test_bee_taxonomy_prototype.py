"""U2 prototype: the declarative bee schema can express the paper's Our-NBCO mix.

The rich schema (search/bee_sets/*) is an adapter back onto bee_colony's numbered
n_type* counts. Before this prototype it could NOT express the paper's neural
full-route rebuild (type-1 + construction model): a neural construction bee was
always classified as type-4 (one extend step). Adding the ``neural_rebuild``
operator closes that gap. This test proves the ``our_nbco`` bee_set produces the
SAME n_type* plan counts as the flat nbco_variants/our_nbco config
(n_type1=5, n_type5=5), which is what drives bee_colony's behaviour.
"""
from pathlib import Path

from omegaconf import OmegaConf

from connectpt.routes_generator.search.bee_plan import BeeColonyPlan
from connectpt.routes_generator.search.bee_specs import parse_bee_specs
from connectpt.routes_generator.search.search_policies import (
    ConstructionSearchPolicy, EditSearchPolicy)

CFG = Path(__file__).resolve().parents[1] / "connectpt" / "routes_generator" / "cfg"
BEE_SET = CFG / "search" / "bee_sets" / "our_nbco.yaml"
FLAT = CFG / "experiments" / "nbco_variants" / "our_nbco_mumford0.yaml"


def _policies():
    # role-only adapters (validation uses ROLE_CAPABILITIES, not the model)
    return {
        "construction": ConstructionSearchPolicy(model=None, name="construction"),
        "edit": EditSearchPolicy(model=None, name="edit"),
    }


def test_our_nbco_bee_set_matches_flat_counts():
    specs = parse_bee_specs(OmegaConf.load(BEE_SET).bees)
    plan = BeeColonyPlan.from_specs(specs, _policies())

    # The flat config's per-type counts are the ground truth.
    flat = OmegaConf.load(FLAT)
    expected = {f"n_type{i}": int(flat.get(f"n_type{i}_bees", 0)) for i in range(1, 8)}
    assert expected["n_type1"] == 5 and expected["n_type5"] == 5  # sanity on the fixture

    assert plan.counts == expected, (plan.counts, expected)
    assert plan.needs_construction is True  # type-1 rebuild drives the construction model
    assert plan.needs_edit is True          # type-5 edit bees drive the edit model


def test_neural_rebuild_is_type1_not_type4():
    """A neural construction rebuild bee must map to type-1 (full rebuild), not
    type-4 (single construction-extend step)."""
    specs = parse_bee_specs([
        {"name": "r", "count": 3, "operator": "neural_rebuild", "policy": "construction"},
    ])
    plan = BeeColonyPlan.from_specs(specs, _policies())
    assert plan.counts["n_type1"] == 3
    assert plan.counts["n_type4"] == 0
    assert plan.needs_construction is True

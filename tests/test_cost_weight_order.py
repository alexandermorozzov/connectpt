"""Cost-weight ordering invariants (M017, final keyed design).

Weights live on the state ONLY as a named dict (state.cost_weights) — the
same way the BCO cost consumes them (lookup by key at each use site). A
positional vector exists only where a neural net needs one and is assembled
in a single place, ``state.get_cost_weights_tensor(key_order)``, from an
EXPLICIT key tuple supplied by the caller:

- models declare their spec themselves (``models.WEIGHT_FEATURE_KEYS`` /
  ``weight_feature_keys`` class attribute) — CHECKPOINT-LOCKED to the order
  the frozen checkpoints were trained with (== the historical implicit
  ``sorted(keys)``);
- the cost/reward math passes COST_WEIGHT_KEY_ORDER.

These tests pin the spec, the keyed-only surface of the state, and that a
wrong order cannot arise implicitly (no default ordering anywhere).
"""

import pytest
import torch

from connectpt.routes_generator.citygraph_dataset import CityGraphData
from connectpt.routes_generator.improvement_learning import D3POValueModule
from connectpt.routes_generator.inductive_route_learning import NNBaseline
from connectpt.routes_generator.models import (
    RouteGeneratorBase,
    RouteScorer,
    WEIGHT_FEATURE_KEYS,
)
from connectpt.routes_generator.transit_time_estimator import (
    COST_WEIGHT_KEY_ORDER,
    MyCostModule,
    RouteGenBatchState,
)


def _make_state(**cost_kwargs):
    n_nodes = 4
    node_locs = torch.stack((
        torch.arange(n_nodes, dtype=torch.float32),
        torch.zeros(n_nodes, dtype=torch.float32),
    ), dim=-1)
    street_adj = torch.full((n_nodes, n_nodes), float("inf"))
    street_adj.fill_diagonal_(0.0)
    for node_idx in range(n_nodes - 1):
        street_adj[node_idx, node_idx + 1] = 1.0
        street_adj[node_idx + 1, node_idx] = 1.0
    demand = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    demand[0, n_nodes - 1] = 5.0
    demand[n_nodes - 1, 0] = 5.0
    graph = CityGraphData.from_tensors(node_locs, street_adj, demand,
                                       pos_only=False)
    cost_obj = MyCostModule(**cost_kwargs)
    return RouteGenBatchState(graph, cost_obj, n_routes_to_plan=1,
                              min_route_len=2, max_route_len=4)


def test_model_feature_spec_is_checkpoint_locked():
    # The models' spec must equal the historical implicit sorted() order the
    # frozen checkpoints were trained with. If this fires, the tuple was
    # reordered or a key renamed — either breaks every existing checkpoint
    # (silent feature permutation).
    assert WEIGHT_FEATURE_KEYS == tuple(sorted(COST_WEIGHT_KEY_ORDER))
    assert WEIGHT_FEATURE_KEYS == (
        'demand_time_weight',
        'median_connectivity_weight',
        'route_time_weight',
    )
    # Same keys as the cost math, different order; must never drift apart.
    assert set(WEIGHT_FEATURE_KEYS) == set(COST_WEIGHT_KEY_ORDER)


def test_models_and_critics_declare_the_spec():
    # Every network that consumes cost-weight features declares its spec as
    # a class attribute (the state never chooses an order itself).
    for cls in (RouteGeneratorBase, RouteScorer, NNBaseline, D3POValueModule):
        assert cls.weight_feature_keys == WEIGHT_FEATURE_KEYS
    # Critics' legacy input tail is keyed by the canonical cost-math order.
    for cls in (NNBaseline, D3POValueModule):
        assert cls.weight_input_keys == COST_WEIGHT_KEY_ORDER


def test_state_exposes_weights_only_as_named_dict():
    state = _make_state()
    # The positional property is gone: no way to get an implicitly-ordered
    # vector off the state.
    assert not hasattr(state, "cost_weights_tensor")
    # get_cost_weights_tensor requires an explicit key order (no default).
    with pytest.raises(TypeError):
        state.get_cost_weights_tensor()
    # get_global_state_features requires the model's spec.
    with pytest.raises(TypeError):
        state.get_global_state_features()


def test_feature_vector_slots_follow_the_model_spec():
    state = _make_state(demand_time_weight=0.0, route_time_weight=1.0,
                        median_connectivity_weight=0.0)
    feats = state.get_cost_weights_tensor(WEIGHT_FEATURE_KEYS)
    # alpha=1 (pure route time): slot layout is (demand, conn, ROUTE).
    assert feats.shape[-1] == 3
    assert feats[0].tolist() == [0.0, 0.0, 1.0]

    state = _make_state(demand_time_weight=0.0, route_time_weight=0.0,
                        median_connectivity_weight=1.0)
    # alpha=0 (pure connectivity): the CONN slot is index 1, not 2.
    feats = state.get_cost_weights_tensor(WEIGHT_FEATURE_KEYS)
    assert feats[0].tolist() == [0.0, 1.0, 0.0]


def test_cost_math_order_is_a_distinct_explicit_view():
    state = _make_state(demand_time_weight=0.25, route_time_weight=0.5,
                        median_connectivity_weight=0.25)
    canonical = state.get_cost_weights_tensor(COST_WEIGHT_KEY_ORDER)
    assert canonical[0].tolist() == [0.25, 0.5, 0.25]
    # Model-feature view is the same weights, permuted (route <-> conn).
    feats = state.get_cost_weights_tensor(WEIGHT_FEATURE_KEYS)
    assert feats[0].tolist() == [0.25, 0.25, 0.5]


def test_global_state_features_embed_weights_in_model_order():
    state = _make_state(demand_time_weight=0.0, route_time_weight=1.0,
                        median_connectivity_weight=0.0)
    gf = state.get_global_state_features(
        weight_feature_keys=WEIGHT_FEATURE_KEYS)
    # First three global features are the cost weights in the model spec
    # order: (demand, conn, route) -> route_time carries the 1.0.
    assert gf[0, :3].tolist() == [0.0, 0.0, 1.0]

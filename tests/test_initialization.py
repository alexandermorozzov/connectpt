from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch

from connectpt.routes_generator.citygraph_dataset import CityGraphData, STOP_KEY
from connectpt.routes_generator.eval_route_generator import sample_from_model
from connectpt.routes_generator.improvement_learning import (
    _collect_lc_improvement_cfg_d3po_rollout,
    _collect_lc_improvement_cfg_ppo_rollout,
    _compute_ppo_returns_and_advantages,
    _make_route_context_state,
    _update_lc_improvement_cfg_d3po_from_rollout,
    _update_lc_improvement_cfg_ppo_from_rollout,
    _update_reward_baseline_cost,
    rollout_lc_improvement,
    train_lc_improvement_cfg_d3po,
)
from connectpt.routes_generator import improvement_learning as il
from connectpt.routes_generator.initialization import (
    prepare_current_routes,
    prepare_init_network,
)
from connectpt.routes_generator.models import (
    PathCombiningRouteGenerator,
    TrimPathCombiningRouteGenerator,
)
from connectpt.routes_generator.transit_time_estimator import (
    MyCostModule,
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_HALT,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_TRIM_START,
    RouteGenBatchState,
)


class NeverCalledModel:
    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        raise AssertionError("Model rollout should have been skipped")


class RevisitOnlyModel(NeverCalledModel):
    def __init__(self):
        self.visited_routes = []

    def plan_new_route(self, state, greedy=False):
        self.visited_routes.append(state.current_routes[0].tolist())
        halt = torch.full((state.batch_size, 2), -1, dtype=torch.long,
                          device=state.device)
        state.shortest_path_action(halt)
        logits = torch.zeros(state.batch_size, device=state.device)
        entropy = torch.zeros(state.batch_size, device=state.device)
        return halt, logits, entropy


class RecordingHaltModel:
    def __init__(self):
        self.current_routes = []
        self.context_counts = []

    def plan_new_route(self, state, greedy=False):
        self.current_routes.append(state.current_routes.detach().cpu().clone())
        self.context_counts.append(state.n_finished_routes.detach().cpu().clone())
        halt = torch.full((state.batch_size, 2), -1, dtype=torch.long,
                          device=state.device)
        state.shortest_path_action(halt)
        logits = torch.zeros(state.batch_size, device=state.device)
        entropy = torch.zeros(state.batch_size, device=state.device)
        return halt, logits, entropy


class FakePlanNewRouteModel:
    def __init__(self):
        self.setup_called = False
        self.encode_saw_features = False

    def setup_planning(self, state):
        self.setup_called = True
        n_nodes = state.graph_data[STOP_KEY].x.shape[0]
        feats = torch.ones((n_nodes, 1), dtype=torch.float32,
                           device=state.device)
        state.set_normalized_features(feats)
        return state

    def _encode_graph(self, state):
        assert state.norm_node_features.shape[1] == 1
        self.encode_saw_features = True
        return "encoding"

    def step(self, state, greedy=False, actions=None, precalc_data=None):
        halt = torch.full((state.batch_size, 2), -1, dtype=torch.long,
                          device=state.device)
        logits = torch.zeros(state.batch_size, device=state.device)
        entropy = torch.zeros(state.batch_size, device=state.device)
        return halt, logits, entropy

    def _log_step_counts(self, counts):
        pass


class FakeTrimPlanNewRouteModel:
    def __init__(self):
        self.encoded_route_lengths = []
        self.step_count = 0

    def setup_planning(self, state):
        return state

    def _encode_graph(self, state):
        self.encoded_route_lengths.append(
            int(state.current_route_n_stops[0].item())
        )
        return f"encoding-{len(self.encoded_route_lengths)}"

    def step_route_action(
        self,
        state,
        greedy=False,
        actions=None,
        precalc_data=None,
        action_kinds=None,
        allow_halt=True,
        allow_trim_start=True,
        allow_trim_end=True,
    ):
        assert precalc_data == f"encoding-{self.step_count + 1}"
        self.step_count += 1
        logits = torch.zeros(state.batch_size, device=state.device)
        entropy = torch.zeros(state.batch_size, device=state.device)

        if self.step_count == 1:
            assert bool(torch.as_tensor(allow_trim_start).all())
            assert bool(torch.as_tensor(allow_trim_end).all())
            kinds = torch.full(
                (state.batch_size,),
                ROUTE_ACTION_TRIM_START,
                dtype=torch.long,
                device=state.device,
            )
            actions = torch.tensor([[0, 1]], dtype=torch.long,
                                   device=state.device)
        else:
            assert not bool(torch.as_tensor(allow_trim_start).any())
            assert not bool(torch.as_tensor(allow_trim_end).any())
            kinds = torch.full(
                (state.batch_size,),
                ROUTE_ACTION_HALT,
                dtype=torch.long,
                device=state.device,
            )
            actions = torch.full((state.batch_size, 2), -1,
                                 dtype=torch.long, device=state.device)
        return kinds, actions, logits, entropy

    def _log_step_counts(self, counts):
        pass


class FakeValueModule:
    def from_state(self, state):
        return torch.zeros(state.batch_size, dtype=torch.float32,
                           device=state.device)

    def update(self, returns):
        self.last_returns = returns.detach().clone()


class FakeVectorValueModule:
    def from_state(self, state):
        return torch.zeros(state.batch_size, 3, dtype=torch.float32,
                           device=state.device)

    def update(self, returns):
        self.last_returns = returns.detach().clone()


class HaltPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_bias = torch.nn.Parameter(torch.zeros(()))

    def step(self, state, greedy=False, actions=None, precalc_data=None):
        if actions is None:
            actions = torch.full(
                (state.batch_size, 2), -1, dtype=torch.long,
                device=state.device)
        else:
            actions = actions.to(device=state.device, dtype=torch.long)
        logits = self.logit_bias.expand(state.batch_size)
        entropy = torch.zeros(state.batch_size, dtype=torch.float32,
                              device=state.device)
        return actions, logits, entropy


class IdentityGraphNet(torch.nn.Module):
    in_node_dim = 2
    in_edge_dim = 14
    gives_edge_features = False

    def forward(self, data):
        return data.x


def make_line_state(n_nodes=4, n_routes_to_plan=1, min_route_len=2,
                    max_route_len=4):
    graph = make_line_graph(n_nodes)
    return RouteGenBatchState(
        graph,
        MyCostModule(),
        n_routes_to_plan=n_routes_to_plan,
        min_route_len=min_route_len,
        max_route_len=max_route_len,
    )


def make_line_graph(n_nodes=4):
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
    return graph


def test_prepare_init_network_broadcasts_single_batch():
    init_network = torch.tensor([[[0, 1, -1], [2, 3, -1]]], dtype=torch.long)

    prepared = prepare_init_network(
        init_network,
        batch_size=3,
        n_routes=3,
        device=torch.device("cpu"),
    )

    assert prepared.shape == (3, 2, 3)
    assert torch.equal(prepared[0], init_network[0])
    assert torch.equal(prepared[1], init_network[0])
    assert torch.equal(prepared[2], init_network[0])


def test_prepare_init_network_rejects_too_many_routes():
    init_network = torch.tensor([[[0, 1, -1], [2, 3, -1]]], dtype=torch.long)

    with pytest.raises(ValueError, match="too many routes"):
        prepare_init_network(
            init_network,
            batch_size=1,
            n_routes=1,
            device=torch.device("cpu"),
        )


def test_prepare_init_network_rejects_batch_mismatch():
    init_network = torch.tensor(
        [
            [[0, 1, -1]],
            [[2, 3, -1]],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="batch size"):
        prepare_init_network(
            init_network,
            batch_size=3,
            n_routes=2,
            device=torch.device("cpu"),
        )


def test_fold_node_descs_uses_per_graph_offsets_in_batch():
    state = SimpleNamespace(
        batch_size=2,
        max_n_nodes=3,
        n_nodes=torch.tensor([2, 3]),
    )
    node_descs = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    folded, pad_mask = PathCombiningRouteGenerator.fold_node_descs(
        node_descs, state)

    assert torch.equal(folded[0, :2], node_descs[:2])
    assert torch.equal(folded[1, :3], node_descs[2:5])
    assert torch.equal(folded[0, 2], torch.zeros(2))
    assert pad_mask.tolist() == [
        [False, False, True],
        [False, False, False],
    ]


def test_make_route_context_state_sets_active_route_and_context_routes():
    graph = make_line_graph(n_nodes=5)
    cost_obj = MyCostModule()
    seed_routes = torch.tensor(
        [[[0, 1, 2, -1], [2, 3, 4, -1], [-1, -1, -1, -1]]],
        dtype=torch.long,
    )

    state = _make_route_context_state(
        cost_obj,
        graph,
        seed_routes,
        route_idx=1,
        min_route_len=2,
        max_route_len=4,
        cost_weights=cost_obj.get_weights(torch.device("cpu")),
    )

    assert [route.tolist() for route in state._finished_routes[0]] == [
        [0, 1, 2],
    ]
    assert state.current_routes[0, :3].tolist() == [2, 3, 4]
    assert torch.isfinite(state.route_mat[0, 0, 1])
    assert torch.isfinite(state.route_mat[0, 2, 3])


def test_lc_improvement_greedy_rollout_keeps_first_graph_context_in_batch():
    graph = make_line_graph(n_nodes=5)
    other_graph = make_line_graph(n_nodes=5)
    cost_obj = MyCostModule()
    single_routes = torch.tensor(
        [[[0, 1, -1, -1], [2, 3, 4, -1]]],
        dtype=torch.long,
    )
    batched_routes = torch.tensor(
        [
            [[0, 1, -1, -1], [2, 3, 4, -1]],
            [[4, 3, -1, -1], [1, 2, 3, -1]],
        ],
        dtype=torch.long,
    )
    cost_weights = cost_obj.get_weights(torch.device("cpu"))
    single_model = RecordingHaltModel()
    batch_model = RecordingHaltModel()

    single_output = rollout_lc_improvement(
        single_model,
        cost_obj,
        Batch.from_data_list([graph]),
        single_routes,
        min_route_len=2,
        max_route_len=4,
        greedy=True,
        cost_weights=cost_weights,
        return_actions=True,
        max_route_edit_steps=4,
    )
    batch_output = rollout_lc_improvement(
        batch_model,
        cost_obj,
        Batch.from_data_list([graph, other_graph]),
        batched_routes,
        min_route_len=2,
        max_route_len=4,
        greedy=True,
        cost_weights=cost_weights,
        return_actions=True,
        max_route_edit_steps=4,
    )

    single_state, *_, single_actions, _ = single_output
    batch_state, *_, batch_actions, _ = batch_output

    assert torch.equal(batch_state.current_routes[0],
                       single_state.current_routes[0])
    assert [route.tolist() for route in batch_state.routes[0]] == [
        route.tolist() for route in single_state.routes[0]
    ]
    for route_idx in range(single_routes.shape[1]):
        assert torch.equal(
            batch_model.current_routes[route_idx][0],
            single_model.current_routes[route_idx][0],
        )
        assert torch.equal(
            batch_model.context_counts[route_idx][0:1],
            single_model.context_counts[route_idx],
        )
        assert torch.equal(
            batch_actions[route_idx][0:1],
            single_actions[route_idx],
        )


def test_prepare_current_routes_broadcasts_and_pads_single_route_list():
    prepared = prepare_current_routes(
        [0, 1, 2],
        batch_size=2,
        max_n_nodes=5,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [0, 1, 2, -1, -1],
            [0, 1, 2, -1, -1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(prepared, expected)


def test_prepare_current_routes_supports_empty_routes_inside_batch():
    prepared = prepare_current_routes(
        [[], [1, 2]],
        batch_size=2,
        max_n_nodes=4,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [-1, -1, -1, -1],
            [1, 2, -1, -1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(prepared, expected)


def test_set_current_routes_can_continue_and_finalize_route():
    node_locs = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    inf = float("inf")
    street_adj = torch.tensor(
        [
            [0.0, 1.0, inf],
            [1.0, 0.0, 1.0],
            [inf, 1.0, 0.0],
        ]
    )
    demand = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    graph = CityGraphData.from_tensors(node_locs, street_adj, demand,
                                       pos_only=True)
    state = RouteGenBatchState(
        graph,
        MyCostModule(),
        n_routes_to_plan=1,
        min_route_len=2,
        max_route_len=3,
    )

    state.set_current_routes([0, 1])

    assert state.has_current_route.item()
    assert state.current_route_n_stops.item() == 2
    assert state.n_routes_left_to_plan.item() == 1
    assert state.total_route_time.item() > 0
    assert state.routes[0][0].tolist() == [0, 1]

    state.shortest_path_action(torch.tensor([[1, 2]], dtype=torch.long))

    assert state.current_routes[0, :3].tolist() == [0, 1, 2]
    assert state.current_route_n_stops.item() == 3
    assert state.routes[0][0].tolist() == [0, 1, 2]

    state.shortest_path_action(torch.tensor([[-1, -1]], dtype=torch.long))

    assert not state.has_current_route.item()
    assert state.current_route_time.item() == 0
    assert state.n_routes_left_to_plan.item() == 0
    assert len(state.routes[0]) == 1
    assert state.routes[0][0].tolist() == [0, 1, 2]


def test_route_state_trim_actions_rebuild_current_route_graph():
    state = make_line_state(n_nodes=4, max_route_len=4)
    state.set_current_routes([0, 1, 2, 3])

    action_kinds = torch.tensor([ROUTE_ACTION_TRIM_START], dtype=torch.long)
    actions = torch.tensor([[0, 2]], dtype=torch.long)
    state.apply_route_actions(action_kinds, actions)

    assert state.current_routes[0, :2].tolist() == [2, 3]
    assert torch.isinf(state.route_mat[0, 0, 1])
    assert torch.isfinite(state.route_mat[0, 2, 3])

    state = make_line_state(n_nodes=4, max_route_len=4)
    state.set_current_routes([0, 1, 2, 3])

    action_kinds = torch.tensor([ROUTE_ACTION_TRIM_END], dtype=torch.long)
    actions = torch.tensor([[1, 3]], dtype=torch.long)
    state.apply_route_actions(action_kinds, actions)

    assert state.current_routes[0, :2].tolist() == [0, 1]
    assert torch.isinf(state.route_mat[0, 2, 3])
    assert torch.isfinite(state.route_mat[0, 0, 1])


def test_route_state_context_masks_track_finished_not_current_routes():
    state = make_line_state(n_nodes=4, n_routes_to_plan=2, max_route_len=4)

    assert not state.context_node_covered_mask.any()
    assert not state.context_edge_covered_mask.any()

    state.add_new_routes(torch.tensor([[[0, 1, 2, -1]]], dtype=torch.long))

    assert state.context_node_covered_mask[0].tolist() == [
        True, True, True, False]
    assert state.context_edge_covered_mask[0, 0, 1]
    assert state.context_edge_covered_mask[0, 1, 0]
    assert state.context_edge_covered_mask[0, 1, 2]
    assert state.context_edge_covered_mask[0, 2, 1]
    assert not state.context_edge_covered_mask[0, 2, 3]

    node_mask_before = state.context_node_covered_mask.clone()
    edge_mask_before = state.context_edge_covered_mask.clone()
    state.set_current_routes([2, 3])

    assert torch.equal(state.context_node_covered_mask, node_mask_before)
    assert torch.equal(state.context_edge_covered_mask, edge_mask_before)
    assert torch.isfinite(state.route_mat[0, 2, 3])

    state.replace_routes(torch.tensor([[[1, 2, 3, -1]]], dtype=torch.long))

    assert state.context_node_covered_mask[0].tolist() == [
        False, True, True, True]
    assert not state.context_edge_covered_mask[0, 0, 1]
    assert state.context_edge_covered_mask[0, 1, 2]
    assert state.context_edge_covered_mask[0, 2, 3]

    state.clear_routes()

    assert not state.context_node_covered_mask.any()
    assert not state.context_edge_covered_mask.any()


def test_route_state_context_masks_include_fixed_routes():
    graph = make_line_graph(n_nodes=4)
    graph.fixed_routes = torch.tensor([[0, 3, -1, -1]], dtype=torch.long)
    state = RouteGenBatchState(
        graph,
        MyCostModule(),
        n_routes_to_plan=1,
        min_route_len=2,
        max_route_len=4,
    )

    assert state.n_finished_routes.item() == 0
    assert state.context_node_covered_mask[0].tolist() == [
        True, False, False, True]
    assert state.context_edge_covered_mask[0, 0, 3]
    assert state.context_edge_covered_mask[0, 3, 0]

    node_mask_before = state.context_node_covered_mask.clone()
    edge_mask_before = state.context_edge_covered_mask.clone()
    state.set_current_routes([1, 2])

    assert torch.equal(state.context_node_covered_mask, node_mask_before)
    assert torch.equal(state.context_edge_covered_mask, edge_mask_before)


def test_untrained_trim_model_can_emit_and_apply_forced_trim_action():
    state = make_line_state(n_nodes=4, max_route_len=4)
    state.set_current_routes([0, 1, 2, 3])
    model = TrimPathCombiningRouteGenerator(
        backbone_net=IdentityGraphNet(),
        mean_stop_time_s=0,
        embed_dim=2,
        n_nodepair_layers=1,
        n_pathscorer_layers=1,
        pathscorer_hidden_dim=8,
        n_trim_scorer_layers=1,
        trim_scorer_hidden_dim=8,
        n_halt_layers=1,
        symmetric_routes=True,
        serial_halting=True,
    )
    state = model.setup_planning(state)

    forced_kind = torch.tensor([ROUTE_ACTION_TRIM_START], dtype=torch.long)
    forced_action = torch.tensor([[0, 2]], dtype=torch.long)
    action_kinds, actions, logits, entropy = model.step_route_action(
        state,
        greedy=True,
        actions=forced_action,
        action_kinds=forced_kind,
    )

    assert action_kinds.tolist() == [ROUTE_ACTION_TRIM_START]
    assert actions.tolist() == [[0, 2]]
    assert logits.shape == (1,)
    assert entropy.shape == (1,)

    state.apply_route_actions(action_kinds, actions)

    assert state.current_routes[0, :2].tolist() == [2, 3]
    assert state.current_route_n_stops.item() == 2


def test_trim_model_action_mask_can_disable_extend_actions():
    state = make_line_state(n_nodes=4, max_route_len=4)
    state.set_current_routes([0, 1, 2, 3])
    model = TrimPathCombiningRouteGenerator(
        backbone_net=IdentityGraphNet(),
        mean_stop_time_s=0,
        embed_dim=2,
        n_nodepair_layers=1,
        n_pathscorer_layers=1,
        pathscorer_hidden_dim=8,
        n_trim_scorer_layers=1,
        trim_scorer_hidden_dim=8,
        n_halt_layers=1,
        symmetric_routes=True,
        serial_halting=True,
    )
    state = model.setup_planning(state)

    action_kinds, actions, logits, entropy = model.step_route_action(
        state,
        greedy=True,
        allow_halt=False,
        allow_extend=False,
        allow_trim_start=True,
        allow_trim_end=True,
    )

    assert action_kinds.item() in {
        ROUTE_ACTION_TRIM_START,
        ROUTE_ACTION_TRIM_END,
    }
    assert action_kinds.item() != ROUTE_ACTION_EXTEND
    assert actions.shape == (1, 2)
    assert logits.shape == (1,)
    assert entropy.shape == (1,)


def test_trim_model_action_mask_can_disable_trim_actions():
    state = make_line_state(n_nodes=4, max_route_len=4)
    state.set_current_routes([0, 1, 2, 3])
    model = TrimPathCombiningRouteGenerator(
        backbone_net=IdentityGraphNet(),
        mean_stop_time_s=0,
        embed_dim=2,
        n_nodepair_layers=1,
        n_pathscorer_layers=1,
        pathscorer_hidden_dim=8,
        n_trim_scorer_layers=1,
        trim_scorer_hidden_dim=8,
        n_halt_layers=1,
        symmetric_routes=True,
        serial_halting=True,
    )
    state = model.setup_planning(state)

    action_kinds, actions, logits, entropy = model.step_route_action(
        state,
        greedy=True,
        allow_extend=False,
        allow_trim_start=False,
        allow_trim_end=False,
    )

    assert action_kinds.tolist() == [ROUTE_ACTION_HALT]
    assert actions.tolist() == [[-1, -1]]
    assert logits.shape == (1,)
    assert entropy.shape == (1,)


def test_trim_model_overlap_features_compare_to_other_routes():
    state = make_line_state(n_nodes=5, n_routes_to_plan=2, max_route_len=5)
    state.add_new_routes(torch.tensor([[[0, 1, 2, -1, -1]]],
                                      dtype=torch.long))
    state.set_current_routes([0, 1, 2, 3, 4])
    model = TrimPathCombiningRouteGenerator(
        backbone_net=IdentityGraphNet(),
        mean_stop_time_s=0,
        embed_dim=2,
        n_nodepair_layers=1,
        n_pathscorer_layers=1,
        pathscorer_hidden_dim=8,
        n_trim_scorer_layers=1,
        trim_scorer_hidden_dim=8,
        n_halt_layers=1,
        symmetric_routes=True,
        serial_halting=True,
    )

    node_masks, edge_masks = model._get_other_route_overlap_masks(state)
    route = state.current_routes[0]
    route = route[route > -1]
    overlap_features = model._get_trim_overlap_features(
        removed_nodes=route[:2],
        removed_edge_nodes=route[:3],
        kept_nodes=route[2:],
        kept_edge_nodes=route[2:],
        other_node_mask=node_masks[0],
        other_edge_mask=edge_masks[0],
        dtype=torch.float32,
    )

    assert model.trim_action_feat_dim == 26
    assert torch.allclose(
        overlap_features,
        torch.tensor([
            1.0, 0.0,        # removed nodes overlap/unique
            1.0, 0.0,        # removed edges overlap/unique
            1.0 / 3.0, 2.0 / 3.0,  # kept nodes overlap/unique
            0.0, 1.0,        # kept edges overlap/unique
        ]),
    )


def test_vectorized_trim_features_match_removed_and_kept_context_overlap():
    state = make_line_state(n_nodes=5, n_routes_to_plan=2, max_route_len=5)
    state.add_new_routes(torch.tensor([[[0, 1, 2, -1, -1]]],
                                      dtype=torch.long))
    state.set_current_routes([0, 1, 2, 3, 4])
    model = TrimPathCombiningRouteGenerator(
        backbone_net=IdentityGraphNet(),
        mean_stop_time_s=0,
        embed_dim=2,
        n_nodepair_layers=1,
        n_pathscorer_layers=1,
        pathscorer_hidden_dim=8,
        n_trim_scorer_layers=1,
        trim_scorer_hidden_dim=8,
        n_halt_layers=1,
        symmetric_routes=True,
        serial_halting=True,
    )
    global_features = state.get_global_state_features().to(dtype=torch.float32)

    features, valid, from_nodes, to_nodes = model._get_trim_candidate_features(
        state, global_features, torch.float32, trim_start=True)
    overlap_start = features[0, 2, 6:14]

    assert valid[0, 2]
    assert from_nodes[0, 2].item() == 0
    assert to_nodes[0, 2].item() == 2
    assert torch.allclose(
        overlap_start,
        torch.tensor([
            1.0, 0.0,
            1.0, 0.0,
            1.0 / 3.0, 2.0 / 3.0,
            0.0, 1.0,
        ]),
    )

    features, valid, from_nodes, to_nodes = model._get_trim_candidate_features(
        state, global_features, torch.float32, trim_start=False)
    overlap_end = features[0, 2, 6:14]

    assert valid[0, 2]
    assert from_nodes[0, 2].item() == 2
    assert to_nodes[0, 2].item() == 4
    assert torch.allclose(
        overlap_end,
        torch.tensor([
            0.0, 1.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
        ]),
    )


def test_zero_trim_reward_keeps_pretrim_reward_baseline():
    prev_cost = torch.tensor([10.0, 20.0, 30.0])
    new_cost = torch.tensor([15.0, 18.0, 25.0])
    action_kinds = torch.tensor([
        ROUTE_ACTION_TRIM_START,
        ROUTE_ACTION_EXTEND,
        ROUTE_ACTION_HALT,
    ])
    active = torch.tensor([True, True, False])

    updated = _update_reward_baseline_cost(
        prev_cost, new_cost, action_kinds, active,
        zero_trim_reward=True)

    assert updated.tolist() == [10.0, 18.0, 30.0]


def test_my_cost_components_reconstruct_scalar_cost_for_simplex_weights():
    cost_obj = MyCostModule(
        demand_time_weight=0.2,
        route_time_weight=0.3,
        median_connectivity_weight=0.5,
        use_weighted_connectivity=True,
    )
    weights = {
        "demand_time_weight": torch.tensor([0.2]),
        "route_time_weight": torch.tensor([0.3]),
        "median_connectivity_weight": torch.tensor([0.5]),
    }
    state = RouteGenBatchState(
        make_line_graph(n_nodes=3),
        cost_obj,
        n_routes_to_plan=1,
        min_route_len=2,
        max_route_len=3,
        cost_weights=weights,
    )
    state.set_current_routes([0, 1])

    result = cost_obj(state)
    components = cost_obj.get_cost_components(state, result=result)
    preferences = cost_obj.get_preference_weights(state, normalize=True)

    assert torch.allclose(
        (components * preferences).sum(dim=-1),
        result.cost,
        atol=1e-6,
    )


def test_disabling_a_cost_component_drops_it_from_the_weighted_cost():
    weights = {
        "demand_time_weight": torch.tensor([0.2]),
        "route_time_weight": torch.tensor([0.3]),
        "median_connectivity_weight": torch.tensor([0.5]),
    }

    def _build(cost_obj):
        state = RouteGenBatchState(
            make_line_graph(n_nodes=3),
            cost_obj,
            n_routes_to_plan=1,
            min_route_len=2,
            max_route_len=3,
            cost_weights=dict(weights),
        )
        state.set_current_routes([0, 1])
        return state

    cost_obj = MyCostModule(
        demand_time_weight=0.2, route_time_weight=0.3,
        median_connectivity_weight=0.5, use_weighted_connectivity=True,
        disabled_components=["connectivity"])

    assert cost_obj.enabled_component_names == ("demand", "route")
    assert cost_obj.disabled_component_names == ("connectivity",)

    state = _build(cost_obj)
    result = cost_obj(state)
    preferences = cost_obj.get_preference_weights(state, normalize=True)
    components = cost_obj.get_cost_components(state, result=result)

    # The connectivity component carries zero preference mass and the
    # demand / route survivors renormalize to sum to one.
    assert preferences[0, 2].item() == 0.0
    assert torch.isclose(preferences.sum(dim=-1),
                         torch.ones(1), atol=1e-6).all()
    # Scalar cost is the weighted sum over the enabled components only.
    assert torch.allclose(
        (components * preferences).sum(dim=-1), result.cost, atol=1e-6)


def test_my_cost_module_all_components_enabled_is_a_noop():
    cost_obj = MyCostModule()
    assert cost_obj.all_components_enabled
    raw = torch.tensor([[0.33, 0.33, 0.33]])
    assert torch.equal(cost_obj.apply_enabled_mask(raw), raw)


def test_vector_ppo_returns_and_advantages_keep_objective_dim():
    rewards = torch.ones((2, 3, 3), dtype=torch.float32)
    value_estimates = torch.zeros_like(rewards)
    dones = torch.zeros((2, 3), dtype=torch.bool)
    final_value_estimates = torch.zeros((3, 3), dtype=torch.float32)

    returns, advantages = _compute_ppo_returns_and_advantages(
        rewards, value_estimates, dones, final_value_estimates,
        gamma=1.0, use_gae=False, gae_lambda=1.0)

    assert returns.shape == (2, 3, 3)
    assert advantages.shape == (2, 3, 3)
    assert torch.equal(returns[0], torch.full((3, 3), 2.0))


def test_train_lc_improvement_cfg_dispatches_by_trainer(monkeypatch):
    calls = []

    def fake_ppo(*args, **kwargs):
        calls.append("ppo")
        return "ppo-result"

    def fake_d3po(*args, **kwargs):
        calls.append("d3po")
        return "d3po-result"

    monkeypatch.setattr(il, "train_lc_improvement_cfg_ppo", fake_ppo)
    monkeypatch.setattr(il, "train_lc_improvement_cfg_d3po", fake_d3po)

    assert il.train_lc_improvement_cfg(
        None, None, None, None, None, SimpleNamespace(trainer="ppo")
    ) == "ppo-result"
    assert il.train_lc_improvement_cfg(
        None, None, None, None, None, SimpleNamespace(trainer="d3po")
    ) == "d3po-result"
    assert calls == ["ppo", "d3po"]


def test_train_lc_improvement_cfg_d3po_rejects_incumbent_reward():
    cfg = SimpleNamespace(
        eval=SimpleNamespace(min_route_len=2, max_route_len=3),
        # Shared PPO-style hyperparameters live in cfg.ppo (both ppo and
        # d3po trainers read them from there). cfg.d3po holds only
        # D3PO-specific knobs.
        ppo=SimpleNamespace(
            n_iterations=1,
            val_period=1,
            horizon=1,
            n_epochs=1,
            minibatch_size=1,
            epsilon=0.2,
            use_gae=False,
            gae_lambda=1.0,
        ),
        d3po=SimpleNamespace(
            n_objectives=3,
            diversity_weight=0.0,
            diversity_alpha=1.0,
            preference_noise_sigma=0.0,
            normalize_advantages_per_objective=True,
        ),
        batch_size=1,
        reward_scale=1.0,
        diff_reward=True,
        incumbent_reward=True,
        return_best_routes=False,
        zero_trim_reward=True,
        keep_rollout_on_device=False,
        discount_rate=1.0,
        edit_step_penalty=0.0,
        forced_halt_penalty=0.0,
        entropy_weight=0.0,
        max_trim_actions_per_route=1,
    )

    with pytest.raises(ValueError, match="incumbent_reward=false"):
        train_lc_improvement_cfg_d3po(
            None, None, [], torch.empty(0), torch.device("cpu"), cfg,
            ".", "d3po_rejects_incumbent")


def test_cfg_ppo_rollout_can_keep_states_on_device_and_update():
    device = torch.device("cpu")
    cost_obj = MyCostModule()
    model = HaltPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    value_module = FakeValueModule()

    def make_next_state(prev_state=None):
        state = make_line_state(n_nodes=3, n_routes_to_plan=1,
                                min_route_len=2, max_route_len=3)
        state.set_current_routes([0, 1])
        start_cost = cost_obj(state).cost.detach()
        return state, start_cost

    rollout = _collect_lc_improvement_cfg_ppo_rollout(
        model, cost_obj, make_next_state, value_module, horizon=1,
        reward_scale=1.0, diff_reward=True, supports_route_actions=False,
        device=device, keep_rollout_on_device=True)

    assert rollout["keep_rollout_on_device"]
    assert rollout["states"][0].device == device

    returns, advantages = _compute_ppo_returns_and_advantages(
        rollout["rewards"], rollout["value_estimates"], rollout["dones"],
        rollout["final_value_estimates"], gamma=1.0, use_gae=False,
        gae_lambda=1.0)
    stats = _update_lc_improvement_cfg_ppo_from_rollout(
        model, optimizer, value_module, rollout, returns, advantages,
        ppo_epochs=1, minibatch_size=1, clip_epsilon=0.2,
        entropy_weight=0.0, device=device)

    assert stats["ratio_count"] == 1


def test_cfg_d3po_rollout_can_keep_states_on_device_and_update():
    device = torch.device("cpu")
    cost_obj = MyCostModule()
    model = HaltPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    value_module = FakeVectorValueModule()

    def make_next_state(prev_state=None):
        state = make_line_state(n_nodes=3, n_routes_to_plan=1,
                                min_route_len=2, max_route_len=3)
        state.set_current_routes([0, 1])
        start_result = cost_obj(state)
        return state, start_result

    rollout = _collect_lc_improvement_cfg_d3po_rollout(
        model, cost_obj, make_next_state, value_module, horizon=1,
        reward_scale=1.0, diff_reward=True, supports_route_actions=False,
        device=device, keep_rollout_on_device=True)

    assert rollout["keep_rollout_on_device"]
    assert rollout["states"][0].device == device
    assert rollout["rewards"].shape == (1, 1, 3)

    returns, advantages = _compute_ppo_returns_and_advantages(
        rollout["rewards"], rollout["value_estimates"], rollout["dones"],
        rollout["final_value_estimates"], gamma=1.0, use_gae=False,
        gae_lambda=1.0)
    stats = _update_lc_improvement_cfg_d3po_from_rollout(
        model, optimizer, value_module, rollout, returns, advantages,
        d3po_epochs=1, minibatch_size=1, clip_epsilon=0.2,
        entropy_weight=0.0, device=device, diversity_weight=0.0)

    assert stats["ratio_count"] == 1


def test_sample_from_model_skips_rollout_for_fully_seeded_network():
    node_locs = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    inf = float("inf")
    street_adj = torch.tensor(
        [
            [0.0, 1.0, inf],
            [1.0, 0.0, 1.0],
            [inf, 1.0, 0.0],
        ]
    )
    demand = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    graph = CityGraphData.from_tensors(node_locs, street_adj, demand,
                                       pos_only=True)
    cost_obj = MyCostModule()
    state = RouteGenBatchState(
        graph,
        cost_obj,
        n_routes_to_plan=1,
        min_route_len=2,
        max_route_len=3,
    )
    init_network = torch.tensor([[[0, 1, 2]]], dtype=torch.long)

    result_state = sample_from_model(
        NeverCalledModel(),
        state,
        cost_obj,
        n_samples=3,
        init_network=init_network,
    )

    assert result_state.is_done().item()
    assert len(result_state.routes[0]) == 1
    assert result_state.routes[0][0].tolist() == [0, 1, 2]


def test_sample_from_model_revisits_seed_routes_before_marking_done():
    node_locs = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    inf = float("inf")
    street_adj = torch.tensor(
        [
            [0.0, 1.0, inf],
            [1.0, 0.0, 1.0],
            [inf, 1.0, 0.0],
        ]
    )
    demand = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    graph = CityGraphData.from_tensors(node_locs, street_adj, demand,
                                       pos_only=True)
    cost_obj = MyCostModule()
    state = RouteGenBatchState(
        graph,
        cost_obj,
        n_routes_to_plan=2,
        min_route_len=2,
        max_route_len=4,
    )
    revisit_routes = torch.tensor(
        [[[0, 1, -1], [1, 2, -1]]],
        dtype=torch.long,
    )
    model = RevisitOnlyModel()

    result_state = sample_from_model(
        model,
        state,
        cost_obj,
        n_samples=3,
        revisit_routes=revisit_routes,
    )

    assert len(model.visited_routes) == 2
    assert model.visited_routes[0][:2] == [0, 1]
    assert model.visited_routes[1][:2] == [1, 2]
    assert result_state.is_done().item()
    assert len(result_state.routes[0]) == 2
    assert result_state.routes[0][0].tolist() == [0, 1]
    assert result_state.routes[0][1].tolist() == [1, 2]


def test_plan_new_route_calls_setup_planning_before_encoding():
    node_locs = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    inf = float("inf")
    street_adj = torch.tensor(
        [
            [0.0, 1.0, inf],
            [1.0, 0.0, 1.0],
            [inf, 1.0, 0.0],
        ]
    )
    demand = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    graph = CityGraphData.from_tensors(node_locs, street_adj, demand,
                                       pos_only=True)
    state = RouteGenBatchState(
        graph,
        MyCostModule(),
        n_routes_to_plan=1,
        min_route_len=2,
        max_route_len=4,
    )
    state.set_current_routes([0, 1])
    model = FakePlanNewRouteModel()

    actions, logits, entropy = PathCombiningRouteGenerator.plan_new_route(
        model,
        state,
        greedy=False,
    )

    assert model.setup_called
    assert model.encode_saw_features
    assert actions.shape == (1, 1, 2)
    assert logits.shape == (1,)
    assert entropy.shape == (1,)


def test_trim_plan_new_route_reencodes_after_route_edits():
    state = make_line_state(n_nodes=4, n_routes_to_plan=1,
                            min_route_len=2, max_route_len=4)
    state.set_current_routes([0, 1, 2])
    model = FakeTrimPlanNewRouteModel()

    actions, logits, entropy = TrimPathCombiningRouteGenerator.plan_new_route(
        model,
        state,
        greedy=True,
        max_steps=4,
    )

    assert model.encoded_route_lengths == [3, 2]
    assert actions.shape == (1, 2, 2)
    assert logits.shape == (1,)
    assert entropy.shape == (1,)
    assert state.is_done().item()

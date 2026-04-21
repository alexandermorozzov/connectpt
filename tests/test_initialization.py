import pytest
import torch

from connectpt.routes_generator.citygraph_dataset import CityGraphData, STOP_KEY
from connectpt.routes_generator.eval_route_generator import sample_from_model
from connectpt.routes_generator.initialization import (
    prepare_current_routes,
    prepare_init_network,
)
from connectpt.routes_generator.models import PathCombiningRouteGenerator
from connectpt.routes_generator.transit_time_estimator import (
    MyCostModule,
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

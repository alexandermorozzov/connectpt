import torch

from connectpt.routes_generator.bee_colony import _get_route_selection_weights


def test_demand_weighted_route_selection_prefers_lower_direct_demand_routes():
    bee_networks = torch.tensor(
        [[
            [
                [0, 1, -1],
                [2, 3, -1],
            ]
        ]],
        dtype=torch.long,
    )
    demand = torch.zeros((1, 4, 4), dtype=torch.float32)
    demand[0, 0, 1] = 10.0
    demand[0, 1, 0] = 10.0
    demand[0, 2, 3] = 1.0
    demand[0, 3, 2] = 1.0

    weights = _get_route_selection_weights(
        bee_networks,
        demand,
        use_demand_weighted_route_selection=True,
    )

    assert weights.shape == (1, 1, 2)
    assert weights[0, 0, 1] > weights[0, 0, 0]


def test_demand_weighted_route_selection_falls_back_to_uniform_when_equal():
    bee_networks = torch.tensor(
        [[
            [
                [0, 1, -1],
                [2, 3, -1],
            ]
        ]],
        dtype=torch.long,
    )
    demand = torch.zeros((1, 4, 4), dtype=torch.float32)
    demand[0, 0, 1] = 4.0
    demand[0, 1, 0] = 4.0
    demand[0, 2, 3] = 4.0
    demand[0, 3, 2] = 4.0

    weights = _get_route_selection_weights(
        bee_networks,
        demand,
        use_demand_weighted_route_selection=True,
    )

    assert torch.equal(weights, torch.ones_like(weights))

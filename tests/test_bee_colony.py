import importlib

import torch

from connectpt.routes_generator.bee_colony import _get_route_selection_weights


bee_colony = importlib.import_module("connectpt.routes_generator.bee_colony")


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


def test_get_mutants_routes_type7_through_compound_helper(monkeypatch):
    bee_networks = torch.tensor(
        [[
            [[0, 1, 2, -1], [3, 4, -1, -1]],
            [[1, 2, 3, -1], [4, 5, -1, -1]],
        ]],
        dtype=torch.long,
    )
    chosen_route_idxs = torch.tensor([[0, 1]], dtype=torch.long)
    replacement_routes = torch.tensor(
        [[
            [9, 8, -1, -1],
            [7, 6, -1, -1],
        ]],
        dtype=torch.long,
    )
    trim_model = object()
    extend_model = object()
    captured = {}

    def fake_trim_then_extend(
            got_trim_model, got_extend_model, env_state, got_bee_networks,
            got_chosen_route_idxs, greedy=False, ignore_max_route_len=False,
            allow_halt=True):
        captured["trim_model"] = got_trim_model
        captured["extend_model"] = got_extend_model
        captured["ignore_max_route_len"] = ignore_max_route_len
        captured["allow_halt"] = allow_halt
        out = got_bee_networks.clone()
        gather_idx = got_chosen_route_idxs[..., None, None].expand(
            -1, -1, -1, out.shape[-1])
        out.scatter_(2, gather_idx, replacement_routes.unsqueeze(2))
        return out

    monkeypatch.setattr(
        bee_colony,
        "get_neural_trim_then_extend_variants",
        fake_trim_then_extend,
    )

    new_networks, mutation_types = bee_colony.get_mutants(
        bee_networks,
        chosen_route_idxs,
        n_type1=0,
        n_type2=0,
        direct_sat_dmd=torch.zeros((1, 1, 1)),
        shorten_prob=0.0,
        street_node_neighbours=torch.zeros((1, 1, 1), dtype=torch.bool),
        shortest_paths=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        force_linking_unlinked=False,
        bee_model=extend_model,
        env_state=object(),
        n_type7=2,
        edit_model=trim_model,
        ignore_type7_max_route_len=True,
        return_mutation_metadata=True,
    )

    assert captured == {
        "trim_model": trim_model,
        "extend_model": extend_model,
        "ignore_max_route_len": True,
        "allow_halt": True,
    }
    assert mutation_types.tolist() == [7, 7]
    assert torch.equal(new_networks[0, 0, 0], replacement_routes[0, 0])
    assert torch.equal(new_networks[0, 1, 1], replacement_routes[0, 1])


def test_get_mutants_forwards_nohalt_flags_to_edit_bees(monkeypatch):
    bee_networks = torch.tensor(
        [[
            [[0, 1, 2, -1], [3, 4, -1, -1]],
            [[1, 2, 3, -1], [4, 5, -1, -1]],
            [[2, 3, 4, -1], [5, 6, -1, -1]],
        ]],
        dtype=torch.long,
    )
    chosen_route_idxs = torch.tensor([[0, 1, 0]], dtype=torch.long)
    edit_model = object()
    extend_model = object()
    captured = {}

    def fake_extend(model, env_state, got_bee_networks, got_chosen_route_idxs,
                    greedy=False, ignore_max_route_len=False,
                    allow_halt=True):
        captured["type4_allow_halt"] = allow_halt
        return got_bee_networks

    def fake_edit(model, env_state, got_bee_networks, got_chosen_route_idxs,
                  greedy=False, ignore_max_route_len=False,
                  allow_extend=True, allow_trim_start=True,
                  allow_trim_end=True, allow_halt=True,
                  adj_condition_target=None, adj_condition_weight=None):
        captured["type5_allow_halt"] = allow_halt
        return got_bee_networks

    def fake_trim(model, env_state, got_bee_networks, got_chosen_route_idxs,
                  greedy=False, ignore_max_route_len=False,
                  allow_halt=True):
        captured["type6_allow_halt"] = allow_halt
        return got_bee_networks

    monkeypatch.setattr(bee_colony, "get_neural_extend_variants", fake_extend)
    monkeypatch.setattr(bee_colony, "get_neural_edit_variants", fake_edit)
    monkeypatch.setattr(bee_colony, "get_neural_trim_variants", fake_trim)

    bee_colony.get_mutants(
        bee_networks,
        chosen_route_idxs,
        n_type1=0,
        n_type2=0,
        direct_sat_dmd=torch.zeros((1, 1, 1)),
        shorten_prob=0.0,
        street_node_neighbours=torch.zeros((1, 1, 1), dtype=torch.bool),
        shortest_paths=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        force_linking_unlinked=False,
        bee_model=extend_model,
        env_state=object(),
        n_type4=1,
        n_type5=1,
        n_type6=1,
        edit_model=edit_model,
        type4_allow_halt=False,
        type5_allow_halt=False,
        type6_allow_halt=False,
    )

    assert captured == {
        "type4_allow_halt": False,
        "type5_allow_halt": False,
        "type6_allow_halt": False,
    }


def test_get_mutants_can_process_type1_neural_bees_sequentially(monkeypatch):
    bee_networks = torch.tensor(
        [[
            [[0, 1, -1, -1], [2, 3, -1, -1]],
            [[4, 5, -1, -1], [6, 7, -1, -1]],
            [[8, 9, -1, -1], [1, 2, -1, -1]],
        ]],
        dtype=torch.long,
    )
    chosen_route_idxs = torch.tensor([[0, 1, 0]], dtype=torch.long)
    calls = []

    def fake_neural_variants(model, env_state, got_bee_networks,
                             got_chosen_route_idxs, greedy=False):
        calls.append(int(got_bee_networks.shape[1]))
        out = got_bee_networks.clone()
        out[:, :, -1] = torch.tensor(
            [99, 98, -1, -1], dtype=torch.long)
        return out

    monkeypatch.setattr(
        bee_colony, "get_neural_variants", fake_neural_variants)

    new_networks, mutation_types = bee_colony.get_mutants(
        bee_networks,
        chosen_route_idxs,
        n_type1=3,
        n_type2=0,
        direct_sat_dmd=torch.zeros((1, 1, 1)),
        shorten_prob=0.0,
        street_node_neighbours=torch.zeros((1, 1, 1), dtype=torch.bool),
        shortest_paths=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        force_linking_unlinked=False,
        bee_model=object(),
        env_state=object(),
        process_neural_bees_sequentially=True,
        return_mutation_metadata=True,
    )

    assert calls == [1, 1, 1]
    assert mutation_types.tolist() == [1, 1, 1]
    for bee_idx, route_idx in enumerate(chosen_route_idxs[0].tolist()):
        assert torch.equal(
            new_networks[0, bee_idx, route_idx],
            torch.tensor([99, 98, -1, -1]),
        )


def test_get_mutants_can_process_type5_edit_bees_sequentially(monkeypatch):
    bee_networks = torch.tensor(
        [[
            [[0, 1, 2, -1], [3, 4, -1, -1]],
            [[1, 2, 3, -1], [4, 5, -1, -1]],
            [[2, 3, 4, -1], [5, 6, -1, -1]],
        ]],
        dtype=torch.long,
    )
    chosen_route_idxs = torch.tensor([[0, 1, 0]], dtype=torch.long)
    calls = []

    def fake_edit(model, env_state, got_bee_networks, got_chosen_route_idxs,
                  greedy=False, ignore_max_route_len=False,
                  allow_extend=True, allow_trim_start=True,
                  allow_trim_end=True, allow_halt=True,
                  adj_condition_target=None, adj_condition_weight=None):
        calls.append(int(got_bee_networks.shape[1]))
        out = got_bee_networks.clone()
        gather_idx = got_chosen_route_idxs[..., None, None].expand(
            -1, -1, -1, out.shape[-1])
        route = out.gather(2, gather_idx).squeeze(2).clone()
        route[..., 0] += 100
        out.scatter_(2, gather_idx, route.unsqueeze(2))
        return out

    monkeypatch.setattr(bee_colony, "get_neural_edit_variants", fake_edit)

    new_networks, mutation_types = bee_colony.get_mutants(
        bee_networks,
        chosen_route_idxs,
        n_type1=0,
        n_type2=0,
        direct_sat_dmd=torch.zeros((1, 1, 1)),
        shorten_prob=0.0,
        street_node_neighbours=torch.zeros((1, 1, 1), dtype=torch.bool),
        shortest_paths=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        force_linking_unlinked=False,
        bee_model=object(),
        env_state=object(),
        n_type5=3,
        edit_model=object(),
        process_neural_bees_sequentially=True,
        return_mutation_metadata=True,
    )

    assert calls == [1, 1, 1]
    assert mutation_types.tolist() == [5, 5, 5]
    for bee_idx, route_idx in enumerate(chosen_route_idxs[0].tolist()):
        expected = bee_networks[0, bee_idx, route_idx].clone()
        expected[0] += 100
        assert torch.equal(new_networks[0, bee_idx, route_idx], expected)


def test_sequential_neural_bees_use_single_bee_env_state(monkeypatch):
    """Per-bee neural calls must run on the single-bee state (batch == n_graphs),
    not the full graphs*n_bees bee state, or the 1-bee input is broadcast back
    up to the whole bee batch and corrupts/crashes the result."""
    bee_networks = torch.tensor(
        [[
            [[0, 1, 2, -1], [3, 4, -1, -1]],
            [[1, 2, 3, -1], [4, 5, -1, -1]],
        ]],
        dtype=torch.long,
    )
    chosen_route_idxs = torch.tensor([[0, 0]], dtype=torch.long)
    seen_states = []

    def fake_edit(model, env_state, got_bee_networks, got_chosen_route_idxs,
                  **kwargs):
        seen_states.append(env_state)
        return got_bee_networks.clone()

    monkeypatch.setattr(bee_colony, "get_neural_edit_variants", fake_edit)

    full_state = object()
    single_state = object()
    bee_colony.get_mutants(
        bee_networks, chosen_route_idxs, n_type1=0, n_type2=0,
        direct_sat_dmd=torch.zeros((1, 1, 1)), shorten_prob=0.0,
        street_node_neighbours=torch.zeros((1, 1, 1), dtype=torch.bool),
        shortest_paths=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        force_linking_unlinked=False, bee_model=object(),
        env_state=full_state, n_type5=2, edit_model=object(),
        process_neural_bees_sequentially=True,
        single_bee_env_state=single_state,
        return_mutation_metadata=True)

    assert seen_states, "edit variant fn was never called"
    assert all(s is single_state for s in seen_states), \
        "sequential edit bees ran on the full bee state, not the single-bee one"


def test_mutation_stats_include_type7_and_worse_acceptance_bucket():
    mutation_stats = {}
    mutation_types = torch.tensor([1, 7], dtype=torch.long)
    accepted = torch.tensor([[True, True], [False, True]])
    worse_accepted = torch.tensor([[False, True], [False, False]])

    bee_colony._record_accepted_mutations(
        mutation_stats, mutation_types, accepted)
    bee_colony._record_worse_accepted_mutations(
        mutation_stats, mutation_types, worse_accepted)

    assert mutation_stats["accepted"]["type1"] == 1
    assert mutation_stats["accepted"]["type7"] == 2
    assert mutation_stats["worse_accepted"]["type7"] == 1


def test_soft_selection_preserves_elite_and_can_sample_nonbest_parent():
    torch.manual_seed(0)
    objective_costs = torch.arange(100, dtype=torch.float32).unsqueeze(0)

    parents = bee_colony._sample_soft_selection_parents(
        objective_costs,
        temperature=1.0,
        uniform_mix=1.0,
        elite_count=1,
    )

    assert parents.shape == objective_costs.shape
    assert parents[0, 0].item() == 0
    assert (parents[0, 1:] != 0).any()


def test_selection_stats_count_nonbest_and_worse_parent_copies():
    mutation_stats = {}
    objective_costs = torch.tensor([[1.0, 2.0, 3.0]])
    parents = torch.tensor([[0, 2, 1]], dtype=torch.long)

    bee_colony._record_selection_stats(
        mutation_stats,
        parents,
        objective_costs,
    )

    assert mutation_stats["selection"]["parent_copies"] == 3
    assert mutation_stats["selection"]["nonbest_parent_copies"] == 2
    assert mutation_stats["selection"]["worse_parent_copies"] == 1

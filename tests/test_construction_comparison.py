from connectpt.routes_generator.run.config import cost_params
from types import SimpleNamespace
from connectpt.routes_generator.env.features import observe
from connectpt.routes_generator.env.masks import ActionMask
from connectpt.routes_generator.env.features import global_state_features, redundancy_global_features
import torch
import pytest
import json
import math
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from connectpt.routes_generator.run.const import CFG_DIR
from connectpt.routes_generator.city.synthetic import DynamicCityGraphDataset
from connectpt.routes_generator.policy.build import build_policy
from connectpt.routes_generator.run.config import policy_profile
from connectpt.routes_generator.objectives.cost import build_cost
from connectpt.routes_generator.env.state import RouteGenBatchState
from connectpt.routes_generator.training.algorithms.d3po import (
    preference_dict as _preference_dict_from_tensor,
    sample_neighbor_preferences as _sample_neighbor_preferences,
    D3PO,
)
from connectpt.routes_generator.training.algorithms.base import (
    returns_and_advantages as _compute_ppo_returns_and_advantages,
    AlgorithmParams,
)

from connectpt.routes_generator.objectives.pareto import hypervolume_2d, nondominated


def evaluate_action(policy, observation, mask, indices, kinds=None):
    from connectpt.routes_generator.env.actions import RouteAction
    from connectpt.routes_generator.env.const import RouteActionKind
    if kinds is None:
        kinds = torch.where(indices[:, 0] < 0, RouteActionKind.HALT, RouteActionKind.EXTEND)
    return policy.evaluate(observation, mask, RouteAction(kinds, indices))


def preference_dict(weights):
    return _preference_dict_from_tensor(torch.nn.functional.pad(weights, (0, 1)))


def test_episode_statistics_keep_partial_episodes_and_separate_terminals():
    from connectpt.routes_generator.training.episodes import EpisodeStatistics

    def _collect_episode_statistics(rewards,dones,totals,lengths):
        stats=EpisodeStatistics(totals,lengths)
        completed=[stats.record(reward,done,torch.ones_like(done)) for reward,done in zip(rewards,dones)]
        return tuple(torch.cat(items) for items in zip(*completed))

    totals = torch.zeros(2)
    lengths = torch.zeros(2, dtype=torch.long)
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]),
        torch.tensor([[False, False], [True, False], [False, False]]),
        totals,
        lengths,
    )
    assert rewards.tolist() == [3.0]
    assert steps.tolist() == [2]
    assert totals.tolist() == [4.0, 70.0]
    assert lengths.tolist() == [1, 3]
    # Two asynchronous environments, including multiple finishes in one rollout
    # and an episode that began in the preceding rollout.
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[8.0, 80.0], [16.0, 160.0], [32.0, 320.0]]),
        torch.tensor([[True, False], [True, True], [False, True]]),
        totals,
        lengths,
    )
    assert rewards.tolist() == [12.0, 16.0, 310.0, 320.0]
    assert steps.tolist() == [2, 1, 5, 1]
    assert totals.tolist() == [32.0, 0.0]
    assert lengths.tolist() == [1, 0]
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[64.0, 640.0]]), torch.zeros(1, 2, dtype=torch.bool), totals, lengths
    )
    assert rewards.numel() == steps.numel() == 0
    assert totals.tolist() == [96.0, 640.0]
    assert lengths.tolist() == [2, 1]


def test_two_objective_metrics_have_known_geometry():
    front = [[1, 2], [2, 1], [2, 2], [1, 2]]
    assert hypervolume_2d(front, [3, 3]) == 3
    assert hypervolume_2d([], [3, 3]) == 0
    assert len(nondominated(front)) == 2


def test_existing_preference_perturbation_preserves_simplex_and_inactive_component():
    weights = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    same = _sample_neighbor_preferences(weights, 0.0, enabled_mask=(True, True, False))
    assert torch.allclose(weights, same, atol=1e-6)
    neighbor = _sample_neighbor_preferences(weights.repeat(100, 1), 3.0, enabled_mask=(True, True, False))
    assert torch.all(neighbor >= 0) and torch.all(neighbor <= 1)
    assert torch.allclose(neighbor.sum(-1), torch.ones(300))
    assert torch.equal(neighbor[:, 2], torch.zeros(300))


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("use_gae", [False, True])
def test_returns_cut_terminal_and_bootstrap_only_truncated_episode(vector, use_gae):
    rewards = torch.tensor([[1.0], [2.0], [100.0], [200.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    final = torch.tensor([999.0])
    expected = torch.tensor([[3.0], [2.0], [300.0], [200.0]])
    if vector:
        rewards, values, expected = [x[..., None].repeat(1, 1, 2) for x in (rewards, values, expected)]
        final = final[:, None].repeat(1, 2)
    dones = torch.tensor([[False], [True], [False], [True]])
    returns, advantages = _compute_ppo_returns_and_advantages(rewards, values, dones, final, 1.0, use_gae, 1.0)
    assert torch.equal(returns, expected)
    assert torch.equal(advantages, expected - values)
    dones[-1] = False
    truncated, _ = _compute_ppo_returns_and_advantages(rewards, values, dones, final, 1.0, use_gae, 1.0)
    assert torch.equal(truncated[:2], expected[:2])
    assert torch.equal(truncated[2:], expected[2:] + 999.0)


def test_common_sampler_matches_holliday_and_respects_fixed_weights():
    from connectpt.routes_generator.run.config import load_experiment

    from connectpt.routes_generator.objectives.preferences import PreferenceSampler,state_preferences
    from connectpt.routes_generator.run.config import sampler_params
    from dataclasses import replace

    parameters=sampler_params(load_experiment("train_construction").preferences)
    sampler=PreferenceSampler(parameters)
    torch.manual_seed(41)
    selector = torch.rand(4096)
    expected = torch.zeros(4096)
    expected[selector < 0.33] = 1.0
    intermediate = selector >= 0.66
    expected[intermediate] = torch.rand(int(intermediate.sum()))
    torch.manual_seed(41)
    sampled = state_preferences(sampler.sample_named(4096))
    assert torch.equal(sampled["demand_time_weight"], expected)
    assert torch.equal(sampled["route_time_weight"], 1 - expected)
    assert torch.equal(sampled["median_connectivity_weight"], torch.zeros(4096))
    sampler = PreferenceSampler(replace(parameters,variable_weights=False,demand_time_weight=0.2,route_time_weight=0.8))
    assert torch.allclose(sampler.sample_named(5)["ATT"], torch.full((5,), 0.2))


def test_replayed_action_and_existing_d3po_diversity_restore_state():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="runs/train_construction")
    model = build_policy(policy_profile(cfg.model, cfg.cost, cfg.common, torch.device("cpu")))
    model.freeze_feature_norms()
    cost = build_cost(cost_params(cfg.cost), low_memory_mode=False, symmetric_routes=True)
    data = DynamicCityGraphDataset(15, 15, data_type="voronoi")
    graphs = [data.transform(data.generate_graph()) for _ in range(2)]
    ws = torch.tensor([[0.2, 0.8], [0.8, 0.2]])
    state = RouteGenBatchState(
        graphs, cost.constants, 3, min_route_len=2, max_route_len=8, preferences=preference_dict(ws)
    )
    from connectpt.routes_generator.env.base import normalize_nodes

    normalize_nodes(state, model.node_norm, model.feature_set)
    from connectpt.routes_generator.run.tasks import TASKS
    limits = SimpleNamespace(n_routes=3, min_route_len=2, max_route_len=8)
    env = TASKS['construction'].binding(model, limits,
        OmegaConf.to_container(cfg.env.params, resolve=True)).bind(state, model, cost, normalize=False)
    phases = set()
    with torch.no_grad():
        for _ in range(30):
            phases.update(state.current_route_n_stops.tolist())
            active = ~state.is_done()
            observation, mask = env.observe()
            out = model.act(observation, mask)
            actions, logp, entropy = out.action.indices, out.logp, out.entropy
            replay_out = evaluate_action(model, observation, mask, actions)
            replay, distribution = replay_out.logp, replay_out.log_probs
            assert torch.allclose((replay - logp).exp(), torch.ones_like(logp), atol=1e-6)
            selected = actions[:, 0] * state.max_n_nodes + actions[:, 1]
            selected[actions[:, 0] < 0] = distribution.shape[-1] - 1
            assert torch.allclose(distribution.exp().sum(-1), torch.ones_like(logp), atol=1e-6)
            assert torch.allclose(distribution.gather(1, selected[:, None]).squeeze(1), replay, atol=1e-6)
            original = {k: v.clone() for k, v in state.cost_weights.items()}
            preferences = cost.get_preference_weights(state, normalize=True)
            values = dict(cfg.algorithm)
            values.pop("critic")
            algorithm = D3PO(AlgorithmParams(**values), cost)
            algorithm.params = __import__("dataclasses").replace(algorithm.params, diversity_weight=0.01)
            loss = algorithm.diversity(model, env, state, replay_out, mask).loss
            assert loss.isfinite() and loss >= 0
            assert all(torch.equal(state.cost_weights[k], v) for k, v in original.items())
            assert entropy.isfinite().all()
            before = cost.get_cost_components(state)[:, :2]
            scalar_before = cost(state).cost
            state.shortest_path_action(actions)
            after = cost.get_cost_components(state)[:, :2]
            scalar_after = cost(state).cost
            # Common constraint penalty in both streams reproduces scalar PPO rewards.
            assert torch.allclose(
                ((before - after) * ws).sum(-1)[active], (scalar_before - scalar_after)[active], atol=2e-6
            )
            if state.is_done().all():
                break
    assert state.is_done().all()
    assert 0 in phases and any(x > 0 for x in phases)


@pytest.mark.parametrize("trainer,pareto", [("ppo", True), ("d3po", True), ("ppo", False)],
                         ids=["ppo", "d3po", "ppo_pareto_disabled"])
def test_construction_uses_legacy_data_setup_and_existing_updates(trainer, pareto, tmp_path):
    import pickle
    from connectpt.routes_generator.city.synthetic import RAW_GRAPH_FILENAME
    from connectpt.routes_generator.run.training import TrainingRun

    torch.set_num_threads(2)
    torch.manual_seed(7)
    folder = tmp_path
    data_dir = folder / "data"
    data_dir.mkdir(parents=True)
    dataset = DynamicCityGraphDataset(9, 9, data_type="voronoi")
    # Raw graphs, as saved by city.synthetic.main; loaders apply augmentation.
    graphs = [dataset.generate_graph() for _ in range(12)]
    (data_dir / RAW_GRAPH_FILENAME).write_bytes(pickle.dumps(graphs))
    test_dir = folder / "test"
    test_dir.mkdir()
    (test_dir / RAW_GRAPH_FILENAME).write_bytes(pickle.dumps([dataset.generate_graph() for _ in range(2)]))
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(
            config_name="runs/train_construction",
            overrides=[
                f"training/algorithm={trainer}",
                "run.cpu=true",
                "train_loop.batch_size=2",
                "train_loop.n_iterations=2",
                "train_loop.horizon=8",
                "train_loop.val_period=1",
                "dataset.legacy_limits.n_routes=1",
                "dataset.legacy_limits.max_route_len=5",
                "evaluation.preference_count=3",
                "evaluation.stochastic_samples=1",
                "evaluation.test_graph_count=1",
                "evaluation.pareto_preference_count=3",
                "evaluation.pareto_period=1",
                f"evaluation.pareto_enabled={str(pareto).lower()}",
                f"dataset.path={data_dir.as_posix()}",
                "dataset.legacy_split.train_fraction=0.75",
                f"dataset.test_path={test_dir.as_posix()}",
                f'paths.output_dir={(folder / "run").as_posix()}',
                f'paths.weights_dir={(folder / "weights").as_posix()}',
            ],
        )
    run = TrainingRun(cfg)
    artifact = run.run()
    info = json.loads((artifact.output_dir / "run.json").read_text())
    assert set(info) == {"commit", "seed", "argv", "status", "evaluation_protocol"} and info["status"] == "iterations"
    import pandas as pd

    history = pd.read_csv(artifact.output_dir / "history.csv", float_precision="round_trip")
    history = history.dropna(subset=["time/total_timesteps"]).to_dict("records")
    assert len(history) == 2 and history[-1]["time/total_timesteps"] == 32
    assert artifact.checkpoint_path.exists()
    assert artifact.checkpoint_path.with_name("best.pt").exists()
    test_protocol = info["evaluation_protocol"]["test"]
    assert test_protocol["split"] == "independent test" and len(test_protocol["indices"]) == 1
    assert not list(artifact.output_dir.glob("test_*.json"))
    assert not list(artifact.output_dir.glob("pareto/*.json"))
    assert test_protocol["outcomes_per_checkpoint"] == {"greedy": 3, "stochastic": 3}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(artifact.output_dir / "tensorboard")).Reload()
    required_tags = (
        "rollout/ep_rew_mean",
        "rollout/att_minutes",
        "rollout/rtt_minutes",
        "eval/cost",
        "eval/test/best/stochastic/cost",
        "eval/test/last/greedy/objective/ATT",
        "eval/test/last/greedy/penalty/constraints",
    )
    assert all(tag in events.Tags()["scalars"] for tag in required_tags)
    if pareto:
        assert {"pareto/greedy/hv", "pareto/stochastic/eu_archive", "pareto/greedy/distinct_networks",
                "pareto/greedy/distinct_pareto_networks"} <= set(events.Tags()["scalars"])
    else:
        assert not any(tag.startswith("pareto/") or tag.endswith("/hv") for tag in events.Tags()["scalars"])
        assert info["evaluation_protocol"]["monitor"]["outcomes_per_call"] == {
            "validation": {"greedy": 0, "stochastic": 0},
        }
    all_history = pd.read_csv(artifact.output_dir / "history.csv", float_precision="round_trip")
    for label in ("last", "best"):
        for mode in ("greedy", "stochastic"):
            tag = f"eval/test/{label}/{mode}/cost"
            value = all_history[tag].dropna().iloc[-1]
            assert math.isfinite(value)
            assert events.Scalars(tag)[-1].value == pytest.approx(value)
    assert not events.Tags()["images"]
    protocol = info["evaluation_protocol"]["validation"]
    assert len(protocol["indices"]) == 3
    assert info["evaluation_protocol"]["monitor"]["reference"] == [3.0, 3.0]
    for tag in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
        recorded = events.Scalars(tag)
        assert len(recorded) == len(history)
        for entry, row in zip(recorded, history):
            value = row[tag]
            if row["rollout/completed_networks"]:
                assert math.isfinite(value)
                assert entry.value == pytest.approx(value)
            else:
                assert math.isnan(value) and math.isnan(entry.value)
    assert any(row["rollout/completed_networks"] for row in history)
    assert all(row["rollout/ep_len_mean"] >= 1 for row in history if row["rollout/completed_networks"])
    for row in history:
        if row["rollout/completed_networks"]:
            assert math.isfinite(row["rollout/att_minutes"]) and math.isfinite(row["rollout/rtt_minutes"])
            # Diff rewards telescope from the empty network to its final cost.
            expected_reward = 5.5 + 2 * row["rollout/completed_alpha_mean"] - row["rollout/cost"]
            assert row["rollout/ep_rew_mean"] == pytest.approx(expected_reward, abs=1e-5)
    if trainer == "ppo":
        assert history[-1]["rollout/initial_ratio_error"] < 2e-4
    elif trainer == "d3po":
        assert history[-1]["train/diversity_loss"] > 0
    else:
        assert history[-1]["train/diversity_loss"] == 0


def test_holliday_feature_layout_and_legacy_checkpoint_compatibility():
    from connectpt.routes_generator.training.critic import ValueHead, CriticParams
    from connectpt.routes_generator.policy.checkpoints import load_model_weights

    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="runs/train_construction")
        legacy = compose(config_name="runs/train_construction", overrides=["policy/model=construction"])
    model = build_policy(policy_profile(cfg.model, cfg.cost, cfg.common, torch.device("cpu")))
    assert cfg.train_loop.batch_size == cfg.algorithm.minibatch_size == cfg.evaluation.graph_count == 64
    assert cfg.algorithm.gamma == 1.0 and cfg.env.params.diff_reward
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        d3po = compose(config_name="runs/train_construction", overrides=["training/algorithm=d3po"])
    assert d3po.algorithm.gamma == 0.95 and d3po.env.params.diff_reward
    assert model.preference_goals == ("ATT", "RTT")
    assert model.edge_feat_dim == 13
    assert model.path_scorer[0].running_mean.numel() == 16
    assert model.halt_scorer.n_extra_feats == 13
    assert model.halt_scorer.preference_goals == model.preference_goals
    critic_params = CriticParams(**dict(cfg.algorithm.critic))
    scalar = ValueHead(('cost',), model.feature_set, critic_params, torch.device("cpu"), scalar=True)
    vector = ValueHead(('ATT', 'RTT'), model.feature_set, critic_params, torch.device("cpu"), scalar=False)
    assert scalar.input_dim == vector.input_dim == 21
    dataset = DynamicCityGraphDataset(9, 9, data_type="voronoi")
    graphs = [dataset.transform(dataset.generate_graph()) for _ in range(2)]
    cost = build_cost(cost_params(cfg.cost), low_memory_mode=False, symmetric_routes=True)
    state = RouteGenBatchState(graphs, cost.constants, 1, 2, 5, preferences=cost.get_weights())
    globals_ = global_state_features(state, include_redundancy=model.feature_set.redundancy,
                                    preference_goals=model.preference_goals)
    assert globals_.shape[-1] == 11
    features = scalar.inputs_from_data(state.graph_data, state.cost_weights)
    degree = features[:, 9].clone()
    features[:, -globals_.shape[-1] :] = globals_
    assert torch.equal(features[:, 9], degree)
    assert torch.equal(vector._features_from_state(state)[:, 9], degree)
    from dataclasses import replace
    permuted = replace(model.feature_set, critic=tuple(reversed(model.feature_set.critic)))
    reordered = ValueHead(('ATT', 'RTT'), permuted, critic_params, torch.device("cpu"), scalar=False)
    assert torch.equal(reordered._features_from_state(state), vector._features_from_state(state).flip(-1))
    legacy_model = build_policy(policy_profile(legacy.model, legacy.cost, legacy.common, torch.device("cpu")))
    assert legacy_model.edge_feat_dim == 14
    load_model_weights(
        legacy_model, "artifacts/model_weights/inductive_random_graphs_weighted_connectivity.pt", strict=True
    )


def test_advantages_use_holliday_minibatch_sample_std_not_rollout():
    from connectpt.routes_generator.training.algorithms.base import (
        normalize_advantages as _normalize_minibatch_advantages,
    )

    # These minibatches have different locations/scales; each normalizes to [-1,0,1].
    a, b = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([100.0, 200.0, 300.0])
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(_normalize_minibatch_advantages(a), expected)
    assert torch.allclose(_normalize_minibatch_advantages(b), expected)
    rollout = _normalize_minibatch_advantages(torch.cat((a, b)))
    assert not torch.allclose(rollout[:3], expected)
    vector = _normalize_minibatch_advantages(torch.stack((a, b), -1), per_objective=True)
    assert torch.allclose(vector, expected[:, None].expand(3, 2))
    assert torch.equal(_normalize_minibatch_advantages(torch.tensor([7.0])), torch.zeros(1))


def test_holliday_feature_norm_sample_variance_and_singleton_safety():
    from connectpt.routes_generator.policy.models import FeatureNorm

    layer = FeatureNorm(1, unbiased=True)
    output = layer(torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.allclose(output, torch.tensor([[-1.0], [0.0], [1.0]]), atol=1e-5)
    assert torch.equal(layer.new_var, torch.ones(1))
    singleton = FeatureNorm(1, unbiased=True)
    assert singleton(torch.tensor([[9.0]])).isfinite().all()
    singleton.update()
    assert singleton.running_var.isfinite().all()


def test_pareto_network_identity_and_per_city_fronts():
    from connectpt.routes_generator.objectives.pareto import canonical_network, summarize_evaluation

    original = canonical_network([[0, 1, 2], [2, 3]])
    assert original == canonical_network([[3, 2], [2, 1, 0, -1]])
    assert original != canonical_network([[0, 1, 2], [2, 3], [2, 3]])
    assert canonical_network([[0, 1]], False) != canonical_network([[1, 0]], False)
    rows = []
    for graph, pairs in [(0, [(1.0, 2.0), (2.0, 1.0)]), (1, [(2.0, 2.0), (2.0, 2.0)])]:
        for alpha, (x, y) in zip((0.0, 1.0), pairs):
            rows.append(
                dict(
                    graph=graph,
                    sample=0,
                    preference=alpha,
                    objective_0=x,
                    objective_1=y,
                    penalized_0=x,
                    penalized_1=y,
                    feasible=True,
                    unserved_fraction=0,
                    network_signature=original,
                )
            )
    report = summarize_evaluation(rows, [3.0, 3.0], 101)
    # Average of city HVs (3 and 1), not the HV of the pooled front (3).
    assert report["summary"]["hv_feasible"] == 2.0
    assert report["summary"]["distinct_networks"] == 1.0
    assert report["summary"]["pareto_points_feasible"] == 1.5
    assert report["summary"]["distinct_pareto_networks"] == 1.0
    assert report["summary"]["distinct_networks_total"] == 2
    assert report["reference"] == [3.0, 3.0]


def test_fixed_preference_evaluation_does_not_consume_training_rng():
    from connectpt.routes_generator.training.validate import evaluate_preferences, EvaluationProfile, EvaluationParams
    from connectpt.routes_generator.env.base import EnvParams
    from connectpt.routes_generator.run.schemas import params
    from connectpt.routes_generator.run.tasks import TASKS

    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(
            config_name="runs/train_construction",
            overrides=[
                "train_loop.batch_size=2",
                "dataset.legacy_limits.n_routes=1",
                "dataset.legacy_limits.max_route_len=5",
                "evaluation.preference_count=3",
                "evaluation.stochastic_samples=2",
            ],
        )
    model = build_policy(policy_profile(cfg.model, cfg.cost, cfg.common, torch.device("cpu")))
    model.freeze_feature_norms()
    cost = build_cost(cost_params(cfg.cost), low_memory_mode=False, symmetric_routes=True)
    dataset = DynamicCityGraphDataset(9, 9, data_type="voronoi")
    graphs = [dataset.transform(dataset.generate_graph()) for _ in range(2)]
    before = torch.random.get_rng_state().clone()
    profile = EvaluationProfile(
        params(EvaluationParams, cfg.evaluation),
            EnvParams(**OmegaConf.to_container(cfg.env.params, resolve=True),
                      **OmegaConf.to_container(cfg.dataset.legacy_limits, resolve=True)),
        cfg.train_loop.batch_size,
        cfg.common.symmetric_routes,
        TASKS["construction"].env_factory,
    )
    rows = evaluate_preferences(
        model, graphs, profile, cost, torch.device("cpu"), mode="stochastic"
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    repeated = evaluate_preferences(
        model, graphs, profile, cost, torch.device("cpu"), mode="stochastic"
    )
    assert rows == repeated
    assert len(rows) == 12
    assert all("network_signature" in row for row in rows)

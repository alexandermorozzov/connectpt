from connectpt.routes_generator.env.features import observe
from connectpt.routes_generator.env.masks import ActionMask
from connectpt.routes_generator.training.replay import evaluate_action
from connectpt.routes_generator.env.features import global_state_features, redundancy_global_features
import torch
import pytest
import json
import math
from pathlib import Path
from uuid import uuid4
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.run.context import CFG_DIR
from connectpt.routes_generator.city.synthetic import DynamicCityGraphDataset
from connectpt.routes_generator.policy.build import load_policy
from connectpt.routes_generator.run.config import policy_profile
from connectpt.routes_generator.objectives.cost import build_cost
from connectpt.routes_generator.env.state import RouteGenBatchState
from connectpt.routes_generator.improvement_learning import (
    _preference_dict_from_tensor, _sample_neighbor_preferences,
    _compute_ppo_returns_and_advantages, _estimate_d3po_diversity_loss,
)
from connectpt.routes_generator.training.construction_comparison import hypervolume_2d, coverage, nondominated


def preference_dict(weights):
    return _preference_dict_from_tensor(torch.nn.functional.pad(weights, (0, 1)))


def test_episode_statistics_keep_partial_episodes_and_separate_terminals():
    from connectpt.routes_generator.inductive_route_learning import _collect_episode_statistics
    totals = torch.zeros(2)
    lengths = torch.zeros(2, dtype=torch.long)
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[1., 10.], [2., 20.], [4., 40.]]),
        torch.tensor([[False, False], [True, False], [False, False]]),
        totals, lengths)
    assert rewards.tolist() == [3.]
    assert steps.tolist() == [2]
    assert totals.tolist() == [4., 70.]
    assert lengths.tolist() == [1, 3]
    # Two asynchronous environments, including multiple finishes in one rollout
    # and an episode that began in the preceding rollout.
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[8., 80.], [16., 160.], [32., 320.]]),
        torch.tensor([[True, False], [True, True], [False, True]]),
        totals, lengths)
    assert rewards.tolist() == [12., 16., 310., 320.]
    assert steps.tolist() == [2, 1, 5, 1]
    assert totals.tolist() == [32., 0.]
    assert lengths.tolist() == [1, 0]
    rewards, steps = _collect_episode_statistics(
        torch.tensor([[64., 640.]]), torch.zeros(1, 2, dtype=torch.bool), totals, lengths)
    assert rewards.numel() == steps.numel() == 0
    assert totals.tolist() == [96., 640.]
    assert lengths.tolist() == [2, 1]


def test_two_objective_metrics_have_known_geometry():
    front = [[1, 2], [2, 1], [2, 2], [1, 2]]
    assert hypervolume_2d(front, [3, 3]) == 3
    assert hypervolume_2d([], [3, 3]) == 0
    assert len(nondominated(front)) == 2
    assert coverage([[1, 1]], [[1, 1], [2, 2], [0, 2], [2, 2]]) == 2 / 3
    assert coverage([], [[1, 1]]) == 0
    assert coverage([[1, 1]], []) is None


def test_existing_preference_perturbation_preserves_simplex_and_inactive_component():
    weights = torch.tensor([[0., 1., 0.], [.5, .5, 0.], [1., 0., 0.]])
    same = _sample_neighbor_preferences(weights, 0., enabled_mask=(True, True, False))
    assert torch.allclose(weights, same, atol=1e-6)
    neighbor = _sample_neighbor_preferences(weights.repeat(100, 1), 3., enabled_mask=(True, True, False))
    assert torch.all(neighbor >= 0) and torch.all(neighbor <= 1)
    assert torch.allclose(neighbor.sum(-1), torch.ones(300))
    assert torch.equal(neighbor[:, 2], torch.zeros(300))


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('use_gae', [False, True])
def test_returns_cut_terminal_and_bootstrap_only_truncated_episode(vector, use_gae):
    rewards = torch.tensor([[1.], [2.], [100.], [200.]])
    values = torch.tensor([[10.], [20.], [30.], [40.]])
    final = torch.tensor([999.])
    expected = torch.tensor([[3.], [2.], [300.], [200.]])
    if vector:
        rewards, values, expected = [x[..., None].repeat(1, 1, 2) for x in (rewards, values, expected)]
        final = final[:, None].repeat(1, 2)
    dones = torch.tensor([[False], [True], [False], [True]])
    returns, advantages = _compute_ppo_returns_and_advantages(rewards, values, dones, final, 1., use_gae, 1.)
    assert torch.equal(returns, expected)
    assert torch.equal(advantages, expected - values)
    dones[-1] = False
    truncated, _ = _compute_ppo_returns_and_advantages(rewards, values, dones, final, 1., use_gae, 1.)
    assert torch.equal(truncated[:2], expected[:2])
    assert torch.equal(truncated[2:], expected[2:] + 999.)


def test_common_sampler_matches_holliday_and_respects_fixed_weights():
    from connectpt.routes_generator.objectives.cost import MyCostModule
    cost = MyCostModule(variable_weights=True, pp_fraction=.33, op_fraction=.33,
                        mcw_fraction=0., objectives=['ATT', 'RTT'])
    torch.manual_seed(41)
    selector = torch.rand(4096)
    expected = torch.zeros(4096)
    expected[selector < .33] = 1.
    intermediate = selector >= .66
    expected[intermediate] = torch.rand(int(intermediate.sum()))
    torch.manual_seed(41)
    sampled = cost.sample_variable_weights(4096)
    assert torch.equal(sampled['demand_time_weight'], expected)
    assert torch.equal(sampled['route_time_weight'], 1 - expected)
    assert torch.equal(sampled['median_connectivity_weight'], torch.zeros(4096))
    cost = cost.with_params(variable_weights=False, demand_time_weight=.2, route_time_weight=.8)
    assert torch.allclose(cost.sample_variable_weights(5)['demand_time_weight'], torch.full((5,), .2))


def test_three_way_comparison_separates_conditioned_utility_and_archive():
    from connectpt.routes_generator.training.construction_comparison import compare_runs
    # Workspace directory avoids Windows sandbox ACLs on pytest's mode-0700
    # temporary directories. This folder contains only this test's JSON files.
    tmp_path = Path('artifacts/comparisons/construction_fixed_checks_20261003') / f'metrics_{uuid4().hex}'
    for name in ('ppo', 'decomposed_ppo', 'd3po'):
        run = tmp_path / 'runs' / name
        run.mkdir(parents=True)
        # Wrong conditioned outputs: the archive can pick the opposite point.
        rows = [dict(graph=0, preference=w, sample=0, penalized_att=x, penalized_rtt=y,
                     att_norm=x, rtt_norm=y, feasible=True, unserved_fraction=0)
                for w, x, y in [(0., 1., 5.), (1., 5., 1.)]]
        (run / 'test_evaluation.json').write_text(json.dumps({'rows': rows}))
        write_run(run, status='iterations', timesteps=2048, max_seconds=None)
    report = compare_runs(tmp_path)
    assert report['summary']['ppo']['eu_conditioned_penalized'] == -5.
    assert report['summary']['ppo']['eu_archive_penalized'] > -5.
    assert report['summary']['ppo']['active_steps'] == 2048
    assert report['summary']['ppo']['iterations'] == 2
    assert len(report['paired_differences']) == 3
    assert not list(tmp_path.rglob('*.md'))
    write_run(tmp_path / 'runs/d3po', status='iterations', timesteps=1024, max_seconds=None)
    with pytest.raises(ValueError, match='budgets differ'):
        compare_runs(tmp_path)
    for name in ('ppo', 'decomposed_ppo', 'd3po'):
        write_run(tmp_path / 'runs' / name, status='time_budget', timesteps=2048, max_seconds=14400)
    report = compare_runs(tmp_path, names=('ppo', 'd3po'), budget_mode='wall_time')
    assert set(report['summary']) == {'ppo', 'd3po'}
    assert report['budget'] == 'equal training wall time'
    assert report['time_limit_seconds'] == 14400
    write_run(tmp_path / 'runs/d3po', status='time_budget', timesteps=2048, max_seconds=7200)
    with pytest.raises(ValueError, match='time budgets differ'):
        compare_runs(tmp_path, names=('ppo', 'd3po'), budget_mode='wall_time')
    write_run(tmp_path / 'runs/d3po', status='time_budget', timesteps=2048, max_seconds=14400,
              commit='other')
    with pytest.raises(ValueError, match='seeds or commits differ'):
        compare_runs(tmp_path, names=('ppo', 'd3po'), budget_mode='wall_time')


def write_run(run, *, status, timesteps, max_seconds, commit='same'):
    """run.json, config.yaml and history.csv as a finished training run leaves them."""
    (run / 'run.json').write_text(json.dumps(dict(commit=commit, seed=42, argv=[], status=status)))
    (run / 'config.yaml').write_text(
        f"train_loop:\n  max_seconds: {'null' if max_seconds is None else max_seconds}\n"
        "ppo:\n  resample_preferences_on_terminal: false\n")
    (run / 'history.csv').write_text(
        "step,val cost,time/total_timesteps,elapsed seconds\n"
        f"0,1.0,{timesteps // 2},0.5\n1,,{timesteps},1.0\n2,0.9,,\n")


def test_replayed_action_and_existing_d3po_diversity_restore_state():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name='runs/train_construction')
    model = load_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, torch.device("cpu")))
    model.freeze_feature_norms()
    cost = build_cost(cfg.cost, symmetric_routes=True)
    data = DynamicCityGraphDataset(15, 15, data_type='voronoi')
    graphs = [data.transform(data.generate_graph()) for _ in range(2)]
    ws = torch.tensor([[.2, .8], [.8, .2]])
    state = RouteGenBatchState(graphs, cost.constants, 3, min_route_len=2, max_route_len=8,
                              preferences=preference_dict(ws))
    model.setup_planning(state)
    phases = set()
    with torch.no_grad():
        for _ in range(30):
            phases.update(state.current_route_n_stops.tolist())
            active = ~state.is_done()
            out = model.act(observe(state, model.feature_set), ActionMask())
            actions, logp, entropy = out.action.indices, out.logp, out.entropy
            replay_out = evaluate_action(model, state, actions)
            replay, distribution = replay_out.logp, replay_out.log_probs
            assert torch.allclose((replay - logp).exp(), torch.ones_like(logp), atol=1e-6)
            selected = actions[:, 0] * state.max_n_nodes + actions[:, 1]
            selected[actions[:, 0] < 0] = distribution.shape[-1] - 1
            assert torch.allclose(distribution.exp().sum(-1), torch.ones_like(logp), atol=1e-6)
            assert torch.allclose(distribution.gather(1, selected[:, None]).squeeze(1), replay, atol=1e-6)
            original = {k: v.clone() for k, v in state.cost_weights.items()}
            preferences = cost.get_preference_weights(state, normalize=True)
            loss = _estimate_d3po_diversity_loss(
                model, state, actions, None, None, replay, preferences, False,
                .15, 1., enabled_mask=cost.enabled_component_mask)
            assert loss.isfinite() and loss >= 0
            assert all(torch.equal(state.cost_weights[k], v) for k, v in original.items())
            assert entropy.isfinite().all()
            before = cost.get_cost_components(state)[:, :2]
            scalar_before = cost(state).cost
            state.shortest_path_action(actions)
            after = cost.get_cost_components(state)[:, :2]
            scalar_after = cost(state).cost
            # Common constraint penalty in both streams reproduces scalar PPO rewards.
            assert torch.allclose(((before - after) * ws).sum(-1)[active],
                                  (scalar_before - scalar_after)[active], atol=2e-6)
            if state.is_done().all():
                break
    assert state.is_done().all()
    assert 0 in phases and any(x > 0 for x in phases)


@pytest.mark.parametrize('trainer', ['ppo_construction', 'decomposed_construction', 'd3po_construction'])
def test_construction_uses_legacy_data_setup_and_existing_updates(trainer):
    import pickle
    from connectpt.routes_generator.city.synthetic import RAW_GRAPH_FILENAME
    from connectpt.routes_generator.training.construction_run import ConstructionTrainingRun
    torch.set_num_threads(2)
    torch.manual_seed(7)
    folder = Path('artifacts/audits/rl_construction_2026-10-04/reuse_checks') / uuid4().hex
    data_dir = folder / 'data'
    data_dir.mkdir(parents=True)
    dataset = DynamicCityGraphDataset(9, 9, data_type='voronoi')
    # Raw graphs, as saved by city.synthetic.main; loaders apply augmentation.
    graphs = [dataset.generate_graph() for _ in range(12)]
    (data_dir / RAW_GRAPH_FILENAME).write_bytes(pickle.dumps(graphs))
    test_dir = folder / 'test'
    test_dir.mkdir()
    (test_dir / RAW_GRAPH_FILENAME).write_bytes(pickle.dumps([dataset.generate_graph() for _ in range(2)]))
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name='runs/train_construction', overrides=[
            f'trainer={trainer}', 'run.cpu=true', 'batch_size=2',
            'train_loop.n_iterations=2', 'ppo.horizon=8', 'ppo.val_period=1',
            'eval.n_routes=1', 'eval.max_route_len=5',
            'evaluation.preference_count=3', 'evaluation.stochastic_samples=1',
            'evaluation.test_graph_count=1',
            'evaluation.pareto_preference_count=3', 'evaluation.pareto_period=1',
            f'dataset.kwargs.path={data_dir.as_posix()}', 'dataset.kwargs.split=0.75',
            f'dataset.test_path={test_dir.as_posix()}',
            f'paths.output_dir={(folder / "run").as_posix()}',
            f'paths.weights_dir={(folder / "weights").as_posix()}'])
    run = ConstructionTrainingRun(cfg)
    artifact = run.run()
    info = json.loads((artifact.output_dir / 'run.json').read_text())
    assert set(info) == {'commit', 'seed', 'argv', 'status'} and info['status'] == 'iterations'
    import pandas as pd
    history = pd.read_csv(artifact.output_dir / 'history.csv', float_precision='round_trip')
    history = history.dropna(subset=['time/total_timesteps']).to_dict('records')
    assert len(history) == 2 and history[-1]['time/total_timesteps'] == 32
    assert artifact.checkpoint_path.exists()
    assert artifact.checkpoint_path.with_name('best.pt').exists()
    test_protocol = json.loads((artifact.output_dir / 'test_protocol.json').read_text())
    assert test_protocol['split'] == 'independent test' and len(test_protocol['indices']) == 1
    assert (artifact.output_dir / 'test_evaluation_best_stochastic.json').exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(artifact.output_dir / 'tensorboard')).Reload()
    required_tags = ('mean episode reward', 'train/att_minutes', 'train/rtt_minutes',
                     'val cost', 'pareto/greedy/hv_feasible',
                     'pareto/stochastic/eu_archive_penalized',
                     'pareto/greedy/distinct_networks', 'pareto/greedy/distinct_pareto_networks')
    assert all(tag in events.Tags()['scalars'] for tag in required_tags)
    assert 'pareto/greedy/front_city_0' in events.Tags()['images']
    protocol = json.loads((artifact.output_dir / 'pareto/validation_protocol.json').read_text())
    assert len(protocol['indices']) == 3
    assert protocol['reference'] == [3., 3.]
    for tag in ('mean episode reward', 'mean episode length'):
        recorded = events.Scalars(tag)
        assert len(recorded) == len(history)
        for entry, row in zip(recorded, history):
            value = row[tag]
            if row['completed networks']:
                assert math.isfinite(value)
                assert entry.value == pytest.approx(value)
            else:
                assert math.isnan(value) and math.isnan(entry.value)
    assert any(row['completed networks'] for row in history)
    assert all(row['mean episode length'] >= 1 for row in history if row['completed networks'])
    for row in history:
        if row['completed networks']:
            assert math.isfinite(row['train/att_minutes']) and math.isfinite(row['train/rtt_minutes'])
            # Diff rewards telescope from the empty network to its final cost.
            expected_reward = 5.5 + 2*row['train/completed_alpha_mean'] - row['train/cost']
            assert row['mean episode reward'] == pytest.approx(expected_reward, abs=1e-5)
    if trainer == 'ppo_construction':
        assert history[-1]['initial ratio error'] < 2e-4
    elif trainer == 'd3po_construction':
        assert history[-1]['diversity loss'] > 0
    else:
        assert history[-1]['diversity loss'] == 0


def test_holliday_feature_layout_and_legacy_checkpoint_compatibility():
    from connectpt.routes_generator.inductive_route_learning import NNBaseline
    from connectpt.routes_generator.improvement_learning import D3POValueModule
    from connectpt.routes_generator.run.checkpoints import CheckpointStore
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name='runs/train_construction')
        legacy = compose(config_name='runs/train_construction', overrides=['model=construction'])
    model = load_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, torch.device("cpu")))
    assert cfg.batch_size == cfg.ppo.minibatch_size == cfg.evaluation.graph_count == 64
    assert cfg.discount_rate == 1. and cfg.diff_reward
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        d3po = compose(config_name='runs/train_construction', overrides=['trainer=d3po_construction'])
    assert d3po.discount_rate == .95 and d3po.diff_reward
    assert model.preference_goals == ('ATT', 'RTT')
    assert model.edge_feat_dim == 13
    assert model.path_scorer[0].running_mean.numel() == 16
    assert model.halt_scorer.n_extra_feats == 13
    assert model.halt_scorer.preference_goals == model.preference_goals
    scalar = NNBaseline(preference_goals=model.preference_goals, holliday_features=True)
    vector = D3POValueModule(n_objectives=2, preference_goals=model.preference_goals, holliday_features=True)
    assert scalar.input_dim == vector.legacy_input_dim == 21
    dataset = DynamicCityGraphDataset(9, 9, data_type='voronoi')
    graphs = [dataset.transform(dataset.generate_graph()) for _ in range(2)]
    cost = build_cost(cfg.cost, symmetric_routes=True)
    state = RouteGenBatchState(graphs, cost.constants, 1, 2, 5, preferences=cost.get_weights())
    globals_ = global_state_features(state, preference_goals=model.preference_goals)
    assert globals_.shape[-1] == 11
    features = scalar.inputs_from_data(state.graph_data, state.cost_weights)
    degree = features[:, 9].clone()
    features[:, -globals_.shape[-1]:] = globals_
    assert torch.equal(features[:, 9], degree)
    assert torch.equal(vector._features_from_state(state)[:, 9], degree)
    legacy_model = load_policy(policy_profile(legacy.model, legacy.cost, legacy.experiment, torch.device("cpu")))
    assert legacy_model.edge_feat_dim == 14
    CheckpointStore.assert_state_dict_compatible(legacy_model,
        'artifacts/model_weights/inductive_random_graphs_weighted_connectivity.pt')


def test_advantages_use_holliday_minibatch_sample_std_not_rollout():
    from connectpt.routes_generator.improvement_learning import _normalize_minibatch_advantages
    # These minibatches have different locations/scales; each normalizes to [-1,0,1].
    a, b = torch.tensor([1., 2., 3.]), torch.tensor([100., 200., 300.])
    expected = torch.tensor([-1., 0., 1.])
    assert torch.allclose(_normalize_minibatch_advantages(a), expected)
    assert torch.allclose(_normalize_minibatch_advantages(b), expected)
    rollout = _normalize_minibatch_advantages(torch.cat((a, b)))
    assert not torch.allclose(rollout[:3], expected)
    vector = _normalize_minibatch_advantages(torch.stack((a, b), -1), per_objective=True)
    assert torch.allclose(vector, expected[:, None].expand(3, 2))
    assert torch.equal(_normalize_minibatch_advantages(torch.tensor([7.])), torch.zeros(1))


def test_holliday_feature_norm_sample_variance_and_singleton_safety():
    from connectpt.routes_generator.policy.models import FeatureNorm
    layer = FeatureNorm(1, unbiased=True)
    output = layer(torch.tensor([[1.], [2.], [3.]]))
    assert torch.allclose(output, torch.tensor([[-1.], [0.], [1.]]), atol=1e-5)
    assert torch.equal(layer.new_var, torch.ones(1))
    singleton = FeatureNorm(1, unbiased=True)
    assert singleton(torch.tensor([[9.]])).isfinite().all()
    singleton.update()
    assert singleton.running_var.isfinite().all()


def test_pareto_network_identity_and_per_city_fronts():
    from connectpt.routes_generator.objectives.pareto import canonical_network, summarize_evaluation
    original = canonical_network([[0, 1, 2], [2, 3]])
    assert original == canonical_network([[3, 2], [2, 1, 0, -1]])
    assert original != canonical_network([[0, 1, 2], [2, 3], [2, 3]])
    assert canonical_network([[0, 1]], False) != canonical_network([[1, 0]], False)
    rows = []
    for graph, pairs in [(0, [(1., 2.), (2., 1.)]), (1, [(2., 2.), (2., 2.)])]:
        for alpha, (x, y) in zip((0., 1.), pairs):
            rows.append(dict(graph=graph, sample=0, preference=alpha,
                att_norm=x, rtt_norm=y, penalized_att=x, penalized_rtt=y,
                feasible=True, unserved_fraction=0, network_signature=original))
    report = summarize_evaluation(rows, [3., 3.])
    # Average of city HVs (3 and 1), not the HV of the pooled front (3).
    assert report['summary']['hv_feasible'] == 2.
    assert report['summary']['distinct_networks'] == 1.
    assert report['summary']['pareto_points_feasible'] == 1.5
    assert report['summary']['distinct_pareto_networks'] == 1.
    assert report['summary']['distinct_networks_total'] == 2
    assert report['reference'] == [3., 3.]


def test_fixed_preference_evaluation_does_not_consume_training_rng():
    from connectpt.routes_generator.training.construction_experiment import evaluate_test
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name='runs/train_construction', overrides=[
            'batch_size=2', 'eval.n_routes=1', 'eval.max_route_len=5',
            'evaluation.preference_count=3', 'evaluation.stochastic_samples=2'])
    model = load_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, torch.device("cpu")))
    model.freeze_feature_norms()
    cost = build_cost(cfg.cost, symmetric_routes=True)
    dataset = DynamicCityGraphDataset(9, 9, data_type='voronoi')
    graphs = [dataset.transform(dataset.generate_graph()) for _ in range(2)]
    output = Path('artifacts/audits/construction_holliday_64_20261005') / f'rng_{uuid4().hex}'
    before = torch.random.get_rng_state().clone()
    rows = evaluate_test(model, graphs, cfg, cost, output, torch.device('cpu'),
                         mode='stochastic', save_artifacts=False)
    assert torch.equal(torch.random.get_rng_state(), before)
    repeated = evaluate_test(model, graphs, cfg, cost, output, torch.device('cpu'),
                             mode='stochastic', save_artifacts=False)
    assert rows == repeated
    assert len(rows) == 12
    assert all('network_signature' in row for row in rows)

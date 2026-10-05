import torch
import pytest
import json
import math
from pathlib import Path
from uuid import uuid4
from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.paths import CFG_DIR
from connectpt.routes_generator.citygraph_dataset import DynamicCityGraphDataset
from connectpt.routes_generator.model_factory import RouteModelFactory
from connectpt.routes_generator.objectives import CostFactory
from connectpt.routes_generator.transit_time_estimator import RouteGenBatchState
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
    from connectpt.routes_generator.transit_time_estimator import MyCostModule
    cost = MyCostModule(variable_weights=True, pp_fraction=.33, op_fraction=.33,
                        mcw_fraction=0., disabled_components=['connectivity'])
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
    cost.variable_weights = False
    cost.set_weights(demand_time_weight=.2, route_time_weight=.8)
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
        status = dict(state='completed', initial_actor_sha256='same', total_active_steps=2048,
                      seed=42, iterations=1, training_seconds=1)
        (run / 'status.json').write_text(json.dumps(status))
    report = compare_runs(tmp_path)
    assert report['summary']['ppo']['eu_conditioned_penalized'] == -5.
    assert report['summary']['ppo']['eu_archive_penalized'] > -5.
    assert len(report['paired_differences']) == 3
    assert not list(tmp_path.rglob('*.md'))
    path = tmp_path / 'runs/d3po/status.json'
    status['total_active_steps'] = 1024
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='budgets differ'):
        compare_runs(tmp_path)
    for name in ('ppo', 'decomposed_ppo', 'd3po'):
        path = tmp_path / 'runs' / name / 'status.json'
        row = json.loads(path.read_text())
        row.update(max_seconds=14400, stop_reason='time_budget')
        path.write_text(json.dumps(row))
    report = compare_runs(tmp_path, names=('ppo', 'd3po'), budget_mode='wall_time')
    assert set(report['summary']) == {'ppo', 'd3po'}
    assert report['budget'] == 'equal training wall time'
    assert report['time_limit_seconds'] == 14400
    path = tmp_path / 'runs/d3po/status.json'
    row.update(max_seconds=7200)
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='time budgets differ'):
        compare_runs(tmp_path, names=('ppo', 'd3po'), budget_mode='wall_time')


def test_replayed_action_and_existing_d3po_diversity_restore_state():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name='runs/train_construction')
    model = RouteModelFactory.build_construction_model(cfg.model, cfg.experiment)
    model.freeze_feature_norms()
    cost = CostFactory.build(cfg.cost, symmetric_routes=True)
    data = DynamicCityGraphDataset(15, 15, data_type='voronoi')
    graphs = [data.transform(data.generate_graph()) for _ in range(2)]
    ws = torch.tensor([[.2, .8], [.8, .2]])
    state = RouteGenBatchState(graphs, cost, 3, min_route_len=2, max_route_len=8,
                              cost_weights=preference_dict(ws))
    model.setup_planning(state)
    phases = set()
    with torch.no_grad():
        for _ in range(30):
            phases.update(state.current_route_n_stops.tolist())
            active = ~state.is_done()
            actions, logp, entropy = model.step(state)
            _, replay, _ = model.step(state, actions=actions)
            assert torch.allclose((replay - logp).exp(), torch.ones_like(logp), atol=1e-6)
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
    from connectpt.routes_generator.citygraph_dataset import RAW_GRAPH_FILENAME
    from connectpt.routes_generator.training.construction_run import ConstructionTrainingRun
    from connectpt.routes_generator.training.construction_experiment import state_dict_hash
    torch.set_num_threads(2)
    torch.manual_seed(7)
    folder = Path('artifacts/audits/rl_construction_2026-10-04/reuse_checks') / uuid4().hex
    data_dir = folder / 'data'
    data_dir.mkdir(parents=True)
    dataset = DynamicCityGraphDataset(9, 9, data_type='voronoi')
    # Raw graphs, as saved by citygraph_dataset.main; old loaders apply augmentation.
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
            f'dataset.kwargs.path={data_dir.as_posix()}', 'dataset.kwargs.split=0.75',
            f'dataset.test_path={test_dir.as_posix()}',
            f'paths.output_dir={(folder / "run").as_posix()}',
            f'paths.weights_dir={(folder / "weights").as_posix()}'])
    run = ConstructionTrainingRun(cfg)
    artifact = run.run()
    assert len(artifact.history) == 2
    status = json.loads((artifact.output_dir / 'status.json').read_text())
    assert status['state'] == 'completed' and status['total_active_steps'] == 32
    assert status['implementation'] == 'inductive_route_learning.setup_and_train'
    assert state_dict_hash(run.model) != status['initial_actor_sha256']
    assert artifact.checkpoint_path.exists()
    assert artifact.checkpoint_path.with_name('best.pt').exists()
    assert (artifact.output_dir / 'test_evaluation_best_stochastic.json').exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import csv
    events = EventAccumulator(str(artifact.output_dir / 'tensorboard')).Reload()
    with (artifact.output_dir / 'history.csv').open(newline='') as stream:
        csv_history = list(csv.DictReader(stream))
    for tag in ('mean episode reward', 'mean episode length'):
        recorded = events.Scalars(tag)
        assert len(recorded) == len(artifact.history)
        for entry, row, csv_row in zip(recorded, artifact.history, csv_history):
            value = row[tag]
            if row['completed networks']:
                assert math.isfinite(value)
                assert entry.value == pytest.approx(value)
                assert float(csv_row[tag]) == pytest.approx(value)
            else:
                assert math.isnan(value) and math.isnan(entry.value) and math.isnan(float(csv_row[tag]))
    assert any(row['completed networks'] for row in artifact.history)
    assert all(row['mean episode length'] >= 1 for row in artifact.history if row['completed networks'])
    if trainer == 'ppo_construction':
        assert artifact.history[-1]['initial ratio error'] < 2e-4
    elif trainer == 'd3po_construction':
        assert artifact.history[-1]['diversity loss'] > 0
    else:
        assert artifact.history[-1]['diversity loss'] == 0

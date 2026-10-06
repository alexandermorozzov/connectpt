import subprocess
from pathlib import Path
from uuid import uuid4

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.core.paths import CFG_DIR


def test_overnight_preserves_measured_batch_and_independent_training_limits():
    for trainer, gamma in [('ppo_construction', 1.0), ('d3po_construction', .95)]:
        with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
            cfg = compose(config_name='runs/train_construction_overnight',
                          overrides=[f'trainer={trainer}'])
        assert cfg.batch_size == cfg.ppo.minibatch_size == 32
        assert cfg.ppo.horizon == 64 and cfg.train_loop.cpu_threads == 2
        assert cfg.evaluation.graph_count == cfg.evaluation.test_graph_count == 64
        assert cfg.evaluation.preference_count == 11
        assert cfg.evaluation.stochastic_samples == 1
        assert cfg.train_loop.max_seconds == 25200
        assert cfg.train_loop.n_iterations == 100000
        assert cfg.discount_rate == gamma and cfg.diff_reward
        assert 'parallel' not in cfg


def test_direct_launcher_dry_run_validates_both_without_starting_workers():
    root = Path('artifacts/comparisons') / f'direct_dry_{uuid4().hex}'
    result = subprocess.run([
        'powershell.exe', '-NoProfile', '-File', 'scripts/run_construction_overnight.ps1',
        '-ComparisonRoot', str(root), '-Hours', '8', '-DryRun',
    ], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('dry-run OK:') == 2
    assert 'trainer=ppo_construction' in result.stdout
    assert 'trainer=d3po_construction' in result.stdout
    assert 'train_loop.max_seconds=25200' in result.stdout
    assert not root.exists()

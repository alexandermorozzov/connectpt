import subprocess

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.run.context import CFG_DIR


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


def test_launcher_dry_run_validates_both_overnight_trainers(tmp_path):
    for trainer in ('ppo_construction', 'd3po_construction'):
        log = tmp_path / f'{trainer}.log'
        result = subprocess.run([
            'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts/launch.ps1',
            '--wait', 'run.py', '--config-name', 'runs/train_construction_overnight',
            f'trainer={trainer}', '--dry-run', '--log', str(log),
        ], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        text = log.read_text(encoding='utf-8')
        assert 'dry-run OK' in text and f'trainer={trainer}' in text

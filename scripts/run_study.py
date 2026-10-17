"""Coordinate a portable CPU study using the existing dataset, trainer and evaluator.

Run via launch.sh/launch.ps1 with --out-dir and --log inside the study folder.
--calibrate-only measures resources; the default continues within --hours.
--replot rebuilds reports from saved candidates without loading an actor.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil


REPO = Path(__file__).resolve().parents[1]
SCHEMA = 1
GB = 1_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    atomic_replace(temporary, path)


def atomic_replace(temporary, destination):
    # Windows readers can briefly hold the destination without FILE_SHARE_DELETE.
    for attempt in range(6):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 5:
                raise
            time.sleep(.02 * (attempt + 1))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def process_memory(processes):
    """Count each PID once; use PSS only when available for every live process."""
    rss, pss, count, complete = 0, 0, 0, True
    seen = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            memory = process.memory_full_info()
            rss += memory.rss
            if hasattr(memory, 'pss'):
                pss += memory.pss
            else:
                complete = False
            count += 1
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            complete = False
            try:
                rss += process.memory_info().rss
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                raise RuntimeError('Cannot measure an owned process; refusing an unsafe memory estimate')
    return dict(memory_bytes=pss if complete else rss, rss_bytes=rss,
                memory_measure='PSS' if complete else 'RSS', process_count=count)


def stop_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(owned):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def verify_files(root, entries):
    for name, expected in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Missing or changed study input: {name}')


def snapshot(root):
    source = root / 'source'
    source.mkdir()
    shutil.copytree(REPO / 'connectpt', source / 'connectpt',
                    ignore=shutil.ignore_patterns('__pycache__', 'graphify-out'))
    (source / 'scripts').mkdir()
    (source / 'scripts/__init__.py').write_text('', encoding='utf-8')
    for name in ('run.py', 'generate_dataset.py', 'evaluate_policy.py', 'run_study.py', 'launch.sh', 'launch.ps1'):
        shutil.copy2(REPO / 'scripts' / name, source / 'scripts' / name)
    for name in ('requirements.txt', 'pyproject.toml', 'README.rst'):
        if (REPO / name).exists():
            shutil.copy2(REPO / name, source / name)
    for filename, command in [('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                              ('git-revision.txt', ['git', 'rev-parse', 'HEAD']),
                              ('git-diff.patch', ['git', 'diff', '--no-ext-diff', '--binary', 'HEAD'])]:
        result = subprocess.run(command, cwd=REPO, capture_output=True)
        (root / 'environment' / filename).write_bytes(result.stdout)
    save_json(root / 'environment/packaging-machine.json', machine_info())
    from importlib.metadata import version
    packages = ('torch', 'torch-geometric', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'networkx',
                'hydra-core', 'omegaconf', 'tensorboard', 'tqdm', 'psutil', 'matplotlib', 'geopandas')
    runtime = ['--extra-index-url https://download.pytorch.org/whl/cpu']
    runtime.extend(f'{name}=={version(name)}' for name in packages)
    (root / 'environment/runtime-requirements.txt').write_text('\n'.join(runtime)+'\n', encoding='utf-8')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(source.rglob('*')) if p.is_file()}


def machine_info():
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                physical_cores=psutil.cpu_count(logical=False), logical_cores=psutil.cpu_count(),
                memory_bytes=psutil.virtual_memory().total)


def variant_name(spec):
    return f"r{spec['routes']}_l{spec['length']}"


def worker_config(root, spec):
    from connectpt.routes_generator.run.config import load_experiment
    from omegaconf import OmegaConf
    cfg = load_experiment('train_construction_holliday', overrides=[f"training/algorithm={spec['algorithm']}"])
    settings = read_json(root / 'manifest.json')['settings']
    variant = variant_name(spec)
    cfg.run.name, cfg.run.seed, cfg.run.cpu = spec['id'], spec['seed'], True
    cfg.paths.output_dir = f"../runs/{spec['id']}"
    cfg.paths.weights_dir = '../runs'
    cfg.paths.checkpoint_path = f"{spec['id']}/checkpoints/last.pt"
    cfg.paths.init_checkpoint_path = None
    cfg.dataset.path = f'../data/{variant}/pool'
    cfg.dataset.test_path = f'../data/{variant}/test'
    cfg.train_loop.batch_size = cfg.train_loop.batch_size_cuda = spec['batch']
    cfg.train_loop.horizon = spec['horizon']
    cfg.train_loop.n_iterations = spec['updates']
    cfg.train_loop.max_seconds = None
    cfg.train_loop.val_period = 1 if settings['smoke'] else 10
    cfg.train_loop.cpu_threads = spec['threads']
    cfg.train_loop.preference_distribution = spec['sampler']
    cfg.train_loop.resample_preferences_on_terminal = False
    cfg.algorithm.gamma, cfg.algorithm.n_epochs = .95, 1
    cfg.algorithm.minibatch_size = spec['batch']
    cfg.algorithm.diversity_weight = spec['weight']
    cfg.algorithm.diversity_detach_neighbor = spec['detach']
    cfg.algorithm.diversity_reduction = 'state_mean'
    cfg.algorithm.diversity_alpha = 1.
    cfg.algorithm.preference_noise_sigma = .15
    cfg.algorithm.gradient_diagnostics_interval = 0
    cfg.evaluation.batch_size = min(32, spec['batch'])
    cfg.evaluation.graph_count = settings['validation_graphs']
    cfg.evaluation.pareto_graph_count = settings['validation_graphs']
    cfg.evaluation.test_graph_count = settings['test_graphs']
    cfg.evaluation.preference_count = settings['preference_count']
    cfg.evaluation.eu_preference_count = settings['preference_count']
    cfg.evaluation.pareto_enabled = cfg.evaluation.final_test = False
    if spec.get('alpha') is not None:
        cfg.preferences.variable_weights = False
        cfg.preferences.demand_time_weight = cfg.cost.kwargs.demand_time_weight = spec['alpha']
        cfg.preferences.route_time_weight = cfg.cost.kwargs.route_time_weight = 1 - spec['alpha']
    OmegaConf.resolve(cfg)
    return cfg


def prepare_data(root):
    """Use the standard generator and manifest writer; variant blobs share hard links."""
    from connectpt.routes_generator.run.config import load_experiment
    from connectpt.routes_generator.run.dataset import prepare_dataset
    from connectpt.routes_generator.training.dataset import load_dataset, dataset_fingerprint
    from connectpt.routes_generator.training.split import evaluation_indices
    settings = read_json(root / 'manifest.json')['settings']
    base = root / 'data/base'
    for part, count, seed in [('pool', settings['pool_graphs'], 1701), ('test', settings['test_graphs'], 1702)]:
        cfg = load_experiment('generate_construction')
        cfg.run.cpu, cfg.run.seed = True, seed
        cfg.dataset_build.n_graphs = count
        cfg.dataset_build.source.min_nodes = settings['nodes']
        cfg.dataset_build.limits.n_routes = settings['routes']
        cfg.dataset_build.limits.max_route_len = settings['length']
        prepare_dataset(cfg, base / part)
    pool = load_dataset(base / 'pool', kind='construction')
    validation = pool.split.validation
    chosen = evaluation_indices(range(len(validation)), settings['validation_graphs'], 9001)
    held = set(validation[chosen].tolist())
    selected = [i for i in validation.tolist() if i not in held][:settings['selection_graphs']]
    if len(selected) != settings['selection_graphs']:
        raise ValueError('Pool held-out split is too small for disjoint validation/selection')
    save_json(root / 'protocols/splits.json', dict(
        train=pool.split.train.tolist(), validation=sorted(held), selection=selected,
        train_sources=[pool.params.source_ids[i] for i in pool.split.train.tolist()],
        validation_sources=[pool.params.source_ids[i] for i in sorted(held)],
        selection_sources=[pool.params.source_ids[i] for i in selected]))
    # No route data live in a construction dataset. Only limits/fingerprint differ.
    for routes, length in {(settings['routes'], settings['length']),
                           (max(1, settings['routes'] // 2), settings['length']),
                           (settings['routes'], min(8, settings['length']))}:
        tag = f'r{routes}_l{length}'
        for part in ('pool', 'test'):
            destination = root / 'data' / tag / part
            destination.mkdir(parents=True, exist_ok=True)
            meta = read_json(base / part / 'manifest.json')
            meta['limits'].update(n_routes=routes, max_route_len=length)
            meta['configuration']['dataset_build']['limits'] = dict(meta['limits'])
            meta['fingerprint'] = dataset_fingerprint(meta['configuration'], meta['inputs'], meta['seed'])
            blob = destination / 'dataset.pt'
            if not blob.exists():
                try:
                    os.link(base / part / 'dataset.pt', blob)
                except OSError:
                    shutil.copy2(base / part / 'dataset.pt', blob)
            save_json(destination / 'manifest.json', meta)
            # The standard loader validates blob integrity and split contracts.
            check = load_dataset(destination, kind='construction')
            if check.params.limits.n_routes != routes:
                raise RuntimeError('Route variant was not applied')
            del check
    save_json(root / 'protocols/data-hashes.json', {
        p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / 'data').rglob('*')) if p.is_file()})


def candidate_summary(rows, preference_count, routes):
    import numpy as np
    from connectpt.routes_generator.objectives.pareto import summarize_evaluation, hypervolume_2d
    result = summarize_evaluation(rows, [3., 3.], max(2, preference_count))
    metrics = result['summary']
    if preference_count == 1:
        # A fixed-alpha specialist does not represent an archive over preferences.
        for item in [metrics] + [record['metrics'] for record in result['per_graph']]:
            item['eu_archive_penalized'] = item['eu_archive_feasible'] = None
    common = []
    for graph in sorted({r['graph'] for r in rows}):
        common.append(hypervolume_2d([[r['objective_0'], r['objective_1'] * routes / 10]
                                    for r in rows if r['graph'] == graph and r['feasible']], [3., 3.]))
    metrics['hv_common_rtt_reference10'] = float(np.mean(common))
    result['stochastic_rule'] = 'One selected network per city/preference; best-of-K uses nested independent candidate prefixes'
    return result, dict(cost=float(np.mean([r['scalar_cost'] for r in rows])),
                       ATT_minutes=float(np.mean([r['ATT'] for r in rows])),
                       RTT_minutes=float(np.mean([r['RTT'] for r in rows])),
                       hv=metrics['hv_feasible'], feasible_fraction=metrics['feasible_fraction'],
                       distinct_networks=metrics.get('distinct_networks'),
                       distinct_pareto_networks=metrics.get('distinct_pareto_networks'),
                       eu_conditioned_penalized=metrics['eu_conditioned_penalized'],
                       eu_archive_penalized=metrics['eu_archive_penalized'],
                       eu_archive_feasible=metrics['eu_archive_feasible'],
                       sparsity_feasible=metrics['sparsity_feasible'],
                       hv_common_rtt_reference10=metrics['hv_common_rtt_reference10'])


def rebuild_run(root, spec):
    """Reuse the existing evaluator's selectors, Pareto plotting and publication."""
    from scripts.evaluate_policy import select_rows, publish, write_csv, pareto_figures, digest
    import numpy as np
    settings = read_json(root / 'manifest.json')['settings']
    out = root / 'evaluation' / spec['id'] / 'best'
    protocol = read_json(out / 'protocol.json')
    pools = []
    for path in sorted((out / 'candidates').glob('*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            chunk = json.load(stream)
        if chunk['fingerprint'] != protocol['fingerprint']:
            raise ValueError(f'Candidate identity mismatch: {path}')
        pools.append(chunk)
    ids = [r['candidate_id'] for pool in pools for r in pool['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate chunks')
    greedy = [r for pool in pools for r in pool['rows'] if r['mode'] == 'greedy']
    sampled = [r for pool in pools for r in pool['rows'] if r['mode'] == 'sample']
    if len(greedy) != protocol['graph_count'] * len(protocol['alphas']):
        raise ValueError(f'Incomplete greedy candidates: {spec["id"]}')
    expected = protocol['k_graph_count'] * len(protocol['alphas']) * max(protocol['ks'])
    if len(sampled) != expected:
        raise ValueError(f'Incomplete K pool: {spec["id"]}: {len(sampled)}/{expected}')
    k_greedy = [r for r in greedy if r['graph'] < protocol['k_graph_count']]
    record = dict(label=spec['id'], identity=protocol['identity'], completed_alphas=len(protocol['alphas']),
                  overall=[], by_alpha=[], per_city=[])
    modes = [('greedy', k_greedy)] + [(f'best_of_{k}', select_rows(k_greedy, sampled, k)) for k in protocol['ks']]
    selections = []
    previous = None
    for mode, rows in modes:
        metrics, brief = candidate_summary(rows, len(protocol['alphas']), spec['routes'])
        k = 0 if mode == 'greedy' else int(mode.rsplit('_', 1)[-1])
        if k:
            costs = {(r['graph'], r['preference']): r['scalar_cost'] for r in rows}
            if previous is not None and any(costs[key] > previous[key] for key in costs):
                raise ValueError('Best-of-K cost increased for a nested pool')
            previous = costs
        seconds = sum(p['seconds'] for p in pools if (p['mode'] == 'greedy' if k == 0 else
                                                    p['mode'] == 'sample' and p['sample'] < k))
        record['overall'].append(dict(model=spec['id'], mode=mode, K=k,
            candidates_per_query=max(1, k), generation_evaluation_seconds=seconds, **brief))
        save_json(out / f'metrics_{mode}.json', metrics)
        for alpha in protocol['alphas']:
            subset = [r for r in rows if r['preference'] == alpha]
            _, brief_alpha = candidate_summary(subset, 1, spec['routes'])
            record['by_alpha'].append(dict(model=spec['id'], mode=mode, alpha=alpha, **brief_alpha))
        for row in rows:
            selections.append(dict(mode=mode, graph=row['graph'], alpha=row['preference'],
                                   candidate_id=row['candidate_id'], sample=row.get('selected_sample', row['sample'])))
            record['per_city'].append(dict(model=spec['id'], mode=mode, graph=row['graph'],
                alpha=row['preference'], cost=row['scalar_cost'], ATT_minutes=row['ATT'],
                RTT_minutes=row['RTT'], feasible=row['feasible'], network_id=digest(row['network_signature'])[:16]))
    write_csv(out / 'selected.csv', selections)
    request = dict(graph_count=protocol['k_graph_count'], preference_count=len(protocol['alphas']),
                   ks=protocol['ks'], seed=protocol['seed'], test_indices=list(range(protocol['k_graph_count'])))
    publish(out, out / 'report.md', [record], complete=True, request=request)
    report_path = out / 'report.md'
    report_path.write_text(report_path.read_text(encoding='utf-8').replace(
        'Greedy+K использует K+1 кандидатов и показан отдельным режимом.',
        'Best-of-K выбирает из K stochastic кандидатов; greedy сохранён отдельно.'), encoding='utf-8')
    metrics, _ = candidate_summary(greedy, len(protocol['alphas']), spec['routes'])
    save_json(out / 'greedy_all_metrics.json', metrics)
    all_rows = [dict(model=spec['id'], mode='greedy', graph=r['graph'], alpha=r['preference'],
                    cost=r['scalar_cost'], ATT_minutes=r['ATT'], RTT_minutes=r['RTT'], feasible=r['feasible'],
                    network_id=digest(r['network_signature'])[:16]) for r in greedy]
    full = out / 'greedy_all'
    full.mkdir(exist_ok=True)
    pareto_figures(full, all_rows, list(range(protocol['graph_count'])), protocol['ks'], complete=True)
    return record


def evaluate_run(root, spec, *, selection=False, probe=False):
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from connectpt.routes_generator.run.training import TrainingRun
    from connectpt.routes_generator.training.data import evaluation_graphs
    from scripts.evaluate_policy import prediction_rows
    from connectpt.routes_generator.policy.predict import predict
    settings = read_json(root / 'manifest.json')['settings']
    directory = root / 'runs' / spec['id']
    out = (root / 'selection' / spec['id']) if selection else root / 'evaluation' / spec['id'] / 'best'
    cfg = OmegaConf.load(directory / 'config.yaml')
    checkpoint = directory / 'checkpoints/best.pt'
    cfg.paths.init_checkpoint_path = f"{spec['id']}/checkpoints/best.pt"
    actor = TrainingRun(cfg)
    actor.setup()
    torch.set_num_threads(spec['threads'])
    actor.policy.eval()
    params = actor.data_input.params
    raw = evaluation_graphs(params.path if selection else params.test_path, params.kind, params.limits)
    indices = read_json(root / 'protocols/splits.json')['selection'] if selection else list(range(len(raw)))
    if probe:
        indices = indices[:1]
    graphs = [actor.dataset.evaluation_pipeline.graph(raw[i]) for i in indices]
    alphas = [spec['alpha']] if spec.get('alpha') is not None else np.linspace(0, 1, settings['preference_count']).tolist()
    if probe:
        alphas = [.5]
    ks = [1] if selection or probe else settings['ks']
    k_count = min(len(graphs), settings['k_graphs'])
    inputs = dict(spec=spec, config=file_hash(directory / 'config.yaml'), checkpoint=file_hash(checkpoint),
                  library=read_json(root / 'manifest.json')['source_hashes'],
                  data=read_json(root / 'protocols/data-hashes.json'), indices=indices, alphas=alphas,
                  ks=ks, k_count=k_count, seed=9001, selection=selection, probe=probe)
    fingerprint = identity(inputs)
    protocol = dict(identity=inputs, fingerprint=fingerprint, graph_count=len(graphs),
                    k_graph_count=k_count, alphas=alphas, ks=ks, seed=9001)
    save_json(out / 'protocol.json', protocol)
    before = {k: v.clone() for k, v in actor.policy.state_dict().items()}
    all_rows = []
    for ai, alpha in enumerate(alphas):
        weights = dict(demand_time_weight=float(alpha), route_time_weight=float(1-alpha), median_connectivity_weight=0.)
        cost = actor.cost.with_params(**weights)
        for mode in ('greedy',) if selection else ('greedy', 'sample'):
            for sample in range(1 if mode == 'greedy' else max(ks)):
                count = len(graphs) if mode == 'greedy' else k_count
                for start in range(0, count, cfg.evaluation.batch_size):
                    end = min(start + cfg.evaluation.batch_size, count)
                    path = out / 'candidates' / f'{mode}_a{ai:02d}_s{sample:03d}_g{start:04d}.json.gz'
                    if path.exists():
                        with gzip.open(path, 'rt', encoding='utf-8') as stream:
                            chunk = json.load(stream)
                        if chunk['fingerprint'] != fingerprint:
                            raise ValueError(f'Cached candidate inputs changed: {path}')
                    else:
                        with torch.random.fork_rng(devices=[]):
                            seed = 9001 + ai * 100_000 + sample * 1000 + start
                            torch.manual_seed(seed)
                            started = time.monotonic()
                            prediction = predict(actor.policy, graphs[start:end], weights, cost,
                                                 type(actor.env), actor.env_params, mode, 1)
                            rows = prediction_rows(prediction, start, alpha, sample)
                            for offset, row in enumerate(rows):
                                row.update(mode=mode, sampling_seed=seed,
                                    source_id=str(getattr(graphs[start+offset], 'source_id', indices[start+offset])),
                                    dataset_index=indices[start+offset],
                                    candidate_id=f'{mode}:{ai}:{sample}:{start+offset}',
                                    routes=prediction.routes[offset].cpu().tolist())
                            chunk = dict(fingerprint=fingerprint, mode=mode, sample=sample,
                                         seconds=time.monotonic()-started, rows=rows)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_name(path.name + '.tmp')
                        with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                            json.dump(chunk, stream, allow_nan=False)
                        atomic_replace(temporary, path)
                    all_rows.extend(chunk['rows'])
                    save_json(directory / 'progress.json', dict(phase='selection' if selection else mode,
                        alpha=alpha, sample=sample+1, samples=1 if mode == 'greedy' else max(ks),
                        graph=end, graphs=count, updated_at=now()))
    for key, value in actor.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)
    if selection:
        result = dict(feasible_fraction=float(np.mean([r['feasible'] for r in all_rows])),
                      cost=float(np.mean([r['scalar_cost'] for r in all_rows])))
        save_json(out / 'result.json', result)
        return result
    if not probe:
        rebuild_run(root, spec)
    return protocol


def study_training_run(cfg):
    """Scope isolated validation RNG to this study; do not alter existing Run behavior."""
    from dataclasses import replace
    import torch
    from connectpt.routes_generator.run.training import TrainingRun
    from connectpt.routes_generator.training.validate import ConstructionValidation
    from connectpt.routes_generator.training.data import ConstructionDataset

    class StudyDataset(ConstructionDataset):
        def prepare(self, cost, batch_size, evaluation, *args, **kwargs):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(evaluation.seed))
                return super().prepare(cost, batch_size, evaluation, *args, **kwargs)

    class StudyValidation(ConstructionValidation):
        def run(self, policy, logger_list, step):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.profile.evaluation.seed))
                return super().run(policy, logger_list, step)

    class StudyTrainingRun(TrainingRun):
        def setup(self):
            super().setup()
            self.task = replace(self.task, validation_factory=StudyValidation)
            self.dataset = StudyDataset(**vars(self.dataset))

        def evaluation_protocol(self, data, val_cost):
            protocol = super().evaluation_protocol(data, val_cost)
            protocol['validation']['action_rng'] = 'isolated CPU RNG reset to evaluation.seed per call'
            protocol['validation']['preference_seed'] = self.evaluation.seed
            protocol['validation']['source_ids'] = [data.validation_source_ids[i] for i in data.validation_indices]
            return protocol

    return StudyTrainingRun(cfg)


def worker(root, job_id):
    """The only heavyweight entry; called by the existing platform launcher."""
    os.chdir(root / 'source')
    spec = read_json(root / 'protocols/jobs' / f'{job_id}.json')
    if spec['kind'] == 'data':
        prepare_data(root)
        return
    if spec['kind'] == 'evaluate':
        evaluate_run(root, spec, probe=True)
        return
    cfg = worker_config(root, spec)
    directory = root / 'runs' / job_id
    result_path = directory / 'training_result.json'
    if result_path.exists():
        completed = read_json(result_path)
        if completed['iterations'] != spec['updates']:
            raise ValueError('Incomplete training cannot resume from actor-only weights; use a new study')
    else:
        if (directory / 'history.csv').exists():
            raise ValueError('Interrupted training is preserved; actor-only checkpoints are not an exact resume')
        save_json(directory / 'progress.json', dict(phase='training', updated_at=now()))
        study_training_run(cfg).run()
    if spec['kind'] == 'probe':
        return
    evaluate_run(root, spec, selection=True)
    evaluate_run(root, spec)
    save_json(directory / 'progress.json', dict(phase='completed', updated_at=now()))


class Coordinator:
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.started = time.monotonic()
        self.deadline = datetime.fromisoformat(state['started']).timestamp() + state['settings']['hours'] * 3600
        self.active = {}
        self.peak = float(state.get('peak_memory_bytes', 0))
        self.limit = state['settings']['memory_gb'] * GB
        self.reserved = .5 * GB if state['settings']['smoke'] else min(10 * GB, psutil.virtual_memory().total * .15)
        self.log = logging.getLogger('study')

    def save(self):
        save_json(self.root / 'manifest.json', self.state)

    def remaining(self):
        return max(0., self.deadline - time.time())

    def sample(self):
        own = psutil.Process()
        memory = process_memory([own] + own.children(recursive=True))
        self.peak = max(self.peak, memory['memory_bytes'])
        self.state['peak_memory_bytes'] = self.peak
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        rows = []
        for job_id, running in self.active.items():
            try:
                parent = psutil.Process(running['process'].pid)
                item = process_memory([parent] + parent.children(recursive=True))
            except psutil.NoSuchProcess:
                item = dict(memory_bytes=0, rss_bytes=0, process_count=0, memory_measure='RSS')
            running['peak'] = max(running['peak'], item['memory_bytes'])
            progress_path = self.root / 'runs' / job_id / 'progress.json'
            progress = read_json(progress_path) if progress_path.exists() else {}
            history = csv_rows(self.root / 'runs' / job_id / 'history.csv')
            if history:
                progress['updates_completed'] = next((r['time/iterations'] for r in reversed(history)
                                                      if r.get('time/iterations')), '')
            row = dict(timestamp=now(), job=job_id, pid=running['process'].pid,
                       threads=running['spec'].get('threads', 0), **item, progress=progress)
            rows.append(row)
            append_csv(self.root / 'runs' / job_id / 'resources.csv', {k: v for k, v in row.items() if k != 'progress'})
        status = dict(timestamp=now(), status=self.state['status'], stage=self.state.get('stage'),
            workers=len(self.active), worker_limit=self.state.get('workers', 1),
            memory_limit_bytes=self.limit, peak_memory_bytes=self.peak, **memory,
            system_available_bytes=vm.available, system_swap_bytes=swap.used, cpu_percent=psutil.cpu_percent(),
            elapsed_hours=(time.time()-datetime.fromisoformat(self.state['started']).timestamp())/3600,
            remaining_hours=self.remaining()/3600, jobs=rows,
            completed=sum(j['status'] == 'completed' for j in self.state['jobs'].values()),
            skipped=sum(j['status'] == 'skipped' for j in self.state['jobs'].values()),
            error=self.state.get('error'), budget=self.state.get('budget'),
            concurrency_reason=self.state.get('concurrency_reason'),
            stage_eta_hours=max(0., self.stage_deadline-time.time())/3600 if hasattr(self, 'stage_deadline') else None)
        save_json(self.root / 'status.json', status)
        lines = [f"{status['timestamp']} | {status['status']} | {status['stage']}",
            f"Workers: {status['workers']}/{status['worker_limit']} | OS processes: {memory['process_count']}",
            f"Study RAM ({memory['memory_measure']}): {memory['memory_bytes']/GB:.2f} / {self.limit/GB:.2f} GB | peak {self.peak/GB:.2f} GB",
            f"System available: {vm.available/GB:.2f} GB | swap {swap.used/GB:.2f} GB | CPU {status['cpu_percent']:.0f}%",
            f"Elapsed: {status['elapsed_hours']:.2f} h | remaining: {status['remaining_hours']:.2f} h | stage ETA: {status['stage_eta_hours']} h",
            f"Completed: {status['completed']} | skipped: {status['skipped']}",
            f"Budget: {status['budget']} | {status['concurrency_reason'] or ''}"]
        for row in rows:
            lines.append(f"  {row['job']} PID={row['pid']} threads={row['threads']} RAM={row['memory_bytes']/GB:.2f} GB {row['progress']}")
        if status['error']:
            lines.append('ERROR: ' + status['error'])
        temporary = self.root / 'status.txt.tmp'
        temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        atomic_replace(temporary, self.root / 'status.txt')
        append_csv(self.root / 'resources.csv', {k: status[k] for k in (
            'timestamp', 'status', 'stage', 'workers', 'worker_limit', 'memory_bytes', 'memory_measure',
            'rss_bytes', 'peak_memory_bytes', 'memory_limit_bytes', 'process_count', 'system_available_bytes',
            'system_swap_bytes', 'cpu_percent', 'elapsed_hours', 'remaining_hours')})
        return memory['memory_bytes'] <= self.limit and vm.available >= self.reserved

    def group(self, specs, stage, *, optional=False, eta=0.):
        self.stage_deadline = time.time() + eta
        self.state['stage'] = stage
        self.state['status'] = 'running'
        pending = []
        for spec in specs:
            job_id = spec['id']
            old = self.state['jobs'].get(job_id)
            if old and old['spec'] != spec:
                raise ValueError(f'Job parameters changed on restart: {job_id}')
            if old and old['status'] == 'completed':
                continue
            self.state['jobs'][job_id] = dict(spec=spec, status='pending')
            save_json(self.root / 'protocols/jobs' / f'{job_id}.json', spec)
            pending.append(spec)
        if not pending:
            return True
        if eta * 1.3 > self.remaining():
            if not optional:
                raise RuntimeError(f'Required group {stage} does not fit: forecast {eta/3600:.2f} h, remaining {self.remaining()/3600:.2f} h')
            for spec in pending:
                self.state['jobs'][spec['id']].update(status='skipped', reason='Insufficient time including per-run K100')
            self.save()
            return False
        self.log.info('Stage %s: %d jobs, max %d worker(s)', stage, len(pending), self.state['workers'])
        group_started = time.monotonic()
        try:
            while pending or self.active:
                safe = self.sample()
                if not safe and self.active:
                    raise RuntimeError('Memory guard: study limit exceeded or insufficient system headroom')
                if self.remaining() <= 0:
                    raise RuntimeError('Study wall-clock budget exhausted')
                while pending and len(self.active) < self.state['workers'] and safe:
                    spec = pending.pop(0)
                    directory = self.root / 'runs' / spec['id']
                    directory.mkdir(parents=True, exist_ok=True)
                    log = directory / 'console.log'
                    args = ['--wait', 'run_study.py', '--worker', '--out-dir', str(self.root),
                            '--job', spec['id'], '--log', str(log)]
                    command = (['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                str(REPO / 'scripts/launch.ps1')] if os.name == 'nt' else
                               ['bash', str(REPO / 'scripts/launch.sh')]) + args
                    environment = os.environ.copy()
                    environment.update(PYTHONPATH=str(self.root / 'source'), PYTHONUNBUFFERED='1',
                        MPLCONFIGDIR=str(self.root / 'environment/matplotlib'), PYTHONDONTWRITEBYTECODE='1',
                        OMP_NUM_THREADS=str(spec.get('threads', 1)), MKL_NUM_THREADS=str(spec.get('threads', 1)))
                    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
                    console = (directory / 'launcher.log').open('a', encoding='utf-8')
                    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=console, stderr=subprocess.STDOUT)
                    self.active[spec['id']] = dict(process=process, console=console, spec=spec, started=time.monotonic(), peak=0)
                    self.state['jobs'][spec['id']].update(status='running', pid=process.pid, started=now())
                    self.save()
                if pending and not self.active and not safe:
                    raise RuntimeError('Not enough available RAM to start a worker safely')
                finished = []
                for job_id, running in self.active.items():
                    exit_code = running['process'].poll()
                    if exit_code is None:
                        continue
                    running['console'].close()
                    entry = self.state['jobs'][job_id]
                    entry.update(status='completed' if exit_code == 0 else 'failed', exit_code=exit_code,
                        finished=now(), seconds=time.monotonic()-running['started'], peak_memory_bytes=running['peak'])
                    finished.append(job_id)
                    self.save()
                    if exit_code:
                        raise RuntimeError(f'Worker {job_id} failed; inspect runs/{job_id}/console.log.err or .out')
                for job_id in finished:
                    del self.active[job_id]
                if pending or self.active:
                    time.sleep(self.state['settings']['poll_seconds'])
        finally:
            for job_id, running in self.active.items():
                stop_process_tree(running['process'].pid)
                running['process'].wait(timeout=10)
                running['console'].close()
                if self.state['jobs'][job_id]['status'] == 'running':
                    self.state['jobs'][job_id].update(status='interrupted', finished=now())
            self.active.clear()
            self.save()
            self.sample()
        if eta > 0:
            factor = self.state.get('forecast_factor', 1.)
            self.state['forecast_factor'] = max(factor, factor * (time.monotonic()-group_started) / eta)
            self.save()
        return True


def training_spec(job_id, settings, **changes):
    spec = dict(id=job_id, kind='training', algorithm='d3po', weight=.01, detach=False,
                batch=settings['batches'][-1], horizon=settings['horizon'], threads=2,
                routes=settings['routes'], length=settings['length'], sampler='sampler',
                seed=0, alpha=None, updates=2, parent=None)
    spec.update(changes)
    return spec


def choose(root, specs):
    values = [(spec, read_json(root / 'selection' / spec['id'] / 'result.json')) for spec in specs]
    winner, result = min(values, key=lambda pair: (-pair[1]['feasible_fraction'], pair[1]['cost']))
    return winner, result


def resource_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = csv_rows(root / 'resources.csv')
    if not rows:
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), layout='constrained')
    x = [float(r['elapsed_hours']) for r in rows]
    axes[0].plot(x, [float(r['memory_bytes']) / GB for r in rows])
    axes[0].axhline(float(rows[0]['memory_limit_bytes']) / GB, color='red', linestyle='--')
    axes[0].set_ylabel('Study memory, GB')
    axes[1].step(x, [int(r['workers']) for r in rows], where='post')
    axes[1].set(xlabel='Elapsed hours', ylabel='Workers', yticks=[0, 1, 2])
    (root / 'figures').mkdir(exist_ok=True)
    fig.savefig(root / 'figures/resources.png', dpi=140)
    plt.close(fig)


def summary(root):
    state = read_json(root / 'manifest.json')
    rows, models = [], []
    for job_id, job in state['jobs'].items():
        result_path = root / 'evaluation' / job_id / 'best/results.json'
        if job['status'] == 'completed' and result_path.exists():
            result = read_json(result_path)
            models.extend(result['models'])
            rows.extend(result['models'][0]['overall'])
    from scripts.evaluate_policy import write_csv, pareto_figures
    directory = root / 'summary'
    directory.mkdir(exist_ok=True)
    if rows:
        write_csv(directory / 'metrics.csv', rows)
    # Overlay only comparable tasks/specialist grids; never pool cities or seeds.
    groups = {}
    for model in models:
        spec = state['jobs'][model['label']]['spec']
        stage = 'screen' if spec['id'].startswith(('lambda_', 'detached_')) else 'main'
        key = f"{variant_name(spec)}_{stage}_" + ('universal' if spec['alpha'] is None else f"alpha{spec['alpha']}")
        groups.setdefault(key, []).extend(model['per_city'])
    for key, values in groups.items():
        labels = list(dict.fromkeys(row['model'] for row in values))
        for start in range(1, max(2, len(labels)), 2):
            subset = [r for r in values if r['model'] in [labels[0]] + labels[start:start+2]]
            out = directory / f'{key}_{start//2+1}'
            out.mkdir(exist_ok=True)
            count = max(r['graph'] for r in subset) + 1
            pareto_figures(out, subset, list(range(count)), state['settings']['ks'], complete=True)
    text = ['# Исследование PPO/D3PO', '', f"Статус: {state['status']}",
            f"Память: не более {state['settings']['memory_gb']} GB; срок: {state['settings']['hours']} ч.",
            '', 'Текущее состояние: ../status.txt; история ресурсов: ../resources.csv.', '',
            '| Задание | Статус | Время, с |', '|---|---|---:|']
    for job_id, job in state['jobs'].items():
        text.append(f"| {job_id} | {job['status']} | {job.get('seconds', 0):.1f} |")
    if state.get('error'):
        text += ['', 'Причина остановки: ' + state['error']]
    text += ['', 'Отбор вариантов использует selection, не test. Pareto строится отдельно по городам.',
             'Screening и основной бюджет различаются; один training seed не устанавливает превосходство.',
             'HV reference [3,3] зафиксирован до test; coverage сохранено в metrics JSON.',
             'best_of_K — выбранные ответы по alpha; greedy+K не смешивается с K.', '']
    (directory / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    resource_figures(root)


def pipeline(controller, calibrate_only=False):
    root, state = controller.root, controller.state
    settings = state['settings']
    controller.group([dict(id='prepare_data', kind='data', threads=1)], 'data')
    verify_files(root, read_json(root / 'protocols/data-hashes.json'))
    cores = psutil.cpu_count(logical=False) or 1
    profiles = []
    for batch in settings['batches']:
        specs = [training_spec(f'probe_b{batch}_{algorithm}', settings, kind='probe', algorithm=algorithm,
                    weight=1. if algorithm == 'd3po' else 0., batch=batch,
                    threads=max(1, min(4, cores // 2))) for algorithm in ('ppo', 'd3po')]
        state['workers'] = 1
        controller.group(specs, f'calibration_b{batch}_single')
        profiles.append((sum(state['jobs'][s['id']]['seconds'] for s in specs) / (4 * batch * settings['horizon']), specs))
    _, best = min(profiles, key=lambda item: item[0])
    profile = best[-1]
    single_seconds = sum(state['jobs'][s['id']]['seconds'] for s in best)
    pair = [dict(spec, id=spec['id'] + '_pair') for spec in best]
    peak = sum(state['jobs'][s['id']]['peak_memory_bytes'] for s in best)
    if settings['workers'] == 2 and cores >= 2 and peak * 1.25 < controller.limit and psutil.virtual_memory().available > peak * 1.25 + controller.reserved:
        state['workers'] = 2
        controller.group(pair, 'calibration_pair')
        span = (max(datetime.fromisoformat(state['jobs'][s['id']]['finished']) for s in pair) -
                min(datetime.fromisoformat(state['jobs'][s['id']]['started']) for s in pair)).total_seconds()
        workers = 2 if span < single_seconds else 1
        multiplier = max(1., sum(state['jobs'][s['id']]['seconds'] for s in pair) / single_seconds)
        reason = 'Measured pair faster' if workers == 2 else 'Single worker has better aggregate throughput'
    else:
        workers, multiplier, reason = 1, 1., 'Pair not safe under memory/core limits'
    state['workers'], state['concurrency_reason'] = workers, reason
    # Actual inference measurement, not a guessed 100x extrapolation of training FPS.
    evaluation_probe = dict(profile, id='probe_inference', kind='evaluate', model_run=profile['id'])
    # Reuse a trained probe actor, without pretending it is a main model.
    probe_dir = root / 'runs/probe_inference'
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'training_result.json'):
        shutil.copy2(root / 'runs' / profile['id'] / name, probe_dir / name)
    (probe_dir / 'checkpoints').mkdir(exist_ok=True)
    shutil.copy2(root / 'runs' / profile['id'] / 'checkpoints/best.pt', probe_dir / 'checkpoints/best.pt')
    controller.group([evaluation_probe], 'calibration_inference')
    inference = state['jobs']['probe_inference']['seconds']
    prediction_seconds = []
    for path in (root / 'evaluation/probe_inference/best/candidates').glob('*.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            chunk = json.load(stream)
        prediction_seconds.append(chunk['seconds'] / len(chunk['rows']))
    train_seconds, validation_seconds = {}, {}
    for spec in best:
        history = csv_rows(root / 'runs' / spec['id'] / 'history.csv')
        elapsed = [float(r['time/time_elapsed']) for r in history if r.get('time/time_elapsed')]
        per_update = max(.001, elapsed[-1] - elapsed[-2]) if len(elapsed) >= 2 else elapsed[-1] / 2
        train_seconds[spec['algorithm']] = per_update * multiplier
        validation_seconds[spec['algorithm']] = max(0., state['jobs'][spec['id']]['seconds'] - 2 * per_update) / 2 * multiplier
    # Startup time is a fixed cost, not a per-network cost. Selection also uses inference.
    queries = settings['preference_count'] * (settings['test_graphs'] + settings['selection_graphs']
                                             + max(settings['ks']) * settings['k_graphs'])
    eval_seconds = max(prediction_seconds) * queries * multiplier + inference + (2 if settings['smoke'] else 60)
    budget = dict(batch=profile['batch'], horizon=profile['horizon'], threads=profile['threads'], workers=workers,
                  train_seconds_per_update=train_seconds, validation_seconds=validation_seconds,
                  inference_seconds_per_network=max(prediction_seconds), evaluation_seconds_per_universal_run=eval_seconds,
                  concurrency_reason=reason, memory_limit_bytes=controller.limit, memory_peak_bytes=controller.peak)
    save_json(root / 'calibration/budget.json', budget)
    state['calibration'] = budget
    if calibrate_only:
        state['status'] = 'calibrated'
        controller.save()
        return
    batch, horizon = profile['batch'], profile['horizon']
    screen_n = 2 * batch * horizon if settings['smoke'] else 307200
    candidates = [2 * batch * horizon] if settings['smoke'] else [3072000, 1536000, 768000]
    def estimate(specs):
        loads = [0.] * workers
        for s in specs:
            if state['jobs'].get(s['id'], {}).get('status') == 'completed':
                continue
            duration = (train_seconds[s['algorithm']] * s['updates'] * (s['batch'] * s['horizon']) / (batch * horizon)
                        + validation_seconds[s['algorithm']] * (math.ceil(s['updates'] / (1 if settings['smoke'] else 10)) + 1)
                        + eval_seconds)
            loads[loads.index(min(loads))] += duration
        return max(loads) * state.get('forecast_factor', 1.)
    def make(job_id, transitions, **changes):
        spec = training_spec(job_id, settings, batch=batch, horizon=horizon, threads=profile['threads'], **changes)
        per = spec['batch'] * spec['horizon']
        if transitions % per:
            raise ValueError('Budget must divide batch*horizon exactly')
        spec['updates'] = transitions // per
        return spec
    sweep = [make(f'lambda_{str(weight).replace(".", "p")}', screen_n, weight=weight) for weight in (0., .01, .1, 1.)]
    sweep.append(make('detached_control', screen_n, weight=1., detach=True))
    selected_n = None
    budget['forecasts'] = []
    for transitions in [state['budget']] if state.get('budget') else candidates:
        minimal = [make('main_ppo', transitions, algorithm='ppo', weight=0.), make('main_d3po', transitions)]
        required = (estimate(sweep) + estimate(minimal)) * 1.3
        budget['forecasts'].append(dict(transitions=transitions, required_hours_with_margin=required/3600,
                                       remaining_hours=controller.remaining()/3600))
        if required < controller.remaining():
            selected_n = transitions
            break
    if selected_n is None:
        budget['decision'] = 'minimum_does_not_fit'
        save_json(root / 'calibration/budget.json', budget)
        raise RuntimeError(f'Minimum required study including per-run K100 exceeds remaining {controller.remaining()/3600:.2f} h; see calibration/budget.json')
    state['budget'] = selected_n
    budget.update(transitions_per_main_run=selected_n, transitions_per_screen=screen_n)
    save_json(root / 'calibration/budget.json', budget)
    controller.group(sweep, 'lambda_sweep', eta=estimate(sweep))
    winner, metric = choose(root, sweep[:-1])
    state['choices']['lambda'] = dict(parent=winner['id'], weight=winner['weight'], selection=metric)
    main = [make('main_ppo', selected_n, algorithm='ppo', weight=0., seed=1),
            make('main_d3po', selected_n, weight=winner['weight'], seed=1)]
    controller.group(main, 'main', eta=estimate(main))
    parent, metric = choose(root, main)
    if not settings['smoke']:
        simplex = [dict(spec, id=spec['id'] + '_simplex', sampler='simplex', parent=spec['id']) for spec in main]
        if controller.group(simplex, 'simplex', optional=True, eta=estimate(simplex)):
            parent, metric = choose(root, main + simplex)
        state['choices']['universal'] = dict(parent=parent['id'], selection=metric)
        specialists = [dict(parent, id=f'specialist_{str(alpha).replace(".", "p")}', alpha=alpha,
                            parent=parent['id']) for alpha in (.4, .5, .6)]
        controller.group(specialists, 'specialists', optional=True, eta=estimate(specialists))
        if settings['extended']:
            second = next(b for b in settings['batches'] if b != parent['batch'])
            alternative = dict(parent, id='batch_alternative', batch=second,
                               updates=selected_n // (second * parent['horizon']), parent=parent['id'])
            if controller.group([alternative], 'batch', optional=True, eta=estimate([alternative])):
                parent, metric = choose(root, [parent, alternative])
            short = dict(parent, id='horizon_alternative', horizon=60,
                         updates=selected_n // (parent['batch'] * 60), parent=parent['id'])
            if controller.group([short], 'horizon', optional=True, eta=estimate([short])):
                parent, metric = choose(root, [parent, short])
            routes = [dict(parent, id='routes_5', routes=5, parent=parent['id']),
                      dict(parent, id='length_8', length=8, parent=parent['id'])]
            controller.group(routes, 'route_limits', optional=True, eta=estimate(routes))
            # A promoted profile needs its missing algorithm control.
            if parent['id'] not in [s['id'] for s in main + simplex]:
                opposite = 'ppo' if parent['algorithm'] == 'd3po' else 'd3po'
                control = dict(parent, id='promoted_control', algorithm=opposite,
                               weight=0. if opposite == 'ppo' else winner['weight'], parent=parent['id'])
                controller.group([control], 'promoted_control', optional=True, eta=estimate([control]))
        if settings['confirm']:
            confirmation = [dict(parent, id=f'confirm_{a}_seed2', algorithm=a, seed=2,
                weight=0. if a == 'ppo' else winner['weight'], parent=parent['id']) for a in ('ppo', 'd3po')]
            controller.group(confirmation, 'confirmation', optional=True, eta=estimate(confirmation))
    state['status'], state['stage'] = 'completed', 'summary'
    controller.save()


def write_instructions(root):
    torch_requirement = next(line for line in (root / 'environment/runtime-requirements.txt').read_text().splitlines()
                             if line.startswith('torch=='))
    (root / 'START_HERE.txt').write_text('''Пакет PPO/D3PO: всё хранится внутри этой папки.

Linux, Python 3.11+ (локальная проверка: Python 3.13, CPU).
Из корня этой папки:
  python3 -m venv source/.venv
  source source/.venv/bin/activate
  python -m pip install TORCH_REQUIREMENT --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r environment/runtime-requirements.txt
  bash source/scripts/launch.sh run_study.py --out-dir "$PWD" --resume --log "$PWD/controller-console.log"

Начнётся генерация данных и измерение ресурсов. Затем автоматически продолжится
основная серия, если минимальная серия с оценкой K100 укладывается в бюджет.
Иначе запуск остановится с объяснением в status.txt и calibration/budget.json.
35 часов отсчитываются с первого запуска, а не с упаковки/копирования.
Предел 50 GB относится к сумме процессов; одновременно не более двух workers.
Монитор памяти опрашивается раз в 5 секунд; это не ограничение cgroup ОС.

Смотреть:
  watch -n 5 cat status.txt
  tail -f controller.log
  calibration/budget.json       — оценка времени и выбранная параллельность
  resources.csv                — память, число workers/процессов, время
  figures/resources.png        — график ресурсов после завершения
  summary/report.md            — список выполненных и пропущенных экспериментов
  summary/metrics.csv          — сводная таблица метрик
  evaluation/<run>/best/        — Pareto, метрики, все кандидаты и выбранные ответы
  runs/<run>/checkpoints/       — best.pt и last.pt
  runs/<run>/console.log.out    — вывод worker; Windows: ошибки в .err

Скопировать всю папку обратно. Графики из сохранённых кандидатов без инференса:
  python source/scripts/run_study.py --out-dir "$PWD" --replot

--resume сохраняет выполненные задания и продолжает оценку по чанкам.
Прерванное обучение не продолжает с actor-only весов: точного состояния
оптимизатора нет. Его артефакты сохраняются, требуется новая папка серии.
Незавершённый запуск и время простоя входят в исходный лимит 35 часов.
Настройки серии зафиксированы в manifest.json; исходники — в source/.
'''.replace('TORCH_REQUIREMENT', torch_requirement), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--hours', type=float, default=35.)
    parser.add_argument('--memory-gb', type=float, default=50.)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Export the portable folder without generating data or starting its clock')
    parser.add_argument('--extended', action='store_true', help='Try batch/horizon/route branches when the budget permits')
    parser.add_argument('--confirm', action='store_true', help='Try a matched fresh-seed confirmation pair')
    parser.add_argument('--replot', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Tiny two-update integration check; not a scientific experiment')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.replot and (args.prepare_only or args.calibrate_only):
        parser.error('replot cannot be combined with preparation/calibration')
    if args.prepare_only and args.resume:
        parser.error('prepare-only requires a new output directory')
    root = Path(args.out_dir).expanduser().resolve()
    if args.worker or ((args.resume or args.replot) and (root / 'manifest.json').exists()):
        sys.path.insert(0, str(root / 'source'))
        frozen = root / 'source/scripts/run_study.py'
        if Path(__file__).resolve() != frozen:
            sys.argv = [str(frozen)] + list(sys.argv[1:] if argv is None else argv)
            runpy.run_path(str(frozen), run_name='__main__')
            return
        if args.worker:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
            started = time.monotonic()
            try:
                worker(root, args.job)
            finally:
                cpu = psutil.Process().cpu_times()
                save_json(root / 'runs' / args.job / 'worker-resources.json', dict(
                    cpu_seconds=cpu.user+cpu.system, wall_seconds=time.monotonic()-started))
            return
    if args.hours <= 0 or args.memory_gb <= 0:
        parser.error('hours and memory-gb must be positive')
    frozen_study = REPO == root / 'source' and (root / 'manifest.json').is_file()
    if (root == REPO or root in REPO.parents) and not frozen_study:
        parser.error('out-dir must be a dedicated study directory')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('environment', 'protocols/jobs', 'calibration', 'summary', 'figures'):
        (root / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'controller.log', encoding='utf-8')], force=True)
    manifest = root / 'manifest.json'
    lock = root / '.controller.lock'
    if lock.exists():
        owner = read_json(lock)
        try:
            live = psutil.Process(owner['pid'])
            if abs(live.create_time() - owner['created']) < 1:
                raise RuntimeError('This study already has a live controller')
        except psutil.NoSuchProcess:
            pass
        lock.unlink()
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    controller = None
    try:
        if manifest.exists():
            if not (args.resume or args.replot):
                raise ValueError('Study already exists; use --resume or --replot, or a new output directory')
            state = read_json(manifest)
            if state['schema'] != SCHEMA:
                raise ValueError('Unsupported study schema')
            verify_files(root, state['source_hashes'])
        else:
            settings = dict(hours=args.hours, memory_gb=args.memory_gb, workers=args.workers,
                smoke=args.smoke, extended=args.extended, confirm=args.confirm,
                pool_graphs=32 if args.smoke else 32768, nodes=9 if args.smoke else 20,
                routes=2 if args.smoke else 10, length=6 if args.smoke else 12,
                validation_graphs=2 if args.smoke else 128, selection_graphs=2 if args.smoke else 32,
                test_graphs=2 if args.smoke else 64, k_graphs=1 if args.smoke else 16,
                preference_count=3 if args.smoke else 11, ks=[1, 3] if args.smoke else [1, 10, 100],
                batches=[2, 4] if args.smoke else [64, 128], horizon=4 if args.smoke else 120,
                poll_seconds=1 if args.smoke else 5)
            state = dict(schema=SCHEMA, created=now(), settings=settings, jobs={}, choices={},
                         source_hashes=snapshot(root), status='preparing', workers=1, peak_memory_bytes=0)
            save_json(manifest, state)
        if args.replot:
            verify_files(root, read_json(root / 'protocols/data-hashes.json'))
            sys.path.insert(0, str(root / 'source'))
            for job in state['jobs'].values():
                if job['status'] == 'completed' and job['spec']['kind'] == 'training':
                    rebuild_run(root, job['spec'])
            summary(root)
            return
        if args.prepare_only:
            state['status'] = 'prepared'
            save_json(manifest, state)
            (root / 'status.txt').write_text('Prepared; no workers started. Budget starts on --resume.\n', encoding='utf-8')
            write_instructions(root)
            print(f'Portable study prepared: {root}', flush=True)
            return
        print(f'Study: {root}\nStatus: {root / "status.txt"}', flush=True)
        print(f"watch -n 5 cat {shlex.quote(str(root / 'status.txt'))}", flush=True)
        sys.path.insert(0, str(root / 'source'))
        state.pop('error', None)
        if 'started' not in state:
            save_json(root / 'environment/execution-machine.json', machine_info())
            freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, check=True)
            (root / 'environment/execution-pip-freeze.txt').write_bytes(freeze.stdout)
        state.setdefault('started', now())
        write_instructions(root)
        controller = Coordinator(root, state)
        if state['status'] != 'completed':
            pipeline(controller, args.calibrate_only)
    except BaseException as error:
        if controller is not None:
            controller.state.update(status='failed', error=f'{type(error).__name__}: {error}')
            controller.save()
        logging.exception('Study stopped; all existing artifacts are preserved')
        raise
    finally:
        if controller is not None:
            controller.save()
            controller.sample()
            try:
                summary(root)
            except Exception:
                logging.exception('Could not regenerate final summary')
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

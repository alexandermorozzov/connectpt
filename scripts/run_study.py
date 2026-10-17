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
SCHEMA = 2
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
    project = source / 'pyproject.toml'
    project.write_text(project.read_text(encoding='utf-8').replace(
        '[tool.setuptools_scm]', '[tool.setuptools_scm]\nfallback_version = ' + json.dumps(version('connectpt'))),
        encoding='utf-8')
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
    cfg.train_loop.val_period = settings['val_period']
    cfg.train_loop.cpu_threads = spec['threads']
    cfg.train_loop.preference_distribution = spec['sampler']
    cfg.train_loop.resample_preferences_on_terminal = spec['resample']
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
    cfg.evaluation.pareto_graph_count = settings['selection_graphs']
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
    settings = read_json(root / 'manifest.json')['settings']
    base = root / 'data/base'
    for part, count, seed in [('pool', settings['pool_graphs'], 1701),
                              ('test', settings['test_graphs'], 1702),
                              ('selection', settings['selection_graphs'], 1703)]:
        cfg = load_experiment('generate_construction')
        cfg.run.cpu, cfg.run.seed = True, seed
        cfg.dataset_build.n_graphs = count
        cfg.dataset_build.source.min_nodes = settings['nodes']
        cfg.dataset_build.limits.n_routes = settings['routes']
        cfg.dataset_build.limits.max_route_len = settings['length']
        prepare_dataset(cfg, base / part)
    pool = load_dataset(base / 'pool', kind='construction')
    validation = pool.split.validation.tolist()
    selection = load_dataset(base / 'selection', kind='construction')
    test = load_dataset(base / 'test', kind='construction')
    sources = [set(pool.params.source_ids), set(selection.params.source_ids), set(test.params.source_ids)]
    if any(sources[i] & sources[j] for i in range(3) for j in range(i)):
        raise ValueError('Source city leaks between pool, selection and test')
    save_json(root / 'protocols/splits.json', dict(
        train=pool.split.train.tolist(), validation=validation, selection=list(range(len(selection.graphs))),
        train_sources=[pool.params.source_ids[i] for i in pool.split.train.tolist()],
        validation_sources=[pool.params.source_ids[i] for i in validation],
        selection_sources=list(selection.params.source_ids), test_sources=list(test.params.source_ids)))
    # No route data live in a construction dataset. Only limits/fingerprint differ.
    for routes, length in {(settings['routes'], settings['length']),
                           (max(1, settings['routes'] // 2), settings['length'])}:
        tag = f'r{routes}_l{length}'
        for part in ('pool', 'test', 'selection'):
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


def common_reference_rows(rows, routes):
    """Recost every candidate before best-of-K selection, keeping physical minutes."""
    result = []
    for row in rows:
        penalty = row['penalized_0'] - row['objective_0']
        rtt = row['objective_1'] * routes / 10
        result.append(dict(row, objective_1=rtt, penalized_1=rtt + penalty,
                           training_normalized_cost=row['scalar_cost'],
                           scalar_cost=row['preference'] * row['objective_0']
                           + (1 - row['preference']) * rtt + penalty))
    return result


def rebuild_run(root, spec, checkpoint_label='best'):
    """Reuse the existing evaluator's selectors, Pareto plotting and publication."""
    from scripts.evaluate_policy import select_rows, publish, write_csv, pareto_figures, digest
    import numpy as np
    settings = read_json(root / 'manifest.json')['settings']
    out = root / 'evaluation' / spec['id'] / checkpoint_label
    protocol = read_json(out / 'protocol.json')
    label = spec['id'] if checkpoint_label == 'best' else f"{spec['id']}@{protocol['updates']}"
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
    record = dict(label=label, training_spec=dict(spec, updates=protocol['updates']),
                  identity=protocol['identity'], completed_alphas=len(protocol['alphas']),
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
        record['overall'].append(dict(model=label, mode=mode, K=k,
            candidates_per_query=max(1, k), generation_evaluation_seconds=seconds, **brief))
        save_json(out / f'metrics_{mode}.json', metrics)
        for alpha in protocol['alphas']:
            subset = [r for r in rows if r['preference'] == alpha]
            _, brief_alpha = candidate_summary(subset, 1, spec['routes'])
            record['by_alpha'].append(dict(model=label, mode=mode, alpha=alpha, **brief_alpha))
        for row in rows:
            selections.append(dict(mode=mode, graph=row['graph'], alpha=row['preference'],
                                   candidate_id=row['candidate_id'], sample=row.get('selected_sample', row['sample'])))
            record['per_city'].append(dict(model=label, mode=mode, graph=row['graph'],
                alpha=row['preference'], cost=row['scalar_cost'], ATT_minutes=row['ATT'],
                RTT_minutes=row['RTT'], feasible=row['feasible'], network_id=digest(row['network_signature'])[:16]))
    write_csv(out / 'selected.csv', selections)
    request = dict(graph_count=protocol['k_graph_count'], preference_count=len(protocol['alphas']),
                   ks=protocol['ks'], seed=protocol['seed'], test_indices=list(range(protocol['k_graph_count'])))
    common_greedy = common_reference_rows(k_greedy, spec['routes'])
    common_sampled = common_reference_rows(sampled, spec['routes'])
    common_out = out / 'common_reference10'
    common_out.mkdir(exist_ok=True)
    common = dict(overall=[], by_alpha=[], per_city=[], selected=[])
    for mode, rows in [('greedy', common_greedy)] + [
            (f'best_of_{k}', select_rows(common_greedy, common_sampled, k)) for k in protocol['ks']]:
        metrics, brief = candidate_summary(rows, len(protocol['alphas']), 10)
        save_json(common_out / f'metrics_{mode}.json', metrics)
        common['overall'].append(dict(model=label, mode=mode, **brief))
        for alpha in protocol['alphas']:
            _, brief_alpha = candidate_summary([r for r in rows if r['preference'] == alpha], 1, 10)
            common['by_alpha'].append(dict(model=label, mode=mode, alpha=alpha, **brief_alpha))
        for row in rows:
            common['selected'].append(dict(mode=mode, graph=row['graph'], alpha=row['preference'],
                candidate_id=row['candidate_id'], cost=row['scalar_cost'],
                training_normalized_cost=row['training_normalized_cost']))
            common['per_city'].append(dict(model=label, mode=mode, graph=row['graph'],
                alpha=row['preference'], cost=row['scalar_cost'], ATT_minutes=row['ATT'],
                RTT_minutes=row['RTT'], feasible=row['feasible'], network_id=digest(row['network_signature'])[:16]))
    for name, rows in common.items():
        write_csv(common_out / f'{name}.csv', rows)
    record['common_reference10'] = common
    publish(out, out / 'report.md', [record], complete=True, request=request)
    report_path = out / 'report.md'
    report_path.write_text(report_path.read_text(encoding='utf-8').replace(
        'Greedy+K использует K+1 кандидатов и показан отдельным режимом.',
        'Best-of-K выбирает из K stochastic кандидатов; greedy сохранён отдельно.'), encoding='utf-8')
    metrics, _ = candidate_summary(greedy, len(protocol['alphas']), spec['routes'])
    save_json(out / 'greedy_all_metrics.json', metrics)
    all_rows = [dict(model=label, mode='greedy', graph=r['graph'], alpha=r['preference'],
                    cost=r['scalar_cost'], ATT_minutes=r['ATT'], RTT_minutes=r['RTT'], feasible=r['feasible'],
                    network_id=digest(r['network_signature'])[:16]) for r in greedy]
    full = out / 'greedy_all'
    full.mkdir(exist_ok=True)
    pareto_figures(full, all_rows, list(range(protocol['graph_count'])), protocol['ks'], complete=True)
    return record


def evaluate_run(root, spec, *, selection=False, probe=False, checkpoint_label='best'):
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from connectpt.routes_generator.run.training import TrainingRun
    from connectpt.routes_generator.training.data import evaluation_graphs
    from scripts.evaluate_policy import prediction_rows
    from connectpt.routes_generator.policy.predict import predict
    settings = read_json(root / 'manifest.json')['settings']
    directory = root / 'runs' / spec['id']
    out = (root / 'selection' / spec['id']) if selection else root / 'evaluation' / spec['id'] / checkpoint_label
    cfg = OmegaConf.load(directory / 'config.yaml')
    checkpoint = directory / 'checkpoints' / f'{checkpoint_label}.pt'
    cfg.paths.init_checkpoint_path = f"{spec['id']}/checkpoints/{checkpoint_label}.pt"
    actor = TrainingRun(cfg)
    actor.setup()
    torch.set_num_threads(spec['threads'])
    actor.policy.eval()
    params = actor.data_input.params
    raw = evaluation_graphs(root / 'data' / variant_name(spec) / 'selection' if selection else params.test_path,
                            params.kind, params.limits)
    splits = read_json(root / 'protocols/splits.json')
    indices = splits['selection'] if selection else list(range(len(raw)))
    source_ids = splits['selection_sources' if selection else 'test_sources']
    if probe:
        indices = indices[:cfg.evaluation.batch_size]
    graphs = [actor.dataset.evaluation_pipeline.graph(raw[i]) for i in indices]
    alphas = [spec['alpha']] if spec.get('alpha') is not None else np.linspace(0, 1, settings['preference_count']).tolist()
    if probe:
        alphas = [.5] if settings['smoke'] else [0., .5, 1.]
    ks = [1] if selection or probe else settings['ks']
    k_count = min(len(graphs), settings['k_graphs'])
    inputs = dict(spec=spec, config=file_hash(directory / 'config.yaml'), checkpoint=file_hash(checkpoint),
                  library=read_json(root / 'manifest.json')['source_hashes'],
                  data=read_json(root / 'protocols/data-hashes.json'), indices=indices, alphas=alphas,
                  ks=ks, k_count=k_count, seed=9001, selection=selection, probe=probe)
    fingerprint = identity(inputs)
    protocol = dict(identity=inputs, fingerprint=fingerprint, graph_count=len(graphs),
                    split='independent selection' if selection else 'independent test',
                    source_ids=[source_ids[i] for i in indices],
                    checkpoint_label=checkpoint_label,
                    updates=spec['comparison_updates'] if checkpoint_label != 'best' else spec['updates'],
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
                                    source_id=source_ids[indices[start+offset]],
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
        from scripts.evaluate_policy import write_csv
        by_alpha = []
        for alpha in alphas:
            rows = [r for r in all_rows if r['preference'] == alpha]
            by_alpha.append(dict(alpha=alpha, cost=float(np.mean([r['scalar_cost'] for r in rows])),
                ATT_minutes=float(np.mean([r['ATT'] for r in rows])),
                RTT_minutes=float(np.mean([r['RTT'] for r in rows])),
                feasible_fraction=float(np.mean([r['feasible'] for r in rows]))))
        write_csv(out / 'by_alpha.csv', by_alpha)
        save_json(out / 'metrics.json', candidate_summary(all_rows, len(alphas), spec['routes'])[0])
        save_json(out / 'result.json', result)
        return result
    if not probe:
        rebuild_run(root, spec, checkpoint_label)
    return protocol


def study_training_run(cfg, comparison_updates=None):
    """Scope isolated validation RNG to this study; do not alter existing Run behavior."""
    from dataclasses import replace
    import torch
    from connectpt.routes_generator.run.training import TrainingRun
    from connectpt.routes_generator.training.validate import ConstructionValidation
    from connectpt.routes_generator.training.data import ConstructionDataset
    from connectpt.routes_generator.policy.checkpoints import save_model_weights
    from omegaconf import OmegaConf

    owner = None

    class StudyDataset(ConstructionDataset):
        def prepare(self, cost, batch_size, evaluation, *args, **kwargs):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(evaluation.seed))
                prepared = super().prepare(cost, batch_size, evaluation, *args, **kwargs)
            if owner is not None and getattr(owner, 'run_dir', None) is not None:
                data, _ = prepared
                histograms = {key: torch.zeros(11, dtype=torch.long) for key in ('transitions', 'completed')}
                prepare_step, record_step = data.prepare_step, data.record_step
                alpha = None

                def step(env):
                    nonlocal alpha
                    prepare_step(env)
                    alpha = env.state.cost_weights['demand_time_weight'].detach()

                def record(reward, ended, active):
                    for name, mask in (('transitions', active), ('completed', ended)):
                        bins = (alpha[mask].clamp(0, 1) * 10).long()
                        histograms[name] += torch.bincount(bins.cpu(), minlength=11)
                    return record_step(reward, ended, active)

                data.prepare_step, data.record_step = step, record
                owner.preference_histograms = histograms
            return prepared

    class StudyValidation(ConstructionValidation):
        def run(self, policy, logger_list, step):
            started = time.monotonic()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.profile.evaluation.seed))
                cost = super().run(policy, logger_list, step)
            directory = getattr(owner, 'run_dir', None)
            if directory is not None:
                append_csv(directory / 'validation.csv', dict(step=step, cost=cost,
                    seconds=time.monotonic()-started, graphs=self.data.validation_count))
                self.validated_cost = cost
                if hasattr(owner, 'preference_histograms'):
                    save_json(directory / 'preferences.json', dict(
                        bins='[0,.1), [.1,.2), ..., [.9,1), {1}',
                        **{k: v.tolist() for k, v in owner.preference_histograms.items()}))
            return cost

        def monitor(self, policy, logger_list, step, **kwargs):
            super().monitor(policy, logger_list, step, **kwargs)
            # Tracker calls monitor after freezing features and updating its best.
            # Its untracked feature-calibration call never enters this hook.
            if (comparison_updates is not None and step <= comparison_updates
                    and self.validated_cost < getattr(self, 'prefix_best', math.inf)):
                self.prefix_best = self.validated_cost
                save_model_weights(policy, owner.best_checkpoint.with_name(f'best_at_{comparison_updates}.pt'),
                                   OmegaConf.to_container(cfg, resolve=True))
                save_json(owner.run_dir / 'comparison_checkpoint.json', dict(
                    budget_updates=comparison_updates, best_step=step, best_cost=self.validated_cost))

    class StudyTrainingRun(TrainingRun):
        def setup(self):
            nonlocal owner
            super().setup()
            owner = self
            self.task = replace(self.task, validation_factory=StudyValidation)
            self.dataset = StudyDataset(**vars(self.dataset))
            if getattr(self, 'run_dir', None) is not None:
                digest = hashlib.sha256()
                for key, value in self.policy.state_dict().items():
                    digest.update(key.encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                save_json(self.run_dir / 'initialization.json', dict(
                    seed=self.run_params.seed, actor_sha256=digest.hexdigest(), init_checkpoint=None))

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
        study_training_run(cfg, spec.get('comparison_updates')).run()
    if spec['kind'] == 'probe':
        return
    evaluate_run(root, spec, selection=True)
    evaluate_run(root, spec)
    if spec.get('comparison_updates') is not None:
        evaluate_run(root, spec, checkpoint_label=f"best_at_{spec['comparison_updates']}")
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
    spec = dict(id=job_id, kind='training', algorithm='d3po', weight=0., detach=False,
                batch=settings['batches'][-1], horizon=settings['horizon'], threads=2,
                routes=settings['routes'], length=settings['length'], sampler='sampler',
                seed=0, alpha=None, updates=2, parent=None, resample=False)
    spec.update(changes)
    return spec


def choose(root, specs, *, prefer=None, relative_tolerance=0.):
    values = [(spec, read_json(root / 'selection' / spec['id'] / 'result.json')) for spec in specs]
    winner, result = min(values, key=lambda pair: (-pair[1]['feasible_fraction'], pair[1]['cost']))
    if prefer is not None:
        reference, metric = next(pair for pair in values if pair[0]['id'] == prefer)
        if (metric['feasible_fraction'] == result['feasible_fraction']
                and metric['cost'] - result['cost'] <= abs(metric['cost']) * relative_tolerance):
            return reference, metric
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
    rows, common_rows, models = [], [], []
    for job_id, job in state['jobs'].items():
        if job['status'] != 'completed':
            continue
        for result_path in sorted((root / 'evaluation' / job_id).glob('*/results.json')):
            result = read_json(result_path)
            for model in result['models']:
                models.append(model)
                spec = model['training_spec']
                metadata = {k: spec[k] for k in ('algorithm', 'weight', 'sampler', 'resample', 'routes', 'updates', 'seed')}
                rows.extend(dict(row, **metadata) for row in model['overall'])
                common_rows.extend(dict(row, **metadata) for row in model['common_reference10']['overall'])
    from scripts.evaluate_policy import write_csv, pareto_figures
    directory = root / 'summary'
    directory.mkdir(exist_ok=True)
    if rows:
        write_csv(directory / 'metrics.csv', rows)
        write_csv(directory / 'metrics_common_reference10.csv', common_rows)
    # Overlay only comparable tasks/specialist grids; never pool cities or seeds.
    groups = {}
    for model in models:
        spec = model['training_spec']
        key = f"l{spec['length']}_u{spec['updates']}_seed{spec['seed']}"
        groups.setdefault(f'{variant_name(spec)}_{key}', []).extend(model['per_city'])
        # Common reference cost and physical ATT/RTT allow overlays across S=5 and S=10.
        groups.setdefault(f'common_reference10_{key}', []).extend(model['common_reference10']['per_city'])
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
             'Обычная валидация: весь validation-сплит, смесь Holliday, каждые 20 обновлений и в конце.',
             '16 selection-городов × 11 alpha: только итоговая диагностика best и отбор настроек.',
             'Основной метод: D3PO div=0; другие seed проверяются последними.',
             'metrics_common_reference10.csv: общий RTT/(10D), best-of-K заново выбран из полного пула.',
             'Screening и основной бюджет различаются; один training seed не устанавливает превосходство.',
             'HV reference [3,3] зафиксирован до test; coverage сохранено в metrics JSON.',
             'best_of_K — выбранные ответы по alpha; greedy+K не смешивается с K.', '']
    (directory / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    resource_figures(root)


def campaign_jobs(settings, batch, threads, protocol=None):
    """Two-stage screening; every training job starts from the common seed, not a parent actor."""
    protocol = protocol or dict(sampler='sampler', resample=False)
    def make(job_id, updates, **changes):
        return training_spec(job_id, settings, batch=batch, threads=threads, updates=updates,
                             **dict(protocol, **changes))
    tournament = [make(f'protocol_{sampler}_resample{int(resample)}', settings['screen_updates'],
                       sampler=sampler, resample=resample)
                  for sampler in ('sampler', 'simplex') for resample in (False, True)]
    main = [make(f'main_{algorithm}', settings['main_updates'], algorithm=algorithm,
                 comparison_updates=settings['route_updates']) for algorithm in ('ppo', 'd3po')]
    routes = [make(f"routes{max(1, settings['routes']//2)}_{algorithm}", settings['route_updates'], algorithm=algorithm,
                   routes=max(1, settings['routes']//2), parent=f'main_{algorithm}')
              for algorithm in ('ppo', 'd3po')]
    return dict(protocol=tournament, main=main, routes=routes,
        baseline=[make('original_ppo', settings['screen_updates'], algorithm='ppo',
                       sampler='sampler', resample=False)],
        diversity=[make(f'lambda_{weight}', settings['screen_updates'], weight=weight)
                   for weight in (.01, .1)],
        confirmation=[make(f'confirm_{algorithm}_seed1', settings['main_updates'], algorithm=algorithm,
                           seed=1, parent=f'main_{algorithm}') for algorithm in ('ppo', 'd3po')])


def probe_timing(root, spec, multiplier=1.):
    history = csv_rows(root / 'runs' / spec['id'] / 'history.csv')
    elapsed = [float(r['time/time_elapsed']) for r in history if r.get('time/time_elapsed')]
    validation = csv_rows(root / 'runs' / spec['id'] / 'validation.csv')
    # Probe validation is before/final only; update-to-update differences exclude it.
    differences = [b-a for a, b in zip(elapsed, elapsed[1:])]
    per_update = max(differences[-2:]) if differences else elapsed[-1]
    total_validation = sum(float(r['seconds']) for r in validation)
    resources = read_json(root / 'runs' / spec['id'] / 'worker-resources.json')
    startup = max(0., resources['wall_seconds'] - per_update * spec['updates'] - total_validation)
    return dict(update=per_update * multiplier,
                validation=max(float(r['seconds']) for r in validation) * multiplier,
                startup=startup * multiplier)


def pipeline(controller, calibrate_only=False):
    root, state = controller.root, controller.state
    settings = state['settings']
    controller.group([dict(id='prepare_data', kind='data', threads=1)], 'data')
    verify_files(root, read_json(root / 'protocols/data-hashes.json'))
    cores = psutil.cpu_count(logical=False) or 1
    batch = settings['batches'][-1]
    probes = [training_spec(f'probe_{algorithm}', settings, kind='probe', algorithm=algorithm,
                           updates=settings['probe_updates'], batch=batch,
                           threads=max(1, min(4, cores//2))) for algorithm in ('ppo', 'd3po')]
    state['workers'] = 1
    controller.group(probes, 'calibration_single')
    single_seconds = sum(state['jobs'][s['id']]['seconds'] for s in probes)
    pair = [dict(spec, id=spec['id'] + '_pair') for spec in probes]
    peak = sum(state['jobs'][s['id']]['peak_memory_bytes'] for s in probes)
    if (settings['workers'] == 2 and cores >= 2 and peak * 1.25 < controller.limit
            and psutil.virtual_memory().available > peak * 1.25 + controller.reserved):
        state['workers'] = 2
        controller.group(pair, 'calibration_pair')
        span = (max(datetime.fromisoformat(state['jobs'][s['id']]['finished']) for s in pair) -
                min(datetime.fromisoformat(state['jobs'][s['id']]['started']) for s in pair)).total_seconds()
        workers = 2 if span < single_seconds else 1
        reason = 'Measured pair faster' if workers == 2 else 'Single worker has better aggregate throughput'
    else:
        workers, reason = 1, 'Pair not safe under memory/core limits'
    state['workers'], state['concurrency_reason'] = workers, reason
    timings = {spec['algorithm']: probe_timing(root, spec) for spec in probes}
    if workers == 2:
        for spec in pair:
            paired = probe_timing(root, spec)
            for key in timings[spec['algorithm']]:
                timings[spec['algorithm']][key] = max(timings[spec['algorithm']][key], paired[key])
    profile = probes[-1]
    evaluation_probe = dict(profile, id='probe_inference', kind='evaluate')
    probe_dir = root / 'runs/probe_inference'
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'training_result.json'):
        shutil.copy2(root / 'runs' / profile['id'] / name, probe_dir / name)
    (probe_dir / 'checkpoints').mkdir(exist_ok=True)
    shutil.copy2(root / 'runs' / profile['id'] / 'checkpoints/best.pt', probe_dir / 'checkpoints/best.pt')
    controller.group([evaluation_probe], 'calibration_inference')
    chunks = []
    for path in (root / 'evaluation/probe_inference/best/candidates').glob('*.json.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            chunks.append(json.load(stream))
    seconds_per_network = {mode: max(c['seconds']/len(c['rows']) for c in chunks if c['mode'] == mode)
                           for mode in ('greedy', 'sample')}
    inference_startup = max(0., state['jobs']['probe_inference']['seconds'] - sum(c['seconds'] for c in chunks))
    inference_multiplier = 1.
    if workers == 2:
        # Validation forward passes provide a measured bound on pair contention.
        inference_multiplier = max(1., max(
            probe_timing(root, p)['validation']/probe_timing(root, s)['validation']
            for s, p in zip(probes, pair)))
    eval_seconds = ((seconds_per_network['greedy'] * (settings['test_graphs'] + settings['selection_graphs'])
                    + seconds_per_network['sample'] * max(settings['ks']) * settings['k_graphs'])
                    * settings['preference_count'] * inference_multiplier
                    + inference_startup + (2 if settings['smoke'] else 60))
    budget = dict(batch=batch, horizon=settings['horizon'], threads=profile['threads'], workers=workers,
        timings=timings, inference_seconds_per_network=seconds_per_network,
        evaluation_seconds_per_universal_run=eval_seconds, inference_pair_multiplier=inference_multiplier,
        concurrency_reason=reason, memory_limit_bytes=controller.limit, memory_peak_bytes=controller.peak,
        validation_graphs=len(read_json(root / 'protocols/splits.json')['validation']),
        screen_updates=settings['screen_updates'], main_updates=settings['main_updates'],
        route_updates=settings['route_updates'], late_training_margin=settings['late_training_margin'],
        selection_graphs=settings['selection_graphs'], validation_period=settings['val_period'])
    state['calibration'] = budget
    state['budget'] = settings['main_updates'] * batch * settings['horizon']
    def estimate(specs):
        loads = [0.] * workers
        for spec in specs:
            if state['jobs'].get(spec['id'], {}).get('status') == 'completed':
                continue
            timing = timings[spec['algorithm']]
            duration = (timing['startup'] + timing['update'] * spec['updates']
                        * settings['late_training_margin'] * (1.5 if spec['weight'] else 1.)
                        + timing['validation'] * (math.ceil(spec['updates']/settings['val_period']) + 1)
                        + eval_seconds * (2 if spec.get('comparison_updates') is not None else 1))
            loads[loads.index(min(loads))] += duration
        return max(loads) * state.get('forecast_factor', 1.)
    def plan_record(jobs):
        return [dict(stage=stage, required=stage in ('protocol', 'main', 'routes', 'baseline'),
                     jobs=specs, forecast_hours=estimate(specs)/3600)
                for stage, specs in jobs.items()]
    jobs = campaign_jobs(settings, batch, profile['threads'])
    required = sum(estimate(jobs[stage]) for stage in ('protocol', 'main', 'routes', 'baseline')) * 1.3
    budget.update(plan=plan_record(jobs), required_hours_with_margin=required/3600,
                  remaining_hours=controller.remaining()/3600,
                  decision='fits' if required < controller.remaining() else 'minimum_does_not_fit')
    save_json(root / 'calibration/budget.json', budget)
    save_json(root / 'protocols/planned-jobs.json', budget['plan'])
    controller.save()
    if calibrate_only:
        state['status'] = 'calibrated'
        controller.save()
        return
    if required >= controller.remaining():
        raise RuntimeError('Minimum study (75-update tournament, 200-update main, matched 100-update routes, '
                           'full validation and per-run K100) does not fit; see calibration/budget.json')
    controller.group(jobs['protocol'], 'protocol_tournament', eta=estimate(jobs['protocol']))
    winner, metric = choose(root, jobs['protocol'], prefer=jobs['protocol'][0]['id'],
                            relative_tolerance=settings['selection_tolerance'])
    settings_protocol = {k: winner[k] for k in ('sampler', 'resample')}
    state['choices']['protocol'] = dict(parent=winner['id'], **settings_protocol, selection=metric,
        relative_cost_tolerance=settings['selection_tolerance'])
    jobs = campaign_jobs(settings, batch, profile['threads'], settings_protocol)
    if settings_protocol == dict(sampler='sampler', resample=False):
        jobs['baseline'] = []
    budget['plan'] = plan_record(jobs)
    save_json(root / 'protocols/planned-jobs.json', budget['plan'])
    # Reserve all remaining mandatory work before admitting a long training group.
    core = [('main', jobs['main']), ('routes', jobs['routes']), ('baseline', jobs['baseline'])]
    for index, (stage, specs) in enumerate(core):
        remaining_core = sum(estimate(group) for _, group in core[index:]) * 1.3
        budget.update(remaining_core_hours_with_margin=remaining_core/3600,
                      remaining_hours=controller.remaining()/3600)
        if remaining_core > controller.remaining():
            budget['decision'] = 'remaining_core_does_not_fit'
        save_json(root / 'calibration/budget.json', budget)
        if remaining_core > controller.remaining():
            raise RuntimeError('Updated forecast: remaining required study does not fit; artifacts preserved')
        controller.group(specs, stage, eta=estimate(specs))
        if stage == 'main':
            initial = [read_json(root / 'runs' / spec['id'] / 'initialization.json')['actor_sha256']
                       for spec in jobs['main']]
            if len(set(initial)) != 1:
                raise RuntimeError('Matched PPO/D3PO actors did not start from identical weights')
    diversity = jobs['diversity']
    if estimate(diversity) * 1.3 <= controller.remaining():
        controller.group(diversity, 'diversity', optional=True, eta=estimate(diversity))
    else:
        for spec in diversity:
            controller.group([spec], spec['id'], optional=True, eta=estimate([spec]))
    candidates = [winner] + [s for s in diversity if state['jobs'][s['id']]['status'] == 'completed']
    div_winner, div_metric = choose(root, candidates, prefer=winner['id'],
                                   relative_tolerance=settings['selection_tolerance'])
    state['choices']['diversity'] = dict(parent=div_winner['id'], weight=div_winner['weight'],
                                        selection=div_metric, evidence=f"{settings['screen_updates']}-update screening")
    if div_winner['weight']:
        promoted = dict(jobs['main'][1], id='main_d3po_diversity', weight=div_winner['weight'],
                        parent=div_winner['id'])
        promoted.pop('comparison_updates')
        controller.group([promoted], 'diversity_confirmation', optional=True, eta=estimate([promoted]))
    if settings['confirm']:
        controller.group(jobs['confirmation'], 'confirmation_last', optional=True, eta=estimate(jobs['confirmation']))
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
  python -m pip install --no-deps -e source
  bash source/scripts/launch.sh run_study.py --out-dir "$PWD" --resume --log "$PWD/controller-console.log"

Начнётся генерация данных и измерение ресурсов. Затем автоматически продолжится
основная серия, если минимальная серия с оценкой K100 укладывается в бюджет.
Иначе запуск остановится с объяснением в status.txt и calibration/budget.json.
35 часов отсчитываются с первого запуска, а не с упаковки/копирования.
Предел 50 GB относится к сумме процессов; одновременно не более двух workers.
Монитор памяти опрашивается раз в 5 секунд; это не ограничение cgroup ОС.
Новая серия: D3PO div=0, четыре протокола по 75 обновлений, основная пара по 200,
5/10 маршрутов на общем бюджете 100; ненулевые lambda далее, другие seed последними.
Валидация: весь validation-сплит, каждые 20 обновлений и в конце, best по смеси Holliday.
Отдельные 16 городов x 11 alpha проверяются только в конце по best checkpoint.
Длинные прогоны начинаются с нуля. Минимальное ядро не сокращается автоматически.

Смотреть:
  watch -n 5 cat status.txt
  tail -f controller.log
  calibration/budget.json       — оценка времени и выбранная параллельность
  protocols/planned-jobs.json   — обязательные и условные задания с оценкой времени
  resources.csv                — память, число workers/процессов, время
  figures/resources.png        — график ресурсов после завершения
  summary/report.md            — список выполненных и пропущенных экспериментов
  summary/metrics.csv          — сводная таблица метрик
  summary/metrics_common_reference10.csv — сравнение по общему RTT/(10D)
  selection/<run>/by_alpha.csv — итоговая проверка 16 x 11 и отбор настроек
  evaluation/<run>/best/        — Pareto, метрики, все кандидаты и выбранные ответы
  runs/<run>/checkpoints/       — best.pt и last.pt
  evaluation/<run>/best_at_100/ — основной прогон при бюджете 100 для сравнения маршрутов
  runs/<run>/validation.csv     — время и cost полной валидации
  runs/<run>/preferences.json  — распределение alpha по переходам и завершённым эпизодам
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
    parser.add_argument('--extended', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--confirm', action=argparse.BooleanOptionalAction, default=True,
                        help='Try fresh-seed confirmation last, after the planned ablations')
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
                smoke=args.smoke, confirm=args.confirm,
                pool_graphs=32 if args.smoke else 32768, nodes=9 if args.smoke else 20,
                routes=2 if args.smoke else 10, length=9 if args.smoke else 12,
                validation_graphs=None, selection_graphs=2 if args.smoke else 16,
                test_graphs=2 if args.smoke else 64, k_graphs=1 if args.smoke else 16,
                preference_count=3 if args.smoke else 11, ks=[1, 3] if args.smoke else [1, 10, 100],
                batches=[2] if args.smoke else [128], horizon=4 if args.smoke else 120,
                screen_updates=2 if args.smoke else 75, main_updates=2 if args.smoke else 200,
                route_updates=2 if args.smoke else 100, probe_updates=2 if args.smoke else 4,
                val_period=1 if args.smoke else 20, late_training_margin=1. if args.smoke else 1.25,
                selection_tolerance=.005,
                poll_seconds=1 if args.smoke else 5)
            state = dict(schema=SCHEMA, created=now(), settings=settings, jobs={}, choices={},
                         source_hashes=snapshot(root), status='preparing', workers=1, peak_memory_bytes=0)
            save_json(manifest, state)
        if args.replot:
            verify_files(root, read_json(root / 'protocols/data-hashes.json'))
            sys.path.insert(0, str(root / 'source'))
            for job in state['jobs'].values():
                if job['status'] == 'completed' and job['spec']['kind'] == 'training':
                    for protocol in (root / 'evaluation' / job['spec']['id']).glob('*/protocol.json'):
                        rebuild_run(root, job['spec'], protocol.parent.name)
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

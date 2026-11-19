"""Prepare and coordinate the self-contained seven-training v5 study."""
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
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil


REPO = Path(__file__).resolve().parents[1]
SCHEMA = 5
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
    cfg.algorithm.gamma, cfg.algorithm.n_epochs = settings.get('gamma', .95), 1
    if 'gae_lambda' in settings:
        cfg.algorithm.gae_lambda = settings['gae_lambda']
    cfg.algorithm.minibatch_size = spec['minibatch']
    cfg.algorithm.diversity_weight = spec['weight']
    cfg.algorithm.diversity_detach_neighbor = spec['detach']
    cfg.algorithm.diversity_reduction = 'state_mean'
    cfg.algorithm.diversity_alpha = 1.
    cfg.algorithm.preference_noise_sigma = .15
    cfg.algorithm.gradient_diagnostics_interval = 0
    cfg.evaluation.batch_size = min(32, settings['batches'][-1])
    cfg.evaluation.graph_count = settings['validation_graphs']
    cfg.evaluation.pareto_graph_count = settings['selection_graphs']
    cfg.evaluation.test_graph_count = settings['test_graphs']
    cfg.evaluation.preference_count = settings['preference_count']
    cfg.evaluation.eu_preference_count = settings['preference_count']
    cfg.evaluation.pareto_enabled = cfg.evaluation.final_test = False
    if spec.get('alpha') is not None:
        # The simplex sampler deliberately ignores fixed-weight parameters.
        # A specialist must use the fixed-weight sampler even if its parent used simplex.
        cfg.train_loop.preference_distribution = 'sampler'
        cfg.preferences.variable_weights = False
        cfg.preferences.demand_time_weight = cfg.cost.kwargs.demand_time_weight = spec['alpha']
        cfg.preferences.route_time_weight = cfg.cost.kwargs.route_time_weight = 1 - spec['alpha']
    OmegaConf.resolve(cfg)
    return cfg


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
    if 'balanced' in checkpoint_label:
        label=f"{spec['id']}@{checkpoint_label}"
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
    record = dict(label=label, checkpoint=checkpoint_label, checkpoint_updates=protocol.get('checkpoint_updates'),
                  training_spec=dict(spec, updates=protocol['updates']),
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
    mode_front_figures(out, record['per_city'], protocol['ks'])
    return record


def mode_front_figures(out, rows, ks):
    """Overlay the three generation modes within each actual city (no mean-point front)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scripts.evaluate_policy import pareto_diagnostics
    cities = sorted({r['graph'] for r in rows})
    annotated, _ = pareto_diagnostics(rows, list(range(max(cities) + 1)))
    styles = [('greedy', 'Greedy', 'C0'), ('best_of_1', 'Sample / LC1', 'C1'),
              (f'best_of_{max(ks)}', f'LC{max(ks)}', 'C2')]
    with PdfPages(out / 'pareto_modes_all_cities.pdf') as pdf:
        for start in range(0, len(cities), 4):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
            for ax, city in zip(axes.flat, cities[start:start+4]):
                for mode, label, color in styles:
                    points = [r for r in annotated if r['graph'] == city and r['mode'] == mode]
                    feasible = [r for r in points if r['feasible']]
                    front = sorted((r for r in feasible if r['nondominated_within_model_mode']),
                                   key=lambda r: r['RTT_minutes'])
                    ax.scatter([r['RTT_minutes'] for r in feasible], [r['ATT_minutes'] for r in feasible],
                               label=label, color=color, s=18, alpha=.6)
                    ax.plot([r['RTT_minutes'] for r in front], [r['ATT_minutes'] for r in front], color=color)
                ax.set(title=f'City {city}', xlabel='RTT (minutes)', ylabel='ATT (minutes)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(cities[start:start+4]):]:
                ax.set_visible(False)
            pdf.savefig(fig)
            if start == 0:
                fig.savefig(out / 'pareto_modes.png', dpi=140)
            plt.close(fig)


def evaluation_tasks(spec):
    """The S5/S10 pair, Mandl and Mumford0 share one trained actor."""
    tasks = [dict(suite=f'synthetic_s{routes}', routes=routes, length=12, minimum=2)
             for routes in (10, 5)]
    for city in ('mandl', 'mumford0'):
        from connectpt.routes_generator.run.instances import instance_spec
        cfg = instance_spec(city)
        tasks.append(dict(suite=city, routes=int(cfg.n_routes), length=int(cfg.max_route_len),
                          minimum=int(cfg.min_route_len), benchmark=city))
    return [dict(spec, **task, actor_id=spec.get('actor_id', spec['id']),
                 training_routes=spec.get('training_routes', spec['routes'])) for task in tasks]


def skip_evaluation(root, spec, checkpoint_label):
    """Cooperative cancellation for new workers, checked between candidate chunks."""
    actor = spec.get('actor_id', spec['id'])
    markers = (root / 'SKIP_EVALUATION', root / 'runs' / actor / 'SKIP_EVALUATION')
    if not any(p.exists() for p in markers):
        return False
    record = dict(phase='evaluation_skipped', checkpoint=checkpoint_label,
                  suite=spec.get('suite'), updated_at=now(), reason='SKIP_EVALUATION marker')
    save_json(root / 'evaluation' / spec['id'] / checkpoint_label / 'skipped.json', record)
    save_json(root / 'runs' / spec['id'] / 'progress.json', record)
    return True


def checkpoint_updates(directory, label, budget):
    """Actual selected weight age, distinct from the campaign training budget."""
    if label.startswith('at_'):
        return int(label.removeprefix('at_'))
    selection = ('checkpoint_selection_at_' + label.removeprefix('best_balanced_at_') + '.json'
                 if label.startswith('best_balanced_at_') else
                 'checkpoint_selection.json' if label == 'best_balanced' else
                 'comparison_checkpoint.json' if label.startswith('best_at_') else None)
    if selection is not None:
        return read_json(directory / selection)['best_step']
    if label == 'best':
        rows = csv_rows(directory / 'validation.csv')
        return int(min(rows, key=lambda row: float(row['cost']))['step']) if rows else None
    return budget if label == 'last' else None


def evaluate_run(root, spec, *, checkpoint_label='best'):
    if skip_evaluation(root, spec, checkpoint_label):
        return dict(skipped=True)
    import torch
    from omegaconf import OmegaConf
    from connectpt.routes_generator.run.training import TrainingRun
    from scripts.evaluate_policy import prediction_rows
    from connectpt.routes_generator.policy.predict import predict
    settings = read_json(root / 'manifest.json')['settings']
    actor_id = spec.get('actor_id', spec['id'])
    directory = root / 'runs' / actor_id
    suite = spec['suite']
    out = root / 'evaluation' / spec['id'] / checkpoint_label
    cfg = OmegaConf.load(directory / 'config.yaml')
    checkpoint = directory / 'checkpoints' / f'{checkpoint_label}.pt'
    cfg.paths.init_checkpoint_path = f"{actor_id}/checkpoints/{checkpoint_label}.pt"
    actor = TrainingRun(cfg)
    actor.setup()
    torch.set_num_threads(spec['threads'])
    actor.policy.eval()
    graphs, source_ids, indices = evaluation_inputs(root, spec, actor)
    from dataclasses import replace
    actor.env_params = replace(actor.env_params, n_routes=spec['routes'],
                               min_route_len=spec.get('minimum', 2), max_route_len=spec['length'])
    alphas = [i / (settings['preference_count']-1) for i in range(settings['preference_count'])]
    ks = settings['ks']
    k_count = len(graphs)
    inputs = dict(spec=spec, config=file_hash(directory / 'config.yaml'), checkpoint=file_hash(checkpoint),
                  library=read_json(root / 'manifest.json')['source_hashes'],
                  data=read_json(root / 'protocols/data-hashes.json'), indices=indices, alphas=alphas,
                  ks=ks, k_count=k_count, seed=9001)
    fingerprint = identity(inputs)
    protocol = dict(identity=inputs, fingerprint=fingerprint, graph_count=len(graphs),
                    split='benchmark' if spec.get('benchmark') else 'independent synthetic test', suite=suite,
                    source_ids=source_ids,
                    checkpoint_label=checkpoint_label,
                    checkpoint_updates=checkpoint_updates(directory, checkpoint_label, spec['updates']),
                    updates=int(checkpoint_label.split('_')[-1]) if checkpoint_label.startswith(('at_', 'best_at_')) else spec['updates'],
                    k_graph_count=k_count, alphas=alphas, ks=ks, seed=9001)
    save_json(out / 'protocol.json', protocol)
    before = {k: v.clone() for k, v in actor.policy.state_dict().items()}
    all_rows = []
    for ai, alpha in enumerate(alphas):
        weights = dict(demand_time_weight=float(alpha), route_time_weight=float(1-alpha), median_connectivity_weight=0.)
        cost = actor.cost.with_params(**weights)
        for mode in ('greedy', 'sample'):
            for sample in range(1 if mode == 'greedy' else max(ks)):
                count = len(graphs) if mode == 'greedy' else k_count
                for start in range(0, count, cfg.evaluation.batch_size):
                    if skip_evaluation(root, spec, checkpoint_label):
                        return dict(skipped=True)
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
                                    source_id=source_ids[start+offset],
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
                    progress = dict(phase='evaluation', mode=mode, suite=suite, checkpoint=checkpoint_label,
                        alpha=alpha, sample=sample+1, samples=1 if mode == 'greedy' else max(ks),
                        graph=end, graphs=count, updated_at=now())
                    save_json(root / 'runs' / spec['id'] / 'progress.json', progress)
                    save_json(root / 'runs' / actor_id / 'progress.json', progress)
    for key, value in actor.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)
    if skip_evaluation(root, spec, checkpoint_label):
        return dict(skipped=True)
    rebuild_run(root, spec, checkpoint_label)
    (out / 'skipped.json').unlink(missing_ok=True)
    return protocol


def study_training_run(cfg, comparison_updates=None):
    """Scope isolated validation RNG to this study; do not alter existing Run behavior."""
    from dataclasses import replace
    import torch
    from connectpt.routes_generator.run.training import TrainingRun
    from connectpt.routes_generator.training.validate import ConstructionValidation
    from connectpt.routes_generator.training.data import ConstructionDataset
    from connectpt.routes_generator.policy.checkpoints import save_model_weights
    from connectpt.routes_generator.training.checkpoints import TrainingRecovery, preserve_rng
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
                data.checkpoint_extra = histograms
                if getattr(owner, 'training_diagnostics', None) is not None:
                    owner.training_diagnostics.attach(data)
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
            diagnostics = getattr(owner, 'training_diagnostics', None)
            if diagnostics is not None:
                balanced=diagnostics.validation(policy, logger_list, step, data=self.data,
                                                cost=owner.cost, env_factory=type(owner.env))
                if balanced is not None:
                    improved=balanced<getattr(self,'balanced_best_cost',math.inf)
                    if improved:
                        self.balanced_best_cost=balanced
                        self.balanced_best_step=step
                        save_model_weights(policy,owner.best_checkpoint.with_name('best_balanced.pt'),
                                           OmegaConf.to_container(cfg,resolve=True))
                        save_json(owner.run_dir/'checkpoint_selection.json',dict(
                            criterion='mean Sample final cost, equally weighted validation cities and uniform alpha grid',
                            best_step=step,best_cost=balanced,alphas=list(diagnostics.params.selection_alphas),
                            validation_indices=self.data.validation_indices[:diagnostics.params.validation_graphs],
                            samples=diagnostics.params.validation_samples,seed=diagnostics.params.seed,
                            baseline_checkpoint='best.pt retains v4 full sample-validation selection',
                            test_and_benchmarks_used=False))
                    for logger in logger_list:
                        logger.scalar('diag/validation/best_balanced_cost',self.balanced_best_cost,step)
                        logger.scalar('diag/validation/best_balanced_step',self.balanced_best_step,step)
                    if comparison_updates is not None and step<=comparison_updates and balanced<getattr(self,'balanced_prefix_best',math.inf):
                        self.balanced_prefix_best=balanced
                        save_model_weights(policy,owner.best_checkpoint.with_name(f'best_balanced_at_{comparison_updates}.pt'),
                                           OmegaConf.to_container(cfg,resolve=True))
                        save_json(owner.run_dir/f'checkpoint_selection_at_{comparison_updates}.json',
                                  dict(best_step=step,best_cost=balanced,criterion='uniform validation alpha grid'))
            if comparison_updates is not None:
                if step <= comparison_updates and self.validated_cost < getattr(self, 'prefix_best', math.inf):
                    self.prefix_best = self.validated_cost
                    save_model_weights(policy, owner.best_checkpoint.with_name(f'best_at_{comparison_updates}.pt'),
                                       OmegaConf.to_container(cfg, resolve=True))
                    save_json(owner.run_dir / 'comparison_checkpoint.json', dict(
                        budget_updates=comparison_updates, best_step=step, best_cost=self.validated_cost))
                if step > 0 and step % comparison_updates == 0:
                    label = f'at_{step}'
                    save_model_weights(policy, owner.best_checkpoint.with_name(label + '.pt'),
                                       OmegaConf.to_container(cfg, resolve=True))
                    root = owner.run_dir.parent.parent
                    spec = read_json(root / 'protocols/jobs' / f'{cfg.run.name}.json')
                    # These held-out observations never select the best checkpoint.
                    with preserve_rng():
                        for task in evaluation_tasks(spec):
                            task['id'] = spec['id'] + '__' + task['suite']
                            evaluate_run(root, task, checkpoint_label=label)
                    save_json(owner.run_dir / 'progress.json', dict(phase='training', updated_at=now()))

    class StudyTrainingRun(TrainingRun):
        def setup(self):
            nonlocal owner
            super().setup()
            owner = self
            self.task = replace(self.task, validation_factory=StudyValidation)
            self.dataset = StudyDataset(**vars(self.dataset))
            root = self.run_dir.parent.parent
            manifest = root / 'manifest.json'
            profile = read_json(manifest)['settings'].get('scalar_diagnostics') if manifest.exists() else None
            if profile:
                from connectpt.routes_generator.training.diagnostics import ScalarDiagnostics, ScalarDiagnosticParams
                self.training_diagnostics = ScalarDiagnostics(ScalarDiagnosticParams(**profile),
                    gamma=self.algorithm.params.gamma, objective_names=self.cost.objectives, env_params=self.env_params)
                self.metadata['scalar_diagnostics'] = profile
            definition = OmegaConf.to_container(self.cfg, resolve=True)
            definition.pop('paths', None)
            if profile:
                definition['scalar_diagnostics'] = profile
            self.training_recovery = TrainingRecovery(
                self.best_checkpoint.with_name('training_state.pt'), identity(definition),
                milestone_period=comparison_updates or self.loop_params.val_period)
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
    """Only the explicit --resume controller dispatches training workers."""
    os.chdir(root / 'source')
    spec = read_json(root / 'protocols/jobs' / f'{job_id}.json')
    if spec['kind'] == 'evaluation':
        labels=read_json(root/'manifest.json')['settings'].get('final_checkpoint_labels',['best'])
        for label in labels:
            evaluate_run(root, spec, checkpoint_label=label)
        return
    cfg = worker_config(root, spec)
    directory = root / 'runs' / job_id
    result = directory / 'training_result.json'
    if result.exists():
        if read_json(result)['iterations'] != spec['updates']:
            raise ValueError('Incomplete training has no exact optimizer/RNG resume; artifacts are preserved')
    else:
        if ((directory / 'history.csv').exists()
                and not (directory / 'checkpoints/training_state.pt').exists()):
            raise ValueError('No full training checkpoint exists; actor-only weights cannot resume training exactly')
        save_json(directory / 'progress.json', dict(phase='training', updated_at=now()))
        study_training_run(cfg, spec['comparison_updates']).run()
    save_json(directory / 'progress.json', dict(phase='completed', updated_at=now()))


class Coordinator:
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.started = time.monotonic()
        self.active = {}
        self.peak = float(state.get('peak_memory_bytes', 0))
        self.limit = state['settings']['memory_gb'] * GB
        self.reserved = .5 * GB if state['settings']['smoke'] else min(10 * GB, psutil.virtual_memory().total * .15)
        self.log = logging.getLogger('study')

    def save(self):
        save_json(self.root / 'manifest.json', self.state)

    def sample(self):
        own = psutil.Process()
        memory = process_memory([own] + own.children(recursive=True))
        self.peak = max(self.peak, memory['memory_bytes'])
        self.state['peak_memory_bytes'] = self.peak
        vm = psutil.virtual_memory()
        try:
            swap_used = psutil.swap_memory().used
        except (OSError, RuntimeError, psutil.Error):
            # Restricted Windows hosts may disable performance counters.
            swap_used = None
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
            system_available_bytes=vm.available, system_swap_bytes=swap_used, cpu_percent=psutil.cpu_percent(),
            elapsed_hours=(time.time()-datetime.fromisoformat(self.state['started']).timestamp())/3600,
            time_budget_enforced=False, jobs=rows,
            completed=sum(j['status'] == 'completed' for j in self.state['jobs'].values()),
            skipped=sum(j['status'] == 'skipped' for j in self.state['jobs'].values()),
            error=self.state.get('error'), budget=self.state.get('budget'),
            concurrency_reason=self.state.get('concurrency_reason'),
            training_budget_updates=self.state['settings']['main_updates'])
        save_json(self.root / 'status.json', status)
        lines = [f"{status['timestamp']} | {status['status']} | {status['stage']}",
            f"Workers: {status['workers']}/{status['worker_limit']} | OS processes: {memory['process_count']}",
            f"Study RAM ({memory['memory_measure']}): {memory['memory_bytes']/GB:.2f} / {self.limit/GB:.2f} GB | peak {self.peak/GB:.2f} GB",
            f"System available: {vm.available/GB:.2f} GB | swap bytes: {swap_used} | CPU {status['cpu_percent']:.0f}%",
            f"Elapsed: {status['elapsed_hours']:.2f} h | no time limit",
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
            'system_swap_bytes', 'cpu_percent', 'elapsed_hours')})
        return memory['memory_bytes'] <= self.limit and vm.available >= self.reserved

    def group(self, specs, stage):
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
        self.log.info('Stage %s: %d jobs, max %d worker(s)', stage, len(pending), self.state['workers'])
        try:
            while pending or self.active:
                safe = self.sample()
                if not safe and self.active:
                    raise RuntimeError('Memory guard: study limit exceeded or insufficient system headroom')
                while pending and len(self.active) < self.state['workers'] and safe:
                    spec = pending.pop(0)
                    directory = self.root / 'runs' / spec['id']
                    directory.mkdir(parents=True, exist_ok=True)
                    log = directory / 'console.log'
                    args = ['--wait', 'run_study.py', '--worker', '--out-dir', str(self.root),
                            '--job', spec['id'], '--log', str(log)]
                    launch_root = Path(os.environ.get('CONNECTPT_STUDY_LAUNCH_ROOT', str(REPO)))
                    command = (['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                str(launch_root / 'scripts/launch.ps1')] if os.name == 'nt' else
                               ['bash', str(launch_root / 'scripts/launch.sh')]) + args
                    environment = os.environ.copy()
                    environment.update(PYTHONPATH=str(self.root / 'source'), PYTHONUNBUFFERED='1',
                        MPLCONFIGDIR=str(self.root / 'environment/matplotlib'), PYTHONDONTWRITEBYTECODE='1',
                        OMP_NUM_THREADS=str(spec.get('threads', 1)), MKL_NUM_THREADS=str(spec.get('threads', 1)))
                    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
                    console = (directory / 'launcher.log').open('a', encoding='utf-8')
                    process = subprocess.Popen(command, cwd=launch_root, env=environment,
                                               stdout=console, stderr=subprocess.STDOUT)
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
                    progress_path = self.root / 'runs' / job_id / 'progress.json'
                    progress = read_json(progress_path) if progress_path.exists() else {}
                    skipped = (exit_code == 0 and running['spec']['kind'] == 'evaluation'
                               and progress.get('phase') == 'evaluation_skipped')
                    entry.update(status='skipped' if skipped else 'completed' if exit_code == 0 else 'failed', exit_code=exit_code,
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
        return True


def resource_figures(root):
    state = read_json(root / 'manifest.json')
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
    axes[1].set(xlabel='Elapsed hours', ylabel='Workers', yticks=list(range(state['settings']['workers'] + 1)))
    (root / 'figures').mkdir(exist_ok=True)
    fig.savefig(root / 'figures/resources.png', dpi=140)
    plt.close(fig)


def summary(root):
    state = read_json(root / 'manifest.json')
    import torch
    torch.set_num_threads(state['settings']['threads'])
    rows, common_rows, models = [], [], []
    for job_id, job in state['jobs'].items():
        for result_path in sorted((root / 'evaluation' / job_id).glob('*/results.json')):
            result = read_json(result_path)
            for model in result['models']:
                models.append(model)
                spec = model['training_spec']
                metadata = {k: spec[k] for k in ('algorithm', 'weight', 'sampler', 'resample', 'routes', 'updates', 'seed',
                                                'batch', 'horizon', 'alpha')}
                metadata.update(suite=spec.get('suite'), training_routes=spec.get('training_routes', spec['routes']), kind=spec['kind'], training_transitions=spec['updates']*spec['batch']*spec['horizon'],checkpoint=model.get('checkpoint',result_path.parent.name),checkpoint_updates=model.get('checkpoint_updates'))
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
        key = f"{spec.get('suite', 'synthetic')}_l{spec['length']}_u{spec['updates']}_seed{spec['seed']}"
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
            f"Память: не более {state['settings']['memory_gb']} GB; ограничения времени отсутствуют.",
            '', 'Текущее состояние: ../status.txt; история ресурсов: ../resources.csv.', '',
            '| Задание | Статус | Время, с | Причина пропуска |', '|---|---|---:|---|']
    for job_id, job in state['jobs'].items():
        text.append(f"| {job_id} | {job['status']} | {job.get('seconds', 0):.1f} | {job.get('reason', '')} |")
    if state.get('error'):
        text += ['', 'Причина остановки: ' + state['error']]
    text += ['', 'Семь обучений с нуля, без турнира и предварительного подбора бюджета.',
             'Checkpoint best выбирается по среднему sample cost полного synthetic validation.',
             'Бенчмарки не участвуют в отборе весов. Все режимы используют одни города.',
             'Greedy, LC1 и LC100 разделены; greedy не входит в stochastic-пул.',
             'Фронты строятся отдельно по городам; затем усредняются метрики фронтов.',
             'Один training seed: устойчивость между seed не проверена.', '']
    (directory / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    resource_figures(root)
    study_figures(root, state, models)


def study_figures(root, state, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.evaluate_policy import write_csv
    directory = root / 'summary'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    groups = [('ppo_', 'PPO'), ('d3po_l0_', 'D3PO without diversity'),
              ('d3po_l0p1_', 'D3PO with diversity'), ('routes5_', 'Five-route training')]
    for ax, (prefix, title) in zip(axes.flat, groups):
        for job_id, job in state['jobs'].items():
            matches = job_id.startswith(prefix)
            if not matches or job['spec']['kind'] != 'training':
                continue
            rows = csv_rows(root / 'runs' / job_id / 'validation.csv')
            points = {int(r['step']): float(r['cost']) for r in rows}
            if points:
                ax.plot(list(points), list(points.values()), marker='.', label=job_id)
        ax.set(title=title, xlabel='Updates', ylabel='Validation cost')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.savefig(directory / 'learning_curves.png', dpi=140)
    plt.close(fig)
    rows = [dict(r) for model in models if model['training_spec'].get('suite') == 'synthetic_s10'
            for r in model['by_alpha'] if r['mode'] in ('greedy', 'best_of_1', f"best_of_{max(state['settings']['ks'])}")]
    if not rows:
        return
    write_csv(directory / 'by_alpha_main.csv', rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, key, ylabel in zip(axes, ('cost', 'ATT_minutes', 'RTT_minutes'), ('Cost', 'ATT, minutes', 'RTT, minutes')):
        for label in dict.fromkeys(r['model'] for r in rows):
            for mode in dict.fromkeys(r['mode'] for r in rows):
                subset = sorted((r for r in rows if r['model'] == label and r['mode'] == mode), key=lambda r:r['alpha'])
                ax.plot([r['alpha'] for r in subset], [r[key] for r in subset],
                        '--' if mode == 'greedy' else '-', marker='.', label=f'{label}: {mode}')
        ax.set(xlabel='ATT preference alpha', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Main policies on the common K-test cities')
    fig.savefig(directory / 'alpha_response_main.png', dpi=140)
    plt.close(fig)



def campaign_jobs(settings):
    """Seven independent starts: six matched S10 conditions and one S5 condition."""
    jobs = []
    for sampler in ('sampler', 'simplex'):
        for algorithm, weight, label in (('ppo', 0., 'ppo'), ('d3po', 0., 'd3po_l0'),
                                         ('d3po', .1, 'd3po_l0p1')):
            jobs.append(dict(id=f'{label}_{sampler}', kind='training', algorithm=algorithm,
                weight=weight, detach=False, sampler=sampler, resample=False, seed=0, alpha=None,
                batch=settings['batches'][0], minibatch=settings['minibatch'],
                horizon=settings['horizon'], updates=settings['main_updates'],
                threads=settings['threads'], routes=10, length=12,
                comparison_updates=settings['monitor_period']))
    jobs.append(dict(jobs[2], id='routes5_d3po_l0p1_sampler', routes=5))
    return jobs


def evaluation_inputs(root, spec, actor):
    """Return prepared graphs, source IDs and indices, without touching train data."""
    from connectpt.routes_generator.training.dataset import load_dataset
    if spec.get('benchmark'):
        from connectpt.routes_generator.run.instances import load_instance
        from connectpt.routes_generator.city.graph import to_batch
        city = load_instance(spec['benchmark'], instances_dir=root / 'data/benchmark', require_routes=False)
        graph = to_batch(city, insert_pos_features=False,
                         scale_dynamically=actor.cfg.dataset.scale_dynamically).to_data_list()[0]
        graphs = [actor.dataset.evaluation_pipeline.graph(graph)]
        source = identity({p.name: file_hash(p) for p in sorted(
            (root / 'data/benchmark' / spec['benchmark']).glob('*.txt'))})
        return graphs, [source], [0]
    path = root / 'data' / variant_name(spec) / 'test'
    data = load_dataset(path, kind='construction')
    settings = read_json(root / 'manifest.json')['settings']
    indices = list(range(min(len(data.graphs), settings.get('synthetic_eval_graphs', 10))))
    graphs = [actor.dataset.evaluation_pipeline.graph(data.graphs[i]) for i in indices]
    return graphs, [data.params.source_ids[i] for i in indices], indices


def import_data(root, source):
    """Copy only immutable datasets; never import settings, measurements or weights."""
    from connectpt.routes_generator.training.dataset import read_manifest
    from connectpt.routes_generator.run.instances import load_instance
    source = Path(source).expanduser().resolve()
    sources = {}
    for variant in ('r10_l12', 'r5_l12'):
        for part in ('pool', 'test', 'selection'):
            origin = source / variant / part
            meta = read_manifest(origin)
            destination = root / 'data' / variant / part
            destination.mkdir(parents=True, exist_ok=True)
            # Internal hard links avoid duplicate variant blobs. External files are copied.
            shared = root / 'data/r10_l12' / part / 'dataset.pt'
            target = destination / 'dataset.pt'
            if target.is_file() and file_hash(target) == meta['artifacts']['dataset.pt']:
                pass
            elif target != shared and shared.exists() and file_hash(shared) == meta['artifacts']['dataset.pt']:
                try:
                    os.link(shared, target)
                except OSError:
                    shutil.copy2(origin / 'dataset.pt', target)
            else:
                shutil.copy2(origin / 'dataset.pt', target)
            shutil.copy2(origin / 'manifest.json', destination / 'manifest.json')
            read_manifest(destination)
            if (len(meta['source_ids']) != meta['count'] or meta['kind'] != 'construction'
                    or meta['limits']['n_routes'] != int(variant[1:].split('_')[0])):
                raise ValueError('Unexpected dataset size or route limits')
            sources[variant, part] = tuple(meta['source_ids'])
            if variant == 'r10_l12' and part == 'pool':
                splits = dict(train=meta['split']['train'], validation=meta['split']['validation'])
    for part in ('pool', 'test', 'selection'):
        if sources['r10_l12', part] != sources['r5_l12', part]:
            raise ValueError('S5/S10 tasks must contain exactly the same cities')
    groups = [set(sources['r10_l12', p]) for p in ('pool', 'test', 'selection')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Synthetic source city leaks between splits')
    pool = sources['r10_l12', 'pool']
    splits.update(train_sources=[pool[i] for i in splits['train']],
                  validation_sources=[pool[i] for i in splits['validation']],
                  selection=list(range(len(sources['r10_l12', 'selection']))),
                  selection_sources=list(sources['r10_l12', 'selection']),
                  test_sources=list(sources['r10_l12', 'test']))
    if set(splits['train']) & set(splits['validation']):
        raise ValueError('Training/validation indices overlap')
    save_json(root / 'protocols/splits.json', splits)
    benchmarks = ('mandl', 'mumford0')
    for city in benchmarks:
        # Mirror the original bytes too (e.g. checkout line endings may differ).
        origin = source / 'benchmark' / city
        if not origin.is_dir():
            origin = REPO / 'datasets/instances' / city
        shutil.copytree(origin, root / 'data/benchmark' / city, dirs_exist_ok=True)
        load_instance(city, instances_dir=root / 'data/benchmark', require_routes=False)
    save_json(root / 'protocols/data-hashes.json', {
        p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / 'data').rglob('*')) if p.is_file()})
    return dict(train=len(splits['train']), validation=len(splits['validation']),
                selection=len(splits['selection_sources']), test=len(splits['test_sources']),
                benchmarks=list(benchmarks), nodes=20)


def pipeline(controller):
    root, state = controller.root, controller.state
    training = campaign_jobs(state['settings'])
    controller.group(training, 'training')
    initial = [read_json(root / 'runs' / s['id'] / 'initialization.json')['actor_sha256'] for s in training]
    if len(set(initial)) != 1:
        raise RuntimeError('Matched actors did not start from identical weights')
    evaluations = []
    for spec in training:
        for task in evaluation_tasks(spec):
            task.update(id=spec['id'] + '__' + task['suite'], kind='evaluation')
            evaluations.append(task)
    controller.group(evaluations, 'final_best_evaluation')
    state.update(status='completed', stage='summary')
    controller.save()


def register_plan(root, state):
    """Write all seven configs and immutable task specifications, without a worker."""
    from omegaconf import OmegaConf
    training = campaign_jobs(state['settings'])
    for spec in training:
        save_json(root / 'protocols/jobs' / (spec['id'] + '.json'), spec)
        cfg = worker_config(root, spec)
        cfg_dir = root / 'protocols/configs'
        cfg_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, cfg_dir / (spec['id'] + '.yaml'))
        state['jobs'][spec['id']] = dict(spec=spec, status='pending')
        for task in evaluation_tasks(spec):
            task.update(id=spec['id'] + '__' + task['suite'], kind='evaluation')
            save_json(root / 'protocols/jobs' / (task['id'] + '.json'), task)
            state['jobs'][task['id']] = dict(spec=task, status='pending')
    save_json(root / 'protocols/plan.json', dict(
        schema=SCHEMA, training=training, training_count=len(training), workers=state['settings']['workers'],
        time_limit_seconds=None, checkpoints=['best', 'best_balanced', 'at_200', 'at_400', 'best_at_200', 'best_balanced_at_200', 'last'],
        training_recovery=['training_state.pt', 'training_at_200.pt', 'training_at_400.pt'],
        periodic_evaluation='at_200 and at_400: 10 fixed synthetic cities S10/S5, Mandl and Mumford0',
        final_evaluation='best and best_balanced checkpoints separately, the same four tasks',
        checkpoint_selection='best: original v4 validation; best_balanced: uniform-alpha mean Sample cost on validation only; benchmark/test never select',
        evaluation_modes=['Greedy', 'Sample / LC1', 'LC100'],
        pareto_rule='one actual selected network per city/alpha/mode, feasible nondominated subset; then mean city metrics',
        candidate_pool='100 stochastic networks per city/alpha; LC1 is the first, no greedy mixed into LC100',
        algorithm_change=dict(gamma=state['settings'].get('gamma', .95),
                              gae_lambda=state['settings'].get('gae_lambda', .95)),
        scalar_diagnostics=state['settings'].get('scalar_diagnostics'),
        checkpoint_selection_change='additional best_balanced selection; legacy best and training updates preserved'))


def verify_v4_mirror(root, source):
    """Require every effective config to differ from v4 in exactly gamma/lambda."""
    from omegaconf import OmegaConf
    source = Path(source).expanduser().resolve()
    reference = root / 'protocols/reference_v4'
    reference.mkdir(exist_ok=True)
    allowed = {'algorithm.gamma', 'algorithm.gae_lambda'}
    def differences(left, right, prefix=''):
        if isinstance(left, dict) and isinstance(right, dict):
            return [item for key in sorted(set(left) | set(right))
                    for item in differences(left.get(key), right.get(key), f'{prefix}.{key}' if prefix else key)]
        return [] if left == right else [dict(key=prefix, v4=left, v5=right)]
    rows=[]
    for config in sorted((root / 'protocols/configs').glob('*.yaml')):
        original = source / 'protocols/configs' / config.name
        if not original.exists():original = source / 'runs' / config.stem / 'config.yaml'
        if not original.exists():raise ValueError(f'Missing v4 reference config: {config.stem}')
        old = OmegaConf.to_container(OmegaConf.load(original), resolve=True)
        new = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
        diff=differences(old,new)
        if {r['key'] for r in diff} != allowed:
            raise ValueError(f'Unexpected v4/v5 configuration difference for {config.stem}: {diff}')
        if any(r['v4'] != .95 or r['v5'] != 1. for r in diff):
            raise ValueError(f'Expected .95 -> 1 for gamma and GAE lambda: {diff}')
        shutil.copy2(original,reference/config.name)
        initialization=source/'runs'/config.stem/'initialization.json'
        rows.append(dict(model=config.stem,differences=diff,reference_sha256=file_hash(original),
            initial_actor_sha256=read_json(initialization)['actor_sha256'] if initialization.exists() else None))
    old_data=read_json(source/'protocols/data-hashes.json')
    new_data=read_json(root/'protocols/data-hashes.json')
    for name,expected in new_data.items():
        if old_data.get(name)!=expected:raise ValueError(f'Dataset differs from v4: {name}')
    save_json(root/'protocols/v4-parity.json',dict(reference_manifest_sha256=file_hash(source/'manifest.json'),
        allowed_training_differences=sorted(allowed),models=rows,data_hashes_identical=True,
        evaluation_change='10 fixed synthetic test cities S10/S5, Mandl and Mumford0; ordinary validation unchanged',
        logging_change='scalar-only observers; no gradient diagnostics or tensor/histogram dumps',
        checkpoint_selection_change='additional best_balanced using uniform validation alpha grid; original best retained'))


def write_instructions(root):
    torch_requirement = next(line for line in (root / 'environment/runtime-requirements.txt').read_text().splitlines()
                             if line.startswith('torch=='))
    (root / 'START_HERE.txt').write_text(f'''Самостоятельный пакет PPO/D3PO v5. Обучение ещё не запускалось при сборке.

7 обучений: PPO / D3PO lambda=0 / D3PO lambda=.1, каждый с Holliday и simplex;
дополнительно D3PO lambda=.1 Holliday с 5 маршрутами. Seed=0 у всех.
400 обновлений, batch=minibatch=128, horizon=120, resample=false.
Единственные изменения обучения относительно v4: gamma=1 и GAE lambda=1.
D3PO diversity_weight остаётся 0/.1. Метрики диагностики — только скаляры.
Максимум 3 worker-процесса по 4 CPU-потока. Ограничений времени нет.
Нормировка модели сохранена. Старые веса и калибровочные замеры не нужны.
Данные и бенчмарки включены; внешние пути к прошлым сериям не используются.
Тяжёлая оценка использует 10 фиксированных synthetic test городов при S5/S10,
а из бенчмарков только Mandl и Mumford0. Обычная sample validation не сокращается.

Сервер: архив распаковывается в корень проекта connectpt, с сохранением путей.
Запуск из корня проекта через существующее окружение .venv:
  bash scripts/launch.sh run_study.py --out-dir "$PWD/artifacts/studies/ppo_d3po_v5" --resume --log "$PWD/artifacts/studies/ppo_d3po_v5/controller-console.log"
Дочерние workers используют launcher и Python того же корня проекта.

Альтернатива: отдельное окружение пакета, из корня этой папки:
  python3 -m venv source/.venv
  source source/.venv/bin/activate
  python -m pip install {torch_requirement} --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r environment/runtime-requirements.txt
  python -m pip install --no-deps -e source
  bash source/scripts/launch.sh run_study.py --out-dir "$PWD" --resume --log "$PWD/controller-console.log"

Windows: окружение source/.venv; запуск существующим source/scripts/launch.ps1
с теми же аргументами --out-dir, --resume и --log.

protocols/plan.json — полный план; protocols/configs — семь эффективных конфигов.
protocols/v4-parity.json — проверка двух изменений и идентичности данных v4.
При незавершённой подготовке: повторить --prepare-only с --retry-preparation
и исходными --data-from / --mirror-from. Обученные серии повтор не перезаписывает.
diag/episode, diag/critic, diag/update, diag/validation — компактные скаляры TB.
Дополнительный best_balanced: 16 validation-городов, равномерная сетка 11 alpha,
4 Sample на город/alpha, раз в 20 обновлений. Отдельно диагностируется alpha=.85.
RNG изолирован, исходный best не меняется. Test/бенчмарки не участвуют в отборе.
status.txt / status.json — прогресс; resources.csv — фактические ресурсы.
runs/<model>/checkpoints — best.pt, best_balanced.pt, last.pt, best_at_200.pt,
best_balanced_at_200.pt, at_200.pt, at_400.pt.
training_state.pt — полное состояние после каждого обновления;
training_at_200.pt / training_at_400.pt — полное состояние перед оценкой границы.
Выбор best: прежний полный sample validation cost, как в v4.
Выбор best_balanced: средний Sample-cost с равным весом 11 alpha на validation.
Оба отбираются независимо; checkpoint_selection.json сохраняет шаг/критерий.
evaluation/<model>__<task>/<checkpoint> — synthetic S5/S10 либо один бенчмарк.
Для каждого города и alpha сохранены все 100 stochastic кандидатов и greedy.
Greedy, best_of_1 (LC1), best_of_100 (LC100) имеют отдельные метрики и фронты.
Периодические оценки at_200/at_400 не влияют на отбор checkpoint.
Итоговые best и best_balanced оцениваются раздельно по всем четырём задачам.
summary/metrics.csv — раздельные режимы; common_reference10 — дополнительная нормировка.
Парето-атласы строятся по городам; график средних координат не является фронтом города.

Перестроение из кандидатов без инференса:
  python source/scripts/run_study.py --out-dir "$PWD" --replot

--resume продолжает очередь и прерванную оценку по сохранённым чанкам.
Прерванное новое обучение продолжается из полного training_state.pt.
Старые actor-only checkpoint не обеспечивают точного продолжения.
Пропуск тяжёлой оценки: --skip-evaluation либо файл SKIP_EVALUATION в корне серии.
Файл runs/<model>/SKIP_EVALUATION пропускает оценку только этой модели.
Маркер проверяется между чанками кандидатов. Удаление маркера возвращает оценку.
Sample validation для выбора best не пропускается.
Уже запущенные старые процессы не получают эти возможности от изменения файлов.
Все существующие артефакты при ошибке сохраняются. Для переноса копируется вся папка.
''', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--memory-gb', type=float, default=50.)
    parser.add_argument('--data-from', help='Directory containing r10_l12 and r5_l12 immutable datasets; copied into package')
    parser.add_argument('--mirror-from', help='Prepared v4 package: verify that only gamma and GAE lambda change')
    parser.add_argument('--prepare-only', action='store_true', help='Package configs, source and data; never starts workers')
    parser.add_argument('--retry-preparation', action='store_true',
                        help='Retry an incomplete prepare-only operation; refuses studies with training outputs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip heavy monitoring/final evaluation; retain sample validation and full checkpoints')
    parser.add_argument('--replot', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.out_dir).expanduser().resolve()
    if args.prepare_only and (args.resume or args.replot or args.worker):
        parser.error('prepare-only cannot start/resume/replot a study')
    if args.retry_preparation and not args.prepare_only:
        parser.error('retry-preparation requires prepare-only; it never starts workers')
    if (args.resume or args.replot or args.worker) and (root / 'manifest.json').exists():
        frozen = root / 'source/scripts/run_study.py'
        prepared = read_json(root / 'manifest.json')
        if prepared.get('schema') == SCHEMA and (
                prepared.get('status') in ('preparing', 'preparation_failed')
                or not prepared.get('source_hashes') or not frozen.is_file()):
            parser.error('Study preparation is incomplete; no workers were started. '
                         'Inspect the prepare-only launcher log, then use --prepare-only '
                         '--retry-preparation with the original data-from and mirror-from paths.')
        if Path(__file__).resolve() != frozen:
            os.environ.setdefault('CONNECTPT_STUDY_LAUNCH_ROOT', str(REPO))
            sys.path.insert(0, str(root / 'source'))
            sys.argv = [str(frozen)] + list(sys.argv[1:] if argv is None else argv)
            runpy.run_path(str(frozen), run_name='__main__')
            return
    if args.worker:
        started = time.monotonic()
        try:
            worker(root, args.job)
        finally:
            cpu = psutil.Process().cpu_times()
            save_json(root / 'runs' / args.job / 'worker-resources.json', dict(
                cpu_seconds=cpu.user+cpu.system, wall_seconds=time.monotonic()-started))
        return
    if args.memory_gb <= 0:
        parser.error('memory-gb must be positive')
    if args.prepare_only:
        if not args.data_from:
            parser.error('preparation requires --data-from DATA_DIRECTORY; no data or workers are generated')
        if args.retry_preparation:
            previous = read_json(root / 'manifest.json')
            if (previous.get('schema') != SCHEMA or previous.get('source_hashes')
                    or previous.get('status') not in ('preparing', 'preparation_failed', 'prepared')
                    or 'started' in previous
                    or any((root / name).exists() for name in ('runs', 'evaluation', '.controller.lock'))):
                parser.error('Retry only supports incomplete preparation with no training outputs or completed snapshot')
            if (previous['settings']['workers'] != args.workers
                    or previous['settings']['memory_gb'] != args.memory_gb):
                parser.error('Retry requires the original workers and memory-gb settings')
        elif root.exists() and any(root.iterdir()):
            parser.error('preparation requires an empty, dedicated output directory')
        root.mkdir(parents=True, exist_ok=True)
        for name in ('environment', 'protocols/jobs', 'summary', 'figures'):
            (root / name).mkdir(parents=True, exist_ok=True)
        settings = dict(memory_gb=args.memory_gb, workers=args.workers, smoke=False,
            pool_graphs=32768, nodes=20, routes=10, length=12,
            validation_graphs=None, selection_graphs=16, test_graphs=64, k_graphs=10, synthetic_eval_graphs=10,
            preference_count=11, ks=[1, 100], batches=[128], minibatch=128, horizon=120,
            main_updates=400, monitor_period=200, val_period=20, threads=4, poll_seconds=5,
            gamma=1., gae_lambda=1., final_checkpoint_labels=['best','best_balanced'],
            scalar_diagnostics=dict(reference_gamma=.95,
                validation_graphs=16, validation_samples=4, validation_alphas=[0.,.5,.85,1.],
                validation_batch_size=32, validation_period=20, seed=9105))
        settings['scalar_diagnostics']['selection_alphas']=[i/(settings['preference_count']-1) for i in range(settings['preference_count'])]
        settings['scalar_diagnostics']['validation_alphas']=sorted(set(settings['scalar_diagnostics']['validation_alphas']+settings['scalar_diagnostics']['selection_alphas']))
        counts = import_data(root, args.data_from)
        if counts != dict(train=29491, validation=3277, selection=16, test=64,
                          benchmarks=['mandl', 'mumford0'], nodes=20):
            raise ValueError(f'v5 requires the agreed synthetic dataset sizes: {counts}')
        state = dict(schema=SCHEMA, created=now(), settings=settings, jobs={}, choices={},
                     status='preparing', workers=args.workers, peak_memory_bytes=0, data_counts=counts)
        save_json(root / 'manifest.json', state)
        register_plan(root, state)
        if args.mirror_from:verify_v4_mirror(root, args.mirror_from)
        if args.retry_preparation and (root / 'source').exists():
            incomplete = root / 'source'
            backup = root / 'environment' / f'incomplete-source-{time.time_ns()}'
            if not incomplete.resolve().is_relative_to(root) or not backup.resolve().is_relative_to(root):
                raise ValueError('Incomplete source must remain inside the study directory')
            incomplete.rename(backup)
        state['source_hashes'] = snapshot(root)
        write_instructions(root)
        if args.skip_evaluation:
            (root / 'SKIP_EVALUATION').touch()
        state['status'] = 'prepared'
        save_json(root / 'manifest.json', state)
        (root / 'status.txt').write_text('Prepared v5; no training, inference or worker processes started.\n', encoding='utf-8')
        print(f'Portable v5 prepared, no workers started: {root}', flush=True)
        return
    if not (args.resume or args.replot):
        parser.error('Use --prepare-only to package or --resume to explicitly start the prepared study')
    state = read_json(root / 'manifest.json')
    if state['schema'] != SCHEMA:
        raise ValueError('This runner requires a v5 package; older studies retain their frozen runner')
    verify_files(root, state['source_hashes'])
    verify_files(root, read_json(root / 'protocols/data-hashes.json'))
    if args.replot:
        for job in state['jobs'].values():
            if job['spec']['kind'] == 'evaluation':
                for protocol in (root / 'evaluation' / job['spec']['id']).glob('*/protocol.json'):
                    if (protocol.parent / 'results.json').exists():
                        rebuild_run(root, job['spec'], protocol.parent.name)
        summary(root)
        return
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
    save_json(lock, dict(pid=os.getpid(), created=psutil.Process().create_time()))
    if args.skip_evaluation:
        (root / 'SKIP_EVALUATION').touch()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'controller.log', encoding='utf-8')], force=True)
    state.pop('error', None)
    if 'started' not in state:
        save_json(root / 'environment/execution-machine.json', machine_info())
    state.setdefault('started', now())
    controller = Coordinator(root, state)
    try:
        if state['status'] != 'completed':
            pipeline(controller)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        controller.save()
        controller.sample()
        lock.unlink(missing_ok=True)
        summary(root)


if __name__ == '__main__':
    main()

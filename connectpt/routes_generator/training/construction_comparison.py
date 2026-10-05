"""Paired two-objective evaluation on shared construction test instances."""
from pathlib import Path
import json
import numpy as np
from itertools import combinations, permutations

from ..core.artifacts import ArtifactStore


def nondominated(points):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    keep = [not np.any(np.all(points <= p, axis=1) & np.any(points < p, axis=1))
            for p in points]
    return points[keep]


def hypervolume_2d(points, reference):
    front = nondominated(points)
    front = front[np.all(front < reference, axis=1)]
    front = front[np.argsort(front[:, 0])]
    volume, y = 0.0, float(reference[1])
    for x, new_y in front:
        volume += (float(reference[0]) - x) * max(y - new_y, 0)
        y = min(y, new_y)
    return float(volume)


def coverage(a, b):
    """Fraction of unique B points weakly dominated by an A point (minimization)."""
    b = np.unique(np.asarray(b).reshape(-1, 2), axis=0)
    a = np.asarray(a).reshape(-1, 2)
    if not len(b):
        return None
    return float(np.mean([np.any(np.all(a <= p, axis=1)) for p in b]))


def compare_runs(root, names=('ppo', 'decomposed_ppo', 'd3po'), *,
                 checkpoint='last', mode='greedy', budget_mode='transitions'):
    root = Path(root)
    evaluations, statuses = {}, {}
    for name in names:
        run = root / 'runs' / name
        suffix = '' if checkpoint == 'last' and mode == 'greedy' else f'_{checkpoint}_{mode}'
        evaluations[name] = json.loads((run / f'test_evaluation{suffix}.json').read_text())['rows']
        statuses[name] = json.loads((run / 'status.json').read_text())
    keys = [{(r['graph'], r['preference'], r.get('sample', 0)) for r in rows}
            for rows in evaluations.values()]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError('Evaluation graph/preference grids differ')
    if len({s['initial_actor_sha256'] for s in statuses.values()}) != 1:
        raise ValueError('Initial actors differ')
    if any(s['state'] != 'completed' for s in statuses.values()):
        raise ValueError('All runs must complete before comparison')
    if budget_mode == 'transitions':
        if len({s['total_active_steps'] for s in statuses.values()}) != 1:
            raise ValueError('Training interaction budgets differ')
    elif budget_mode == 'wall_time':
        limits = {s.get('max_seconds') for s in statuses.values()}
        if len(limits) != 1 or None in limits or next(iter(limits)) <= 0:
            raise ValueError('Training time budgets differ or are missing')
        if any(s.get('stop_reason') != 'time_budget' for s in statuses.values()):
            raise ValueError('All runs must reach the training time budget')
    else:
        raise ValueError(f'Unknown budget mode: {budget_mode}')
    if len({s.get('seed') for s in statuses.values()}) != 1:
        raise ValueError('Training seeds differ')
    # Shared deterministic bounds, selected from the pooled evaluation outcomes.
    pooled = np.array([[r['penalized_att'], r['penalized_rtt']]
                      for rows in evaluations.values() for r in rows])
    reference = (pooled.max(0) * 1.1 + .01).tolist()
    pure_pool = np.array([[r['att_norm'], r['rtt_norm']]
                         for rows in evaluations.values() for r in rows])
    pure_reference = (pure_pool.max(0) * 1.1 + .01).tolist()
    prefs = np.linspace(0, 1, 101)
    query_weights = np.stack((prefs, 1 - prefs), -1)
    per_graph = []
    for graph, sample in sorted({(r['graph'], r.get('sample', 0))
                                 for r in evaluations[names[0]]}):
        points, feasible_points, metrics = {}, {}, {}
        for name in names:
            rows = [r for r in evaluations[name]
                    if r['graph'] == graph and r.get('sample', 0) == sample]
            points[name] = np.array([[r['penalized_att'], r['penalized_rtt']] for r in rows])
            feasible_points[name] = np.array([[r['att_norm'], r['rtt_norm']]
                                              for r in rows if r['feasible']]).reshape(-1, 2)
            feasible = feasible_points[name]
            metrics[name] = dict(hv_penalized=hypervolume_2d(points[name], reference),
                eu_conditioned_penalized=float(np.mean([
                    -r['preference'] * r['penalized_att']
                    - (1 - r['preference']) * r['penalized_rtt'] for r in rows])),
                eu_archive_penalized=float((-np.min(query_weights @ points[name].T, axis=1)).mean()),
                hv_feasible=hypervolume_2d(feasible, pure_reference),
                eu_archive_feasible=float((-np.min(query_weights @ feasible.T, axis=1)).mean()) if len(feasible) else None,
                feasible_fraction=float(np.mean([r['feasible'] for r in rows])),
                pareto_points_penalized=len(nondominated(points[name])),
                pareto_points_feasible=len(nondominated(feasible)),
                mean_unserved_fraction=float(np.mean([r['unserved_fraction'] for r in rows])))
        per_graph.append(dict(graph=graph, sample=sample, metrics=metrics,
            coverage_penalized={f'{a}_dominates_{b}': coverage(points[a], points[b])
                                for a, b in permutations(names, 2)},
            coverage_feasible={f'{a}_dominates_{b}': coverage(feasible_points[a], feasible_points[b])
                               for a, b in permutations(names, 2)}))
    summary = {}
    for name in names:
        fields = per_graph[0]['metrics'][name]
        summary[name] = {field: float(np.mean([g['metrics'][name][field] for g in per_graph
                                              if g['metrics'][name][field] is not None]))
                         if any(g['metrics'][name][field] is not None for g in per_graph) else None
                         for field in fields}
        summary[name].update(iterations=statuses[name]['iterations'],
            training_seconds=statuses[name]['training_seconds'],
            active_steps=statuses[name]['total_active_steps'],
            graphs_with_feasible_points=len({g['graph'] for g in per_graph
                if g['metrics'][name]['pareto_points_feasible'] > 0}))
    paired = {}
    for a, b in combinations(names, 2):
        paired[f'{b}_minus_{a}'] = {}
        for field in ('hv_penalized', 'eu_conditioned_penalized', 'hv_feasible', 'feasible_fraction'):
            # Samples are repeated measurements, not independent test graphs.
            differences = np.array([np.mean([g['metrics'][b][field] - g['metrics'][a][field]
                for g in per_graph if g['graph'] == graph])
                for graph in sorted({g['graph'] for g in per_graph})])
            rng = np.random.default_rng(1503)
            bootstrap = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(1)
            paired[f'{b}_minus_{a}'][field] = dict(difference=float(differences.mean()),
                ci95_graph_bootstrap=np.quantile(bootstrap, [.025, .975]).tolist())
    report = dict(summary=summary, per_graph=per_graph, paired_differences=paired,
        reference_penalized=reference, reference_feasible=pure_reference,
        direction='minimize costs; maximize HV and EU (EU is negative cost)',
        hv_reference_rule='1.1 * pooled coordinate maximum + 0.01; same reference for both runs',
        eu_conditioned_rule='Mean utility of the network generated with the requested preference',
        eu_archive_rule='Best linear utility among evaluated preferences, 101 uniform query weights',
        coverage_rule='Weak domination of unique points, computed within each graph',
        checkpoint=checkpoint, evaluation_mode=mode,
        budget='equal environment transitions' if budget_mode == 'transitions' else 'equal training wall time',
        time_limit_seconds=statuses[names[0]].get('max_seconds') if budget_mode == 'wall_time' else None,
        stochastic_rule='Mean metrics of independent preference-grid draws; no pooling or best-of-N',
        n_training_seeds=1)
    suffix = '' if checkpoint == 'last' and mode == 'greedy' else f'_{checkpoint}_{mode}'
    ArtifactStore(root).save_json(report, f'comparison{suffix}.json')
    return report

"""Compare frozen construction actors with nested best-of-K sampling.

Uses the saved test protocol, actor configuration and original scalar cost.
Writes resumable candidate pools, physical ATT/RTT, CSV tables and a report.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from connectpt.routes_generator.objectives.pareto import canonical_network, summarize_evaluation, hypervolume_2d
from connectpt.routes_generator.objectives.preferences import scalarize
from connectpt.routes_generator.policy.predict import predict
from connectpt.routes_generator.run.context import resolve_under_root
from connectpt.routes_generator.run.training import TrainingRun
from connectpt.routes_generator.training.data import evaluation_graphs


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def library_fingerprint():
    root = resolve_under_root('connectpt/routes_generator')
    return digest({p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob('*.py'))})


def prediction_rows(prediction, graph_start, alpha, sample):
    result = prediction.evaluation
    penalty = torch.zeros_like(result.cost)
    for j, weight in enumerate(result.penalty_weights):
        penalty += result.penalties[:, j] * weight
    feasible = ~result.are_constraints_violated()
    rows = []
    for i in range(len(result.cost)):
        row = dict(graph=graph_start+i, preference=float(alpha), sample=sample,
                   objective_0=float(result.goals[i, 0]), objective_1=float(result.goals[i, 1]),
                   penalized_0=float(result.goals[i, 0] + penalty[i]),
                   penalized_1=float(result.goals[i, 1] + penalty[i]),
                   scalar_cost=float(result.cost[i]), feasible=bool(feasible[i]),
                   unserved_fraction=float(result.unserved_demand[i] / result.total_demand[i]),
                   ATT=float(result.mean_demand_time[i] / 60), RTT=float(result.total_route_time[i] / 60),
                   network_signature=canonical_network(prediction.routes[i].cpu().tolist(), True))
        if not all(math.isfinite(row[key]) for key in ('scalar_cost', 'ATT', 'RTT')):
            raise ValueError('Nonfinite candidate evaluation')
        rows.append(row)
    return rows


def select_rows(greedy, sampled, k=None, *, include_greedy=False):
    """Select one actual candidate per (city, alpha), retaining its ATT and RTT."""
    if k is None:
        return [dict(row, sample=0) for row in greedy]
    groups = {}
    for row in sampled:
        if row['sample'] < k:
            groups.setdefault((row['graph'], row['preference']), []).append(row)
    for group in groups.values():
        if len(group) != k:
            raise ValueError('Incomplete candidate prefix')
    if include_greedy:
        for row in greedy:
            groups[(row['graph'], row['preference'])].append(row)
    selected = []
    for key in sorted(groups):
        best = min(groups[key], key=lambda row: row['scalar_cost'])
        selected.append(dict(best, selected_sample=best['sample'], sample=0))
    return selected


def row_summary(rows):
    metrics = summarize_evaluation(rows, [3., 3.], 101)['summary']
    return dict(cost=float(np.mean([r['scalar_cost'] for r in rows])),
                ATT_minutes=float(np.mean([r['ATT'] for r in rows])),
                RTT_minutes=float(np.mean([r['RTT'] for r in rows])),
                hv=metrics['hv_feasible'], feasible_fraction=metrics['feasible_fraction'],
                distinct_networks=metrics['distinct_networks'],
                distinct_pareto_networks=metrics['distinct_pareto_networks'])


def summarize_pool(label, pools, ks):
    greedy = [row for pool in pools for row in pool['greedy']]
    sampled = [row for pool in pools for row in pool['sampled']]
    modes = [('greedy', None, False)] + [(f'best_of_{k}', k, False) for k in ks]
    modes += [(f'greedy_plus_{max(ks)}', max(ks), True)]
    overall, by_alpha, per_city = [], [], []
    previous = None
    for mode, k, hybrid in modes:
        selected = select_rows(greedy, sampled, k, include_greedy=hybrid)
        if k is not None and not hybrid:
            values = {(r['graph'], r['preference']): r['scalar_cost'] for r in selected}
            if previous is not None and any(values[key] > previous[key] for key in values):
                raise ValueError('Best-of-K cost increased for a nested pool')
            previous = values
        elapsed = sum(pool['greedy_seconds'] if k is None else pool['prefix_seconds'][str(k)]
                      + (pool['greedy_seconds'] if hybrid else 0.) for pool in pools)
        overall.append(dict(model=label, mode=mode, K=0 if k is None else k,
                            candidates_per_query=1 if k is None else k+int(hybrid),
                            generation_evaluation_seconds=elapsed, **row_summary(selected)))
        for alpha in sorted({r['preference'] for r in selected}):
            subset = [r for r in selected if r['preference'] == alpha]
            by_alpha.append(dict(model=label, mode=mode, alpha=alpha,
                cost=float(np.mean([r['scalar_cost'] for r in subset])),
                ATT_minutes=float(np.mean([r['ATT'] for r in subset])),
                RTT_minutes=float(np.mean([r['RTT'] for r in subset])),
                ATT_std_between_cities=float(np.std([r['ATT'] for r in subset])),
                RTT_std_between_cities=float(np.std([r['RTT'] for r in subset])),
                feasible_fraction=float(np.mean([r['feasible'] for r in subset]))))
        for row in selected:
            per_city.append(dict(model=label, mode=mode, graph=row['graph'], alpha=row['preference'],
                                 cost=row['scalar_cost'], ATT_minutes=row['ATT'], RTT_minutes=row['RTT'],
                                 feasible=row['feasible'], network_id=digest(row['network_signature'])[:16]))
    return overall, by_alpha, per_city


def paired_comparisons(per_city):
    labels = list(dict.fromkeys(r['model'] for r in per_city))
    if len(labels) < 2:
        return []
    result = []
    reference = labels[0]
    baseline = {(r['mode'], r['graph'], r['alpha']): r for r in per_city if r['model'] == reference}
    for candidate in labels[1:]:
        other = {(r['mode'], r['graph'], r['alpha']): r for r in per_city if r['model'] == candidate}
        common = sorted(set(baseline) & set(other))
        for mode in sorted({k[0] for k in common}):
            modes = [key for key in common if key[0] == mode]
            for alpha in ['all'] + sorted({key[2] for key in modes}):
                keys = modes if alpha == 'all' else [key for key in modes if key[2] == alpha]
                row = dict(reference=reference, candidate=candidate, mode=mode, alpha=alpha, city_queries=len(keys))
                for field in ['cost', 'ATT_minutes', 'RTT_minutes']:
                    row[field+'_difference'] = float(np.mean([other[key][field]-baseline[key][field] for key in keys]))
                row['cost_better_fraction'] = float(np.mean([other[key]['cost'] < baseline[key]['cost'] for key in keys]))
                result.append(row)
    return result


def figures(out, overall, by_alpha, ks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = list(dict.fromkeys(row['model'] for row in overall))
    largest = f'best_of_{max(ks)}'
    for color, label in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'], labels):
        values = [r for r in overall if r['model'] == label and r['mode'].startswith('best_of_')]
        axes[0, 0].plot([r['K'] for r in values], [r['cost'] for r in values], 'o-', color=color, label=label)
        for mode, style in [('greedy', '--'), (largest, '-')]:
            rows = [r for r in by_alpha if r['model'] == label and r['mode'] == mode]
            legend = f'{label}, {mode}'
            axes[0, 1].plot([r['alpha'] for r in rows], [r['ATT_minutes'] for r in rows], style, color=color, label=legend)
            axes[1, 0].plot([r['alpha'] for r in rows], [r['RTT_minutes'] for r in rows], style, color=color)
            axes[1, 1].plot([r['RTT_minutes'] for r in rows], [r['ATT_minutes'] for r in rows],
                            style+'o', color=color, markersize=3, label=legend)
    axes[0, 0].set(xlabel='K (nested stochastic candidates)', ylabel='Mean normalized cost', xscale='log')
    axes[0, 1].set(xlabel='ATT preference alpha', ylabel='ATT (minutes)')
    axes[1, 0].set(xlabel='ATT preference alpha', ylabel='RTT (minutes)')
    axes[1, 1].set(xlabel='RTT (minutes)', ylabel='ATT (minutes)', title='Means at each requested alpha')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(out/'comparison.png', dpi=160)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)


def pareto_diagnostics(per_city, test_indices):
    """Compare feasible outcomes within a city/model/mode, never across cities.

    Inputs are selected networks, with physical minutes and a canonical network ID.
    Numerical changes <=1e-4 minutes count as ties; no policy, RNG or input mutation.
    """
    groups = {}
    for row in per_city:
        groups.setdefault((row['model'], row['mode'], row['graph']), []).append(row)
    annotated, diagnostics = [], []
    tolerance = 1e-4
    for (model, mode, graph), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['alpha'])
        points = np.array([[r['RTT_minutes'], r['ATT_minutes']] for r in rows])
        feasible = np.array([r['feasible'] for r in rows], dtype=bool)
        front = []
        for index, point in enumerate(points):
            dominates = (np.all(points <= point+tolerance, axis=1)
                         & np.any(points < point-tolerance, axis=1) & feasible)
            front.append(bool(feasible[index] and not dominates.any()))
        for row, keep in zip(rows, front):
            annotated.append(dict(row, dataset_index=test_indices[graph], nondominated_within_model_mode=keep))
        adjacent = len(rows)-1
        feasible_pairs = feasible[:-1] & feasible[1:]
        att = np.diff(points[:, 1])
        rtt = np.diff(points[:, 0])
        diagnostics.append(dict(model=model, mode=mode, graph=graph, dataset_index=test_indices[graph],
            alpha_count=len(rows), feasible_alpha_count=int(feasible.sum()),
            unique_networks=len({r['network_id'] for r in rows}),
            unique_nondominated_networks=len({r['network_id'] for r, keep in zip(rows, front) if keep}),
            unchanged_adjacent_alpha=sum(a['network_id'] == b['network_id'] for a, b in zip(rows, rows[1:])),
            adjacent_alpha_pairs=adjacent,
            adjacent_feasible_alpha_pairs=int(feasible_pairs.sum()),
            ATT_increases_with_alpha=int(((att > tolerance) & feasible_pairs).sum()),
            RTT_decreases_with_alpha=int(((rtt < -tolerance) & feasible_pairs).sum()),
            ATT_last_minus_first_minutes=float(points[-1, 1]-points[0, 1]),
            RTT_last_minus_first_minutes=float(points[-1, 0]-points[0, 0])))
    return annotated, diagnostics


def pareto_figures(out, per_city, test_indices, ks, *, complete):
    """Export empirical fronts and alpha trajectories; full atlas only on completion."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    rows, diagnostics = pareto_diagnostics(per_city, test_indices)
    write_csv(out/'pareto_points.csv', rows)
    write_csv(out/'alpha_response.csv', diagnostics)
    labels = list(dict.fromkeys(r['model'] for r in per_city))
    styles = {label: (f'C{i}', ['o', '^', 's', 'D'][i % 4]) for i, label in enumerate(labels)}
    cities = sorted({r['graph'] for r in rows})
    modes = ['greedy', f'best_of_{max(ks)}']
    if 8 in ks and max(ks) != 8:
        modes.insert(1, 'best_of_8')
    modes = [mode for mode in modes if any(r['mode'] == mode for r in rows)]
    norm, cmap = Normalize(0, 1), plt.get_cmap('viridis')
    for mode in modes:
        def page(page_cities):
            fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), layout='constrained')
            for ax, graph in zip(axes.flat, page_cities):
                for label in labels:
                    values = sorted((r for r in rows if r['model'] == label and r['mode'] == mode
                                     and r['graph'] == graph), key=lambda r: r['alpha'])
                    if not values:
                        continue
                    color, marker = styles[label]
                    # NaN gaps prevent connecting trajectories through infeasible outcomes.
                    x = [r['RTT_minutes'] if r['feasible'] else np.nan for r in values]
                    y = [r['ATT_minutes'] if r['feasible'] else np.nan for r in values]
                    ax.plot(x, y, '--', color=color, lw=.9, alpha=.6)
                    feasible = [r for r in values if r['feasible']]
                    ax.scatter([r['RTT_minutes'] for r in feasible], [r['ATT_minutes'] for r in feasible],
                               c=[r['alpha'] for r in feasible], cmap=cmap, norm=norm, marker=marker,
                               edgecolors=color, linewidths=1.3, s=65, zorder=3)
                    front = sorted((r for r in values if r['nondominated_within_model_mode']),
                                   key=lambda r: r['RTT_minutes'])
                    ax.plot([r['RTT_minutes'] for r in front], [r['ATT_minutes'] for r in front],
                            '-', color=color, lw=2, alpha=.8)
                    # Group repeated networks: alpha plateaus retain every requested weight.
                    groups = {}
                    for r in feasible:
                        groups.setdefault(r['network_id'], []).append(r)
                    for group in groups.values():
                        r = group[0]
                        annotation = ','.join(f"{p['alpha']:.1f}" for p in group)
                        ax.annotate(annotation, (r['RTT_minutes'], r['ATT_minutes']),
                                    xytext=(4, 6 if label == labels[0] else -12),
                                    textcoords='offset points', fontsize=6, color=color)
                counts = []
                for label in labels:
                    d = next((d for d in diagnostics if d['model'] == label and d['mode'] == mode
                              and d['graph'] == graph), None)
                    if d:
                        counts.append(f"{label}: {d['unique_nondominated_networks']}/{d['unique_networks']} ND/unique")
                ax.set(title=f"Test city {test_indices[graph]} | " + '; '.join(counts),
                       xlabel='RTT (minutes) - minimize', ylabel='ATT (minutes) - minimize')
                ax.grid(alpha=.18)
                ax.margins(.14)
            for ax in list(axes.flat)[len(page_cities):]:
                ax.set_visible(False)
            handles = [Line2D([], [], color=color, marker=marker, label=label)
                       for label, (color, marker) in styles.items()]
            handles += [Line2D([], [], color='0.4', lw=2, label='Empirical nondominated front'),
                        Line2D([], [], color='0.4', ls='--', label='Order of requested alpha')]
            axes.flat[0].legend(handles=handles, fontsize=7, loc='best')
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat), shrink=.8,
                         label='ATT preference alpha: 0 = RTT, 1 = ATT')
            fig.suptitle(f'{mode}: alpha response and feasible empirical fronts'
                         + ('' if complete else ' (partial / pilot)'), fontsize=13)
            return fig
        preview = page(cities[:4])
        preview.savefig(out/f'pareto_{mode}.png', dpi=170)
        preview.savefig(out/f'pareto_{mode}.svg')
        plt.close(preview)
        if complete:
            with PdfPages(out/f'pareto_{mode}_all_cities.pdf') as pdf:
                for start in range(0, len(cities), 4):
                    fig = page(cities[start:start+4])
                    pdf.savefig(fig)
                    plt.close(fig)


def preference_tests(label, greedy, sampled, ks, test_indices):
    """Offline cross-preference selection, using the original scalarization owner.

    Retains source cost; query_cost evaluates saved penalized objectives at query alpha.
    Never mutates candidates or RNG. Archives are separate, larger-budget diagnostics.
    """
    alphas = sorted({r['preference'] for r in greedy})
    if not any(abs(a-.5) < 1e-8 for a in alphas):
        raise ValueError('Fixed-alpha control requires saved alpha=0.5 candidates')
    fixed = min(alphas, key=lambda a: abs(a-.5))
    cities = sorted({r['graph'] for r in greedy})
    network_ids = {id(r): digest(r['network_signature'])[:16] for r in greedy+sampled}
    normalized = {(r['graph'], r['preference'], 'greedy' if i < len(greedy) else 'sample', r['sample']): r
                  for i, r in enumerate(greedy+sampled)}
    selected, matrices = [], []
    maximum_error = 0.
    for graph in cities:
        sources = [r for r in greedy+sampled if r['graph'] == graph]
        goals = torch.tensor([[r['penalized_0'], r['penalized_1']] for r in sources], dtype=torch.float64)
        weights = torch.tensor([[a, 1-a] for a in alphas], dtype=torch.float64)
        costs = scalarize(goals[None], weights[:, None], goals.new_empty((1, 0)), []).numpy()
        if not np.isfinite(costs).all():
            raise ValueError('Nonfinite cross-preference cost')
        generated = np.array([r['preference'] for r in sources])
        physical = np.array([[r['ATT'], r['RTT']] for r in sources])
        feasible_source = np.array([r['feasible'] for r in sources], dtype=bool)
        own = np.array([costs[alphas.index(r['preference']), i] for i, r in enumerate(sources)])
        error = float(np.max(np.abs(own-np.array([r['scalar_cost'] for r in sources]))))
        maximum_error = max(maximum_error, error)
        if error > 2e-6:
            raise ValueError('Saved scalar cost disagrees with ATT/RTT penalized objective scalarization')
        is_greedy = np.arange(len(sources)) < sum(r['graph'] == graph for r in greedy)
        # Sources were concatenated greedy then sampled, with one greedy per city/alpha.
        for k in [0]+list(ks):
            available = is_greedy if k == 0 else ~is_greedy & np.array([r['sample'] < k for r in sources])
            expected = 1 if k == 0 else k
            for alpha in alphas:
                ids = np.flatnonzero(available & (generated == alpha))
                if len(ids) != expected or (k and {sources[i]['sample'] for i in ids} != set(range(k))):
                    raise ValueError('Incomplete or duplicate source-alpha candidate prefix')
            for qi, alpha in enumerate(alphas):
                conditioned_ids = np.flatnonzero(available & (generated == alpha))
                fixed_ids = np.flatnonzero(available & (generated == fixed))
                archive_ids = np.flatnonzero(available)
                for strategy, ids in [('conditioned', conditioned_ids), ('fixed_0.5', fixed_ids), ('archive', archive_ids)]:
                    index = ids[np.argmin(costs[qi, ids])]
                    r = sources[index]
                    dominating = (np.all(physical <= physical[index]+1e-4, axis=1)
                                  & np.any(physical < physical[index]-1e-4, axis=1)
                                  & feasible_source & available)
                    selected.append(dict(model=label, strategy=strategy, K=k, graph=graph,
                        dataset_index=test_indices[graph], alpha=float(alpha), generated_alpha=r['preference'],
                        source_sample=r['sample'], source_scalar_cost=r['scalar_cost'], query_cost=float(costs[qi, index]),
                        source_mode='greedy' if k == 0 else 'sample',
                        candidates_per_query=len(ids), ATT_minutes=r['ATT'], RTT_minutes=r['RTT'],
                        feasible=r['feasible'], network_id=network_ids[id(r)],
                        dominated_in_source_archive=bool(dominating.any()) if r['feasible'] else None))
                for source_alpha in alphas:
                    ids = np.flatnonzero(available & (generated == source_alpha))
                    matrices.append(dict(model=label, K=k, graph=graph, query_alpha=float(alpha),
                        generated_alpha=float(source_alpha), query_cost=float(costs[qi, ids].min())))
    summary, gaps, by_alpha = [], [], []
    for k in [0]+list(ks):
        for strategy in ['conditioned', 'fixed_0.5', 'archive']:
            rows = [r for r in selected if r['K'] == k and r['strategy'] == strategy]
            original = [dict(r, preference=r['alpha'], sample=0, scalar_cost=r['query_cost'],
                ATT=r['ATT_minutes'], RTT=r['RTT_minutes'], objective_0=0., objective_1=0.,
                penalized_0=0., penalized_1=0., unserved_fraction=0.) for r in rows]
            # Physical fronts and counts use original, saved normalized objectives below.
            for r in original:
                source = normalized[(r['graph'], r['generated_alpha'], r['source_mode'], r['source_sample'])]
                r.update({key: source[key] for key in ['objective_0', 'objective_1', 'penalized_0', 'penalized_1',
                                                      'unserved_fraction', 'network_signature']})
            summary.append(dict(model=label, strategy=strategy, K=k, candidates_per_query=rows[0]['candidates_per_query'],
                                **row_summary(original)))
            for alpha in alphas:
                values = [r for r in rows if r['alpha'] == alpha]
                by_alpha.append(dict(model=label, strategy=strategy, K=k, alpha=float(alpha),
                    query_cost=float(np.mean([r['query_cost'] for r in values])),
                    ATT_minutes=float(np.mean([r['ATT_minutes'] for r in values])),
                    RTT_minutes=float(np.mean([r['RTT_minutes'] for r in values]))))
        for graph in cities:
            for alpha in alphas:
                group = {r['strategy']: r for r in selected if r['K'] == k and r['graph'] == graph and r['alpha'] == alpha}
                base = group['conditioned']
                for strategy in ['fixed_0.5', 'archive']:
                    other = group[strategy]
                    delta = base['query_cost']-other['query_cost']
                    if strategy == 'archive' and delta < -1e-10:
                        raise ValueError('Archive lost an included conditioned candidate')
                    gaps.append(dict(model=label, K=k, graph=graph, dataset_index=test_indices[graph],
                        alpha=float(alpha), comparator=strategy,
                        conditioned_minus_comparator_query_cost=delta,
                        relative_gain=float(delta/base['query_cost']) if base['query_cost'] else 0.,
                        comparator_better=bool(delta > 2e-6),
                        ATT_comparator_minus_conditioned=other['ATT_minutes']-base['ATT_minutes'],
                        RTT_comparator_minus_conditioned=other['RTT_minutes']-base['RTT_minutes']))
    # At alpha=.5, fixed and conditioned must select the identical saved prefix.
    if any(abs(r['conditioned_minus_comparator_query_cost']) > 1e-10 for r in gaps
           if r['comparator'] == 'fixed_0.5' and abs(r['alpha']-.5) < 1e-8):
        raise ValueError('Fixed and conditioned controls disagree at alpha=0.5')
    return dict(label=label, scalarization_max_abs_error=maximum_error,
                summary=summary, by_alpha=by_alpha, gaps=gaps, matrix=matrices, selected=selected)


def publish_preference_tests(out, report, records, ks, test_indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for field in ['summary', 'by_alpha', 'gaps', 'matrix', 'selected']:
        write_csv(out/f'{field}.csv', [r for record in records for r in record[field]])
    save_json(out/'results.json', dict(models=records, test_indices=test_indices, ks=ks,
        interpretation='Saved single sampling pool; archive budget is larger; fixed pool is reused across queries.'))
    fig, axes = plt.subplots(len(records), 2, figsize=(12, 4.5*len(records)), squeeze=False, layout='constrained')
    for ri, record in enumerate(records):
        for ci, k in enumerate([0, max(ks)]):
            rows = [r for r in record['matrix'] if r['K'] == k]
            alphas = sorted({r['query_alpha'] for r in rows})
            mean = np.array([[np.mean([r['query_cost'] for r in rows if r['generated_alpha'] == a
                                                   and r['query_alpha'] == b]) for b in alphas] for a in alphas])
            base = np.diag(mean)
            gain = (base[None]-mean)/np.maximum(base[None], 1e-12)*100
            limit = max(1., float(np.abs(gain).max()))
            ax = axes[ri, ci]
            im = ax.imshow(gain, origin='lower', cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set(xticks=range(len(alphas)), xticklabels=[f'{a:.1f}' for a in alphas],
                   yticks=range(len(alphas)), yticklabels=[f'{a:.1f}' for a in alphas],
                   xlabel='Requested alpha', ylabel='Generation alpha',
                   title=f"{record['label']} | {'greedy' if k == 0 else f'best-of-{k}'}")
            ax.axhline(alphas.index(min(alphas, key=lambda a: abs(a-.5))), color='black', lw=1)
            fig.colorbar(im, ax=ax, label='Gain vs generation at requested alpha (%)')
    fig.savefig(out/'cross_alpha.png', dpi=170)
    fig.savefig(out/'cross_alpha.svg')
    plt.close(fig)
    for record in records:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        for ax, graph in zip(axes.flat, range(min(4, len(test_indices)))):
            for strategy, style, marker in [('conditioned', '-', 'o'), ('fixed_0.5', '--', '^'), ('archive', ':', 's')]:
                rows = sorted([r for r in record['selected'] if r['graph'] == graph and r['K'] == max(ks)
                               and r['strategy'] == strategy], key=lambda r: r['alpha'])
                ax.plot([r['RTT_minutes'] if r['feasible'] else np.nan for r in rows],
                        [r['ATT_minutes'] if r['feasible'] else np.nan for r in rows], style, label=strategy, alpha=.7)
                feasible = [r for r in rows if r['feasible']]
                sc = ax.scatter([r['RTT_minutes'] for r in feasible], [r['ATT_minutes'] for r in feasible],
                    c=[r['alpha'] for r in feasible], cmap='viridis', vmin=0, vmax=1, marker=marker, edgecolors='0.3')
            ax.set(title=f'Test city {test_indices[graph]}', xlabel='RTT (minutes)', ylabel='ATT (minutes)')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[min(4, len(test_indices)):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.colorbar(sc, ax=list(axes.flat), label='Requested ATT alpha')
        fig.suptitle(f"{record['label']}: alpha response, K={max(ks)}; archive has {len(set(r['alpha'] for r in record['selected']))}x candidates")
        fig.savefig(out/f"response_{record['label']}.png", dpi=170)
        fig.savefig(out/f"response_{record['label']}.svg")
        plt.close(fig)
    text = ['# Проверка соответствия α: архив и фиксированная генерация', '',
        f'Городов: {len(test_indices)}; K: {ks}. Данные взяты из одного сохранённого sampling-пула.',
        'Это диагностика замороженных моделей, без нового обучения и генерации.', '',
        '1. conditioned: генерация при запрошенном α, отбор K сетей.',
        '2. fixed_0.5: генерация всегда при α=0,5, отбор тех же K сетей под каждый запрос.',
        '3. archive: отбор по объединению всех исходных α; бюджет в число α раз больше.', '',
        'Архив — внутренний ориентир доступных решений, не независимый эталон и не сравнение при равных затратах.',
        'Фиксированный пул переиспользуется между запросами; сравнение равное по кандидатам на запрос,',
        'но не по общему числу сгенерированных сетей на всей сетке.',
        'Исходный scalar_cost сохранён как source_scalar_cost. query_cost получен штатной scalarize',
        'из сохранённых нормированных целей со штрафами при новом α; физические метрики не пересчитывались.',
        'На исходных α проверено совпадение с сохранённым cost с допуском 2e-6.', '',
        '| Модель | K (0=greedy) | Сравнение | Средний выигрыш query cost | Запросов лучше |',
        '|---|---:|---|---:|---:|']
    for record in records:
        for k in [0]+list(ks):
            for comparator in ['fixed_0.5', 'archive']:
                rows = [r for r in record['gaps'] if r['K'] == k and r['comparator'] == comparator]
                text.append(f"| {record['label']} | {k} | {comparator} | {np.mean([r['relative_gain'] for r in rows]):.2%} | {np.mean([r['comparator_better'] for r in rows]):.2%} |")
    text += ['', '## Качество реакции и покрытие', '',
        '| Модель | K | Режим | HV выбранных ответов | Разных сетей на город | Доминируемых ответов внутри 0<α<1 |',
        '|---|---:|---|---:|---:|---:|']
    for record in records:
        for s in record['summary']:
            if s['K'] not in (0, max(ks)):
                continue
            interior = [r for r in record['selected'] if r['K'] == s['K'] and r['strategy'] == s['strategy']
                        and 0 < r['alpha'] < 1 and r['feasible']]
            fraction = np.mean([r['dominated_in_source_archive'] for r in interior])
            text.append(f"| {s['model']} | {s['K']} | {s['strategy']} | {s['hv']:.4f} | {s['distinct_networks']:.2f} | {fraction:.2%} |")
    rel = os.path.relpath(out, report.parent).replace('\\', '/')
    text += ['', 'Положительный выигрыш означает, что сравниваемый способ лучше conditioned.',
        'Столбец процентов — среднее относительных изменений по запросам, не относительное изменение общего среднего.',
        'Стоимость здесь измеряет соответствие запросу; ATT/RTT по каждому α и городу сохранены в CSV.', '',
        f'![Перенос между α]({rel}/cross_alpha.png)', '']
    for record in records:
        text += [f"![Реакция {record['label']}]({rel}/response_{record['label']}.png)", '']
    text += ['`matrix.csv`: оценка каждого α генерации под каждым α запроса по городам.',
        '`selected.csv`: выбранная сеть и её ATT/RTT; `by_alpha.csv`: физические средние по α.',
        '`gaps.csv`: парные разности; `summary.csv`: качество и HV каждого режима.',
        'При α=0,5 фиксированный и условный режимы обязаны совпасть; этот инвариант проверен.',
        'Доминирование проверено по допустимым сетям собственного архива того же K, с допуском 0,0001 минуты.',
        'Крайние α исключены из сводной доли доминируемых ответов: там одна из целей имеет нулевой вес.',
        'HV относится к выбранным ответам на сетке α, а не ко всему архиву. Переотбор не обязан увеличить этот HV.',
        'Разные α генерации использовали разные sampling seeds; stochastic-сравнение смешивает эффект весов и одного случайного пула.',
        'Greedy-контроль не содержит этой случайности; для stochastic-вывода нужны повторные пулы.',
        'Один training seed и один sampling-пул не устанавливают устойчивое преимущество.', '']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(text), encoding='utf-8')


def publish(out, report_path, records, *, complete, request):
    overall = [row for r in records for row in r['overall']]
    by_alpha = [row for r in records for row in r['by_alpha']]
    per_city = [row for r in records for row in r['per_city']]
    write_csv(out/'summary.csv', overall)
    write_csv(out/'by_alpha.csv', by_alpha)
    write_csv(out/'per_city.csv', per_city)
    paired = paired_comparisons(per_city)
    write_csv(out/'paired_differences.csv', paired)
    save_json(out/'results.json', dict(status='completed' if complete else 'partial', request=request,
                                      models=records, generated_at=datetime.now().astimezone().isoformat()))
    if overall:
        figures(out, overall, by_alpha, request['ks'])
        pareto_figures(out, per_city, request['test_indices'], request['ks'], complete=complete)
    text = ['# PPO и D3PO: реакция на α, Парето-фронты и best-of-K', '',
            '**Статус: ' + ('завершено' if complete else 'промежуточные результаты') + '.**', '',
            f"Городов: {request['graph_count']}; весов: {request['preference_count']}; seed sampling: {request['seed']}.",
            'Выбор сети выполняется по прежней стоимости с ограничениями при запрошенном α.',
            'ATT и RTT относятся к той же выбранной сети и выражены в минутах.', '',
            '| Модель | Режим | Cost ↓ | ATT, мин ↓ | RTT, мин ↓ | HV ↑ | Допустимость |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in overall:
        text.append(f"| {r['model']} | {r['mode']} | {r['cost']:.5f} | {r['ATT_minutes']:.2f} | {r['RTT_minutes']:.2f} | {r['hv']:.5f} | {r['feasible_fraction']:.2%} |")
    if not complete:
        text += ['', 'Промежуточные средние рассчитаны только по завершённым α.',
                 'Модели с разным числом завершённых α напрямую не сравнивать.']
    text += ['', 'Завершено α по моделям: ' + ', '.join(f"{r['label']}: {r['completed_alphas']}/{request['preference_count']}" for r in records) + '.',
             '', '## Сравнение при каждом α', '',
             'Общие средние по α являются вспомогательными: при разных весах выбираются разные компромиссы.',
             'Основное физическое сравнение приведено ниже при одинаковом α.', '',
             '| Модель | Режим | α | Cost ↓ | ATT, мин ↓ | RTT, мин ↓ |',
             '|---|---|---:|---:|---:|---:|']
    for r in by_alpha:
        if r['mode'] in ('greedy', f"best_of_{max(request['ks'])}"):
            text.append(f"| {r['model']} | {r['mode']} | {r['alpha']:.1f} | {r['cost']:.5f} | {r['ATT_minutes']:.2f} | {r['RTT_minutes']:.2f} |")
    image_path = os.path.relpath(out/'comparison.png', report_path.parent).replace('\\', '/')
    text += ['', f'![Сравнение стоимости и физических целей]({image_path})', '',
             'Полные таблицы: `summary.csv`, `by_alpha.csv`, `per_city.csv`, `paired_differences.csv` рядом с `results.json`.',
             'В paired_differences знак разности: кандидат минус первая модель; отрицательное значение означает меньшую стоимость/ATT/RTT.',
             'Кандидаты разных α не смешиваются; HV строится из одной выбранной сети на α.',
             'Пулы вложенные. Для каждого города/α проверяется невозрастание cost с ростом K.',
             'Greedy+K использует K+1 кандидатов и показан отдельным режимом.',
             'Время включает генерацию и оценку кандидатов; цифры для префиксов сняты в общем последовательном проходе.',
             'Один training seed и один общий пул sampling не устанавливают устойчивое ранжирование алгоритмов.', '']
    front_text = ['## Реакция на α и эмпирические Парето-фронты', '',
             'Каждая панель — отдельный город; разные города не объединяются в один фронт.',
             'Оси: RTT и ATT в минутах, обе цели минимизируются. Цвет точки задаёт α.',
             'Сплошная линия соединяет недоминируемые допустимые решения отдельно для каждой модели и режима;',
             'пунктир соединяет ответы в порядке α, включая доминируемые ответы. Линии не означают достижимость промежуточных сетей.',
             'Подписи перечисляют α одной и той же сети. Численная погрешность до 0,0001 минуты считается равенством.',
             'Рост α в идеальном случае уменьшает ATT и увеличивает RTT; локальные нарушения и плато вынесены в `alpha_response.csv`.',
             'Это эмпирические фронты найденных решений, не истинный Парето-фронт задачи.',
             'Best-of-K показывает совместный эффект условной политики и отбора по весам; greedy лучше отделяет прямую реакцию политики.', '']
    for mode in ['greedy'] + (["best_of_8"] if 8 in request['ks'] and max(request['ks']) != 8 else []) + [f"best_of_{max(request['ks'])}"]:
        prefix = os.path.relpath(out/f'pareto_{mode}', report_path.parent).replace('\\', '/')
        front_text += [f'![Парето и α: {mode}]({prefix}.png)', '']
        if complete:
            front_text += [f'[Все города: {mode}, PDF]({prefix}_all_cities.pdf)', '']
    text[8:8] = front_text + ['## Вспомогательное сравнение стоимости', '']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(text), encoding='utf-8')
    temporary.replace(report_path)


def route_structure(signature, minimum, maximum):
    routes = [tuple(route) for route in signature]
    lengths = [len(route) for route in routes]
    legs = [tuple(sorted((a, b))) for route in routes for a, b in zip(route, route[1:])]
    return dict(mean_route_length=float(np.mean(lengths)),
                duplicate_route_fraction=1-len(set(routes))/len(routes),
                repeated_leg_fraction=1-len(set(legs))/len(legs) if legs else 0.,
                min_length_route_fraction=float(np.mean(np.array(lengths) == minimum)),
                max_length_route_fraction=float(np.mean(np.array(lengths) == maximum)))


def front_evaluation(args):
    """Deterministic responses at a shared, explicit set of test cities/preferences.

    No stochastic candidates, best-of-K, training data or checkpoint selection.
    City fronts stay separate; common RTT normalization is an additional diagnostic.
    """
    out = resolve_under_root(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preferences_path = resolve_under_root(args.front_preferences)
    protocol = json.loads(preferences_path.read_text(encoding='utf-8'))
    alphas, indices = protocol['alphas'], protocol['test_indices']
    if not alphas or len(set(alphas)) != len(alphas) or not all(0 <= a <= 1 for a in alphas):
        raise ValueError('Front preferences must be unique values in [0,1]')
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError('Front test indices must be unique nonnegative integers')
    labels = args.labels or [Path(p).name for p in args.run_dirs]
    if len(labels) != len(args.run_dirs) or len(set(labels)) != len(labels):
        raise ValueError('Provide one unique label per source run')
    records, all_rows, all_metrics = [], [], []
    source_library = library_fingerprint()
    inputs = None
    for directory, label in zip(args.run_dirs, labels):
        run_dir = resolve_under_root(directory)
        cfg = OmegaConf.load(run_dir/'config.yaml')
        comparable = OmegaConf.to_container(OmegaConf.create(dict(cost=cfg.cost, common=cfg.common,
            env=cfg.env, dataset=cfg.dataset)), resolve=True)
        if inputs is not None and comparable != inputs:
            raise ValueError('Compare identical task geometry, data and cost within one front evaluation')
        inputs = comparable
        checkpoint = (resolve_under_root(args.weights_dir)/cfg.paths.checkpoint_path).with_name(args.checkpoint+'.pt')
        identity = dict(config_sha256=file_sha(run_dir/'config.yaml'), checkpoint_sha256=file_sha(checkpoint),
                        evaluator_sha256=file_sha(__file__), library_sha256=source_library,
                        preferences_sha256=file_sha(preferences_path), batch_size=args.batch_size,
                        checkpoint=str(checkpoint), seed=int(cfg.run.seed))
        with open_dict(cfg):
            cfg.evaluation.batch_size = args.batch_size
            cfg.paths.output_dir = str(out/'setup'/label)
            cfg.paths.init_checkpoint_path = checkpoint.as_posix()
            cfg.run.cpu = True
        actor = TrainingRun(cfg)
        actor.setup()
        torch.set_num_threads(args.cpu_threads)
        actor.policy.eval()
        if tuple(actor.cost.objectives) != ('ATT', 'RTT'):
            raise ValueError('Front evaluation requires ATT/RTT in that order')
        before = {key: value.clone() for key, value in actor.policy.state_dict().items()}
        params = actor.data_input.params
        raw = evaluation_graphs(params.test_path, params.kind, params.limits)
        if max(indices) >= len(raw):
            raise ValueError('Front test index is out of range')
        graphs = [actor.dataset.evaluation_pipeline.graph(raw[i]) for i in indices]
        if args.dry_run:
            logging.info('front dry-run OK: %s, %d cities x %d preferences', label, len(indices), len(alphas))
            del actor, before, graphs, raw
            gc.collect()
            continue
        cache = out/'candidates'/f'{label}.json'
        fingerprint = digest(identity)
        prediction = None
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['fingerprint'] != fingerprint:
                raise ValueError('Front cache protocol or source changed')
            rows = saved['rows']
        else:
            rows = []
            queries = [(g, float(alpha)) for alpha in alphas for g in range(len(graphs))]
            with torch.random.fork_rng():
                torch.manual_seed(args.seed)
                for start in range(0, len(queries), args.batch_size):
                    chunk = queries[start:start+args.batch_size]
                    weights = dict(demand_time_weight=[a for _, a in chunk],
                                   route_time_weight=[1-a for _, a in chunk], median_connectivity_weight=0.)
                    prediction = predict(actor.policy, [graphs[g] for g, _ in chunk], weights, actor.cost,
                                         type(actor.env), actor.env_params, 'greedy', 1)
                    produced = prediction_rows(prediction, 0, 0., 0)
                    for row, (g, alpha) in zip(produced, chunk):
                        row.update(graph=g, preference=alpha)
                    rows.extend(produced)
                    save_json(out/'progress.json', dict(status='running', model=label,
                        completed_queries=len(rows), total_queries=len(queries), completed_models=len(records)))
                    logging.info('front %s: %d/%d queries', label, len(rows), len(queries))
            save_json(cache, dict(fingerprint=fingerprint, identity=identity, rows=rows))
        if len(rows) != len(alphas)*len(indices) or len({(r['graph'], r['preference']) for r in rows}) != len(rows):
            raise ValueError('Incomplete or duplicate front queries')
        for key, value in actor.policy.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)
        if file_sha(checkpoint) != identity['checkpoint_sha256'] or library_fingerprint() != source_library:
            raise ValueError('Front source changed during evaluation')
        metrics = summarize_evaluation(rows, [3., 3.], 1001)
        route_count = actor.env_params.n_routes
        if not isinstance(route_count, int):
            raise ValueError('Front comparison requires a fixed route count')
        common_hv = []
        for graph_record in metrics['per_graph']:
            city_rows = [r for r in rows if r['graph'] == graph_record['graph'] and r['feasible']]
            points = np.array([[r['objective_0'], r['objective_1']*route_count/10] for r in city_rows]).reshape(-1, 2)
            hv = hypervolume_2d(points, [3., 3.])
            common_hv.append(hv)
            all_metrics.append(dict(model=label, seed=identity['seed'], dataset_index=indices[graph_record['graph']],
                                    hv_common_rtt_reference10=hv, **graph_record['metrics']))
        structures = []
        for row in rows:
            structure = route_structure(row['network_signature'], actor.env_params.min_route_len, actor.env_params.max_route_len)
            structures.append(structure)
            all_rows.append(dict(model=label, seed=identity['seed'], dataset_index=indices[row['graph']],
                alpha=row['preference'], query_cost=row['scalar_cost'], feasible=row['feasible'],
                ATT_minutes=row['ATT'], RTT_minutes=row['RTT'], **structure,
                network_id=digest(row['network_signature'])[:16]))
        summary = metrics['summary']
        summary.update(hv_common_rtt_reference10=float(np.mean(common_hv)),
                       **{key: float(np.mean([r[key] for r in structures])) for key in structures[0]})
        records.append(dict(label=label, identity=identity, n_routes=route_count,
                            min_route_len=actor.env_params.min_route_len, max_route_len=actor.env_params.max_route_len,
                            summary=summary, policy_state_unchanged=True))
        del actor, before, graphs, raw, prediction, rows
        gc.collect()
    if args.dry_run:
        return
    save_json(out/'results.json', dict(protocol=protocol, models=records))
    write_csv(out/'per_query.csv', all_rows)
    write_csv(out/'per_city.csv', all_metrics)
    write_csv(out/'summary.csv', [dict(model=r['label'], seed=r['identity']['seed'], **r['summary']) for r in records])
    front_rows = [dict(r, graph=indices.index(r['dataset_index']), mode='greedy',
                       network_id=r['network_id']) for r in all_rows]
    # The shared plotter uses the explicit alpha sequence and physical per-city outcomes.
    pareto_figures(out, front_rows, indices, [1], complete=True)
    text = ['# Детерминированные фронты PPO и D3PO', '',
        f'Городов: {len(indices)}; весов на город: {len(alphas)}; ответов на модель: {len(indices)*len(alphas)}.',
        f'Checkpoint: {args.checkpoint}. Все модели оценены при одинаковых городах и весах.',
        'Один greedy-ответ на запрос; best-of-K и архивный переотбор при генерации не применяются.', '',
        '| Модель | Seed | HV ↑ | EU ответа ↑ | EU архива ↑ | Разных НД сетей | Feasible |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        s = r['summary']
        text.append(f"| {r['label']} | {r['identity']['seed']} | {s['hv_feasible']:.5f} | {s['eu_conditioned_penalized']:.5f} | {s['eu_archive_penalized']:.5f} | {s['distinct_pareto_networks']:.2f} | {s['feasible_fraction']:.2%} |")
    rel = os.path.relpath(out, resolve_under_root(args.report).parent).replace('\\', '/')
    text += ['', f'![ATT–RTT: первые четыре города]({rel}/pareto_greedy.png)', '',
        f'[Все города, PDF]({rel}/pareto_greedy_all_cities.pdf)', '',
        'HV и фронты рассчитываются по отдельным городам, затем усредняются.',
        'EU ответа относится к запрошенным весам; EU архива использует независимую сетку 1001 весов.',
        'При изменении числа маршрутов основной нормированный RTT меняет масштаб. Дополнительный',
        'hv_common_rtt_reference10 использует общий знаменатель, соответствующий 10 маршрутам.',
        'Физические ATT/RTT и повторы маршрутов/рёбер доступны в per_query.csv.',
        'Эта таблица содержит отдельные обученные модели; статистическое сравнение требует всех training seeds.', '']
    report = resolve_under_root(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(text), encoding='utf-8')
    save_json(out/'progress.json', dict(status='completed', completed_models=len(records)))


def run(args):
    out = resolve_under_root(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(args.ks))
    if min(ks) < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError('K, batch size and CPU thread count must be positive')
    if args.graph_count is not None and args.graph_count < 1:
        raise ValueError('graph-count must be positive')
    labels = args.labels or [Path(p).name for p in args.run_dirs]
    if len(labels) != len(args.run_dirs) or len(set(labels)) != len(labels):
        raise ValueError('Provide one unique label per source run')
    protocols = [json.loads((resolve_under_root(p)/'run.json').read_text(encoding='utf-8'))['evaluation_protocol']['test']
                 for p in args.run_dirs]
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError('Source test protocols differ')
    indices = protocols[0]['indices'][:args.graph_count] if args.graph_count else protocols[0]['indices']
    count = args.preference_count or protocols[0]['preference_count']
    if not indices or count < 2:
        raise ValueError('Need cities and >=2 preferences')
    request = dict(run_dirs=args.run_dirs, labels=labels, checkpoint=args.checkpoint, ks=ks,
                   graph_count=len(indices), test_indices=indices, preference_count=count,
                   seed=args.seed, batch_size=args.batch_size, cpu_threads=args.cpu_threads,
                   selection='minimum original penalized scalar cost at the requested alpha')
    save_json(out/'request.json', request)
    records = []
    source_library = library_fingerprint()
    evaluation_inputs = None
    preference_records = []
    try:
        for run_dir, label in zip(args.run_dirs, labels):
            run_dir = resolve_under_root(run_dir)
            cfg = OmegaConf.load(run_dir/'config.yaml')
            inputs = OmegaConf.to_container(OmegaConf.create(dict(cost=cfg.cost, common=cfg.common,
                env=cfg.env, test_path=cfg.dataset.test_path, evaluation_pipeline=cfg.dataset.evaluation_pipeline)), resolve=True)
            if evaluation_inputs is not None and inputs != evaluation_inputs:
                raise ValueError('Source cost, environment or test preparation differs')
            evaluation_inputs = inputs
            checkpoint = resolve_under_root(args.weights_dir)/cfg.paths.checkpoint_path
            checkpoint = checkpoint.with_name(args.checkpoint+'.pt')
            checkpoint_sha = file_sha(checkpoint)
            with open_dict(cfg):
                cfg.evaluation.batch_size = args.batch_size
                cfg.paths.output_dir = str(out/'setup'/label)
                cfg.paths.init_checkpoint_path = checkpoint.as_posix()
                cfg.run.cpu = True
            actor = TrainingRun(cfg)
            actor.setup()
            if actor.env_name != 'construction' or len(actor.cost.objectives) != 2:
                raise ValueError('This evaluator requires two-objective construction policies')
            torch.set_num_threads(args.cpu_threads)
            actor.policy.eval()
            before = {key: value.clone() for key, value in actor.policy.state_dict().items()}
            params = actor.data_input.params
            raw = evaluation_graphs(params.test_path, params.kind, params.limits)
            graphs = [actor.dataset.evaluation_pipeline.graph(raw[i]) for i in indices]
            identity = dict(request=request, model=label, checkpoint_sha256=checkpoint_sha,
                            config_sha256=file_sha(run_dir/'config.yaml'),
                            evaluator_sha256=file_sha(__file__),
                            library_sha256=source_library, evaluation_inputs=inputs,
                            graph_sources=[str(getattr(graph, 'source_id', index)) for index, graph in zip(indices, graphs)])
            fingerprint = digest(identity)
            if args.dry_run:
                logging.info('dry-run OK: %s, %d cities, K=%s, checkpoint=%s', label, len(graphs), ks, checkpoint)
                del actor, raw, graphs, before
                gc.collect()
                continue
            pools = []
            prediction = None
            for pref_index, alpha in enumerate(np.linspace(0., 1., count)):
                cache = out/'candidates'/label/f'alpha_{pref_index:03d}.json.gz'
                if cache.exists():
                    with gzip.open(cache, 'rt', encoding='utf-8') as stream:
                        pool = json.load(stream)
                    if pool['fingerprint'] != fingerprint:
                        raise ValueError(f'Cached protocol/checkpoint differs: {cache}')
                    logging.info('Reusing %s alpha %.3f', label, alpha)
                else:
                    weights = dict(demand_time_weight=float(alpha), route_time_weight=float(1-alpha), median_connectivity_weight=0.)
                    cost = actor.cost.with_params(**weights)
                    pool = dict(fingerprint=fingerprint, alpha_index=pref_index, alpha=float(alpha),
                                greedy=[], sampled=[], prefix_seconds={})
                    with torch.random.fork_rng():
                        torch.manual_seed(args.seed + pref_index*1000)
                        started = time.monotonic()
                        for start in range(0, len(graphs), args.batch_size):
                            prediction = predict(actor.policy, graphs[start:start+args.batch_size], weights, cost,
                                                 type(actor.env), actor.env_params, 'greedy', 1)
                            pool['greedy'].extend(prediction_rows(prediction, start, alpha, 0))
                        pool['greedy_seconds'] = time.monotonic()-started
                        started = time.monotonic()
                        for sample in range(max(ks)):
                            for start in range(0, len(graphs), args.batch_size):
                                prediction = predict(actor.policy, graphs[start:start+args.batch_size], weights, cost,
                                                     type(actor.env), actor.env_params, 'sample', 1)
                                pool['sampled'].extend(prediction_rows(prediction, start, alpha, sample))
                            if sample+1 in ks:
                                pool['prefix_seconds'][str(sample+1)] = time.monotonic()-started
                            if (sample+1) % 8 == 0 or sample+1 == max(ks):
                                save_json(out/'progress.json', dict(status='running', model=label, alpha=float(alpha),
                                    completed_alphas=pref_index, completed_samples_at_alpha=sample+1,
                                    max_k=max(ks), model_count=len(args.run_dirs), completed_models=len(records),
                                    timestamp=datetime.now().astimezone().isoformat()))
                                logging.info('%s alpha=%.2f K=%d/%d elapsed=%.1fs', label, alpha, sample+1,
                                             max(ks), time.monotonic()-started)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix('.gz.tmp')
                    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                        json.dump(pool, stream, allow_nan=False)
                    temporary.replace(cache)
                if len(pool['sampled']) != len(indices)*max(ks) or len(pool['greedy']) != len(indices):
                    raise ValueError('Incomplete candidate cache')
                pools.append(pool)
                overall, by_alpha, per_city = summarize_pool(label, pools, ks)
                partial = dict(label=label, identity=identity, completed_alphas=len(pools),
                               overall=overall, by_alpha=by_alpha, per_city=per_city)
                publish(out, resolve_under_root(args.report), records+[partial], complete=False, request=request)
            for key, value in actor.policy.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)
            if file_sha(checkpoint) != checkpoint_sha:
                raise ValueError('Source checkpoint changed during evaluation')
            if library_fingerprint() != source_library:
                raise ValueError('Library source changed during evaluation')
            records.append(dict(label=label, identity=identity, completed_alphas=count,
                                policy_state_unchanged=True, overall=overall, by_alpha=by_alpha, per_city=per_city))
            publish(out, resolve_under_root(args.report), records, complete=False, request=request)
            if any(abs(pool['alpha']-.5) < 1e-8 for pool in pools):
                preference_records.append(preference_tests(label,
                    [r for pool in pools for r in pool['greedy']],
                    [r for pool in pools for r in pool['sampled']], ks, indices))
                preference_report = resolve_under_root(args.report).with_name(Path(args.report).stem+'-preferences.md')
                publish_preference_tests(out/'preference_tests', preference_report, preference_records, ks, indices)
            del actor, raw, graphs, before, pools, prediction
            gc.collect()
        if not args.dry_run:
            publish(out, resolve_under_root(args.report), records, complete=True, request=request)
            save_json(out/'progress.json', dict(status='completed', completed_models=len(records),
                                               timestamp=datetime.now().astimezone().isoformat()))
    except BaseException as error:
        try:
            save_json(out/'progress.json', dict(status='failed', error=repr(error),
                                               timestamp=datetime.now().astimezone().isoformat()))
        except Exception:
            logging.exception('Could not persist evaluator failure metadata')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+')
    parser.add_argument('--weights-dir')
    parser.add_argument('--candidate-files', nargs='+', help='Offline analysis: LABEL=JSON with greedy and sampled lists')
    parser.add_argument('--test-indices', nargs='+', type=int, help='Dataset indices for offline candidate files')
    parser.add_argument('--front-preferences', help='Deterministic front protocol JSON with alphas and test_indices')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    parser.add_argument('--ks', nargs='+', type=int, required=True)
    parser.add_argument('--graph-count', type=int)
    parser.add_argument('--preference-count', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9001)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log')
    args = parser.parse_args(argv)
    handlers = [logging.StreamHandler()]
    if args.log:
        log_path = resolve_under_root(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    if args.candidate_files:
        if args.run_dirs or not args.test_indices or args.dry_run:
            parser.error('Offline analysis requires --test-indices and no --run-dirs/--dry-run')
        records, manifest = [], []
        if min(args.ks) < 1:
            parser.error('K must be positive')
        ks = sorted(set(args.ks))
        torch.set_num_threads(args.cpu_threads)
        for entry in args.candidate_files:
            label, separator, path = entry.partition('=')
            if not separator or not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
                parser.error('Use LABEL=FILE, with a simple alphanumeric label')
            path = resolve_under_root(path)
            data = json.loads(path.read_text(encoding='utf-8'))
            records.append(preference_tests(label, data['greedy'], data['sampled'], ks, args.test_indices))
            manifest.append(dict(label=label, path=str(path), sha256=file_sha(path)))
        if len({r['label'] for r in records}) != len(records):
            parser.error('Duplicate labels')
        out = resolve_under_root(args.out_dir)
        publish_preference_tests(out, resolve_under_root(args.report), records, ks, args.test_indices)
        save_json(out/'sources.json', manifest)
        logging.info('Offline preference tests completed for %d models', len(records))
    else:
        if not args.run_dirs or not args.weights_dir:
            parser.error('Inference requires --run-dirs and --weights-dir')
        if args.front_preferences:
            front_evaluation(args)
        else:
            run(args)


if __name__ == '__main__':
    main()

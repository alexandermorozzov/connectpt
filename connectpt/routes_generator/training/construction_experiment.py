"""Construction test evaluation and artifact helpers; training lives in setup_and_train."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ..run.artifacts import ArtifactStore
from ..run.logger import flush, log
from ..eval_route_generator import eval_model
from ..transit_time_estimator import RouteGenBatchState
from .construction_metrics import canonical_network, summarize_evaluation
from .records import ParetoRecord


def evaluate_test(model, data, cfg, cost, output, device, *, checkpoint='last',
                  mode='greedy', preference_count=None, stochastic_samples=None,
                  save_artifacts=True):
    artifacts = ArtifactStore(output)
    rows, networks = [], []
    count = int(preference_count if preference_count is not None else cfg.evaluation.preference_count)
    samples = 1 if mode == 'greedy' else int(stochastic_samples if stochastic_samples is not None
                                          else cfg.evaluation.stochastic_samples)
    if mode not in ('greedy', 'stochastic') or count < 2 or samples < 1 or not len(data):
        raise ValueError('Evaluation needs data, >=2 preferences and >=1 sample in a supported mode')
    for pref_idx, preference in enumerate(np.linspace(0, 1, count)):
        fixed_cost = copy.deepcopy(cost)
        fixed_cost.set_weights(demand_time_weight=float(preference),
                               route_time_weight=float(1 - preference), median_connectivity_weight=0.)
        for sample in range(samples):
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(cfg.evaluation.seed) + pref_idx * 1000 + sample)
                for start in range(0, len(data), int(cfg.batch_size)):
                    chunk = [data[i] for i in range(start, min(start + int(cfg.batch_size), len(data)))]
                    # Reuse the original evaluator. Repeated n_samples=1 gives
                    # independent draws without the multi-sample state slicing
                    # limitation in sample_from_model; no best-of-N selection.
                    loader = DataLoader(chunk, batch_size=int(cfg.batch_size))
                    _, _, _, routes = eval_model(model, loader, cfg.eval, fixed_cost,
                        n_samples=None if mode == 'greedy' else 1,
                        return_routes=True, silent=True, device=device)
                    state = RouteGenBatchState(Batch.from_data_list(chunk).to(device), fixed_cost,
                        int(cfg.eval.n_routes), int(cfg.eval.min_route_len), int(cfg.eval.max_route_len))
                    state.add_new_routes(routes.to(device))
                    result = fixed_cost(state)
                    pure = fixed_cost.get_cost_components(state, result, include_constraint=False)[:, :2]
                    penalized = fixed_cost.get_cost_components(state, result)[:, :2]
                    feasible = ~result.are_constraints_violated()
                    for i in range(len(chunk)):
                        route_list = [r.cpu().tolist() for r in state.routes[i]]
                        rows.append(dict(graph=start + i, preference=float(preference), sample=sample,
                            att_minutes=float(result.mean_demand_time[i] / 60),
                            rtt_minutes=float(result.total_route_time[i] / 60),
                            att_norm=float(pure[i, 0]), rtt_norm=float(pure[i, 1]),
                            penalized_att=float(penalized[i, 0]), penalized_rtt=float(penalized[i, 1]),
                            scalar_cost=float(result.cost[i]), feasible=bool(feasible[i]),
                            unserved_fraction=float(result.unserved_demand[i] / result.total_demand[i]),
                            network_signature=canonical_network(route_list, cfg.experiment.symmetric_routes)))
                        if save_artifacts:
                            networks.append(dict(graph=start + i, preference=float(preference), sample=sample,
                                                 routes=route_list))
    suffix = '' if checkpoint == 'last' and mode == 'greedy' else f'_{checkpoint}_{mode}'
    if save_artifacts:
        artifacts.save_json({'rows': rows,
               'checkpoint': checkpoint, 'mode': mode, 'samples_per_preference': samples,
               'hv_reference': list(cfg.evaluation.get('hv_reference', [3., 3.])),
               'eu_preference_count': int(cfg.evaluation.get('eu_preference_count', 101)),
               'protocol': dict(graph_count=len(data), preference_count=count,
                                seed=int(cfg.evaluation.seed), n_routes=int(cfg.eval.n_routes),
                                min_route_len=int(cfg.eval.min_route_len), max_route_len=int(cfg.eval.max_route_len)),
               'mean_cost': float(np.mean([r['scalar_cost'] for r in rows]))},
               f'test_evaluation{suffix}.json')
        artifacts.save_json(networks, f'test_routes{suffix}.json')
    return rows


def log_pareto_metrics(rows, cfg, logger_list, step, prefix):
    report = summarize_evaluation(rows, cfg.evaluation.hv_reference,
                                  int(cfg.evaluation.get('eu_preference_count', 101)))
    summary = {k: float('nan') if v is None else v for k, v in report['summary'].items()}
    reference_att, reference_rtt = cfg.evaluation.hv_reference
    log(logger_list, ParetoRecord(**summary, reference_att=reference_att,
                                  reference_rtt=reference_rtt), step, prefix=prefix)
    return report


def evaluate_pareto(model, data, cfg, cost, output, device, logger_list, step):
    """Fixed held-out cities/preferences/action seeds, isolated from training RNG."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    artifacts = ArtifactStore(output)
    for mode in ('greedy', 'stochastic'):
        started = time.monotonic()
        rows = evaluate_test(model, data, cfg, cost, output, device, mode=mode,
            preference_count=int(cfg.evaluation.pareto_preference_count),
            stochastic_samples=int(cfg.evaluation.pareto_stochastic_samples), save_artifacts=False)
        report = log_pareto_metrics(rows, cfg, logger_list, step, f'pareto/{mode}')
        report['evaluation_seconds'] = time.monotonic()-started
        log(logger_list, ParetoRecord(evaluation_seconds=report['evaluation_seconds']), step,
            prefix=f'pareto/{mode}')
        example = [r for r in rows if r['graph'] == 0 and r['sample'] == 0]
        report['example_front'] = example
        artifacts.save_json(report, f'pareto/iter{step:05d}_{mode}.json')
        if cfg.evaluation.get('log_pareto_front', True):
            from .construction_metrics import nondominated
            fig = Figure(figsize=(5, 4))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            points = np.array([[r['att_norm'], r['rtt_norm']] for r in example])
            feasible = np.array([r['feasible'] for r in example])
            scatter = ax.scatter(points[:, 0], points[:, 1], c=[r['preference'] for r in example],
                                 vmin=0, vmax=1, cmap='viridis', label='Generated networks')
            front = nondominated(points[feasible])
            if len(front):
                front = front[np.argsort(front[:, 0])]
                ax.plot(front[:, 0], front[:, 1], 'k.-', label='Feasible Pareto front')
            ax.set(xlabel='Normalized ATT', ylabel='Normalized RTT',
                   title=f'Validation city 0, {mode}, iteration {step}')
            ax.legend(loc='best')
            fig.colorbar(scatter, ax=ax, label='Passenger weight alpha')
            fig.tight_layout()
            for logger in logger_list:
                logger.figure(f'pareto/{mode}/front_city_0', fig, step)
    flush(logger_list)


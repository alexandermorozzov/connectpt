"""Construction test evaluation and artifact helpers; training lives in setup_and_train."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..core.artifacts import ArtifactStore
from ..eval_route_generator import eval_model
from ..transit_time_estimator import RouteGenBatchState


class MetricWriter(SummaryWriter):
    """Keep the original TensorBoard tags and mirror their latest values to CSV."""
    def __init__(self, path):
        super().__init__(str(path))
        self.latest = {}

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self.latest[tag] = float(scalar_value)
        return super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)


def state_dict_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_test(model, data, cfg, cost, output, device, *, checkpoint='last',
                  mode='greedy'):
    artifacts = ArtifactStore(output)
    rows, networks = [], []
    samples = 1 if mode == 'greedy' else int(cfg.evaluation.stochastic_samples)
    for pref_idx, preference in enumerate(np.linspace(0, 1, int(cfg.evaluation.preference_count))):
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
                        rows.append(dict(graph=start + i, preference=float(preference), sample=sample,
                            att_minutes=float(result.mean_demand_time[i] / 60),
                            rtt_minutes=float(result.total_route_time[i] / 60),
                            att_norm=float(pure[i, 0]), rtt_norm=float(pure[i, 1]),
                            penalized_att=float(penalized[i, 0]), penalized_rtt=float(penalized[i, 1]),
                            scalar_cost=float(result.cost[i]), feasible=bool(feasible[i]),
                            unserved_fraction=float(result.unserved_demand[i] / result.total_demand[i])))
                        networks.append(dict(graph=start + i, preference=float(preference), sample=sample,
                            routes=[r.cpu().tolist() for r in state.routes[i]]))
    suffix = '' if checkpoint == 'last' and mode == 'greedy' else f'_{checkpoint}_{mode}'
    artifacts.save_json({'rows': rows,
               'checkpoint': checkpoint, 'mode': mode, 'samples_per_preference': samples,
               'mean_cost': float(np.mean([r['scalar_cost'] for r in rows]))},
               f'test_evaluation{suffix}.json')
    artifacts.save_json(networks, f'test_routes{suffix}.json')
    return rows


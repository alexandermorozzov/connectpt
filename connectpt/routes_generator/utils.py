"""Legacy search evaluation adapter, removed with the colony migration."""
import logging as log
import time

import torch
from tqdm import tqdm

from .env.state import RouteGenBatchState
from .initialization import init_from_cfg


def test_method(method_fn, dataloader, eval_cfg, init_cfg, cost_obj,
                logger_list=(), silent=False, return_routes=False,
                device=None, routes_tensor=None, return_histories=False,
                *method_args, **method_kwargs):
    if method_fn is not None:
        log.debug(f"evaluating {method_fn.__name__} on dataset")
    cost_histories = []
    final_costs = []
    all_metrics = None

    try:
        _single_sample = len(dataloader) <= 1
    except TypeError:
        _single_sample = False
    # Hide the per-sample outer bar when there is only one graph (the useless
    # "0/1"); the algorithm's own per-iteration tqdm shows real progress.
    for data in tqdm(dataloader, disable=silent or _single_sample):
        if device is not None and device.type != 'cpu':
            data = data.cuda()
        
        start_time = time.time()

        state = RouteGenBatchState(data, cost_obj.constants, eval_cfg.n_routes,
                                   eval_cfg.min_route_len, 
                                   eval_cfg.max_route_len, preferences=cost_obj.get_weights())

        init_network = init_from_cfg(state, init_cfg, routes_tensor)
        assert init_network is None or init_network.shape[1] <= eval_cfg.n_routes, \
            "initial solution has wrong number of routes "\
            f"{init_network.shape[1]}, should be at most {eval_cfg.n_routes}"

        if method_fn is not None:
            state, cost_history = method_fn(state, cost_obj, silent=silent,
                                            init_network=init_network,
                                            logger_list=logger_list, *method_args, 
                                            **method_kwargs)
        else:
            # just evaluate the initial solution
            state.add_new_routes(init_network)
            cost_history = None

        duration = time.time() - start_time
        if cost_history is not None:
            n_iters = cost_history.shape[1]
            cost_histories.append(cost_history)
            final_costs.append(cost_history[:, -1])
        else:
            final_costs.append(cost_obj(state).cost)
            n_iters = 0

        # compute the metrics for this iteration
        result = cost_obj(state)
        metrics = result.get_metrics()
        metrics['average wall-clock duration'] = \
            torch.full_like(metrics['ATT'], duration)
        metrics['average # iterations'] = \
            torch.full_like(metrics['ATT'], n_iters)
        # metrics['demand weight'] = cost_obj.demand_time_weight
        # metrics['route weight'] = cost_obj.route_time_weight
        # metrics['unserved weight'] = cost_obj.unserved_weight

        # add this iteration's metrics to the collection of metrics        
        if all_metrics is None:
            all_metrics = metrics
        else:
            for key, val in metrics.items():
                all_metrics[key] = torch.cat((all_metrics[key], val))

        # Print CSV for each element if csv flag is set and average_metrics is False
        if not silent and eval_cfg.csv and not eval_cfg.get('average_metrics', True):
            # Print metrics for each element in the current batch
            batch_size = next(iter(metrics.values())).shape[0]
            for i in range(batch_size):
                parts = [eval_cfg.n_routes]
                for stat in metrics.values():
                    parts.append(stat[i].item())
                csv_row = ',' + ','.join([f"{pp:.3f}" for pp in parts])
                print(csv_row)

    # compute some aggregate statistics
    final_costs = torch.cat(final_costs)
    mean_metrics = {key: val.mean() for key, val in all_metrics.items()}
    if not silent:
        if eval_cfg.csv and eval_cfg.get('average_metrics', True):
            # print averaged statistics in csv format
            parts = [eval_cfg.n_routes]
            for stat in all_metrics.values():
                parts.append(stat.mean().item())
                if stat.numel() > 1: 
                    parts.append(stat.std().item())
            csv_row = ',' + ','.join([f"{pp:.3f}" for pp in parts])
            print(csv_row)
        elif not eval_cfg.csv:
            # print overall statistics normally
            print(f"average cost: {final_costs.mean():.3f}")
            for name, stat_value in mean_metrics.items():
                print(f"{name}: {stat_value:.3f}")

    unserved_demand = cost_obj(state).unserved_demand_matrix
    # unbiased=False avoids the "degrees of freedom <= 0" warning when a single
    # network is evaluated (batch_size 1); population std of one sample is 0.
    out_stats = (final_costs.mean(), final_costs.std(unbiased=False),
                 unserved_demand, all_metrics)
    if return_routes:
        out_stats = out_stats + (state.routes,)
    if return_histories:
        out_stats = out_stats + (cost_histories,)
    return out_stats

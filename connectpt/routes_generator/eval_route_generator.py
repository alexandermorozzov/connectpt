import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging as log

import torch
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
import hydra

from torch_geometric.loader import DataLoader
from .citygraph_dataset import CityGraphData, \
    get_dataset_from_config, STOP_KEY
from .transit_time_estimator import RouteGenBatchState
from . import utils as lrnu
from .initialization import prepare_current_routes, prepare_init_network
from .torch_utils import get_batch_tensor_from_routes, dump_routes


def sample_from_model(model, state, cost_obj, n_samples=20, 
                      sample_batch_size=None, init_network=None,
                      init_current_routes=None, revisit_routes=None):
    model.eval()
    # duplicate the state across the samples
    graph_data = state.graph_data
    data_list = graph_data.to_data_list()
    flat_sample_inputs = data_list * n_samples

    if sample_batch_size is None:
        if state.batch_size > 1:
            sample_batch_size = state.batch_size
        else:
            sample_batch_size = n_samples

    all_plans = []
    all_costs = []
    sampled_init_network = None
    sampled_current_routes = None
    sampled_revisit_routes = None
    if init_network is not None:
        sampled_init_network = init_network.repeat(n_samples, 1, 1)
    if init_current_routes is not None:
        prepared_current_routes = prepare_current_routes(
            init_current_routes,
            batch_size=state.batch_size,
            max_n_nodes=state.max_n_nodes,
            device=state.device,
        )
        sampled_current_routes = prepared_current_routes.repeat(n_samples, 1)
    if revisit_routes is not None:
        sampled_revisit_routes = revisit_routes.repeat(n_samples, 1, 1)

    for ii in range(0, len(flat_sample_inputs), sample_batch_size):
        chunk = flat_sample_inputs[ii:ii+sample_batch_size]
        n_routes = state.n_routes_to_plan
        if state.batch_size == 1:
            min_route_len = state.min_route_len.squeeze()
            max_route_len = state.max_route_len.squeeze()
        else:
            # TODO this doesn't actually work right now, the indexing will go
             # past the end of the tensor because it's only as long as one 
             # batch.
            min_route_len = state.min_route_len[ii:ii+sample_batch_size]
            max_route_len = state.max_route_len[ii:ii+sample_batch_size]
        batch_state = RouteGenBatchState(chunk, cost_obj, 
                                         state.n_routes_to_plan,
                                         min_route_len, max_route_len)
        if sampled_init_network is not None:
            init_chunk = prepare_init_network(
                sampled_init_network[ii:ii+sample_batch_size],
                batch_size=batch_state.batch_size,
                n_routes=int(batch_state.n_routes_to_plan.min().item()),
                device=batch_state.device,
            )
            if init_chunk.shape[1] > 0:
                batch_state.add_new_routes(init_chunk)
        if sampled_current_routes is not None:
            current_chunk = prepare_current_routes(
                sampled_current_routes[ii:ii+sample_batch_size],
                batch_size=batch_state.batch_size,
                max_n_nodes=batch_state.max_n_nodes,
                device=batch_state.device,
            )
            if (current_chunk > -1).any():
                batch_state.set_current_routes(current_chunk)
        if sampled_revisit_routes is not None:
            revisit_chunk = prepare_init_network(
                sampled_revisit_routes[ii:ii+sample_batch_size],
                batch_size=batch_state.batch_size,
                n_routes=int(batch_state.n_routes_to_plan.min().item()),
                device=batch_state.device,
            )
            if revisit_chunk.shape[1] > 0:
                batch_state = _revisit_seeded_routes(
                    model,
                    batch_state,
                    revisit_chunk,
                    greedy=False,
                )
        if batch_state.is_done().all():
            all_plans += batch_state.routes
            all_costs.append(cost_obj(batch_state).cost)
            continue
        with torch.no_grad():
            plan_out = model(batch_state, greedy=False)
            batch_costs = cost_obj(plan_out.state).cost
        all_plans += plan_out.state.routes
        all_costs.append(batch_costs)

    all_costs = torch.cat(all_costs, dim=0).reshape(n_samples, -1)

    # # plot a histogram of the costs, with 1/10th as many bins as samples
    # import matplotlib.pyplot as plt
    # plt.hist(all_costs.cpu().numpy().flatten(), bins= n_samples // 1)
    # plt.savefig("cost_hist.png")
    # print(f"Average cost of samples: {all_costs.mean()}")

    _, min_indices = all_costs.min(0)
    batch_size = len(flat_sample_inputs) // n_samples
    best_plans = [all_plans[mi * batch_size + ii] \
                  for ii, mi in enumerate(min_indices)]
    best_plans_tensor = get_batch_tensor_from_routes(best_plans)
    state.add_new_routes(best_plans_tensor)
    return state


def _seed_state_with_init_network(state, init_network, init_current_routes=None):
    if init_network is not None:
        init_network = prepare_init_network(
            init_network,
            batch_size=state.batch_size,
            n_routes=int(state.n_routes_to_plan.min().item()),
            device=state.device,
        )
        if init_network.shape[1] > 0:
            state.add_new_routes(init_network)

    if init_current_routes is None:
        return state

    init_current_routes = prepare_current_routes(
        init_current_routes,
        batch_size=state.batch_size,
        max_n_nodes=state.max_n_nodes,
        device=state.device,
    )
    if (init_current_routes > -1).any():
        state.set_current_routes(init_current_routes)
    return state


def _revisit_seeded_routes(model, state, revisit_routes, greedy=False):
    if revisit_routes is None:
        return state

    if not hasattr(model, "plan_new_route"):
        raise ValueError(
            "This model does not support revisiting seeded routes"
        )

    revisit_routes = prepare_init_network(
        revisit_routes,
        batch_size=state.batch_size,
        n_routes=int(state.n_routes_to_plan.min().item()),
        device=state.device,
    )
    if revisit_routes.shape[1] == 0:
        return state

    for route_idx in range(revisit_routes.shape[1]):
        route_batch = revisit_routes[:, route_idx]
        has_route = (route_batch > -1).any(dim=-1)
        if not has_route.any():
            continue
        if not has_route.all():
            raise NotImplementedError(
                "Revisiting seeded routes with uneven route counts across "
                "the batch is not supported"
            )
        state.set_current_routes(route_batch)
        model.plan_new_route(state, greedy=greedy)

    return state


def eval_model(model, eval_dataloader, eval_cfg, cost_obj, sum_writer=None, 
               iter_num=0, n_samples=None, silent=False, 
               sample_batch_size=None, return_routes=False, device=None,
               init_cfg=None, routes_tensor=None, current_routes_tensor=None,
               revisit_routes_tensor=None):
    log.debug("evaluating our model on test set")
    model.eval()
    if current_routes_tensor is not None and revisit_routes_tensor is not None:
        raise ValueError(
            "Only one of current_routes_tensor or revisit_routes_tensor may "
            "be provided"
        )
    if n_samples is None:
        def method_fn(state, *args, init_network=None, **kwargs):
            state = _seed_state_with_init_network(
                state,
                init_network,
                init_current_routes=current_routes_tensor,
            )
            state = _revisit_seeded_routes(
                model,
                state,
                revisit_routes_tensor,
                greedy=True,
            )
            if state.is_done().all():
                return state, None
            return model(state, greedy=True).state, None
    else:
        def method_fn(state, cost_obj, *args, init_network=None, **kwargs):
            sampled_state = sample_from_model(
                model,
                state,
                cost_obj,
                n_samples=n_samples,
                sample_batch_size=sample_batch_size,
                init_network=init_network,
                init_current_routes=current_routes_tensor,
                revisit_routes=revisit_routes_tensor,
            )
            return sampled_state, None
    cost, _, unserved_demand, metrics, routes = \
        lrnu.test_method(method_fn, eval_dataloader, eval_cfg, init_cfg, cost_obj, 
                         sum_writer, device=device, silent=silent, 
                         iter_num=iter_num, return_routes=True,
                         routes_tensor=routes_tensor)
    if return_routes:
        routes = get_batch_tensor_from_routes(routes)
        return cost, unserved_demand, metrics, routes
    else:
        return cost, metrics


# @hydra.main(version_base=None, config_path="../cfg", 
#             config_name="eval_model_mumford")
def main(cfg: DictConfig, tensors:dict):
    global DEVICE
    assert 'model' in cfg, "Must provide config for model!"
    DEVICE, run_name, _, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'nn_construction_', 
                                             weights_required=True)

    # load the data
    test_ds = get_dataset_from_config(cfg.eval.dataset, tensors=tensors)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    # evaluate the model on the dataset
    n_samples = cfg.get('n_samples', None)
    sbs = cfg.get('sample_batch_size', cfg.batch_size)
    _, unserved_demand, metrics, routes = eval_model(
        model,
        test_dl,
        cfg.eval,
        cost_obj,
        n_samples=n_samples,
        sample_batch_size=sbs,
        return_routes=True,
        silent=True,
        device=DEVICE,
        init_cfg=cfg.get('init', None),
    )
    
    dump_routes(run_name, routes.cpu())
    return metrics, unserved_demand, routes


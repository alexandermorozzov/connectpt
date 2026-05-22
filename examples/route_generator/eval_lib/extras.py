"""Leftover helper definitions: the NX-dataset graph loader and the
route-change visualisation helpers.

Extracted from the notebook's §8b loader cell and §10 visualisation cells --
only the ``def`` blocks; the imports / constants / execution stay in the
notebook. Free names resolve via the eval_lib star-imports below.
"""
import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from connectpt.routes_generator.citygraph_dataset import STOP_KEY

from .context import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403
from .sweeps import *  # noqa: F401,F403
from . import plots as _plots


# === section 8b (NX-heuristic dataset graph loader) ===


def load_single_nx_heuristic_graph_and_routes(raw_graphs_path, results_dir, graph_index):
    graph_index = int(graph_index)
    # raw_graphs_1000.pkl is a single pickle, so loading the file reads the
    # list, but this sweep uses only the selected graph and one route file.
    with Path(raw_graphs_path).open("rb") as file:
        graphs = pickle.load(file)
    graph = graphs[graph_index]

    graph_dir = Path(results_dir) / f"graph_{graph_index:04d}"
    route_path = graph_dir / f"lc_lc_graph_{graph_index:04d}_routes_routes.pkl"
    if not route_path.exists():
        matches = sorted(graph_dir.glob("lc_*_routes_routes.pkl"))
        if not matches:
            raise FileNotFoundError(route_path)
        route_path = matches[0]

    with route_path.open("rb") as file:
        routes = pickle.load(file)
    if isinstance(routes, list):
        routes = routes[0]
    routes = torch.as_tensor(routes, dtype=torch.long)
    if routes.ndim == 3 and routes.shape[0] == 1:
        routes = routes.squeeze(0)
    if routes.ndim != 2:
        raise ValueError(f"Expected routes with shape (n_routes, len), got {tuple(routes.shape)}")

    used_columns = (routes >= 0).any(dim=0).nonzero().flatten()
    route_width = int(used_columns[-1].item()) + 1 if used_columns.numel() else 1
    route_width = min(route_width, MAX_ROUTE_LEN)
    routes = routes[:, :route_width]

    route_batch = torch.full(
        (1, N_ROUTES, MAX_ROUTE_LEN),
        -1,
        dtype=torch.long,
    )
    n_copy_routes = min(N_ROUTES, routes.shape[0])
    n_copy_stops = min(MAX_ROUTE_LEN, routes.shape[1])
    route_batch[0, :n_copy_routes, :n_copy_stops] = routes[:n_copy_routes, :n_copy_stops]
    return graph, route_batch, route_path


def nx_graph_to_tensor_dataset(graph):
    n_nodes = graph[STOP_KEY].pos.shape[0]
    return {
        "node_locs": graph[STOP_KEY].pos[:n_nodes].detach().cpu(),
        "street_adj": graph.street_adj[:n_nodes, :n_nodes].detach().cpu(),
        "demand": graph.demand[:n_nodes, :n_nodes].detach().cpu(),
    }


# === section 10 (route-change visualisation) ===


def plot_plain_route_set(ax, routes, coords, street_adj, title, subtitle=None):
    return _plots.plot_plain_route_set(
        ax, routes, coords, street_adj,
        title=title, subtitle=subtitle,
        palette="tab10", with_overlap_curves=False,
    )


def plot_route_diff(ax, routes, reference_routes, coords, street_adj,
                    title, subtitle=None):
    return _plots.plot_route_diff(
        ax, routes, reference_routes, coords, street_adj,
        title=title, subtitle=subtitle,
        palette="tab10", with_overlap_curves=False,
    )


def get_worse_plot_result(sweep_results: dict, accept_key: str, variant_key: str, seed: int):
    variant_results = sweep_results["results"][accept_key][variant_key]
    for result in variant_results:
        if int(result["seed"]) == int(seed):
            return result
    return variant_results[0]


# === section 10 (NX-dataset routes evaluation) ===
def evaluate_routes_on_tensors(cfg, tensors, routes_tensor):
    dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix="dataset_seed_eval_",
        weights_required=True,
    )
    output = lrnu.test_method(
        None,
        dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=True,
        device=device,
        return_routes=True,
        routes_tensor=routes_tensor,
    )
    _, _, unserved_demand, metrics, routes = output
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    return run_name, metrics, unserved_demand, routes_tensor

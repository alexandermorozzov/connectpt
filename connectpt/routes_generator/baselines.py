"""Comparison-baseline runners: SA / GA / hyper-heuristic / NSGA-II.

Library home of the notebook's baseline optimizers (was eval_lib.baselines).
All are purely heuristic (no neural model): they run the ported
``simulated_annealing`` / ``genetic_algorithm`` / ``hyperheuristic`` / ``NSGAII``
algorithms through the shared ``test_method`` contract on the same LC/NX-seeded
initial routes as the BCO experiments, so their metrics are directly comparable.

Imports are explicit library dependencies (no eval_lib glue); the unified cost
weights come from the single objective source, benchmark tensors + specs from the
data layer, and the cost breakdown from the evaluation layer.
"""
from __future__ import annotations

import gc

import numpy as np  # noqa: F401  (kept: baseline plotting cells expect it present)
import pandas as pd  # noqa: F401
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from . import heuristics as _hs
from . import utils as lrnu
from .citygraph_dataset import get_dataset_from_config
from .core.paths import CFG_DIR
from .data.loaders import BENCHMARK_SPECS, load_benchmark_tensors  # noqa: F401
from .data.routes import as_route_tensor
from .evaluation.cost_breakdown import add_cost_breakdown_to_metrics
from .genetic_algorithm import run as genetic_algorithm
from .hyperheuristics import hyperheuristic
from .nsgaii import NSGAII
from .nx_heuristic import build_nx_heuristic_routes
from .citygraph_dataset import CityGraphData
from .objectives import load_unified_objective
from .simulated_annealing import simulated_annealing_with_reheating
from .transit_time_estimator import RouteGenBatchState

# Unified cost weights (single source: cfg/objective/*.yaml).
_OBJ = load_unified_objective()
DEMAND_TIME_WEIGHT = _OBJ.demand_time_weight
ROUTE_TIME_WEIGHT = _OBJ.route_time_weight
MEDIAN_CONNECTIVITY_WEIGHT = _OBJ.median_connectivity_weight

CITY_NAME = "Mumford0"

# Compute budgets -- AHolliday/transit_learning reference values, halved.
SA_N_ITERATIONS = 20000
GA_N_ITERATIONS = 200
GA_POP_SIZE = 10
HH_N_ITERATIONS = 20000
NSGAII_N_ITERATIONS = 1000
NSGAII_POP_SIZE = 200
BENCHMARK_NX_SEED = 0


def safe_run_name(run_name: str) -> str:
    """Filesystem/run-name-safe slug (ascii alnum + ._-, collapsed underscores)."""
    safe = "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "_"
                   for c in str(run_name))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe.strip("._-") or "run"


def make_test_dataloader(dataset_cfg):
    return DataLoader(get_dataset_from_config(dataset_cfg), batch_size=1)


def make_tensor_dataloader(dataset_cfg, tensors):
    return DataLoader(get_dataset_from_config(dataset_cfg, tensors=tensors), batch_size=1)


def _baseline_cfg_overrides(run_name, n_routes, min_route_len, max_route_len,
                            connectivity_mode="median_weighted"):
    return [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",  # no empty TensorBoard run dir for eval runs
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={DEMAND_TIME_WEIGHT}",
        f"++experiment.cost_function.kwargs.route_time_weight={ROUTE_TIME_WEIGHT}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={MEDIAN_CONNECTIVITY_WEIGHT}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={safe_run_name(run_name)}",
    ]


def _compose_baseline_cfg(config_name, overrides):
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    cfg.batch_size = 1
    return cfg


def _early_stop_overrides(early_stop_patience, early_stop_min_delta):
    overrides = [f"++early_stop_min_delta={float(early_stop_min_delta)}"]
    if early_stop_patience is not None:
        overrides.append(f"++early_stop_patience={int(early_stop_patience)}")
    return overrides


def build_sa_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=SA_N_ITERATIONS,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++alg_args.n_iterations={n_iterations}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("sa_mumford", overrides)


def build_ga_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=GA_N_ITERATIONS, population_size=GA_POP_SIZE,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++n_iterations={n_iterations}")
    overrides.append(f"++population_size={population_size}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("ga_mumford", overrides)


def build_hh_cfg(run_name, n_routes, min_route_len, max_route_len,
                 n_iterations=HH_N_ITERATIONS, max_repair_iters=None,
                 early_stop_patience=None, early_stop_min_delta=0.0,
                 connectivity_mode="median_weighted"):
    overrides = _baseline_cfg_overrides(
        run_name, n_routes, min_route_len, max_route_len, connectivity_mode)
    overrides.append(f"++n_iterations={n_iterations}")
    if max_repair_iters is not None:
        overrides.append(f"++max_repair_iters={int(max_repair_iters)}")
    overrides += _early_stop_overrides(early_stop_patience, early_stop_min_delta)
    return _compose_baseline_cfg("hh_mumford", overrides)


def _run_baseline(method_fn, cfg, init_routes, prefix, method_kwargs, *,
                  tensors=None, return_history=False, use_weighted_connectivity=None,
                  connectivity_mode=None,
                  adjustment_seed_routes=None,
                  adjustment_degree_weight=0.0, adjustment_degree_target=0.2,
                  adjustment_degree_objective='raw', adjustment_degree_gap=0.1,
                  adjustment_degree_mode='paper'):
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=prefix, weights_required=False)
    if use_weighted_connectivity is not None:
        cost_obj.use_weighted_connectivity = bool(use_weighted_connectivity)
    if connectivity_mode is not None:
        cost_obj.connectivity_mode = connectivity_mode
    _adj_seed_src = adjustment_seed_routes if adjustment_seed_routes is not None else init_routes
    _adj_on = bool(adjustment_degree_weight) and adjustment_degree_weight > 0 \
        and _adj_seed_src is not None
    if _adj_on:
        cost_obj.adjustment_degree_weight = float(adjustment_degree_weight)
        cost_obj.adjustment_degree_target = float(adjustment_degree_target)
        cost_obj.adjustment_degree_objective = adjustment_degree_objective
        cost_obj.adjustment_degree_gap = float(adjustment_degree_gap)
        cost_obj.adjustment_degree_mode = adjustment_degree_mode
        _seed = as_route_tensor(_adj_seed_src)
        cost_obj.adjustment_seed = (_seed[None] if _seed.dim() == 2 else _seed).to(device)
    output = lrnu.test_method(
        method_fn,
        dataloader,
        cfg.eval,
        OmegaConf.create({"method": "tensor"}),
        cost_obj,
        silent=False,
        device=device,
        return_routes=True,
        return_histories=return_history,
        routes_tensor=init_routes,
        **method_kwargs,
    )
    if _adj_on:
        cost_obj.adjustment_seed = None
        cost_obj.adjustment_degree_weight = 0.0
    if return_history:
        _, _, unserved_demand, metrics, routes, cost_histories = output
    else:
        _, _, unserved_demand, metrics, routes = output
        cost_histories = None
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(
        metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    result = (run_name, metrics, unserved_demand, routes_tensor)
    if return_history:
        result = result + (cost_histories,)
    return result


def _early_stop_kwargs(cfg):
    return dict(
        early_stop_patience=cfg.get("early_stop_patience", None),
        early_stop_min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
    )


def run_sa(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = OmegaConf.to_container(cfg.alg_args, resolve=True)
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        simulated_annealing_with_reheating, cfg, init_routes,
        f"{run_name_scope}sa_", method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


def run_ga(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = dict(
        pop_size=int(cfg.population_size),
        n_iterations=int(cfg.n_iterations),
        shorten_prob=float(cfg.get("shorten_prob", 0.2)),
        force_linking_unlinked=bool(cfg.get("force_linking_unlinked", False)),
    )
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        genetic_algorithm, cfg, init_routes, f"{run_name_scope}ga_", method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


def run_hh(cfg, init_routes, *, tensors=None, run_name_scope="",
           return_history=False, **adj):
    method_kwargs = dict(f_0=float(cfg.f_0), n_steps=int(cfg.n_iterations))
    _mri = cfg.get("max_repair_iters", None)
    if _mri is not None:
        method_kwargs["max_repair_iters"] = int(_mri)
    method_kwargs.update(_early_stop_kwargs(cfg))
    return _run_baseline(
        hyperheuristic, cfg, init_routes, f"{run_name_scope}hh_", method_kwargs,
        tensors=tensors, return_history=return_history, **adj)


def build_nsgaii_cfg(run_name, n_routes, min_route_len, max_route_len,
                     n_iterations=NSGAII_N_ITERATIONS, pop_size=NSGAII_POP_SIZE,
                     connectivity_mode="median_weighted"):
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={safe_run_name(run_name)}",
        f"++n_iterations={n_iterations}",
        f"++pop_size={pop_size}",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="nsgaii_mumford", overrides=overrides)
    cfg.batch_size = 1
    return cfg


def run_nsgaii(cfg, *, tensors=None, init_routes=None, run_name_scope="",
               use_weighted_connectivity=False,
               connectivity_mode=None,
               adjustment_degree_weight=0.0, adjustment_degree_target=0.2,
               adjustment_degree_objective='raw', adjustment_degree_gap=0.1,
               adjustment_degree_mode='paper'):
    """Run NSGA-II and return ``(run_name, output)``. See the section-9c notes."""
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, _ = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=f"{run_name_scope}nsgaii_", weights_required=False)
    cost_obj.use_weighted_connectivity = bool(use_weighted_connectivity)
    if connectivity_mode is not None:
        cost_obj.connectivity_mode = connectivity_mode
    if adjustment_degree_weight and adjustment_degree_weight > 0 and init_routes is not None:
        cost_obj.adjustment_degree_weight = float(adjustment_degree_weight)
        cost_obj.adjustment_degree_target = float(adjustment_degree_target)
        cost_obj.adjustment_degree_objective = adjustment_degree_objective
        cost_obj.adjustment_degree_gap = float(adjustment_degree_gap)
        cost_obj.adjustment_degree_mode = adjustment_degree_mode
        _seed = as_route_tensor(init_routes)
        cost_obj.adjustment_seed = (_seed[None] if _seed.dim() == 2 else _seed).to(device)
    data = next(iter(dataloader))
    if device.type != "cpu":
        data = data.cuda()
    state = RouteGenBatchState(data, cost_obj, cfg.eval.n_routes,
                               cfg.eval.min_route_len, cfg.eval.max_route_len)
    mutators = [_hs.add_terminal, _hs.delete_terminal, _hs.add_inside,
                _hs.delete_inside, _hs.invert_nodes, _hs.exchange_routes,
                _hs.replace_node, _hs.donate_node]
    if cfg.get("use_cost_based_heuristics", True):
        mutators += [_hs.cost_based_grow, _hs.cost_based_trim]
    optimizer = NSGAII(
        cost_obj, init_models=[], mutators=mutators,
        n_iterations=int(cfg.n_iterations), pop_size=int(cfg.pop_size),
        p_crossover=float(cfg.p_crossover), p_mutation=float(cfg.p_mutation),
        mutator_p_t=float(cfg.mutator_p_t),
        batch_size=int(cfg.get("gen_batch_size", cfg.pop_size)),
        device=device)
    with torch.no_grad():
        output = optimizer.run(state, cfg.get("init_mode", "husselmann"),
                               sum_writer=None, seed_routes=init_routes)
    return run_name, output


def reduce_pareto_front(output, demand_weight, route_weight):
    """Collapse the Pareto front to its lowest weighted-sum member."""
    pareto = output["pareto_pop"]
    return min(pareto, key=lambda m: (demand_weight * float(m["cost"][0])
                                      + route_weight * float(m["cost"][1])))


def load_benchmark_graph(spec, init_mode="nx"):
    """Load a benchmark city + build its benchmark initial route set.

    ``init_mode="nx"`` uses the NX heuristic (library-native). ``"rpc"`` uses the
    random-path-combiner init (learned-construction plumbing, imported lazily from
    the notebook LC helpers -- the paper run uses nx init).
    """
    tensors = load_benchmark_tensors(spec["city"])
    if init_mode == "rpc":
        from eval_lib.helpers import build_rpc_routes  # transitional (LC plumbing)
        init_routes = build_rpc_routes(
            spec, tensors, run_name=f"benchmark_rpc_init_{spec['city']}", n_samples=1)
    elif init_mode == "nx":
        graph = CityGraphData.from_tensors(
            tensors["node_locs"], tensors["street_adj"], tensors["demand"],
            pos_only=False)
        init_routes = build_nx_heuristic_routes(
            graph, num_routes=spec["n_routes"], min_len=spec["min_route_len"],
            max_len=spec["max_route_len"], seed=BENCHMARK_NX_SEED)
    else:
        raise ValueError(f"Unknown benchmark init_mode: {init_mode!r}")
    return tensors, init_routes

"""Learned-construction (LC) evaluation: build an eval cfg + run the model.

Library home of the notebook's LC helpers (was eval_lib.helpers). ``build_lc_cfg``
composes the eval config (weights + route bounds + unified cost weights);
``run_lc`` / ``run_lc_batch`` run the construction model through ``eval_model`` and
annotate the metrics with the cost breakdown. ``build_rpc_cfg`` / ``build_rpc_routes``
back the random-path-combiner init (no trained weights). Imports are explicit
library dependencies -- the objective weights come from the single source, the
disabled-component injection is inlined (was compat.bco_config).
"""
from __future__ import annotations

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from . import utils as lrnu
from .baselines import safe_run_name
from .citygraph_dataset import get_dataset_from_config
from .core.paths import CFG_DIR, CONSTRUCTION_MODEL_WEIGHTS_PATH as MODEL_WEIGHTS_PATH
from .data.routes import as_route_tensor
from .eval_route_generator import eval_model
from .evaluation.cost_breakdown import add_cost_breakdown_to_metrics
from .objectives import load_unified_objective
from .utils import get_eval_cfg

_OBJ = load_unified_objective()
DEMAND_TIME_WEIGHT = _OBJ.demand_time_weight
ROUTE_TIME_WEIGHT = _OBJ.route_time_weight
MEDIAN_CONNECTIVITY_WEIGHT = _OBJ.median_connectivity_weight
LC_SAMPLES = 100


def _apply_disabled_components(cfg):
    """Inject the objective's disabled cost components into the cost kwargs (so
    the built cost drops them + renormalizes the survivors). Was compat."""
    disabled = list(_OBJ.disabled_components)
    if not disabled:
        return cfg
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.cost_function.kwargs.disabled_components = disabled
    OmegaConf.set_struct(cfg, True)
    return cfg


def make_test_dataloader(dataset_cfg):
    return DataLoader(get_dataset_from_config(dataset_cfg), batch_size=1)


def make_tensor_dataloader(dataset_cfg, tensors):
    return DataLoader(get_dataset_from_config(dataset_cfg, tensors=tensors), batch_size=1)


def build_lc_cfg(run_name, n_routes, min_route_len, max_route_len,
                 demand_time_weight=DEMAND_TIME_WEIGHT,
                 route_time_weight=ROUTE_TIME_WEIGHT,
                 median_connectivity_weight=MEDIAN_CONNECTIVITY_WEIGHT,
                 connectivity_mode="median_weighted"):
    run_name = safe_run_name(run_name)
    params = {
        "dataset_name": "tensor",
        "n_routes": n_routes,
        "min_route_len": min_route_len,
        "max_route_len": max_route_len,
        "demand_time_weight": demand_time_weight,
        "route_time_weight": route_time_weight,
        "median_connectivity_weight": median_connectivity_weight,
        "connectivity_mode": connectivity_mode,
        "run_name": run_name,
        "model_weights": str(MODEL_WEIGHTS_PATH),
    }
    cfg = get_eval_cfg(str(CFG_DIR), "eval_model_mumford", params)
    cfg.batch_size = 1
    return _apply_disabled_components(cfg)


def build_rpc_cfg(run_name, n_routes, min_route_len, max_route_len,
                  demand_time_weight=DEMAND_TIME_WEIGHT,
                  route_time_weight=ROUTE_TIME_WEIGHT,
                  median_connectivity_weight=MEDIAN_CONNECTIVITY_WEIGHT,
                  connectivity_mode="median_weighted"):
    """LC-style eval cfg backed by random_path_combiner (no trained weights)."""
    run_name = safe_run_name(run_name)
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={demand_time_weight}",
        f"++experiment.cost_function.kwargs.route_time_weight={route_time_weight}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={median_connectivity_weight}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={run_name}",
        "model=random_path_combiner",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="eval_model_mumford", overrides=overrides)
    cfg.batch_size = 1
    return _apply_disabled_components(cfg)


def _pad_routes_to_spec(routes, n_routes, max_route_len):
    routes = as_route_tensor(routes).long()
    if routes.ndim == 2:
        routes = routes[None]
    if routes.shape[1] != n_routes:
        raise ValueError(
            f"Expected {n_routes} routes, got tensor shape {tuple(routes.shape)}")
    current_len = routes.shape[-1]
    if current_len > max_route_len:
        raise ValueError(
            f"RPC init produced route tensor length {current_len}, "
            f"but max_route_len={max_route_len}")
    if current_len < max_route_len:
        routes = torch.nn.functional.pad(
            routes, (0, max_route_len - current_len), value=-1)
    return routes


def run_lc(cfg, init_routes=None, revisit_routes=None, *,
           tensors=None, run_name_prefix="lc_", n_samples=None):
    if tensors is None:
        dataloader = make_test_dataloader(cfg.eval.dataset)
    else:
        dataloader = make_tensor_dataloader(cfg.eval.dataset, tensors)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=run_name_prefix, weights_required=True)
    init_cfg = OmegaConf.create({"method": "tensor"}) if init_routes is not None else None
    if hasattr(model, "clear_step_counts_log"):
        model.clear_step_counts_log()
    effective_n_samples = LC_SAMPLES if n_samples is None else int(n_samples)
    _, unserved_demand, metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj, n_samples=effective_n_samples,
        return_routes=True, silent=True, device=device, init_cfg=init_cfg,
        routes_tensor=init_routes, revisit_routes_tensor=revisit_routes)
    routes_tensor = as_route_tensor(routes)
    metrics = add_cost_breakdown_to_metrics(
        metrics, dataloader, cfg.eval, cost_obj, routes_tensor, device)
    step_counts = getattr(model, "route_step_counts_log", [])
    return run_name, metrics, unserved_demand, routes_tensor, list(step_counts)


def run_lc_batch(cfg, tensors_list, *, run_name_prefix="lc_", n_samples=None,
                 batch_size=None):
    """Batched learned-construction over many graphs (one GPU forward per batch)."""
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg, run_name_prefix=run_name_prefix, weights_required=True)
    if hasattr(model, "clear_step_counts_log"):
        model.clear_step_counts_log()
    datalist = []
    for tn in tensors_list:
        ds = get_dataset_from_config(cfg.eval.dataset, tensors=tn)
        datalist.append(ds[0] if isinstance(ds, (list, tuple)) else ds)
    bs = int(batch_size or len(datalist))
    dataloader = DataLoader(datalist, batch_size=bs)
    effective_n_samples = LC_SAMPLES if n_samples is None else int(n_samples)
    _, _unserved, _metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj, n_samples=effective_n_samples,
        sample_batch_size=bs, return_routes=True, silent=True, device=device)
    return as_route_tensor(routes)


def build_rpc_routes(spec, tensors, run_name=None, n_samples=1):
    """Generate benchmark initial routes with RPC/pi_random (one benchmark_spec)."""
    run_name = run_name or f"rpc_init_{spec['city']}"
    cfg = build_rpc_cfg(run_name=run_name, n_routes=spec["n_routes"],
                        min_route_len=spec["min_route_len"],
                        max_route_len=spec["max_route_len"])
    _, _metrics, _unserved, routes, _step_counts = run_lc(
        cfg, tensors=tensors, run_name_prefix="rpc_", n_samples=n_samples)
    return _pad_routes_to_spec(routes, spec["n_routes"], spec["max_route_len"])

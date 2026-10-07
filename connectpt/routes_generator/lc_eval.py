"""Learned-construction (LC) evaluation: build an eval cfg + run the model.

Library home of the notebook's LC helpers (was eval_lib.helpers). ``build_lc_cfg``
composes the eval config (weights + route bounds + unified cost weights);
``run_lc`` / ``run_lc_batch`` run the construction model through ``eval_model`` and
annotate the metrics with the cost breakdown. Imports are explicit
library dependencies -- the objective weights come from the single source, the
disabled-component injection is inlined (was compat.bco_config).
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from . import utils as lrnu
from .city.graph import to_batch
from .city.instances import load_instance
from .run.context import CFG_DIR, CONSTRUCTION_MODEL_WEIGHTS_PATH as MODEL_WEIGHTS_PATH, safe_run_name
from .network.routes import to_routes
from .eval_route_generator import eval_model
from .evaluation.cost_breakdown import add_cost_breakdown_to_metrics
from .objectives import cost_kwargs
from .utils import get_eval_cfg

LC_SAMPLES = 100


def _apply_disabled_components(cfg):
    """Inject the objective's disabled cost components into the cost kwargs (so
    the built cost drops them + renormalizes the survivors). Was compat."""
    disabled = list(cfg.cost.kwargs.disabled_components)
    if not disabled:
        return cfg
    OmegaConf.set_struct(cfg, False)
    cfg.cost.kwargs.disabled_components = disabled
    OmegaConf.set_struct(cfg, True)
    return cfg






def build_lc_cfg(run_name, n_routes, min_route_len, max_route_len,
                 demand_time_weight=None, route_time_weight=None,
                 median_connectivity_weight=None,
                 connectivity_mode="median_weighted"):
    defaults = cost_kwargs()
    demand_time_weight = (defaults.demand_time_weight if demand_time_weight is None
                          else demand_time_weight)
    route_time_weight = (defaults.route_time_weight if route_time_weight is None
                         else route_time_weight)
    median_connectivity_weight = (
        defaults.median_connectivity_weight if median_connectivity_weight is None
        else median_connectivity_weight)
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
    cfg = get_eval_cfg(str(CFG_DIR), "stacks/model_eval", params)
    return _apply_disabled_components(cfg)


def run_lc(cfg, init_routes=None, revisit_routes=None, *,
           tensors=None, run_name_prefix="lc_", n_samples=None):
    if tensors is None:
        dataloader = DataLoader(to_batch(load_instance(cfg.eval.dataset.city.lower())).to_data_list(), batch_size=1)
    else:
        dataloader = DataLoader(to_batch(tensors).to_data_list(), batch_size=1)
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
    routes_tensor = to_routes(routes)
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
    datalist = [to_batch(tn).to_data_list()[0] for tn in tensors_list]
    bs = int(batch_size or len(datalist))
    dataloader = DataLoader(datalist, batch_size=bs)
    effective_n_samples = LC_SAMPLES if n_samples is None else int(n_samples)
    _, _unserved, _metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj, n_samples=effective_n_samples,
        sample_batch_size=bs, return_routes=True, silent=True, device=device)
    return to_routes(routes)

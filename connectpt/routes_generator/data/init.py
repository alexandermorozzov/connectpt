"""Build an initial route network for a benchmark instance.

A run never builds one on the fly -- it reads ``Routes.txt`` from the instance
folder. This is what ``scripts/make_init_routes.py`` calls to produce that file.
"""
from __future__ import annotations

import torch

from .routes import as_route_tensor


def build_init_routes(spec, tensors, *, seed: int = 0, device=None,
                      clean: bool = False) -> torch.Tensor:
    """Build an initial network: one stochastic sample of the learned
    construction model, plus the realistic corruption tier unless ``clean``."""
    import random

    from .route_copies import inject_realistic_tier, pad_routes

    if device is None:
        from ..core.runtime import resolve_device
        device = resolve_device()

    routes = _lc_construct(spec, tensors, device=device)
    routes = pad_routes(routes, spec["n_routes"], spec["max_route_len"])
    if not clean:
        rng = random.Random(seed)
        routes, _ = inject_realistic_tier(
            routes, rng, spec["min_route_len"], spec["max_route_len"],
            street_adj=tensors["street_adj"], demand=tensors["demand"],
            n_nodes=tensors["node_locs"].shape[0])
    routes = as_route_tensor(routes)
    return routes[None] if routes.ndim == 2 else routes


def _lc_construct(spec, tensors, *, device) -> torch.Tensor:
    """One learned-construction rollout for ``spec`` on ``tensors``."""
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    from ..citygraph_dataset import get_dataset_from_config
    from ..core.checkpoints import CheckpointStore
    from ..core.paths import CFG_DIR, CONSTRUCTION_MODEL_WEIGHTS_PATH
    from ..core.runtime import seed_everything
    from ..eval_route_generator import eval_model
    from ..model_factory import RouteModelFactory
    from ..objectives import CostFactory, cost_kwargs
    from ..utils import get_eval_cfg

    kw = cost_kwargs()
    params = {
        "dataset_name": "tensor",
        "n_routes": spec["n_routes"],
        "min_route_len": spec["min_route_len"],
        "max_route_len": spec["max_route_len"],
        "connectivity_mode": kw.connectivity_mode,
        "run_name": f"lc_init_{spec.get('city', 'inst')}",
        "model_weights": str(CONSTRUCTION_MODEL_WEIGHTS_PATH),
        "demand_time_weight": kw.demand_time_weight,
        "route_time_weight": kw.route_time_weight,
        "median_connectivity_weight": kw.median_connectivity_weight,
    }
    cfg = get_eval_cfg(str(CFG_DIR), "stacks/model_eval", params)
    disabled = list(kw.disabled_components)
    if disabled:
        OmegaConf.set_struct(cfg, False)
        cfg.cost.kwargs.disabled_components = disabled
        OmegaConf.set_struct(cfg, True)

    seed_everything(cfg.experiment.get("seed"))
    dataloader = DataLoader(
        get_dataset_from_config(cfg.eval.dataset, tensors=tensors), batch_size=1)
    cost_obj = CostFactory.build(
        cfg.cost,
        symmetric_routes=bool(cfg.experiment.symmetric_routes))
    cost_obj.to(device)
    model = RouteModelFactory.build_construction_model(cfg.model, cfg.experiment)
    CheckpointStore.load_model_weights(model, cfg.model.weights, map_location=device)
    model.to(device)

    _, _unserved, _metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj, n_samples=1,
        return_routes=True, silent=True, device=device)
    return as_route_tensor(routes)

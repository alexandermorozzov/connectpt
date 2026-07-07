"""Resolve the initial route network for an experiment instance.

Three sources, in priority order:

* ``init_dump``       -- a ``paper_results`` ``*_routes.pt`` dump; the route set
  whose label starts with "Initial" is used (pins a machine-independent init a
  previous run saved);
* ``init_routes_path``-- a plain route-tensor file (``torch_utils`` format);
* from scratch        -- learned-construction (LC) network + realistic-tier
  corruption, the E1 init pipeline.

The from-scratch path is written against the library factories
(``RouteModelFactory`` / ``CostFactory`` / ``eval_model``) -- it does NOT go
through the legacy ``process_standard_experiment_cfg``.
"""
from __future__ import annotations

import torch

from .routes import as_route_tensor


def resolve_init_routes(spec, tensors, *, init_dump=None, init_routes_path=None,
                        seed: int = 0, device=None) -> torch.Tensor:
    """Return the batched init route tensor for ``spec`` on ``tensors``."""
    if init_dump:
        return _init_from_dump(init_dump, spec)
    if init_routes_path:
        from ..torch_utils import load_routes_tensor
        return as_route_tensor(load_routes_tensor(init_routes_path))
    if device is None:
        from ..core.runtime import resolve_device
        device = resolve_device()
    return _init_from_scratch(spec, tensors, seed=seed, device=device)


def _init_from_dump(init_dump, spec) -> torch.Tensor:
    from .route_copies import pad_routes
    dump = torch.load(init_dump, map_location="cpu", weights_only=False)
    routes = dump.get("routes", {})
    key = next((k for k in routes if str(k).startswith("Initial")), None)
    if key is None:
        raise ValueError(
            f"init_dump {init_dump}: no 'Initial*' route set among {list(routes)}")
    init = pad_routes(as_route_tensor(routes[key]), spec["n_routes"],
                      spec["max_route_len"])
    return init[None] if init.ndim == 2 else init


def _init_from_scratch(spec, tensors, *, seed: int, device) -> torch.Tensor:
    import random

    from .route_copies import inject_realistic_tier, pad_routes

    clean = _lc_construct(spec, tensors, device=device)
    clean = pad_routes(clean, spec["n_routes"], spec["max_route_len"])
    rng = random.Random(seed)
    init, _ = inject_realistic_tier(
        clean, rng, spec["min_route_len"], spec["max_route_len"],
        street_adj=tensors["street_adj"], demand=tensors["demand"],
        n_nodes=tensors["node_locs"].shape[0])
    init = as_route_tensor(init)
    return init[None] if init.ndim == 2 else init


def _lc_construct(spec, tensors, *, device) -> torch.Tensor:
    """Build a clean learned-construction network for ``spec`` on ``tensors``.

    Config-first: composes the ``eval_model_mumford`` eval cfg with the unified
    objective's weights + the construction checkpoint, then builds model/cost
    through the factories and rolls out one sample via ``eval_model``.
    """
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader

    from ..citygraph_dataset import get_dataset_from_config
    from ..core.checkpoints import CheckpointStore
    from ..core.paths import CFG_DIR, CONSTRUCTION_MODEL_WEIGHTS_PATH
    from ..core.runtime import seed_everything
    from ..eval_route_generator import eval_model
    from ..model_factory import RouteModelFactory
    from ..objectives import CostFactory, load_unified_objective
    from ..utils import get_eval_cfg

    obj = load_unified_objective()
    params = {
        "dataset_name": "tensor",
        "n_routes": spec["n_routes"],
        "min_route_len": spec["min_route_len"],
        "max_route_len": spec["max_route_len"],
        "connectivity_mode": obj.connectivity_mode,
        "run_name": f"lc_init_{spec.get('city', 'inst')}",
        "model_weights": str(CONSTRUCTION_MODEL_WEIGHTS_PATH),
        **obj.weights,
    }
    cfg = get_eval_cfg(str(CFG_DIR), "eval_model_mumford", params)
    cfg.batch_size = 1
    disabled = list(obj.disabled_components)
    if disabled:
        OmegaConf.set_struct(cfg, False)
        cfg.experiment.cost_function.kwargs.disabled_components = disabled
        OmegaConf.set_struct(cfg, True)

    seed_everything(cfg.experiment.get("seed"))
    dataloader = DataLoader(
        get_dataset_from_config(cfg.eval.dataset, tensors=tensors), batch_size=1)
    cost_obj = CostFactory.build(
        cfg.experiment.cost_function,
        symmetric_routes=bool(cfg.experiment.symmetric_routes))
    cost_obj.to(device)
    model = RouteModelFactory.build_construction_model(cfg.model, cfg.experiment)
    CheckpointStore.load_model_weights(model, cfg.model.weights, map_location=device)
    model.to(device)

    _, _unserved, _metrics, routes = eval_model(
        model, dataloader, cfg.eval, cost_obj, n_samples=1,
        return_routes=True, silent=True, device=device)
    return as_route_tensor(routes)

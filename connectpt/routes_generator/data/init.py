"""Build an initial route network for a benchmark instance.

A run never builds one on the fly -- it reads ``Routes.txt`` from the instance
folder. This is what ``scripts/make_init_routes.py`` calls to produce that file.
"""
from __future__ import annotations

import torch

from ..network.routes import to_routes


def build_init_routes(spec, tensors, *, seed: int = 0, device=None,
                      clean: bool = False) -> torch.Tensor:
    """Build an initial network: one stochastic sample of the learned
    construction model, plus the realistic corruption tier unless ``clean``."""
    import random

    from ..network.corruption import inject_realistic_tier

    if device is None:
        from ..run.context import resolve_device
        device = resolve_device()

    routes = _lc_construct(spec, tensors, device=device)
    routes = to_routes(routes, n_routes=spec["n_routes"], width=spec["max_route_len"])[0]
    if not clean:
        rng = random.Random(seed)
        routes, _ = inject_realistic_tier(
            routes, rng, spec["min_route_len"], spec["max_route_len"],
            street_adj=tensors["street_adj"], demand=tensors["demand"],
            n_nodes=tensors["node_locs"].shape[0])
    routes = to_routes(routes)
    return routes[None] if routes.ndim == 2 else routes


def _lc_construct(spec, tensors, *, device) -> torch.Tensor:
    from torch_geometric.loader import DataLoader
    from ..city.graph import to_batch
    from ..env.state import RouteGenBatchState
    from ..policy.generate import generate
    from ..policy.rollout import bind_env
    from ..run.policy import PolicyRun
    from ..run.config import compose_policy_config

    run = PolicyRun(compose_policy_config(), device)
    run.setup()
    data = next(iter(DataLoader(to_batch(tensors).to_data_list(), batch_size=1))).to(device)
    state = RouteGenBatchState(data, run.cost.constants,
        spec["n_routes"], spec["min_route_len"], spec["max_route_len"],
        preferences=run.cost.get_weights())
    return generate(bind_env(state, run.policy, run.cost, normalize=False), run.policy, "sample", 1)

"""Configuration boundary for scoring an existing route network."""
import torch

from ..objectives.cost import build_cost
from ..objectives.metrics import evaluate_network, evaluate_routes
from .config import cost_params
from .schemas import AdjustmentParams


class NetworkRun:
    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = torch.device("cpu") if device is None else device

    def setup(self):
        cfg = self.cfg
        self.cost = build_cost(
            cost_params(cfg.cost), symmetric_routes=cfg.experiment.symmetric_routes, low_memory_mode=cfg.experiment.low_memory_mode
        )
        self.adjustment = AdjustmentParams.from_cost(cfg.cost.kwargs)

    def score(
        self,
        routes,
        tensors,
        spec,
        *,
        alpha=None,
        adj_target=None,
        adj_weight=None,
        adj_objective=None,
        seed_routes=None,
    ):
        overrides = {}
        if alpha is not None:
            overrides.update(route_time_weight=float(alpha), median_connectivity_weight=float(1 - alpha))
        if adj_target is not None and seed_routes is not None:
            overrides["adjustment_degree_target"] = float(adj_target)
            if adj_weight is not None:
                overrides["adjustment_degree_weight"] = float(adj_weight)
            if adj_objective is not None:
                overrides["adjustment_degree_objective"] = adj_objective
        else:
            overrides["adjustment_degree_weight"] = 0.0
        return evaluate_routes(
            routes,
            tensors,
            spec,
            cost=self.cost.with_params(**overrides),
            device=self.device,
            seed_routes=seed_routes if adj_target is not None else None,
        )

    def table(self, instance, *, alpha=None, adj_target=None, adj_weight=None):
        metrics, routes = self.score(
            instance.init_routes,
            instance.tensors,
            instance.spec,
            alpha=alpha,
            adj_target=adj_target,
            adj_weight=adj_weight,
            seed_routes=instance.init_routes,
        )
        return evaluate_network(metrics, routes=routes, seed=instance.init_routes, adjustment=self.adjustment)

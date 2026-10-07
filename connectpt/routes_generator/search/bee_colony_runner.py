"""BeeColonyRunner -- drive the bee-colony engine from an ExecutablePlan.

The runner holds the cost and the ``ExecutablePlan`` (the bee taxonomy:
operator groups + models + halt / max-len flags) plus the run-level values
``BeeColonySearchRun.setup`` read from the config; it never reads the config. ``run_seeded`` improves an existing network with one
``test_method(run_bee_colony_plan, ...)`` call -- the trained models live in the
plan, not a wrapper. This layer must NOT import training.
"""
from __future__ import annotations


class BeeColonyRunner:
    def __init__(self, *, cost_obj, plan, device, seed: int, silent: bool,
                 sequential: bool, adjustment: dict):
        self.cost_obj = cost_obj
        self.plan = plan
        self.device = device
        self.seed = seed
        self.silent = silent
        self.sequential = sequential
        self.adjustment = adjustment

    def plan_summary(self) -> dict:
        # bee counts keyed by bee name (the plan group's name), not legacy slot.
        return {
            "counts": self.plan.summary(),
            "needs_construction": self.plan.needs_construction,
            "needs_edit": self.plan.needs_edit,
            "n_bees": int(self.plan.total_bees),
        }

    def run_seeded(self, init_routes, tensors, *, eval_dims, n_iterations,
                   alpha, adj_target, adj_weight, sum_writer=None):
        """Seeded improvement of an EXISTING network (init from ``init_routes``).

        ``alpha`` (RTT/WMC trade-off) sets route_time_weight=alpha and
        median_connectivity_weight=1-alpha. ``adj_target`` / ``adj_weight``
        override the target and weight of the adjustment-degree penalty
        (``0`` = penalty off). Reseeds from ``seed`` immediately before the run,
        so the search does not depend on the model-init RNG. Returns
        ``(routes, unserved, metrics)``.
        """
        from omegaconf import OmegaConf
        from torch_geometric.loader import DataLoader

        from ..bee_colony import run_bee_colony_plan
        from ..citygraph_dataset import get_dataset_from_config
        from ..core.runtime import seed_everything
        from ..utils import test_method

        if alpha is not None:
            self.cost_obj.route_time_weight = float(alpha)
            self.cost_obj.median_connectivity_weight = float(1.0 - alpha)
        adjustment = dict(self.adjustment)
        if adj_weight is not None:
            adjustment["adjustment_degree_weight"] = float(adj_weight)
        if adj_target is not None:
            adjustment["adjustment_degree_target"] = float(adj_target)

        dataloader = DataLoader(
            get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
            batch_size=1)
        # csv default: the non-silent eval path reads eval_cfg.csv strictly.
        eval_cfg = OmegaConf.create({"csv": False, **dict(eval_dims)})

        seed_everything(self.seed)
        _mean, _std, unserved, metrics, routes = test_method(
            run_bee_colony_plan, dataloader, eval_cfg,
            OmegaConf.create({"method": "tensor"}), self.cost_obj,
            sum_writer=sum_writer, silent=self.silent,
            device=self.device, return_routes=True, routes_tensor=init_routes,
            # keep only the per-iteration ``best *`` curves in TB
            log_eval_summary=False,
            n_bees=self.plan.total_bees, n_iterations=int(n_iterations), plan=self.plan,
            process_neural_bees_sequentially=self.sequential,
            **adjustment)
        return routes, unserved, metrics

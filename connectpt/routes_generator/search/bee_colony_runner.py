"""BeeColonyRunner -- drive the bee-colony engine from an ExecutablePlan.

The runner holds the cost, the loaded models, the policy adapters and the
``ExecutablePlan`` (the bee taxonomy: operator groups + models + halt / max-len
flags). ``run_suite`` builds the benchmark dataloader + the run schedule and
drives the plan-based engine through ``test_method`` -- the trained models live
in the plan, not a wrapper. This layer must NOT import training.
"""
from __future__ import annotations

from ..objectives import CostFactory


class BeeColonyRunner:
    def __init__(self, cfg, *, cost_obj, models: dict, policies: dict, plan,
                 data_module, device):
        self.cfg = cfg
        self.cost_obj = cost_obj
        self.models = models
        self.policies = policies
        self.plan = plan
        self.data = data_module
        self.device = device

    def plan_summary(self) -> dict:
        search = self.cfg.get("search", {})
        # bee counts keyed by bee name (the plan group's name), not legacy slot.
        counts = self.plan.summary()
        return {
            "counts": counts,
            "needs_construction": self.plan.needs_construction,
            "needs_edit": self.plan.needs_edit,
            "n_bees": int(search.get("n_bees", self.plan.total_bees)),
        }

    def _adjustment_kwargs(self) -> dict:
        """The adjustment-degree penalty kwargs from the unified objective."""
        adj = CostFactory.load_objective("rtt_wmc_no_demand").adjustment
        return dict(
            adjustment_degree_weight=float(adj.weight),
            adjustment_degree_target=float(adj.target),
            adjustment_degree_objective=str(adj.objective),  # two-sided "target"
            adjustment_degree_gap=float(adj.gap),
            adjustment_degree_mode=str(adj.mode),
        )

    def run_seeded(self, init_routes, tensors, *, eval_dims, n_iterations=None,
                   alpha=None, adj_target=None):
        """Seeded improvement of an EXISTING network (init from ``init_routes``).

        Builds the tensor dataloader from ``tensors``, assembles the run
        schedule (:func:`build_bco_schedule_cfg`) and runs the seeded executor on
        the ExecutablePlan. Reseeds from ``cfg.run.seed`` immediately before the
        run so the search is reproducible independent of model-init RNG.

        ``alpha`` (RTT/WMC trade-off) reconfigures the cost weights in place
        (route_time_weight=alpha, median_connectivity_weight=1-alpha), matching the
        notebook's per-sweep-point weight override; ``adj_target`` overrides the
        adjustment-degree target the bee-colony optimizes. Returns
        ``(routes, unserved, metrics)``.
        """
        from omegaconf import OmegaConf
        from torch_geometric.loader import DataLoader

        from ..citygraph_dataset import get_dataset_from_config
        from ..core.runtime import seed_everything
        from .plan_kwargs import build_bco_schedule_cfg
        from .seeded_search import run_seeded_bee_colony

        if alpha is not None:
            self.cost_obj.route_time_weight = float(alpha)
            self.cost_obj.median_connectivity_weight = float(1.0 - alpha)

        dataloader = DataLoader(
            get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
            batch_size=1)
        eval_cfg = OmegaConf.create(dict(eval_dims))
        search = self.cfg.search
        acceptance = search.get("acceptance")
        search_cfg = build_bco_schedule_cfg(
            n_bees=int(search.n_bees),
            n_iterations=int(search.n_iterations if n_iterations is None else n_iterations),
            acceptance=None if acceptance is None else dict(acceptance))
        if adj_target is not None:
            search_cfg.adjustment_degree_target = float(adj_target)

        seed_everything(int(self.cfg.run.get("seed", 0)))
        out = run_seeded_bee_colony(
            dataloader, eval_cfg, self.cost_obj, init_routes, search_cfg=search_cfg,
            plan=self.plan, device=self.device, silent=True)
        _mean, _std, unserved, metrics, routes = out
        return routes, unserved, metrics

    def run_sweep(self, init_routes, tensors, *, eval_dims, alpha_grid=(None,),
                  adj_targets=(None,), n_iterations=None):
        """Run a seeded (alpha x adj_target) sweep -- one seeded search per point.

        Replaces the notebook's ``run_experiment`` alpha/adj loop: for each
        (alpha, adj_target) it reconfigures the cost trade-off + adjustment target
        and runs the seeded search from the shared initial network. Returns a list
        of ``{"alpha", "adj_target", "routes", "unserved", "metrics"}`` rows.
        """
        rows = []
        for alpha in alpha_grid:
            for adj_target in adj_targets:
                routes, unserved, metrics = self.run_seeded(
                    init_routes, tensors, eval_dims=eval_dims,
                    n_iterations=n_iterations, alpha=alpha, adj_target=adj_target)
                rows.append({"alpha": alpha, "adj_target": adj_target,
                             "routes": routes, "unserved": unserved, "metrics": metrics})
        return rows

    def run_suite(self):
        """Run the full bee-colony search on the benchmark instance."""
        from omegaconf import OmegaConf
        from torch_geometric.loader import DataLoader

        from ..bee_colony import run_bee_colony_plan
        from ..utils import test_method

        if not self.data.dataset:
            self.data.setup()
        dataloader = DataLoader(self.data.dataset, batch_size=1)

        search = self.cfg.search
        eval_cfg = OmegaConf.create({
            "n_routes": int(self.data.n_routes),
            "min_route_len": int(self.data.min_route_len),
            "max_route_len": int(self.data.max_route_len),
            "csv": False,
        })
        # from-scratch initial network (no precomputed routes needed)
        init_cfg = OmegaConf.create(
            {"method": "john", "alpha": 0.0, "prioritize_direct_connections": True}
        )

        out = test_method(
            run_bee_colony_plan, dataloader, eval_cfg, init_cfg, self.cost_obj,
            silent=True, return_routes=True, device=self.device,
            n_bees=int(search.n_bees), n_iterations=int(search.n_iterations),
            plan=self.plan, **self._adjustment_kwargs(),
        )
        mean_cost, std_cost, unserved, metrics, routes = out
        return {
            "mean_cost": float(mean_cost),
            "routes": routes.detach().cpu() if hasattr(routes, "detach") else routes,
            "metrics": {k: float(v.mean()) for k, v in metrics.items()},
        }

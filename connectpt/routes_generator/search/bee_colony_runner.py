"""BeeColonyRunner -- drive the existing bee_colony from a translated plan.

The runner holds the cost, the loaded models, the policy adapters and the
``BeeColonyPlan`` (per-type bee counts). ``run_suite`` builds the benchmark
dataloader + the bee_colony kwargs from the plan and runs the existing
``bee_colony`` through ``test_method`` -- the trained models are passed straight
in (no wrapper owns their state_dict). This layer must NOT import training.
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
        return {
            "counts": dict(self.plan.counts),
            "needs_construction": self.plan.needs_construction,
            "needs_edit": self.plan.needs_edit,
            "n_bees": int(search.get("n_bees", sum(self.plan.counts.values()))),
        }

    def _bco_kwargs(self) -> dict:
        """Translate the plan + objective into bee_colony keyword arguments."""
        counts = self.plan.counts
        obj = CostFactory.load_objective("rtt_wmc_no_demand")
        adj = obj.adjustment
        # bee_colony accepts n_type1/2/4/5/6/7 (type3 is not a kwarg).
        kwargs = {f"n_type{i}_bees": int(counts[f"n_type{i}"])
                  for i in (1, 2, 4, 5, 6, 7)}
        kwargs.update(
            bee_model=self.models.get("construction"),  # type1/type4 neural rebuild/extend
            edit_model=self.models.get("edit"),         # type5/6/7 edit bees
            adjustment_degree_weight=float(adj.weight),
            adjustment_degree_target=float(adj.target),
            adjustment_degree_objective=str(adj.objective),  # search -> two-sided "target"
            adjustment_degree_gap=float(adj.gap),
            adjustment_degree_mode=str(adj.mode),
        )
        return kwargs

    def run_seeded(self, init_routes, tensors, *, eval_dims, n_iterations=None):
        """Seeded improvement of an EXISTING network (init from ``init_routes``).

        Builds the tensor dataloader from ``tensors``, translates the declarative
        plan into the flat bee_colony search cfg (:func:`plan_to_search_cfg`) and
        runs the seeded executor. Reseeds from ``cfg.run.seed`` immediately before
        the run so the search is reproducible independent of model-init RNG.
        Returns ``(routes, unserved, metrics)``.
        """
        from omegaconf import OmegaConf
        from torch_geometric.loader import DataLoader

        from ..citygraph_dataset import get_dataset_from_config
        from ..core.runtime import seed_everything
        from .plan_kwargs import plan_to_search_cfg
        from .seeded_search import run_seeded_bee_colony

        dataloader = DataLoader(
            get_dataset_from_config(OmegaConf.create({"type": "tensor"}), tensors=tensors),
            batch_size=1)
        eval_cfg = OmegaConf.create(dict(eval_dims))
        search = self.cfg.search
        acceptance = search.get("acceptance")
        search_cfg = plan_to_search_cfg(
            self.plan, n_bees=int(search.n_bees),
            n_iterations=int(search.n_iterations if n_iterations is None else n_iterations),
            acceptance=None if acceptance is None else dict(acceptance))

        seed_everything(int(self.cfg.run.get("seed", 0)))
        out = run_seeded_bee_colony(
            dataloader, eval_cfg, self.cost_obj, init_routes, search_cfg=search_cfg,
            bee_model=self.models.get("construction"), edit_model=self.models.get("edit"),
            device=self.device, silent=True)
        _mean, _std, unserved, metrics, routes = out
        return routes, unserved, metrics

    def run_suite(self):
        """Run the full bee-colony search on the benchmark instance."""
        from omegaconf import OmegaConf
        from torch_geometric.loader import DataLoader

        from ..bee_colony import bee_colony
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
            bee_colony, dataloader, eval_cfg, init_cfg, self.cost_obj,
            silent=True, return_routes=True, device=self.device,
            n_bees=int(search.n_bees), n_iterations=int(search.n_iterations),
            **self._bco_kwargs(),
        )
        mean_cost, std_cost, unserved, metrics, routes = out
        return {
            "mean_cost": float(mean_cost),
            "routes": routes.detach().cpu() if hasattr(routes, "detach") else routes,
            "metrics": {k: float(v.mean()) for k, v in metrics.items()},
        }

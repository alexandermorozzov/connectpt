"""BeeColonySearchRun -- config-driven flexible bee-colony search.

setup() builds the cost (unified objective), loads each enabled model through
the factory + strict checkpoint load (no wrapper, so old weights load), builds
the policy adapters and translates the BeeSpecs into a BeeColonyPlan. The search
layer never imports the training layer; it consumes checkpoints + configs only.

``run(dry_run=True)`` validates the whole wiring without running the BCO loop:
benchmark config, cost, models (strict-loaded), policies, bee operators + plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactStore
from ..core.checkpoints import CheckpointStore
from ..core.paths import ROOT_DIR
from ..core.runs import ExperimentRun, RunArtifact
from ..core.runtime import RunContext
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from .bee_colony_runner import BeeColonyRunner
from .bee_plan import BeeColonyPlan
from .bee_specs import parse_bee_specs
from .benchmark_data import BenchmarkDataModule
from .search_policies import build_policies


@dataclass
class SearchArtifact(RunArtifact):
    result: Any = None
    plan: dict = field(default_factory=dict)


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


class BeeColonySearchRun(ExperimentRun):
    _ROLE_BUILDER = {
        "construction": RouteModelFactory.build_construction_model_by_name,
        "edit": RouteModelFactory.build_edit_model_by_name,
    }

    def setup(self) -> None:
        cfg = self.cfg
        self.context = RunContext.create(
            cfg.run.name, cfg.paths.output_dir,
            seed=int(cfg.run.seed), cpu=bool(cfg.run.get("cpu", False)),
        )
        device = self.context.device

        # cost: built config-first from the YAML cost base + unified objective
        # in its search/eval form (two-sided adjustment "target").
        self.cost_obj = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False)
        self.cost_obj.to(device)

        # models: build (role-checked) + strict-load the checkpoint into the class
        self.models: dict = {}
        for role in ("construction", "edit"):
            m_cfg = cfg.models.get(role)
            if m_cfg is None or not m_cfg.get("enabled", False):
                continue
            model = self._ROLE_BUILDER[role](m_cfg.config)
            CheckpointStore.load_model_weights(
                model, _resolve(m_cfg.checkpoint_path),
                strict=bool(m_cfg.get("strict_load", True)), map_location=device,
            )
            model.to(device)
            self.models[role] = model

        self.policies = build_policies(cfg.policies, self.models)
        self.specs = parse_bee_specs(cfg.bees)
        self.plan = BeeColonyPlan.from_specs(self.specs, self.policies)

        self.data = BenchmarkDataModule(
            city=cfg.data.city,
            n_routes=int(cfg.data.n_routes),
            min_route_len=int(cfg.data.min_route_len),
            max_route_len=int(cfg.data.max_route_len),
        )
        self.runner = BeeColonyRunner(
            cfg, cost_obj=self.cost_obj, models=self.models,
            policies=self.policies, plan=self.plan, data_module=self.data,
            device=device,
        )

    def run(self, *, dry_run: bool = False, init_routes=None, tensors=None,
            eval_dims=None) -> SearchArtifact:
        self.setup()
        summary = self.runner.plan_summary()
        if dry_run:
            return SearchArtifact(
                run_name=self.cfg.run.name, output_dir=self.context.output_dir,
                plan=summary,
                metadata={"dry_run": True,
                          "models_loaded": sorted(self.models),
                          "policies": sorted(self.policies)},
            )

        # Seeded mode: improve an EXISTING network passed in (init_routes on the
        # provided graph tensors), the paper contract. eval_dims gives the route
        # bounds of that network. Falls back to from-scratch benchmark search.
        if init_routes is not None:
            if eval_dims is None or tensors is None:
                raise ValueError("seeded run requires both tensors and eval_dims")
            routes, unserved, metrics = self.runner.run_seeded(
                init_routes, tensors, eval_dims=eval_dims)
            return SearchArtifact(
                run_name=self.cfg.run.name, output_dir=self.context.output_dir,
                result={"routes": routes, "unserved": unserved, "metrics": metrics},
                plan=summary, metadata={"seeded": True},
            )

        result = self.runner.run_suite()

        # persist artifacts so the reports layer can read them back without
        # re-running the search.
        store = ArtifactStore(self.context.output_dir)
        store.save_json({"mean_cost": result["mean_cost"], "metrics": result["metrics"],
                         "plan": summary}, f"{self.cfg.run.name.replace('/', '_')}_search")
        if result.get("routes") is not None:
            store.save_routes({"routes": result["routes"]},
                              f"{self.cfg.run.name.replace('/', '_')}_routes")

        return SearchArtifact(
            run_name=self.cfg.run.name, output_dir=self.context.output_dir,
            result=result, plan=summary,
            metadata={"mean_cost": result["mean_cost"]},
        )

"""BeeColonySearchRun -- config-driven flexible bee-colony search.

setup() builds the cost (unified objective), loads each enabled model through
the factory + strict checkpoint load (no wrapper, so old weights load), builds
the policy adapters and translates the BeeSpecs into an ExecutablePlan. The
search layer never imports the training layer; it consumes checkpoints + configs
only.

``execute()`` improves the instance's initial network at ONE point (alpha,
adj_target) of ``cfg.sweep``; the script loops over the grid. The point has its
own run directory under ``output_dir``.
``run(dry_run=True)`` validates the whole wiring without running the BCO loop:
cost, models (strict-loaded), policies, bee operators + plan.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("connectpt.sweep")


from ..run.checkpoints import CheckpointStore
from ..run.context import resolve_weights_path, safe_run_name
from ..run.runs import ExperimentRun, RunArtifact
from ..run.context import RunContext
from ..run.schemas import (AdjustmentParams, InstanceParams, SearchPaths, SearchRunParams,
                           SweepParams, params)
from ..city.instances import load_instance
from ..objectives.metrics import full_metric_row, select_metrics
from ..policy.build import ModelProfileParams, load_policy
from ..run.config import compose_policy_model, policy_profile
from ..objectives.cost import build_cost
from .bee_colony_runner import BeeColonyRunner
from .bee_specs import parse_bee_specs
from .executable_plan import ExecutablePlan
from .search_policies import build_policies


@dataclass
class SearchArtifact(RunArtifact):
    result: Any = None
    plan: dict = field(default_factory=dict)
    table: Any = None                # one-row pd.DataFrame of the point
    routes: dict = field(default_factory=dict)
    instance: Any = None             # the loaded Instance


class BeeColonySearchRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.run_params = params(SearchRunParams, cfg.run)
        self.paths = params(SearchPaths, cfg.paths)
        self.instance = params(InstanceParams, cfg.instance)
        self.sweep = params(SweepParams, cfg.sweep)
        self.adjustment = AdjustmentParams.from_cost(cfg.cost.kwargs)
        run = self.run_params
        self.context = RunContext.create(run.name, self.paths.output_dir,
                                         seed=run.seed, cpu=run.cpu)
        device = self.context.device

        # cost: the /cost group this run composed (single source).
        self.cost_obj = build_cost(cfg.cost, symmetric_routes=cfg.experiment.symmetric_routes,
                                   low_memory_mode=cfg.experiment.low_memory_mode)
        self.cost_obj.to(device)

        # models: build (role-checked) + strict-load the checkpoint into the class
        self.models: dict = {}
        for role in ("construction", "edit"):
            m_cfg = cfg.models.get(role)
            if m_cfg is None:
                continue
            profile = ModelProfileParams(**m_cfg)
            model = load_policy(policy_profile(compose_policy_model(profile.config), cfg.cost,
                cfg.experiment, device, checkpoint_path=resolve_weights_path(
                    profile.checkpoint_path, self.paths.weights_dir)))
            if model.action_spec.role != role:
                raise ValueError(f"profile for {role} has actions {model.action_spec}")
            self.models[role] = model

        self.policies = build_policies(self.models)
        self.specs = parse_bee_specs(cfg.bees)
        self.plan = ExecutablePlan.from_specs(
            self.specs, self.policies, models=self.models,
            algo_cfg=cfg.search)

        self.runner = BeeColonyRunner(
            cost_obj=self.cost_obj, plan=self.plan, device=device,
            seed=run.seed, silent=run.silent,
            sequential=bool(cfg.search.process_neural_bees_sequentially),
            adjustment=asdict(self.adjustment))

        self.alpha = self.sweep.alpha
        self.adj_target = None if self.sweep.adj_target is None else float(self.sweep.adj_target)
        self.run_dir = (Path(self.context.output_dir) / safe_run_name(
            f"{run.label}_seed{run.seed}_a{self.alpha}_t{self.adj_target}"))

    def dry_run_artifact(self) -> SearchArtifact:
        return SearchArtifact(
            run_name=self.run_params.name, output_dir=self.context.output_dir,
            plan=self.runner.plan_summary(),
            metadata={"dry_run": True,
                      "models_loaded": sorted(self.models),
                      "policies": sorted(self.policies)},
        )

    def execute(self) -> SearchArtifact:
        import pandas as pd

        inst = load_instance(self.instance)
        alpha, adj_target = self.alpha, self.adj_target
        n_iterations = self.sweep.n_iterations
        label = self.run_params.label
        _log.info("search point | %s | alpha=%s adj_target=%s", label, alpha, adj_target)
        routes, _unserved, metrics = self.runner.run_seeded(
            inst.init_routes, inst.tensors, eval_dims=inst.spec,
            n_iterations=n_iterations, alpha=alpha, adj_target=adj_target,
            adj_weight=self.sweep.adj_weight, logger_list=self.logger_list)

        keep = list(self.cfg.get("metrics", []))
        row = full_metric_row(metrics, routes, inst.init_routes, adjustment=self.adjustment)
        row = select_metrics(row, keep) if keep else row
        row.update(method=label, alpha=alpha, adj_target=adj_target,
                   n_iterations=n_iterations)
        return SearchArtifact(
            run_name=self.run_params.name, output_dir=self.context.output_dir,
            table=pd.DataFrame([row]), plan=self.runner.plan_summary(), instance=inst,
            routes={"Initial": inst.init_routes,
                    f"{label} a={alpha} t={adj_target}": routes},
        )

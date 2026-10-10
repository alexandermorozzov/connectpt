"""Build typed search parameters and strict-loaded policy objects."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any

from .context import resolve_weights_path, safe_run_name
from .runs import ExperimentRun, RunArtifact
from .context import RunContext
from ..city.instances import InstanceParams
from .schemas import AdjustmentParams, SearchPaths, SearchRunParams, SweepParams, params
from .instances import load_instance
from ..objectives.metrics import select_metrics
from .evaluation import export_evaluation
from ..policy.build import build_policy
from .config import cost_params, policy_profile
from ..objectives.cost import build_cost
from ..search.plan import MethodParams, ExecutablePlan
from ..search.search import run_search
from .tasks import TASKS


_log = logging.getLogger("connectpt.sweep")

@dataclass
class SearchArtifact(RunArtifact):
    result: Any = None
    plan: dict = field(default_factory=dict)
    table: Any = None  # one-row pd.DataFrame of the point
    routes: dict = field(default_factory=dict)
    instance: Any = None  # the loaded Instance


class SearchRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.run_params = params(SearchRunParams, cfg.run)
        self.paths = params(SearchPaths, cfg.paths)
        self.instance = params(InstanceParams, cfg.instance)
        self.sweep = params(SweepParams, cfg.sweep)
        self.adjustment = AdjustmentParams.from_cost(cfg.cost.kwargs)
        run = self.run_params
        self.context = RunContext.create(run.name, self.paths.output_dir, seed=run.seed, cpu=run.cpu)
        device = self.context.device

        # cost: the /cost group this run composed (single source).
        self.cost_obj = build_cost(
            cost_params(cfg.cost), symmetric_routes=cfg.experiment.symmetric_routes, low_memory_mode=cfg.experiment.low_memory_mode
        )
        self.cost_obj.to(device)

        # models: build (role-checked) + strict-load the checkpoint into the class
        self.models: dict = {}
        self.bindings={}
        for role in ("construction", "edit"):
            if role not in cfg.models:
                continue
            m_cfg = cfg.models[role]
            model = build_policy(
                policy_profile(
                    m_cfg.model,
                    cfg.cost,
                    cfg.experiment,
                    device,
                    checkpoint_path=resolve_weights_path(m_cfg.checkpoint_path, self.paths.weights_dir),
                )
            )
            if model.action_spec.role != role:
                raise ValueError(f"profile for {role} has actions {model.action_spec}")
            self.models[role] = model
            self.bindings[role]=TASKS[m_cfg.env.name].binding(
                model,self.instance,OmegaConf.to_container(m_cfg.env.params,resolve=True))

        self.policies = self.models
        self.method = MethodParams.from_values(OmegaConf.to_container(cfg.method, resolve=True))
        self.plan = ExecutablePlan.from_method(self.method, self.policies,self.bindings)
        self.keep_metrics = list(cfg.metrics)

        self.alpha = self.sweep.alpha
        self.adj_target = None if self.sweep.adj_target is None else float(self.sweep.adj_target)
        self.run_dir = Path(self.context.output_dir) / safe_run_name(
            f"{run.label}_seed{run.seed}_a{self.alpha}_t{self.adj_target}"
        )

    def dry_run_artifact(self) -> SearchArtifact:
        return SearchArtifact(
            run_name=self.run_params.name,
            output_dir=self.context.output_dir,
            plan=self.plan_summary(),
            metadata={"dry_run": True, "models_loaded": sorted(self.models), "policies": sorted(self.policies)},
        )

    def execute(self) -> SearchArtifact:
        import pandas as pd

        inst = load_instance(self.instance)
        alpha, adj_target = self.alpha, self.adj_target
        n_iterations = self.sweep.n_iterations
        label = self.run_params.label
        _log.info("search point | %s | alpha=%s adj_target=%s", label, alpha, adj_target)
        adjustment = asdict(self.adjustment)
        if self.sweep.adj_weight is not None:
            adjustment["adjustment_degree_weight"] = float(self.sweep.adj_weight)
        if adj_target is not None:
            adjustment["adjustment_degree_target"] = adj_target
        result = run_search(
            inst,
            self.method,
            self.plan,
            alpha,
            adjustment,
            self.run_params.seed,
            n_iterations,
            cost=self.cost_obj,
            device=self.context.device,
            silent=self.run_params.silent,
            logger_list=self.logger_list,
        )
        routes = result.routes
        keep = self.keep_metrics
        row = export_evaluation(result.evaluation, reduction="mean")
        evaluation = result.evaluation
        self.metadata["evaluation_protocol"] = dict(
            goals=list(evaluation.goal_names), penalties=list(evaluation.penalty_names),
            preferences=evaluation.preferences.cpu().tolist(),
            penalty_weights=[value.cpu().tolist() for value in evaluation.penalty_weights],
            reference_routes=evaluation.reference_routes.cpu().tolist(),
            adjustment=adjustment,
        )
        row = select_metrics(row, keep) if keep else row
        row.update(method=label, alpha=alpha, adj_target=adj_target, n_iterations=n_iterations)
        return SearchArtifact(
            run_name=self.run_params.name,
            output_dir=self.context.output_dir,
            result=result,
            table=pd.DataFrame([row]),
            plan=self.plan_summary(),
            instance=inst,
            routes={"Initial": inst.init_routes, f"{label} a={alpha} t={adj_target}": routes},
        )

    def plan_summary(self):
        return {
            "counts": self.plan.summary(),
            "needs_construction": self.plan.needs_construction,
            "needs_edit": self.plan.needs_edit,
            "n_bees": self.plan.total_bees,
        }

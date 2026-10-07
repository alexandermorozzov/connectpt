"""BeeColonySearchRun -- config-driven flexible bee-colony search.

setup() builds the cost (unified objective), loads each enabled model through
the factory + strict checkpoint load (no wrapper, so old weights load), builds
the policy adapters and translates the BeeSpecs into an ExecutablePlan. The
search layer never imports the training layer; it consumes checkpoints + configs
only.

``run()`` improves the instance's initial network at ONE point (alpha,
adj_target) of ``cfg.sweep``; the script loops over the grid.
``run(dry_run=True)`` validates the whole wiring without running the BCO loop:
cost, models (strict-loaded), policies, bee operators + plan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("connectpt.sweep")


def _fs_key(text: Any) -> str:
    """Filesystem-safe token from a sweep point key (spaces / '=' -> '_')."""
    return "".join(c if c.isalnum() else "_" for c in str(text))

from ..core.checkpoints import CheckpointStore
from ..core.paths import resolve_weights_path
from ..core.runs import ExperimentRun, RunArtifact
from ..core.runtime import RunContext
from ..data import load_instance
from ..evaluation import full_metric_row, select_metrics
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
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

        # cost: the /cost group this run composed (single source).
        self.cost_obj = CostFactory.build(cfg.cost)
        self.cost_obj.to(device)

        # models: build (role-checked) + strict-load the checkpoint into the class
        self.models: dict = {}
        for role in ("construction", "edit"):
            m_cfg = cfg.models.get(role)
            if m_cfg is None:
                continue
            model = self._ROLE_BUILDER[role](m_cfg.config)
            CheckpointStore.load_model_weights(
                model,
                resolve_weights_path(m_cfg.checkpoint_path, cfg.paths.weights_dir),
                strict=True, map_location=device)
            model.to(device)
            self.models[role] = model

        self.policies = build_policies(self.models)
        self.specs = parse_bee_specs(cfg.bees)
        self.plan = ExecutablePlan.from_specs(
            self.specs, self.policies, models=self.models,
            algo_cfg=cfg.search)

        self.runner = BeeColonyRunner(
            cost_obj=self.cost_obj, plan=self.plan, device=device,
            seed=int(cfg.run.seed), silent=bool(cfg.run.get("silent", True)),
            sequential=bool(cfg.search.process_neural_bees_sequentially),
            adjustment={k: v for k, v in cfg.cost.kwargs.items()
                        if k.startswith("adjustment_degree_")})

    def run(self, *, dry_run: bool = False) -> SearchArtifact:
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

        import pandas as pd

        inst = load_instance(self.cfg.instance)
        sweep = self.cfg.sweep
        alpha = sweep.alpha
        adj_target = sweep.get("adj_target")
        adj_target = None if adj_target is None else float(adj_target)
        n_iterations = int(sweep.n_iterations)
        label = self.cfg.run.get("label", self.cfg.run.name)
        _log.info("search point | %s | alpha=%s adj_target=%s", label, alpha, adj_target)

        writer = None
        if bool((self.cfg.get("logging") or {}).get("tensorboard", False)):
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(self.context.output_dir) / "tb" / _fs_key(f"a{alpha}_t{adj_target}")
            writer = SummaryWriter(str(tb_dir))
        try:
            routes, _unserved, metrics = self.runner.run_seeded(
                inst.init_routes, inst.tensors, eval_dims=inst.spec,
                n_iterations=n_iterations, alpha=alpha, adj_target=adj_target,
                adj_weight=sweep.get("adj_weight"), sum_writer=writer)
        finally:
            if writer is not None:
                writer.close()

        keep = list(self.cfg.get("metrics", []))
        row = full_metric_row(metrics, routes, inst.init_routes,
                              adjustment=self.cfg.cost.kwargs)
        row = select_metrics(row, keep) if keep else row
        row.update(method=label, alpha=alpha, adj_target=adj_target,
                   n_iterations=n_iterations)
        return SearchArtifact(
            run_name=self.cfg.run.name, output_dir=self.context.output_dir,
            table=pd.DataFrame([row]), plan=summary, instance=inst,
            routes={"Initial": inst.init_routes,
                    f"{label} a={alpha} t={adj_target}": routes},
        )

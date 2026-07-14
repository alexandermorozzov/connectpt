"""BeeColonySearchRun -- config-driven flexible bee-colony search.

setup() builds the cost (unified objective), loads each enabled model through
the factory + strict checkpoint load (no wrapper, so old weights load), builds
the policy adapters and translates the BeeSpecs into an ExecutablePlan. The
search layer never imports the training layer; it consumes checkpoints + configs
only.

``run(dry_run=True)`` validates the whole wiring without running the BCO loop:
benchmark config, cost, models (strict-loaded), policies, bee operators + plan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import ListConfig

_log = logging.getLogger("connectpt.sweep")


def _fs_key(text: Any) -> str:
    """Filesystem-safe token from a sweep point key (spaces / '=' -> '_')."""
    return "".join(c if c.isalnum() else "_" for c in str(text))

from ..core.artifacts import ArtifactStore
from ..core.checkpoints import CheckpointStore
from ..core.paths import resolve_under_root
from ..core.runs import ExperimentRun, RunArtifact
from ..core.runtime import RunContext
from ..data import BenchmarkDataSource, create_data_source
from ..evaluation import full_metric_row, select_metrics
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory, load_bco_algo_config
from .bee_colony_runner import BeeColonyRunner
from .bee_specs import parse_bee_specs
from .executable_plan import ExecutablePlan
from .benchmark_data import BenchmarkDataModule
from .search_policies import build_policies
from .sweep import run_sweep_table


@dataclass
class SearchArtifact(RunArtifact):
    result: Any = None
    plan: dict = field(default_factory=dict)
    table: Any = None                # pd.DataFrame for a sweep (else None)
    routes: dict = field(default_factory=dict)
    instance: Any = None             # the loaded Instance for a sweep


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
                model, resolve_under_root(m_cfg.checkpoint_path),
                strict=bool(m_cfg.get("strict_load", True)), map_location=device,
            )
            model.to(device)
            self.models[role] = model

        self.policies = build_policies(cfg.policies, self.models)
        self.specs = parse_bee_specs(cfg.bees)
        self.plan = ExecutablePlan.from_specs(
            self.specs, self.policies, models=self.models,
            algo_cfg=load_bco_algo_config())

        # from-scratch benchmark suite needs a BenchmarkDataModule; a sweep
        # (``cfg.data.source`` set, or ekb/macsa) loads its instance in run()
        # instead, so skip the module here.
        self.data = None
        if (cfg.data.get("source") is None
                and cfg.data.get("city") is not None
                and cfg.data.get("n_routes") is not None):
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

        # Config-driven sweep: load the instance from ``cfg.data`` and improve it
        # across the alpha x adj_target grid (the notebook's per-experiment loop).
        if self.cfg.get("sweep") is not None:
            return self._run_sweep(summary)

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

    # -- config-driven sweep ------------------------------------------------

    def _load_instance(self):
        """Load the experiment instance from ``cfg.data``.

        Supports the source shape (``source: ekb|macsa|benchmark`` + params) and
        the plain benchmark shape (``city:`` + bounds, the from-scratch data
        group) -- the latter is loaded as a benchmark source so a benchmark
        sweep and an ekb/macsa sweep share one code path.
        """
        d = self.cfg.data
        if d.get("source") is not None:
            return create_data_source(d).load()
        return BenchmarkDataSource(
            city=d.city, n_routes=d.get("n_routes"),
            min_route_len=d.get("min_route_len"),
            max_route_len=d.get("max_route_len"),
        ).load()

    def _sweep_grid(self):
        """Parse the ``cfg.sweep`` block into (alphas, adj_targets, n_iterations).

        ``adj_target`` may be a scalar or a list (2D alpha x target Pareto sweep).
        """
        sweep = self.cfg.sweep
        alphas = list(sweep.get("alpha", [None]))
        _at = sweep.get("adj_target")
        if isinstance(_at, (list, ListConfig)):
            adj_targets = [float(t) for t in _at]
        else:
            adj_targets = [None if _at is None else float(_at)]
        n_iterations = (int(sweep.n_iterations)
                        if sweep.get("n_iterations") is not None else None)
        return alphas, adj_targets, n_iterations

    def _run_sweep(self, summary) -> SearchArtifact:
        import pandas as pd
        import torch

        inst = self._load_instance()
        alphas, adj_targets, n_iterations = self._sweep_grid()
        keep = list(self.cfg.get("metrics", []))
        label = self.cfg.run.get("label", self.cfg.run.name)
        # adj_weight override (0 = adjustment penalty off, e.g. E2 5-model).
        adj_weight = self.cfg.sweep.get("adj_weight")

        # Crash-safety + online visibility: each grid point is persisted the
        # moment it finishes (partial CSV row + route dump) and, when enabled,
        # streamed to TensorBoard per BCO iteration -- a crash never loses the
        # points already done. Owned here (not the script) so notebooks share it.
        out_dir = Path(self.context.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.cfg.run.name.replace("/", "_")
        partial_csv = out_dir / f"{stem}_partial.csv"
        if partial_csv.exists():
            partial_csv.unlink()          # fresh run (resume is opt-in future work)
        routes_dir = out_dir / "partial_routes"
        routes_dir.mkdir(exist_ok=True)
        logging_cfg = self.cfg.get("logging") or {}
        tb_enabled = bool(logging_cfg.get("tensorboard", False))
        tb_root = out_dir / "tb"

        def _writer(alpha, adj_target):
            if not tb_enabled:
                return None
            from torch.utils.tensorboard import SummaryWriter
            tag = _fs_key(f"a{alpha}_t{adj_target}")
            return SummaryWriter(str(tb_root / tag))

        def run_point(_label, _method, alpha, adj_target):
            _log.info("sweep point | %s | alpha=%s adj_target=%s", _label, alpha,
                      adj_target)
            writer = _writer(alpha, adj_target)
            try:
                routes, _unserved, metrics = self.runner.run_seeded(
                    inst.init_routes, inst.tensors, eval_dims=inst.spec,
                    n_iterations=n_iterations, alpha=alpha, adj_target=adj_target,
                    adj_weight=adj_weight, sum_writer=writer)
            finally:
                if writer is not None:
                    writer.close()
            return routes, metrics

        def score(m, routes):
            row = full_metric_row(m, routes, inst.init_routes)
            return select_metrics(row, keep) if keep else row

        def on_point(row, key, out_routes):
            pd.DataFrame([row]).to_csv(
                partial_csv, mode="a", header=not partial_csv.exists(), index=False)
            try:
                torch.save(out_routes, routes_dir / f"{_fs_key(key)}.pt")
            except Exception:                       # a dump failure must not kill the run
                _log.warning("could not dump partial routes for %s", key)

        total = len(alphas) * len(adj_targets)
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total, desc=stem, unit="pt")
        except Exception:
            pbar = None
        try:
            table, routes = run_sweep_table(
                methods=[(label, None)], alpha_grid=alphas, adj_targets=adj_targets,
                init_routes=inst.init_routes, run_point=run_point, score=score,
                extra=({} if n_iterations is None else {"n_iterations": n_iterations}),
                on_point=on_point, progress=pbar)
        finally:
            if pbar is not None:
                pbar.close()

        return SearchArtifact(
            run_name=self.cfg.run.name, output_dir=self.context.output_dir,
            table=table, routes=routes, instance=inst, plan=summary,
            metadata={"sweep": True, "label": inst.label, "n_points": len(table)},
        )

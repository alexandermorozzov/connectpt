"""ModelEvaluationRun -- evaluate a checkpoint into structured artifacts.

setup() builds the edit model (factory) + strict-loads the checkpoint + cost
(unified objective, eval form). run(dry_run=True) validates the wiring; a real
run loads the eval graphs/seed routes, evaluates and saves an EvaluationResult.
Evaluation consumes the LC graphs/seed routes directly (it does not use the
search benchmark data module -- a different data shape).
"""
from __future__ import annotations

from types import SimpleNamespace

from ..core.artifacts import ArtifactStore
from ..core.checkpoints import CheckpointStore
from ..core.paths import DATASETS_DIR, resolve_weights_path
from ..core.runs import ExperimentRun, RunArtifact
from ..core.runtime import RunContext
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from .evaluators import EditModelEvaluator


class ModelEvaluationRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.context = RunContext.create(
            cfg.run.name, cfg.paths.output_dir,
            seed=int(cfg.run.seed), cpu=bool(cfg.run.get("cpu", False)),
        )
        device = self.context.device

        self.model = RouteModelFactory.build_edit_model_by_name(cfg.model.config)
        CheckpointStore.load_model_weights(
            self.model,
            resolve_weights_path(cfg.model.checkpoint_path,
                                 cfg.paths.get("weights_dir")),
            strict=bool(cfg.model.get("strict_load", True)), map_location=device,
        )
        self.model.to(device)

        self.cost_obj = CostFactory.build_unified("rtt_wmc_no_demand", for_training=False)
        self.cost_obj.to(device)

        self.dataset_dir = DATASETS_DIR / cfg.data.dataset_dirname
        self.evaluator = EditModelEvaluator()
        self.store = ArtifactStore(self.context.output_dir)

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        self.setup()
        if dry_run:
            return RunArtifact(
                run_name=self.cfg.run.name, output_dir=self.context.output_dir,
                metadata={"dry_run": True, "model_class": type(self.model).__name__},
            )

        from ..improvement_learning import load_raw_graphs_and_lc_routes
        graphs, seed_routes = load_raw_graphs_and_lc_routes(
            self.dataset_dir / "raw_graphs_1000.pkl", self.dataset_dir)
        data = SimpleNamespace(graphs=graphs, seed_routes=seed_routes)
        indices = list(range(len(graphs)))
        result = self.evaluator.evaluate(
            self.model, self.cost_obj, data, indices,
            min_route_len=int(self.cfg.data.min_route_len),
            max_route_len=int(self.cfg.data.max_route_len),
            metadata={"run_name": self.cfg.run.name},
        )
        saved = result.save(self.store, self.cfg.run.name)
        return RunArtifact(
            run_name=self.cfg.run.name, output_dir=self.context.output_dir,
            metadata={"saved": saved},
        )

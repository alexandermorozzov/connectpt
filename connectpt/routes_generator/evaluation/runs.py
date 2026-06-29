"""ModelEvaluationRun -- evaluate a checkpoint into structured artifacts.

setup() builds the edit model (factory) + strict-loads the checkpoint + cost
(unified objective, eval form) + benchmark data module. run(dry_run=True)
validates the wiring; a real run evaluates and saves an EvaluationResult.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ..core.artifacts import ArtifactStore
from ..core.checkpoints import CheckpointStore
from ..core.paths import ROOT_DIR
from ..core.runs import ExperimentRun, RunArtifact
from ..core.runtime import RunContext
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from ..search.benchmark_data import BenchmarkDataModule
from .evaluators import EditModelEvaluator


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


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
            self.model, _resolve(cfg.model.checkpoint_path),
            strict=bool(cfg.model.get("strict_load", True)), map_location=device,
        )
        self.model.to(device)

        cost_cfg = OmegaConf.create({"type": "mine",
                                     "kwargs": {"use_weighted_connectivity": True}})
        self.cost_obj = CostFactory.build(cost_cfg)
        CostFactory.apply_objective(self.cost_obj, "rtt_wmc_no_demand", for_training=False)
        self.cost_obj.to(device)

        self.data = BenchmarkDataModule(
            benchmark_dirname=cfg.data.benchmark_dirname,
            min_route_len=int(cfg.data.min_route_len),
            max_route_len=int(cfg.data.max_route_len),
        )
        self.evaluator = EditModelEvaluator()
        self.store = ArtifactStore(self.context.output_dir)

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        self.setup()
        if dry_run:
            return RunArtifact(
                run_name=self.cfg.run.name, output_dir=self.context.output_dir,
                metadata={"dry_run": True, "model_class": type(self.model).__name__},
            )

        self.data.setup()
        indices = list(range(len(self.data.graphs)))
        result = self.evaluator.evaluate(
            self.model, self.cost_obj, self.data, indices,
            min_route_len=int(self.cfg.data.min_route_len),
            max_route_len=int(self.cfg.data.max_route_len),
            metadata={"run_name": self.cfg.run.name},
        )
        saved = result.save(self.store, self.cfg.run.name)
        return RunArtifact(
            run_name=self.cfg.run.name, output_dir=self.context.output_dir,
            metadata={"saved": saved},
        )

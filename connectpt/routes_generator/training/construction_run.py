"""ConstructionTrainingRun -- skeleton for training the construction model.

Construction (from-scratch route building) is not currently retrained in the
paper workflow, so this run is a thin skeleton: it builds the construction model
(via the factory) and cost, and can warm-start from / validate a checkpoint. The
full PPO loop is wired the same way as EditTrainingRun when needed.
"""
from __future__ import annotations

from ..core.checkpoints import CheckpointStore
from ..core.paths import resolve_weights_path
from ..core.runtime import RunContext
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from .runs import ExperimentRun, TrainingArtifact


class ConstructionTrainingRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.context = RunContext.create(
            cfg.run.name, cfg.paths.output_dir,
            seed=int(cfg.run.seed), cpu=bool(cfg.run.get("cpu", False)),
        )
        device = self.context.device

        self.model = RouteModelFactory.build_construction_model(cfg.model, cfg.experiment)
        self.model.to(device)

        init_ckpt = cfg.paths.get("init_checkpoint_path")
        if init_ckpt:
            CheckpointStore.load_model_weights(
                self.model,
                resolve_weights_path(init_ckpt, cfg.paths.get("weights_dir")),
                strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
                map_location=device,
            )

        self.cost_obj = CostFactory.build(
            cfg.experiment.cost_function,
            symmetric_routes=bool(cfg.experiment.symmetric_routes),
        )
        CostFactory.apply_objective(self.cost_obj, "rtt_wmc_no_demand", for_training=True)
        self.cost_obj.to(device)

    def run(self, *, dry_run: bool = False) -> TrainingArtifact:
        self.setup()
        if dry_run:
            return TrainingArtifact(
                run_name=self.cfg.run.name,
                output_dir=self.context.output_dir,
                metadata={"dry_run": True, "model_class": type(self.model).__name__},
            )
        raise NotImplementedError(
            "ConstructionTrainingRun full training is not wired yet; use --dry-run "
            "or EditTrainingRun for the paper workflow."
        )

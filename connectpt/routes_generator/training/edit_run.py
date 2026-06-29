"""EditTrainingRun -- config-driven training of the edit / trim model.

Builds the data module, model (via the factory, so checkpoints still load with
strict=True), cost (via CostFactory + the unified objective) and the PPO trainer
from a composed ``train/edit`` cfg, then trains and saves a v2 checkpoint.

``run(dry_run=True)`` builds everything and validates the wiring WITHOUT loading
the (generated) dataset or running the long PPO loop -- the smoke check used by
``scripts/train_edit.py --dry-run``.
"""
from __future__ import annotations

from pathlib import Path

from ..core.checkpoints import CheckpointStore
from ..core.paths import DATASETS_DIR
from ..core.runtime import RunContext
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from .edit_ppo_trainer import EditPPOTrainer
from .runs import ExperimentRun, TrainingArtifact
from .training_data import TrainingDataModule


class EditTrainingRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.context = RunContext.create(
            cfg.run.name, cfg.paths.output_dir,
            seed=int(cfg.run.seed), cpu=bool(cfg.run.get("cpu", False)),
        )
        device = self.context.device

        # model -- built through the factory; never wrapped, so strict-load holds
        self.model = RouteModelFactory.build_edit_model(cfg.model, cfg.experiment)
        self.model.to(device)

        # optional warm-start checkpoint
        init_ckpt = cfg.paths.get("init_checkpoint_path")
        if init_ckpt:
            CheckpointStore.load_model_weights(
                self.model, init_ckpt,
                strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
                map_location=device,
            )

        # cost -- built then configured from the unified objective (training form)
        self.cost_obj = CostFactory.build(
            cfg.experiment.cost_function,
            symmetric_routes=bool(cfg.experiment.symmetric_routes),
        )
        CostFactory.apply_objective(self.cost_obj, "rtt_wmc_no_demand", for_training=True)
        self.cost_obj.to(device)

        # data module (not loaded yet -- loading happens in run())
        dataset_dir = DATASETS_DIR / cfg.data.dataset_dirname
        self.data = TrainingDataModule(
            raw_graphs_path=dataset_dir / "raw_graphs_1000.pkl",
            lc_results_dir=dataset_dir,
            device=device,
            min_route_len=int(cfg.data.min_route_len),
            max_route_len=int(cfg.data.max_route_len),
            target_n_routes=cfg.data.get("target_n_routes"),
        )

        self.best_path = Path(cfg.paths.checkpoint_path)
        self.trainer = EditPPOTrainer(
            cfg=cfg, model=self.model, cost_obj=self.cost_obj,
            data_module=self.data, run_name=cfg.run.name,
            output_dir=self.context.output_dir, device=device,
            best_model_path=self.best_path,
        )

    def run(self, *, dry_run: bool = False) -> TrainingArtifact:
        self.setup()
        if dry_run:
            return TrainingArtifact(
                run_name=self.cfg.run.name,
                output_dir=self.context.output_dir,
                metadata={"dry_run": True,
                          "model_class": type(self.model).__name__,
                          "dataset_dir": str(DATASETS_DIR / self.cfg.data.dataset_dirname)},
            )

        self.data.setup()
        history = self.trainer.fit()

        CheckpointStore.save_model_weights(self.model, self.best_path, cfg=self.cfg)
        return TrainingArtifact(
            run_name=self.cfg.run.name,
            output_dir=self.context.output_dir,
            checkpoint_path=self.best_path,
            history=history,
            metadata={"model_class": type(self.model).__name__},
        )

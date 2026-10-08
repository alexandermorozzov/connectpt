"""EditTrainingRun -- config-driven training of the edit / trim model.

Builds the data module, model (via the factory, so checkpoints still load with
strict=True), cost (via build_cost + the unified objective) and the PPO trainer
from a composed ``training/edit`` cfg, then trains and saves a v2 checkpoint.

``run(dry_run=True)`` builds everything and validates the wiring WITHOUT loading
the (generated) dataset or running the long PPO loop -- the smoke check used by
``scripts/train_edit.py --dry-run``.
"""
from __future__ import annotations

from ..run.checkpoints import CheckpointStore
from ..run.context import DATASETS_DIR, resolve_weights_path
from ..run.context import RunContext
from ..run.schemas import (CurriculumParams, EditDatasetParams, TrainingPaths,
                           TrainingRunParams, params)
from ..policy.build import load_policy
from ..run.config import policy_profile
from ..objectives.cost import build_cost
from .edit_ppo_trainer import EditPPOTrainer
from .runs import ExperimentRun, TrainingArtifact
from .training_data import TrainingDataModule


class EditTrainingRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.run_params = params(TrainingRunParams, cfg.run)
        self.paths = params(TrainingPaths, cfg.paths)
        self.dataset = params(EditDatasetParams, cfg.dataset)
        self.curriculum = params(CurriculumParams, cfg.curriculum)
        run = self.run_params
        self.context = RunContext.create(run.name, self.paths.output_dir,
                                         seed=run.seed, cpu=run.cpu)
        device = self.context.device

        # model -- built through the factory; never wrapped, so strict-load holds
        self.model = load_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, device))
        from ..env.actions import ActionSpec
        ActionSpec("edit").validate(self.model.action_spec)
        self.model.to(device)

        # optional warm-start checkpoint
        if self.paths.init_checkpoint_path:
            CheckpointStore.load_model_weights(
                self.model,
                resolve_weights_path(self.paths.init_checkpoint_path, self.paths.weights_dir),
                strict=True, map_location=device)

        # cost -- everything (weights, connectivity, adjustment, sampling
        # fractions) comes from cfg.experiment.cost_function
        self.cost_obj = build_cost(
            cfg.cost,
            symmetric_routes=bool(cfg.experiment.symmetric_routes),
            low_memory_mode=cfg.experiment.low_memory_mode,
        )
        self.cost_obj.to(device)

        # data module (not loaded yet -- loading happens in run())
        dataset_dir = DATASETS_DIR / self.dataset.dataset_dirname
        self.data = TrainingDataModule(
            raw_graphs_path=dataset_dir / "raw_graphs_1000.pkl",
            lc_results_dir=dataset_dir,
            device=device,
            min_route_len=self.dataset.min_route_len,
            max_route_len=self.dataset.max_route_len,
            target_n_routes=self.dataset.target_n_routes,
        )

        # Resolve against the weights root (CWD-independent; the notebook runs
        # from examples/route_generator) and ensure the dir exists -- the PPO
        # loop's intermediate best-checkpoint torch.save does not create it.
        self.best_path = resolve_weights_path(self.paths.checkpoint_path,
                                              self.paths.weights_dir)
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer = EditPPOTrainer(
            cfg=cfg, model=self.model, cost_obj=self.cost_obj,
            data_module=self.data, dataset=self.dataset, seed=run.seed,
            run_name=run.name, output_dir=self.context.output_dir, device=device,
            best_model_path=self.best_path,
        )
        self.run_dir = self.context.output_dir

    def _build_curriculum(self):
        """Build (train_idx, val_idx, curriculum_fn, val_curriculum_fn) from the
        curriculum group: split by tier, then the stage schedule."""
        cur = self.curriculum
        import pandas as pd
        dataset_dir = DATASETS_DIR / self.dataset.dataset_dirname
        meta = pd.read_csv(dataset_dir / "meta.csv")
        tier_of = dict(zip(meta["graph_index"].astype(int), meta["tier"]))
        tiers = list(cur.tiers)
        train_idx, val_idx, _monitor, train_by_tier, val_by_tier = self.data.stratified_split(
            tier_of, tiers, train_fraction=self.dataset.train_fraction,
            n_val_per_tier=cur.n_val_per_tier, seed=self.dataset.split_seed)
        n_iter = int(self.cfg.get("train_loop", {}).get("n_iterations",
                                                        self.cfg.ppo.n_iterations))
        schedule = [(round(float(frac) * n_iter), list(t), label)
                    for frac, t, label in cur.schedule]
        cfn, vfn = self.data.build_curriculum(schedule, train_by_tier, val_by_tier)
        return train_idx, val_idx, cfn, vfn

    def dry_run_artifact(self) -> TrainingArtifact:
        return TrainingArtifact(
            run_name=self.run_params.name,
            output_dir=self.context.output_dir,
            metadata={"dry_run": True,
                      "model_class": type(self.model).__name__,
                      "dataset_dir": str(DATASETS_DIR / self.dataset.dataset_dirname)},
        )

    def execute(self) -> TrainingArtifact:
        self.data.setup()
        train_idx, val_idx, curriculum_fn, val_curriculum_fn = self._build_curriculum()
        self.trainer.fit(
            logger_list=self.logger_list, train_indices=train_idx, val_indices=val_idx,
            curriculum_fn=curriculum_fn, val_curriculum_fn=val_curriculum_fn)

        CheckpointStore.save_model_weights(self.model, self.best_path, cfg=self.cfg)
        return TrainingArtifact(
            run_name=self.run_params.name,
            output_dir=self.context.output_dir,
            checkpoint_path=self.best_path,
            metadata={"model_class": type(self.model).__name__},
        )

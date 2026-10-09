"""Training run artifacts (lifecycle base lives in run.runs)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..common.logging import log
from ..env.actions import ActionSpec
from ..env.base import EnvParams
from ..network.corruption import CopyTierParams
from ..objectives.cost import build_cost
from ..policy.build import load_policy
from ..policy.checkpoints import load_model_weights, save_model_weights
from ..policy.rollout import ENVS
from ..training.algorithms.base import AlgorithmParams
from ..training.algorithms.d3po import D3PO
from ..training.algorithms.ppo import PPO
from ..training.critic import CriticParams, ValueHead
from ..training.data import ConstructionDataset
from ..training.edit_dataset import CopyTierConfig, EditDataset, build
from ..training.loop import Budget, train
from ..training.params import CurriculumParams, DatasetBuildParams, EditDatasetParams, LoopParams, OptimizerParams
from ..training.records import DataCounts
from ..training.validate import (
    ConstructionValidation, EditValidation, EvaluationParams, EvaluationProfile,
    evaluate_preferences, log_pareto_metrics,
)
from .artifacts import ArtifactStore
from .comparison import compare_runs
from .config import compose_policy_config, cost_params, policy_profile
from .const import DATASETS_DIR
from .context import RunContext, resolve_device, resolve_under_root, resolve_weights_path
from .instances import load_instance
from .policy import PolicyRun
from .runs import ExperimentRun, RunArtifact
from .schemas import ComparisonParams, OutputPaths, RunParams, TrainingPaths, TrainingRunParams, params


@dataclass
class TrainingArtifact(RunArtifact):
    """Result of a training run: the saved checkpoint (metrics: ``history.csv``)."""

    checkpoint_path: Path | None = None


class ComparisonRun(ExperimentRun):
    """Compare finished construction runs (``compare_runs``) on their test grid."""

    def setup(self) -> None:
        self.run_params = params(RunParams, self.cfg.run)
        self.comparison = params(ComparisonParams, self.cfg.comparison)
        output_dir = params(OutputPaths, self.cfg.paths).output_dir
        if output_dir is None:
            raise ValueError("set paths.output_dir to the comparison folder")
        self.run_dir = resolve_under_root(output_dir)

    def dry_run_artifact(self) -> RunArtifact:
        return RunArtifact(self.run_params.name, self.run_dir, {"dry_run": True})

    def execute(self) -> RunArtifact:
        c = self.comparison
        report = compare_runs(
            self.run_dir, tuple(c.names), checkpoint=c.checkpoint, mode=c.mode, budget_mode=c.budget_mode
        )
        return RunArtifact(self.run_params.name, self.run_dir, {"summary": report["summary"]})


class TrainingRun(ExperimentRun):
    def setup(self):
        cfg = self.cfg
        self.run_params, self.paths = params(TrainingRunParams, cfg.run), params(TrainingPaths, cfg.paths)
        self.context = RunContext.create(
            self.run_params.name, self.paths.output_dir, seed=self.run_params.seed, cpu=self.run_params.cpu
        )
        self.run_dir = self.context.output_dir
        device = self.context.device
        self.loop_params = params(LoopParams, cfg.train_loop)
        if self.loop_params.cpu_threads is not None:
            torch.set_num_threads(self.loop_params.cpu_threads)
        self.policy = load_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, device))
        self.env_name = cfg.env.name
        ActionSpec(self.env_name).validate(self.policy.action_spec)
        if self.paths.init_checkpoint_path:

            load_model_weights(
                self.policy,
                resolve_weights_path(self.paths.init_checkpoint_path, self.paths.weights_dir),
                strict=True,
                map_location=device,
            )
        self.cost = build_cost(
            cost_params(cfg.cost), symmetric_routes=cfg.experiment.symmetric_routes, low_memory_mode=cfg.experiment.low_memory_mode
        ).to(device)
        self.env_params = params(EnvParams, cfg.env.params)
        self.env = ENVS[self.env_name](
            self.cost,
            self.policy.feature_set,
            self.env_params,
            self.policy.action_spec,
            node_normalizer=self.policy.node_norm,
        )
        algorithm_values = OmegaConf.to_container(cfg.algorithm, resolve=True)
        self.critic_params = CriticParams(**algorithm_values.pop("critic"))
        algorithm_params = AlgorithmParams(**algorithm_values)
        self.algorithm = {"ppo": PPO, "d3po": D3PO}[algorithm_params.name](algorithm_params, self.cost)
        self.optimizer_params = params(OptimizerParams, cfg.optimizer)
        self.evaluation = params(EvaluationParams, cfg.evaluation)
        self.batch_size = self.loop_params.batch_size_cuda if device.type == "cuda" else self.loop_params.batch_size
        self.evaluation_profile = EvaluationProfile(
            self.evaluation, self.env_params, self.batch_size, cfg.experiment.symmetric_routes
        )
        dataset_values = OmegaConf.to_container(cfg.dataset, resolve=True)

        if self.env_name == "edit":
            dataset_values["dataset_dirname"] = resolve_under_root(DATASETS_DIR / dataset_values["dataset_dirname"])
        else:
            dataset_values["test_path"] = resolve_under_root(dataset_values["test_path"])
            for key in ("path", "val_path"):
                if key in dataset_values["kwargs"]:
                    dataset_values["kwargs"][key] = resolve_under_root(dataset_values["kwargs"][key])
            if dataset_values["type"] == "mumford":
                dataset_values["kwargs"]["city"] = load_instance(dataset_values["kwargs"]["city"])
        self.dataset = {"construction": ConstructionDataset, "edit": EditDataset}[self.env_name](**dataset_values)
        self.curriculum = params(CurriculumParams, cfg.curriculum)
        self.checkpoint = resolve_weights_path(self.paths.checkpoint_path, self.paths.weights_dir)
        self.best_checkpoint = self.checkpoint.with_name(
            "best.pt" if self.checkpoint.name == "last.pt" else self.checkpoint.stem + "_best.pt"
        )

    def dry_run_artifact(self):
        return TrainingArtifact(
            self.run_params.name,
            self.run_dir,
            {
                "dry_run": True,
                "model_class": type(self.policy).__name__,
                "environment": self.env_name,
                "algorithm": self.algorithm.params.name,
            },
            self.checkpoint,
        )

    def execute(self):
        device, lp = self.context.device, self.loop_params
        opt = self.optimizer_params
        optimizer = getattr(torch.optim, opt.name)(
            self.policy.parameters(), lr=opt.learning_rate, weight_decay=opt.decay, maximize=True
        )
        data, val_cost = self.dataset.prepare(
            self.cost,
            self.batch_size,
            self.evaluation,
            device,
            self.env_params,
            self.curriculum,
            lp.resample_preferences_on_terminal,
        )
        log(
            self.logger_list,
            DataCounts(training_graphs=data.training_count, validation_graphs=data.validation_count),
            0,
        )
        val_env = ENVS[self.env_name](
            val_cost,
            self.policy.feature_set,
            self.env_params,
            self.policy.action_spec,
            node_normalizer=self.policy.node_norm,
        )
        data.validation = {"construction": ConstructionValidation, "edit": EditValidation}[self.env_name](
            data, val_cost, val_env, self.evaluation_profile, self.run_dir, device,
            save_json=ArtifactStore(self.run_dir).save_json,
        )
        critic = ValueHead(
            self.algorithm.value_dim,
            self.policy.feature_set,
            self.critic_params,
            device,
            scalar=self.algorithm.scalar_values,
            actor_model=self.policy if self.critic_params.shared else None,
            observe=self.env.observe if self.critic_params.shared else None,
        )
        budget = Budget(
            lp.n_iterations,
            lp.horizon,
            lp.val_period,
            lp.validate_before_update,
            lp.max_seconds,
            lp.warmup_batches,
            lp.max_rollout_samples,
            lp.calibrate_before_validation,
            lp.action_seed,
            lp.force_nonhalt_until,
            self.run_dir,
            self.run_params.name,
        )
        result = train(self.env, self.policy, self.algorithm, critic, optimizer, self.logger_list, data, budget)
        checkpoint_config = OmegaConf.to_container(self.cfg, resolve=True)
        save_model_weights(self.policy, self.checkpoint, checkpoint_config)
        save_model_weights(result.best_policy, self.best_checkpoint, checkpoint_config)
        torch.save(result.critic_snapshot, self.run_dir / "critic_snapshot.pt")
        ArtifactStore(self.run_dir).save_json(
            {"iterations": result.iterations, "best_cost": result.best_cost}, "training_result.json"
        )
        if self.evaluation.final_test:
            test_data, test_seeds = data.test()
            if self.evaluation.test_graph_count is not None:
                generator = torch.Generator().manual_seed(self.evaluation.seed)
                indices = (
                    torch.randperm(len(test_data), generator=generator)[: self.evaluation.test_graph_count]
                    .sort()
                    .values.tolist()
                )
                test_data = [test_data[index] for index in indices]
                if test_seeds is not None:
                    test_seeds = test_seeds[indices]
                ArtifactStore(self.run_dir).save_json(
                    dict(
                        indices=indices, seed=self.evaluation.seed, split="independent test", graph_count=len(test_data)
                    ),
                    "test_protocol.json",
                )
            for label, actor in (("last", self.policy), ("best", result.best_policy)):
                for mode in ("greedy", "stochastic"):
                    rows = evaluate_preferences(
                        actor,
                        test_data,
                        self.evaluation_profile,
                        self.cost,
                        self.run_dir,
                        device,
                        checkpoint=label,
                        mode=mode,
                        seed_networks=test_seeds,
                        save_json=ArtifactStore(self.run_dir).save_json,
                    )
                    if self.evaluation.pareto_enabled:
                        report = log_pareto_metrics(
                            rows,
                            self.evaluation_profile,
                            self.logger_list,
                            result.iterations,
                            f"eval/test/{label}/{mode}",
                        )
                        ArtifactStore(self.run_dir).save_json(report, f"test_pareto_{label}_{mode}.json")
        if result.iterations < lp.n_iterations:
            self.stop_reason = "time_budget"
        return TrainingArtifact(self.run_params.name, self.run_dir, {}, self.checkpoint)


def copytier_config(dg, data, policy_profile, policy_cost, policy_seed) -> "CopyTierConfig":
    """The dataset-generation config from the ``dataset_build`` and ``dataset``
    parameters."""

    ddir = DATASETS_DIR / data.dataset_dirname
    tiers = list(dg.tiers)
    combos = [list(c) for c in dg.lc_combos]
    return CopyTierConfig(
        new_dataset_dir=ddir,
        subset_pkl=ddir / "raw_graphs_1000.pkl",
        meta_csv=ddir / "meta.csv",
        raw_graph_seed=int(dg.raw_graph_seed),
        raw_n_nodes=int(dg.raw_n_nodes),
        raw_graph_type=str(dg.raw_graph_type),
        n_graphs=int(dg.n_graphs),
        tiers=tiers,
        lc_combos=combos,
        target_n_routes=int(data.target_n_routes),
        min_route_len=int(data.min_route_len),
        max_route_len=int(data.max_route_len),
        lc_n_samples=int(dg.lc_n_samples),
        tier_cfg={name: CopyTierParams(**values) for name, values in dg.tier_cfg.items()},
        force_regen=dg.force_regen,
        policy_profile=policy_profile,
        policy_cost=policy_cost,
        policy_seed=policy_seed,
        batch_size_cpu=dg.batch_size_cpu,
        batch_size_cuda=dg.batch_size_cuda,
        corruption_seed=dg.corruption_seed,
    )


class DatasetRun(ExperimentRun):
    """Generate the dataset described by a composed ``runs/prepare_dataset`` cfg."""

    def setup(self) -> None:
        cfg = self.cfg
        self.run_params = params(TrainingRunParams, cfg.run)

        policy_run = PolicyRun(compose_policy_config(), resolve_device(cpu=self.run_params.cpu))
        policy_run.setup()
        self.dataset = copytier_config(
            params(DatasetBuildParams, cfg.dataset_build),
            params(EditDatasetParams, cfg.dataset),
            policy_run.profile,
            policy_run.cost,
            policy_run.seed,
        )
        self.run_dir = resolve_under_root(params(OutputPaths, cfg.paths).output_dir)

    def _artifact(self, seconds, dry_run):
        return RunArtifact(
            run_name=self.run_params.name,
            output_dir=Path(self.dataset.new_dataset_dir),
            metadata={"dry_run": dry_run, "seconds_per_graph": seconds, "dataset": self.dataset},
        )

    def dry_run_artifact(self):
        return self._artifact(None, True)

    def execute(self):
        return self._artifact(build(self.dataset), False)

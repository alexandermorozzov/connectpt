"""Training run artifacts (lifecycle base lives in run.runs)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..common.logging import log
from ..env.actions import ActionSpec
from ..env.base import EnvParams
from ..objectives.cost import build_cost
from ..policy.build import build_policy
from ..policy.checkpoints import save_model_weights
from ..policy.rollout import ENVS
from ..training.algorithms.base import AlgorithmParams
from ..training.algorithms.d3po import D3PO
from ..training.algorithms.ppo import PPO
from ..training.critic import CriticParams, ValueHead
from ..training.data import ConstructionDataset
from ..training.edit_dataset import EditDataset
from ..training.loop import Budget, train
from ..training.params import CurriculumParams, LoopParams, OptimizerParams
from ..training.const import DatasetKind
from ..training.records import DataCounts
from ..training.validate import (
    ConstructionValidation, EditValidation, EvaluationParams, EvaluationProfile,
    evaluate_preferences, log_evaluation_metrics, log_pareto_metrics,
)
from .artifacts import ArtifactStore
from .config import cost_params, policy_profile
from .context import RunContext, resolve_weights_path
from .dataset import prepare_data_input
from .runs import ExperimentRun, RunArtifact
from .schemas import TrainingPaths, TrainingRunParams, params


@dataclass
class TrainingArtifact(RunArtifact):
    """Result of a training run: the saved checkpoint (metrics: ``history.csv``)."""

    checkpoint_path: Path | None = None


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
        checkpoint_path = (resolve_weights_path(self.paths.init_checkpoint_path, self.paths.weights_dir)
                           if self.paths.init_checkpoint_path else None)
        self.policy = build_policy(policy_profile(cfg.model, cfg.cost, cfg.experiment, device,
                                                  checkpoint_path=checkpoint_path))
        self.env_name = cfg.env.name
        ActionSpec(self.env_name).validate(self.policy.action_spec)
        self.cost = build_cost(
            cost_params(cfg.cost), symmetric_routes=cfg.experiment.symmetric_routes, low_memory_mode=cfg.experiment.low_memory_mode
        ).to(device)
        self.data_input = prepare_data_input(cfg.dataset, DatasetKind(self.env_name))
        limits = self.data_input.params.limits
        env_values = OmegaConf.to_container(cfg.env.params, resolve=True)
        for key, value in asdict(limits).items():
            if key in env_values and env_values[key] != value:
                raise ValueError(f'env.{key} conflicts with dataset limits')
            env_values[key] = value
        self.env_params = EnvParams(**env_values)
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
        prepared = self.data_input
        self.dataset = (EditDataset(prepared.params, prepared.train_pipeline, prepared.evaluation_pipeline)
                        if self.env_name == 'edit' else
                        ConstructionDataset(prepared.params, prepared.train_pipeline, prepared.evaluation_pipeline,
                                            prepared.online_source, prepared.benchmark))
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

    def evaluation_protocol(self, data, val_cost):
        """Describe the actual split, fixed evaluation inputs and configured budgets."""
        evaluation = self.evaluation
        if self.env_name == "construction":
            validation = dict(split="held-out validation", indices=data.validation_indices, n_routes=list(data.n_routes))
            monitor_counts = {"validation": data.validation_count}
        else:
            validation = dict(
                split="held-out validation", indices=data.all_val_indices.tolist(),
                monitor_indices=data.monitor_indices.tolist(),
            )
            monitor_counts = {'validation': len(data.monitor_indices)}
        validation.update(
            graph_count=data.validation_count, mode=evaluation.validation_mode,
            action_rng="training torch RNG" if evaluation.validation_mode != "greedy" else "greedy",
            preference_weights={key: torch.as_tensor(value).cpu().tolist() for key, value in val_cost.get_weights().items()},
        )
        protocol = dict(
            seed=evaluation.seed,
            grid_action_seed_rule="seed + preference_index * 1000 + sample_index",
            objectives=list(self.cost.objectives),
            env=asdict(self.env_params),
            validation=validation,
            monitor=dict(
                enabled=evaluation.pareto_enabled, period=evaluation.pareto_period,
                preference_count=evaluation.pareto_preference_count,
                preference_grid="linspace(0, 1, preference_count)",
                samples_per_preference=dict(greedy=1, stochastic=evaluation.pareto_stochastic_samples),
                reference=evaluation.hv_reference, eu_preference_count=evaluation.eu_preference_count,
                outcomes_per_call={
                    stage: {
                        mode: count * evaluation.pareto_preference_count * samples if evaluation.pareto_enabled else 0
                        for mode, samples in (("greedy", 1), ("stochastic", evaluation.pareto_stochastic_samples))
                    }
                    for stage, count in monitor_counts.items()
                },
            ),
            budget=dict(
                training=asdict(self.loop_params),
                validation_period=self.loop_params.val_period,
                validation_before_update=self.loop_params.validate_before_update,
            ),
        )
        protocol[data.final_split] = dict(
            enabled=evaluation.final_test, preference_count=evaluation.preference_count,
            preference_grid="linspace(0, 1, preference_count)",
            samples_per_preference=dict(greedy=1, stochastic=evaluation.stochastic_samples),
            checkpoints=["last", "best"],
        )
        protocol['source_splits'] = dict(train=data.training_source_ids, validation=data.validation_source_ids,
                                        test=data.test_source_ids)
        return protocol

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
        self.metadata["evaluation_protocol"] = self.evaluation_protocol(data, val_cost)
        self.write_metadata()
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
            data, val_cost, val_env, self.evaluation_profile, device,
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
            split_name = data.final_split
            test_data, test_seeds, source_indices = data.final_evaluation()
            indices = list(range(len(test_data)))
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
            test_protocol = self.metadata["evaluation_protocol"][split_name]
            test_protocol.update(
                split="independent test" if split_name == 'test' else "held-out validation",
                indices=[source_indices[i] for i in indices],
                graph_count=len(test_data),
                outcomes_per_checkpoint={
                    mode: len(test_data) * self.evaluation.preference_count * samples
                    for mode, samples in test_protocol["samples_per_preference"].items()
                },
            )
            self.write_metadata()
            for label, actor in (("last", self.policy), ("best", result.best_policy)):
                for mode in ("greedy", "stochastic"):
                    rows = evaluate_preferences(
                        actor,
                        test_data,
                        self.evaluation_profile,
                        self.cost,
                        device,
                        mode=mode,
                        seed_networks=test_seeds,
                    )
                    prefix = f"eval/{split_name}/{label}/{mode}"
                    log_evaluation_metrics(rows, self.logger_list, result.iterations, prefix)
                    if self.evaluation.pareto_enabled:
                        log_pareto_metrics(
                            rows,
                            self.evaluation_profile,
                            self.logger_list,
                            result.iterations,
                            prefix,
                        )
        if result.iterations < lp.n_iterations:
            self.stop_reason = "time_budget"
        return TrainingArtifact(self.run_params.name, self.run_dir, {}, self.checkpoint)

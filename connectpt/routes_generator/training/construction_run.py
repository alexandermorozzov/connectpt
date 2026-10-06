"""Construction model setup and preference-conditioned PPO/D3PO training."""
from __future__ import annotations

import logging
import time
from omegaconf import OmegaConf

from ..core.checkpoints import CheckpointStore
from ..core.artifacts import ArtifactStore
from ..core.paths import resolve_weights_path, resolve_under_root
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
            cfg.cost,
            symmetric_routes=bool(cfg.experiment.symmetric_routes),
        )
        self.cost_obj.to(device)

    def run(self, *, dry_run: bool = False) -> TrainingArtifact:
        self.setup()
        if dry_run:
            return TrainingArtifact(
                run_name=self.cfg.run.name,
                output_dir=self.context.output_dir,
                metadata={"dry_run": True, "model_class": type(self.model).__name__},
            )
        from ..citygraph_dataset import CityGraphDataset
        from ..inductive_route_learning import setup_and_train
        from ..improvement_learning import _write_history_checkpoint
        from .construction_experiment import MetricWriter, evaluate_test, state_dict_hash, log_pareto_metrics

        cfg, context = self.cfg, self.context
        output = context.output_dir
        artifacts = ArtifactStore(output)
        checkpoint = resolve_weights_path(cfg.paths.checkpoint_path, cfg.paths.weights_dir)
        writer = MetricWriter(output / "tensorboard")
        history = []
        initial_hash = state_dict_hash(self.model)
        started = time.monotonic()
        torch_threads = cfg.get("train_loop", {}).get("cpu_threads")
        if torch_threads is not None:
            import torch
            torch.set_num_threads(int(torch_threads))
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)

        def callback(iteration, stats, actor, critic, optimizer):
            history.append(dict(writer.latest, iteration=iteration))
            _write_history_checkpoint(history, output / "history.csv")
            if iteration % int(cfg.ppo.val_period) == 0:
                CheckpointStore.save_model_weights(actor, checkpoint, cfg)
            artifacts.save_json(dict(state="training", phase="training", algorithm=str(cfg.trainer),
                iteration=iteration, total_active_steps=stats["env transitions"]), "status.json")
            logging.getLogger(__name__).info("%s iteration=%d transitions=%d",
                cfg.trainer, iteration, stats["env transitions"])

        try:
            artifacts.save_json(dict(state="training", phase="initializing", algorithm=str(cfg.trainer), iteration=0), "status.json")
            test_data = None
            if bool(cfg.evaluation.final_test):
                test_data = CityGraphDataset(resolve_under_root(cfg.dataset.test_path))
                if len(test_data) == 0:
                    raise ValueError('Construction test dataset is empty')
                test_count = cfg.evaluation.get('test_graph_count')
                if test_count is not None:
                    import torch
                    if int(test_count) < 1:
                        raise ValueError('evaluation.test_graph_count must be positive or null')
                    generator = torch.Generator().manual_seed(int(cfg.evaluation.seed))
                    indices = torch.randperm(len(test_data), generator=generator)[:int(test_count)].sort().values.tolist()
                    test_data = [test_data[i] for i in indices]
                    artifacts.save_json(dict(indices=indices, seed=int(cfg.evaluation.seed),
                        split='independent test', graph_count=len(test_data)), 'test_protocol.json')
            best_model, best_cost = setup_and_train(
                cfg, model=self.model, cost_obj=self.cost_obj, device=context.device,
                run_name=cfg.run.name, sum_writer=writer, output_dir=output,
                iteration_callback=callback, return_training_result=True)
            training_seconds = time.monotonic() - started
            CheckpointStore.save_model_weights(self.model, checkpoint, cfg)
            CheckpointStore.save_model_weights(best_model, checkpoint.with_name("best.pt"), cfg)
            if test_data is not None:
                artifacts.save_json(dict(state="evaluating", phase="test", algorithm=str(cfg.trainer),
                    iterations=len(history), training_seconds=training_seconds,
                    total_active_steps=history[-1]['env transitions']), "status.json")
                for checkpoint_name, actor in (("last", self.model), ("best", best_model)):
                    for mode in ("greedy", "stochastic"):
                        rows = evaluate_test(actor, test_data, cfg, self.cost_obj, output, context.device,
                                             checkpoint=checkpoint_name, mode=mode)
                        if cfg.evaluation.get('pareto_enabled', False):
                            report = log_pareto_metrics(rows, cfg, writer, len(history),
                                                       f'test/{checkpoint_name}/{mode}')
                            artifacts.save_json(report, f'test_pareto_{checkpoint_name}_{mode}.json')
            limit = cfg.train_loop.get("max_seconds")
            artifacts.save_json(dict(state="completed", algorithm=str(cfg.trainer),
                iterations=len(history), training_seconds=training_seconds,
                total_active_steps=history[-1]["env transitions"],
                max_seconds=limit, stop_reason="iteration_limit"
                if len(history) >= int(cfg.ppo.n_iterations) else "time_budget",
                best_val_cost=float(best_cost), initial_actor_sha256=initial_hash,
                seed=int(cfg.run.seed), checkpoint=str(checkpoint),
                dataset_path=cfg.dataset.get('path', cfg.dataset.get('kwargs', {}).get('path')),
                evaluation_protocol="Holliday stochastic validation; shared preference-grid test",
                preference_reset_policy='episode' if cfg.ppo.get('resample_preferences_on_terminal', cfg.trainer != 'ppo') else 'batch',
                advantage_normalization=str(cfg.ppo.get('advantage_normalization', 'minibatch')),
                implementation="inductive_route_learning.setup_and_train"), "status.json")
            return TrainingArtifact(run_name=cfg.run.name, output_dir=output,
                                    checkpoint_path=checkpoint, history=history)
        except Exception as exc:
            artifacts.save_json(dict(state="failed", error=str(exc)), "status.json")
            raise
        finally:
            writer.close()

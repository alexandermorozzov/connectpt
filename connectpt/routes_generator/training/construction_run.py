"""Construction model setup and preference-conditioned PPO/D3PO training."""
from __future__ import annotations

import logging

from ..run.checkpoints import CheckpointStore
from ..run.artifacts import ArtifactStore
from ..run.logger import flush
from ..run.context import resolve_weights_path, resolve_under_root
from ..run.context import RunContext
from ..run.schemas import ConstructionDatasetParams, TrainingPaths, TrainingRunParams, params
from ..model_factory import RouteModelFactory
from ..objectives import CostFactory
from .runs import ExperimentRun, TrainingArtifact


class ConstructionTrainingRun(ExperimentRun):
    def setup(self) -> None:
        cfg = self.cfg
        self.run_params = params(TrainingRunParams, cfg.run)
        self.paths = params(TrainingPaths, cfg.paths)
        # still read by setup_and_train from cfg.dataset until Э7
        self.dataset = params(ConstructionDatasetParams, cfg.dataset)
        run = self.run_params
        self.context = RunContext.create(run.name, self.paths.output_dir,
                                         seed=run.seed, cpu=run.cpu)
        device = self.context.device

        self.model = RouteModelFactory.build_construction_model(cfg.model, cfg.experiment)
        self.model.to(device)

        if self.paths.init_checkpoint_path:
            CheckpointStore.load_model_weights(
                self.model,
                resolve_weights_path(self.paths.init_checkpoint_path, self.paths.weights_dir),
                strict=True, map_location=device)

        self.cost_obj = CostFactory.build(
            cfg.cost,
            symmetric_routes=bool(cfg.experiment.symmetric_routes),
        )
        self.cost_obj.to(device)
        self.run_dir = self.context.output_dir

    def dry_run_artifact(self) -> TrainingArtifact:
        return TrainingArtifact(
            run_name=self.run_params.name,
            output_dir=self.context.output_dir,
            metadata={"dry_run": True, "model_class": type(self.model).__name__},
        )

    def execute(self) -> TrainingArtifact:
        from ..citygraph_dataset import CityGraphDataset
        from ..inductive_route_learning import setup_and_train
        from .construction_experiment import evaluate_test, log_pareto_metrics

        cfg, context = self.cfg, self.context
        output = context.output_dir
        artifacts = ArtifactStore(output)
        checkpoint = resolve_weights_path(self.paths.checkpoint_path, self.paths.weights_dir)
        torch_threads = cfg.get("train_loop", {}).get("cpu_threads")
        if torch_threads is not None:
            import torch
            torch.set_num_threads(int(torch_threads))
        iterations = 0

        def callback(iteration, record, actor, critic, optimizer):
            nonlocal iterations
            iterations = iteration
            flush(self.logger_list)
            if iteration % int(cfg.ppo.val_period) == 0:
                CheckpointStore.save_model_weights(actor, checkpoint, cfg)
            logging.getLogger(__name__).info("%s iteration=%d transitions=%d",
                cfg.trainer, iteration, record.env_transitions)

        test_data = None
        if bool(cfg.evaluation.final_test):
            test_data = CityGraphDataset(resolve_under_root(self.dataset.test_path))
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
            run_name=self.run_params.name, logger_list=self.logger_list, output_dir=output,
            iteration_callback=callback, return_training_result=True)
        CheckpointStore.save_model_weights(self.model, checkpoint, cfg)
        CheckpointStore.save_model_weights(best_model, checkpoint.with_name("best.pt"), cfg)
        if test_data is not None:
            for checkpoint_name, actor in (("last", self.model), ("best", best_model)):
                for mode in ("greedy", "stochastic"):
                    rows = evaluate_test(actor, test_data, cfg, self.cost_obj, output, context.device,
                                         checkpoint=checkpoint_name, mode=mode)
                    if cfg.evaluation.get('pareto_enabled', False):
                        report = log_pareto_metrics(rows, cfg, self.logger_list, iterations,
                                                   f'test/{checkpoint_name}/{mode}')
                        artifacts.save_json(report, f'test_pareto_{checkpoint_name}_{mode}.json')
        if iterations < int(cfg.ppo.n_iterations):
            self.stop_reason = "time_budget"
        return TrainingArtifact(run_name=self.run_params.name, output_dir=output,
                                checkpoint_path=checkpoint)

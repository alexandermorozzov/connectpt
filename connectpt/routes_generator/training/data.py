"""Construction graph sources and episodes, independent of the algorithm."""
import random
from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

from ..city.synthetic import load_graph_collection
from .const import DatasetKind, DataSourceKind, DATASET_MANIFEST
from .dataset import load_dataset
from .params import DatasetInputParams
from .pipeline import GraphPipeline, GraphView, OnlineGraphs
from .split import graph_source_id, evaluation_indices


def evaluation_graphs(path, kind, limits):
    if (path / DATASET_MANIFEST).exists():
        return load_dataset(path, kind=kind, legacy_limits=limits).graphs
    return load_graph_collection(path)


@dataclass(frozen=True)
class ConstructionDataset:
    params: DatasetInputParams
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline
    online_source: object | None
    benchmark: object | None

    def prepare(self, cost, batch_size, evaluation, device, env_params, curriculum, resample_preferences):
        if curriculum.enabled:
            raise ValueError('construction data has no curriculum tiers; use curriculum.enabled=false')
        data = self.load(batch_size, evaluation, device, env_params, resample_preferences=resample_preferences)
        val_cost = cost.with_params(**cost.sample_variable_weights(data.validation_batch_size, device))
        return data, val_cost

    def load(self, batch_size, evaluation, device, env_params, *, resample_preferences):
        source = self.params.source
        test_graphs, test_ids = None, ()
        if self.params.test_path is not None:
            raw_test = evaluation_graphs(self.params.test_path, self.params.kind, self.params.limits)
            test_ids = tuple(graph_source_id(g) for g in raw_test)
            test_graphs = GraphView(raw_test, range(len(raw_test)), self.evaluation_pipeline)
        if source == DataSourceKind.FILE:
            dataset = load_dataset(self.params.path, kind=self.params.kind, legacy_limits=self.params.limits,
                                   legacy_split=self.params.legacy_split)
            graphs, split = dataset.graphs, dataset.split
            sources = dataset.params.source_ids
            train = GraphView(graphs, split.train, self.train_pipeline)
            validation = GraphView(graphs, split.validation, self.evaluation_pipeline)
            train_ids = tuple(sources[int(i)] for i in split.train)
            validation_ids = tuple(sources[int(i)] for i in split.validation)
            if len(split.test):
                if test_graphs is not None:
                    raise ValueError('choose either manifest test or a separate test_path')
                test_graphs = GraphView(graphs, split.test, self.evaluation_pipeline)
                test_ids = tuple(sources[int(i)] for i in split.test)
            if set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
                raise ValueError('source city leaks into independent test')
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            raw_validation = evaluation_graphs(self.params.validation_path, self.params.kind, self.params.limits)
            validation_ids = tuple(graph_source_id(g) for g in raw_validation)
            validation = GraphView(raw_validation, range(len(raw_validation)), self.evaluation_pipeline)
            if set(validation_ids) & set(test_ids):
                raise ValueError('source city leaks between validation and independent test')
            if source == DataSourceKind.DYNAMIC:
                train_ids = ()
                train = OnlineGraphs(self.online_source, self.train_pipeline, forbidden_sources=[*validation_ids,*test_ids])
            elif source == DataSourceKind.BENCHMARK:
                train_ids = (graph_source_id(self.benchmark),)
                if set(train_ids) & set((*validation_ids,*test_ids)):
                    raise ValueError('benchmark city leaks into validation or independent test')
                train = GraphView([self.benchmark], [0] * batch_size, self.train_pipeline)
            else:
                raise ValueError(f'unknown graph source: {source}')
            train_loader = DataLoader(train, batch_size=batch_size)
        full_validation = validation
        indices = evaluation_indices(range(len(validation)), evaluation.graph_count, evaluation.seed).tolist()
        if evaluation.graph_count is not None:
            validation = [validation[index] for index in indices]
        if not len(validation) or (source != DataSourceKind.DYNAMIC and len(train) < batch_size):
            raise ValueError('training needs a full batch and nonempty validation')
        validation_batch_size = batch_size * 4
        if evaluation.graph_count is not None:
            validation_batch_size = min(validation_batch_size, len(validation))
        counts = env_params.n_routes
        counts = (counts,) if isinstance(counts, int) else tuple(counts)
        data = ConstructionData(train_loader, DataLoader(validation,batch_size=validation_batch_size),device,
                                n_routes=counts,resample_preferences_on_terminal=resample_preferences)
        data.validation_indices = indices
        data.validation_batch_size = validation_batch_size
        data.training_count = len(train) if source != DataSourceKind.DYNAMIC else None
        data.validation_count = len(validation)
        data.test_graphs = test_graphs
        data.full_validation = full_validation
        data.final_split = 'test' if test_graphs is not None else 'final_validation'
        data.training_source_ids, data.validation_source_ids, data.test_source_ids = train_ids, validation_ids, test_ids
        return data


class ConstructionData:
    def __init__(self, train_loader, validation_loader, device, *, n_routes, resample_preferences_on_terminal):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.n_routes = tuple(n_routes)
        self.resample_preferences = resample_preferences_on_terminal
        self.batch_size = train_loader.batch_size
        self.generator = None
        self.finished = None
        self.route_count = None
        self.graph_instances_loaded = self.graph_batches_loaded = 0
        self.episode_rewards = self.episode_lengths = None

    def initialize(self):
        self.generator = iter(self.train_loader)

    def warmup(self, count):
        return ()

    def get(self, progress):
        self.route_count = (
            max(self.n_routes) if self.route_count is None else random.randint(min(self.n_routes), max(self.n_routes))
        )
        return self

    def start(self, env):
        if self.finished is None or self.finished.all():
            try:
                graphs = next(self.generator)
            except StopIteration:
                self.generator = iter(self.train_loader)
                graphs = next(self.generator)
            graphs = graphs.to(self.device)
            self.graph_instances_loaded += graphs.num_graphs
            self.graph_batches_loaded += 1
            from dataclasses import replace

            env.params = replace(env.params, n_routes=self.route_count)
            env.reset(graphs, env.cost.sample_variable_weights(graphs.num_graphs, self.device))
            self.finished = env.is_done().clone()
            self.episode_rewards = torch.zeros(graphs.num_graphs, device=self.device)
            self.episode_lengths = torch.zeros(graphs.num_graphs, dtype=torch.long, device=self.device)

    def prepare_step(self, env):
        pass

    def after_step(self, env, done):
        self.finished |= done
        env.state.reset_dones()
        if self.resample_preferences and done.any():
            sampled = env.cost.sample_variable_weights(int(done.sum()), self.device)
            merged = {key: value.clone() for key, value in env.state.cost_weights.items()}
            for key, value in sampled.items():
                merged[key][done] = value
            env.state.set_cost_weights(merged)
        env.result = env.cost(env.state)

    def bootstrap_mask(self, env):
        return torch.ones(env.state.batch_size, dtype=torch.bool, device=self.device)

    def final_evaluation(self):
        graphs = self.test_graphs if self.final_split == 'test' else self.full_validation
        return graphs, None, list(range(len(graphs)))

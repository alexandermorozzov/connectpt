"""Construction graph sources and episodes, independent of the algorithm."""
import random
from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

from ..city.synthetic import load_graph_collection
from ..objectives.preferences import PreferenceSampler,state_preferences
from .const import DatasetKind, DataSourceKind, DATASET_MANIFEST
from .dataset import load_dataset
from .episodes import EpisodeSource
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
    sampler: PreferenceSampler

    def prepare(self, cost, batch_size, evaluation, device, env_params, curriculum, resample_preferences):
        if curriculum.enabled:
            raise ValueError('construction data has no curriculum tiers; use curriculum.enabled=false')
        data = self.load(batch_size, evaluation, device, env_params, resample_preferences=resample_preferences)
        val_cost = cost.with_params(**state_preferences(self.sampler.sample_named(data.validation_batch_size, device)))
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
        return ConstructionData(
            train_loader, DataLoader(validation,batch_size=validation_batch_size),device,
            n_routes=counts,resample_preferences_on_terminal=resample_preferences,
            validation_indices=indices,validation_batch_size=validation_batch_size,
            test_graphs=test_graphs,full_validation=full_validation,
            training_source_ids=train_ids,validation_source_ids=validation_ids,test_source_ids=test_ids,
            online=source==DataSourceKind.DYNAMIC,
            sampler=self.sampler,
        )


class ConstructionData(EpisodeSource):
    def __init__(self, train_loader, validation_loader, device, *, n_routes, resample_preferences_on_terminal,
                 validation_indices, validation_batch_size, test_graphs, full_validation,
                 training_source_ids, validation_source_ids, test_source_ids, online,sampler):
        super().__init__()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.n_routes = tuple(n_routes)
        self.resample_preferences = resample_preferences_on_terminal
        self.validation_indices, self.validation_batch_size = validation_indices, validation_batch_size
        self.test_graphs, self.full_validation = test_graphs, full_validation
        self.training_source_ids, self.validation_source_ids, self.test_source_ids = training_source_ids, validation_source_ids, test_source_ids
        self.online = online
        self.sampler=sampler
        self.batch_size = train_loader.batch_size
        self.generator = None
        self.finished = None
        self.route_count = None
        self.graph_instances_loaded = self.graph_batches_loaded = 0

    @property
    def training_count(self):
        return None if self.online else len(self.train_loader.dataset)

    @property
    def validation_count(self):
        return len(self.validation_loader.dataset)

    @property
    def final_split(self):
        return 'test' if self.test_graphs is not None else 'final_validation'

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
            env.set_route_count(self.route_count)
            env.reset(graphs, state_preferences(self.sampler.sample_named(graphs.num_graphs, self.device)))
            self.finished = env.is_done().clone()
            self.start_episode(graphs.num_graphs,self.device)

    def prepare_step(self, env):
        pass

    def after_step(self, env, done):
        self.finished |= done
        sampled = None
        if self.resample_preferences and done.any():
            sampled = state_preferences(self.sampler.sample_named(int(done.sum()), self.device))
        env.reset_completed(done, sampled)

    def bootstrap_mask(self, env):
        return torch.ones(env.state.batch_size, dtype=torch.bool, device=self.device)

    def final_evaluation(self):
        graphs = self.test_graphs if self.final_split == 'test' else self.full_validation
        return graphs, None, list(range(len(graphs)))

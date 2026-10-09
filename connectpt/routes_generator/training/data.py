"""Construction batches and episode resets, independent of the algorithm."""
import random
import torch
from dataclasses import dataclass
from pathlib import Path

from .pipeline import GraphPipeline, GraphView, OnlineGraphs
from .split import SplitParams, graph_source_id, prepare_split


class RawGraphs:
    """Expose collated legacy graphs without their attached transforms."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get(index)


@dataclass(frozen=True)
class ConstructionDataset:
    type: str
    kwargs: dict
    test_path: Path
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline

    def prepare(self, cost, batch_size, evaluation, device, env_params, curriculum, resample_preferences):
        data = self.load(batch_size, evaluation, device, env_params, resample_preferences=resample_preferences)
        data.training_count = len(data.train_loader.dataset) if self.type != "dynamic" else None
        data.validation_count = len(data.validation_loader.dataset)
        val_cost = cost.with_params(**cost.sample_variable_weights(data.validation_batch_size, device))
        return data, val_cost

    def load(self, batch_size, evaluation, device, env_params, *, resample_preferences):
        from torch_geometric.loader import DataLoader
        from ..city.synthetic import CityGraphDataset, DynamicCityGraphDataset
        from ..city.graph import to_batch

        validation_batch_size = batch_size * 4
        test_graphs = RawGraphs(CityGraphDataset(self.test_path))
        test_sources = [graph_source_id(g) for g in test_graphs]
        if self.type == "pickle":
            graphs = RawGraphs(CityGraphDataset(self.kwargs['path']))
            sources = [graph_source_id(g) for g in graphs]
            split = prepare_split(sources, (),
                                  SplitParams('ordered', self.kwargs['split'], 0, 0))
            train = GraphView(graphs, split.train, self.train_pipeline)
            validation = GraphView(graphs, split.validation, self.evaluation_pipeline)
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        elif self.type == "dynamic":
            values = dict(self.kwargs)
            raw_validation = RawGraphs(CityGraphDataset(values.pop("val_path")))
            sources = [graph_source_id(g) for g in raw_validation]
            validation = GraphView(raw_validation, range(len(raw_validation)), self.evaluation_pipeline)
            train = OnlineGraphs(DynamicCityGraphDataset(**values), self.train_pipeline,
                                 forbidden_sources=[*sources, *test_sources])
            train_loader = DataLoader(train, batch_size=batch_size)
        elif self.type == "mumford":
            graph = to_batch(
                self.kwargs["city"], scale_dynamically=self.kwargs["scale_dynamically"]
            ).to_data_list()[0]
            if 'val_path' not in self.kwargs:
                raise ValueError('benchmark training requires a separate fixed validation source')
            raw_validation = RawGraphs(CityGraphDataset(self.kwargs['val_path']))
            source = graph_source_id(graph)
            sources = [source, *[graph_source_id(g) for g in raw_validation]]
            if source in sources[1:]:
                raise ValueError('benchmark city leaks between train and validation')
            train = GraphView([graph], [0] * batch_size, self.train_pipeline)
            validation = GraphView(raw_validation, range(len(raw_validation)), self.evaluation_pipeline)
            train_loader = DataLoader(train, batch_size=batch_size)
        else:
            raise ValueError(f"unknown construction dataset type: {self.type}")
        if set(sources) & set(test_sources):
            raise ValueError('source city leaks between train/validation and independent test')
        indices = list(range(len(validation)))
        if evaluation.graph_count is not None:
            if evaluation.graph_count < 1:
                raise ValueError("evaluation.graph_count must be positive or null")
            generator = torch.Generator().manual_seed(evaluation.seed)
            indices = (
                torch.randperm(len(validation), generator=generator)[: evaluation.graph_count].sort().values.tolist()
            )
            validation = [validation[index] for index in indices]
            validation_batch_size = min(validation_batch_size, len(validation))
        if not len(validation) or (self.type != "dynamic" and len(train) < batch_size):
            raise ValueError("training needs a full batch and nonempty validation")
        counts = env_params.n_routes
        counts = (counts,) if isinstance(counts, int) else tuple(counts)
        data = ConstructionData(
            train_loader,
            DataLoader(validation, batch_size=validation_batch_size),
            device,
            n_routes=counts,
            resample_preferences_on_terminal=resample_preferences,
        )
        data.validation_indices = indices
        data.validation_batch_size = validation_batch_size
        data.test_graphs = GraphView(test_graphs, range(len(test_graphs)), self.evaluation_pipeline)
        data.source_ids = tuple(sources)
        data.test_source_ids = tuple(test_sources)
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

    def test(self):
        return self.test_graphs, None

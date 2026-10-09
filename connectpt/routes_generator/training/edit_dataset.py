"""Stored edit seed networks, batch preparation and route episodes."""
from dataclasses import dataclass

import torch

from ..city.const import STOP_KEY
from ..network.routes import to_routes
from .curriculum import prepare_curriculum
from .dataset import Dataset, load_dataset
from .params import DatasetInputParams
from .pipeline import GraphPipeline
from .split import evaluation_indices


@dataclass(frozen=True)
class EditDataset:
    params: DatasetInputParams
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline

    def prepare(self, cost, batch_size, evaluation, device, env_params, curriculum, resample_preferences):
        dataset = load_dataset(self.params.path, kind=self.params.kind, legacy_limits=self.params.limits,
                               legacy_split=self.params.legacy_split)
        module = SeedDataModule(dataset,device,self.train_pipeline,self.evaluation_pipeline)
        final_module = module
        final_indices = dataset.split.test if len(dataset.split.test) else dataset.split.validation
        final_split = 'test' if len(dataset.split.test) else 'final_validation'
        test_ids = tuple(dataset.params.source_ids[int(i)] for i in dataset.split.test)
        if self.params.test_path is not None:
            if len(dataset.split.test):
                raise ValueError('choose either manifest test or a separate test_path')
            independent = load_dataset(self.params.test_path,kind=self.params.kind,legacy_limits=self.params.limits,
                                       legacy_split=self.params.legacy_split)
            if set(dataset.params.source_ids) & set(independent.params.source_ids):
                raise ValueError('edit source city leaks into independent test')
            final_module = SeedDataModule(independent,device,self.train_pipeline,self.evaluation_pipeline)
            final_indices = torch.arange(len(independent.graphs))
            final_split, test_ids = 'test', independent.params.source_ids
        stages = prepare_curriculum(dataset.split.train,dataset.params.tiers,curriculum)
        monitor = evaluation_indices(dataset.split.monitor,evaluation.graph_count,evaluation.seed)
        data = EditData(module,stages,dataset.split.train,dataset.split.validation,monitor,batch_size,
                        final_module,final_indices,final_split)
        data.training_count, data.validation_count = len(dataset.split.train),len(monitor)
        sources=dataset.params.source_ids
        data.training_source_ids=tuple(sources[int(i)] for i in dataset.split.train)
        data.validation_source_ids=tuple(sources[int(i)] for i in dataset.split.validation)
        data.test_source_ids=test_ids
        return data,cost


@dataclass(frozen=True)
class SeedDataModule:
    dataset: Dataset
    device: torch.device
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline

    def make_batch(self, indices, *, training=False):
        pipeline=self.train_pipeline if training else self.evaluation_pipeline
        graphs=pipeline.prepare([self.dataset.graphs[int(i)] for i in indices],self.device)
        node_capacity=int((graphs[STOP_KEY].ptr[1:]-graphs[STOP_KEY].ptr[:-1]).max())
        seeds=self.dataset.seeds
        routes=to_routes(seeds[torch.as_tensor(indices,dtype=torch.long)],n_routes=self.dataset.params.limits.n_routes,
                         width=min(seeds.shape[-1],node_capacity)).to(self.device)
        return graphs,routes


class EditData:
    def __init__(self, module, curriculum, train_indices, val_indices, monitor_indices, batch_size,
                 final_module, final_indices, final_split):
        self.module, self.curriculum = module, curriculum
        self.train_indices, self.val_indices = train_indices, val_indices
        self.all_val_indices = val_indices
        self.monitor_indices = monitor_indices
        self.final_module, self.final_indices, self.final_split = final_module, final_indices, final_split
        self.batch_size = min(batch_size, len(train_indices))
        self.device = module.device
        self.indices = self.order = None
        self.label = "all"
        self.index_cursor = self.route_cursor = 0
        self.graphs = self.working = self.reference = self.preferences = None
        self.fresh = True
        self.graph_instances_loaded = self.graph_batches_loaded = 0
        self.episode_rewards = self.episode_lengths = None

    def initialize(self):
        self.indices = self.train_indices
        self.order = self.indices[torch.randperm(len(self.indices))]

    def warmup(self, count):
        for indices in list(self.train_indices.split(self.batch_size))[:count]:
            yield self.module.make_batch(indices, training=True)

    def get(self, progress):
        indices, label = self.curriculum.get(progress)
        if self.label != label:
            self.indices, self.label = indices, label
            self.order = indices[torch.randperm(len(indices))]
            self.index_cursor = 0
        self.fresh = True
        return self

    def start(self, env):
        self.prepare_step(env)

    def prepare_step(self, env):
        if not self.fresh and not env.is_done().all():
            return
        if not self.fresh:
            self.working = to_routes(env.updated_network(), width=self.reference.shape[-1])
        if self.fresh or self.route_cursor == 0:
            if self.index_cursor + self.batch_size > len(self.order):
                self.order = self.indices[torch.randperm(len(self.indices))]
                self.index_cursor = 0
            indices = self.order[self.index_cursor : self.index_cursor + self.batch_size]
            self.index_cursor += self.batch_size
            self.graphs, self.working = self.module.make_batch(indices, training=True)
            width = max(self.working.shape[-1], env.params.max_route_len)
            self.working = to_routes(self.working, width=width)
            self.reference = self.working.clone()
            self.preferences = env.cost.sample_variable_weights(self.graphs.num_graphs, self.device)
            self.route_cursor = 0
            self.graph_instances_loaded += self.graphs.num_graphs
            self.graph_batches_loaded += 1
        route_idx = self.route_cursor
        self.route_cursor = (route_idx + 1) % self.working.shape[1]
        env.reset(self.graphs, self.preferences, self.working, route_idx=route_idx, reference_networks=self.reference)
        self.episode_rewards = torch.zeros(self.graphs.num_graphs, device=self.device)
        self.episode_lengths = torch.zeros(self.graphs.num_graphs, dtype=torch.long, device=self.device)
        self.fresh = False

    def after_step(self, env, done):
        pass

    def bootstrap_mask(self, env):
        return ~env.is_done()

    def final_evaluation(self):
        graphs, networks = self.final_module.make_batch(self.final_indices)
        return graphs.to_data_list(), networks, self.final_indices.tolist()

"""Task assembly at the configuration boundary."""
from dataclasses import dataclass

from ..env.base import EnvParams, EditEnvParams, EnvironmentBinding
from ..env.construction import ConstructionEnv
from ..env.edit import EditEnv
from ..training.const import DatasetKind
from ..training.data import ConstructionDataset
from ..training.edit_dataset import EditDataset
from ..training.validate import ConstructionValidation,EditValidation


def construction_source(prepared,sampler):
    return ConstructionDataset(prepared.params,prepared.train_pipeline,prepared.evaluation_pipeline,
                               prepared.online_source,prepared.benchmark,sampler)


def edit_source(prepared,sampler):
    # Edit validation uses the fixed cost weights; training ω comes from EpisodePreferences.
    source=EditDataset(prepared.params,prepared.train_pipeline,prepared.evaluation_pipeline)
    source.validate_input()
    return source


@dataclass(frozen=True)
class Task:
    kind: DatasetKind
    env_factory: object
    source_factory: object
    validation_factory: object
    params_type: type[EnvParams]
    requires_seed: bool

    def validate_policy(self,policy):
        if policy.action_spec.role!=self.kind.value:
            raise ValueError(f'task {self.kind.value} requires compatible policy actions')

    def binding(self,policy,limits,values):
        self.validate_policy(policy)
        values=dict(values)
        physical = dict(n_routes=limits.n_routes, min_route_len=limits.min_route_len, max_route_len=limits.max_route_len)
        for key,value in physical.items():
            if key in values and values[key]!=value:
                raise ValueError(f'env.{key} conflicts with dataset limits')
            values[key]=value
        params=self.params_type(**values)
        return EnvironmentBinding(self.env_factory,params)

    def source(self,prepared,sampler):
        if prepared.params.kind!=self.kind:
            raise ValueError('dataset kind does not match task')
        return self.source_factory(prepared,sampler)


TASKS={
    'construction':Task(DatasetKind.CONSTRUCTION,ConstructionEnv,construction_source,ConstructionValidation,
                        EnvParams,False),
    'edit':Task(DatasetKind.EDIT,EditEnv,edit_source,EditValidation,
                EditEnvParams,True),
}

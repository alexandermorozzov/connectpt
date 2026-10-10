"""Task assembly at the configuration boundary."""
from dataclasses import asdict,dataclass

from ..env.base import EnvParams,EnvironmentBinding
from ..env.construction import ConstructionEnv
from ..env.edit import EditEnv
from ..training.const import DatasetKind
from ..training.data import ConstructionDataset
from ..training.edit_dataset import EditDataset
from ..training.validate import ConstructionValidation,EditValidation


def construction_source(prepared):
    return ConstructionDataset(prepared.params,prepared.train_pipeline,prepared.evaluation_pipeline,
                               prepared.online_source,prepared.benchmark)


def edit_source(prepared):
    return EditDataset(prepared.params,prepared.train_pipeline,prepared.evaluation_pipeline)


@dataclass(frozen=True)
class Task:
    kind: DatasetKind
    env_factory: object
    source_factory: object
    validation_factory: object
    decoder_fields: tuple[str,...]
    requires_seed: bool

    def validate_policy(self,policy):
        if policy.action_spec.role!=self.kind.value:
            raise ValueError(f'task {self.kind.value} requires compatible policy actions')

    def training_binding(self,policy,limits,values):
        self.validate_policy(policy)
        values=dict(values)
        for key,value in asdict(limits).items():
            if key in values and values[key]!=value:
                raise ValueError(f'env.{key} conflicts with dataset limits')
            values[key]=value
        params=EnvParams(**values)
        for field in self.decoder_fields:
            if getattr(params,field)!=getattr(policy,field):
                raise ValueError(f'env.{field} conflicts with policy decoder rules')
        return EnvironmentBinding(self.env_factory,params)

    def rollout_binding(self,policy,limits,*,max_route_edit_steps=None,max_trim_actions=None,
                        force_nonhalt_first_step=False):
        self.validate_policy(policy)
        params=EnvParams(limits.n_routes,limits.min_route_len,limits.max_route_len,True,1.0,
                         policy.force_linking_unlinked,policy.max_act_len,max_route_edit_steps,
                         max_trim_actions,force_nonhalt_first_step,
                         policy.allow_trim_below_min if self.requires_seed else False,
                         policy.forbid_halt_when_overlong if self.requires_seed else False)
        return EnvironmentBinding(self.env_factory,params)

    def source(self,prepared):
        if prepared.params.kind!=self.kind:
            raise ValueError('dataset kind does not match task')
        return self.source_factory(prepared)


TASKS={
    'construction':Task(DatasetKind.CONSTRUCTION,ConstructionEnv,construction_source,ConstructionValidation,
                        ('force_linking_unlinked','max_act_len'),False),
    'edit':Task(DatasetKind.EDIT,EditEnv,edit_source,EditValidation,
                ('force_linking_unlinked','max_act_len','allow_trim_below_min','forbid_halt_when_overlong'),True),
}

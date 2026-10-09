"""Resolve dataset inputs at the configuration boundary."""
from dataclasses import asdict, dataclass

from omegaconf import OmegaConf

from ..city.graph import to_batch
from ..city.synthetic import DynamicCityGraphDataset, SyntheticParams
from ..common.random import seed_everything
from ..env.base import EnvParams
from ..env.construction import ConstructionEnv
from ..network.corruption import CopyTierParams, copy_tier
from ..objectives.cost import build_cost
from ..policy.build import build_policy
from ..training.const import DatasetKind, DataSourceKind, DATASET_MANIFEST
from ..training.dataset import (DatasetLimits, GenerationParams, EditGeneration, PreparedPolicy, read_manifest,
                                generate_dataset, save_dataset, dataset_fingerprint, existing_dataset,
                                file_identity, convert_dataset)
from ..training.params import DatasetInputParams
from ..training.pipeline import GraphPipeline, PipelineParams
from ..training.split import SplitParams, prepare_split
from .context import resolve_under_root, resolve_device, resolve_weights_path
from .config import cost_params, policy_profile
from .const import ROOT_DIR, MODEL_WEIGHTS_DIR
from .instances import load_instance
from .runs import current_commit


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    source: str
    path: str | None
    legacy_limits: dict | None
    legacy_split: dict
    validation_path: str | None
    test_path: str | None
    source_params: dict | None
    benchmark_city: str | None
    scale_dynamically: bool
    train_pipeline: dict
    evaluation_pipeline: dict


@dataclass(frozen=True)
class PreparedDataInput:
    params: DatasetInputParams
    train_pipeline: GraphPipeline
    evaluation_pipeline: GraphPipeline
    online_source: object | None
    benchmark: object | None


def prepare_data_input(node, task: DatasetKind):
    config = DatasetConfig(**OmegaConf.to_container(node, resolve=True))
    kind, source = DatasetKind(config.kind), DataSourceKind(config.source)
    if kind != task:
        raise ValueError(f'dataset kind {kind} does not match task {task}')
    path = resolve_under_root(config.path) if source == DataSourceKind.FILE and config.path is not None else None
    if source == DataSourceKind.FILE and path is None:
        raise ValueError('file dataset needs dataset.path')
    if source == DataSourceKind.FILE and config.validation_path is not None:
        raise ValueError('file validation must be prepared in its dataset split, not validation_path')
    if path is not None and (path / DATASET_MANIFEST).exists():
        manifest = read_manifest(path)
        if DatasetKind(manifest['kind']) != task:
            raise ValueError('dataset manifest kind does not match the task')
        limits = DatasetLimits(**manifest['limits'])
    elif config.legacy_limits is not None:
        limits = DatasetLimits(**config.legacy_limits)
    else:
        raise ValueError('dataset limits require a manifest or explicit legacy_limits')
    train_pipeline = PipelineParams(**config.train_pipeline)
    evaluation_pipeline = PipelineParams(**config.evaluation_pipeline)
    if train_pipeline.edit_features != (task == DatasetKind.EDIT) or evaluation_pipeline.edit_features != (task == DatasetKind.EDIT):
        raise ValueError('graph feature profile does not match dataset task')
    online, benchmark = None, None
    if source == DataSourceKind.DYNAMIC:
        if config.source_params is None:
            raise ValueError('online dataset needs source_params')
        online = DynamicCityGraphDataset(**asdict(SyntheticParams(**config.source_params)))
    elif source == DataSourceKind.BENCHMARK:
        if config.benchmark_city is None:
            raise ValueError('benchmark dataset needs benchmark_city')
        benchmark = to_batch(load_instance(config.benchmark_city),scale_dynamically=config.scale_dynamically,
                             insert_pos_features=False).to_data_list()[0]
    if task == DatasetKind.EDIT and source != DataSourceKind.FILE:
        raise ValueError('edit training requires stored seed networks')
    if source != DataSourceKind.FILE and config.validation_path is None:
        raise ValueError('online and benchmark training require a separate fixed validation source')
    return PreparedDataInput(
        DatasetInputParams(kind, source, path, limits, SplitParams(**config.legacy_split),
                           resolve_under_root(config.validation_path) if config.validation_path is not None else None,
                           resolve_under_root(config.test_path) if config.test_path is not None else None),
        GraphPipeline(train_pipeline), GraphPipeline(evaluation_pipeline), online, benchmark,
    )


@dataclass(frozen=True)
class GenerationConfig:
    kind: str
    n_graphs: int
    fixed_nodes: int | None
    source: dict
    limits: dict
    split: dict


@dataclass(frozen=True)
class EditGenerationConfig(GenerationConfig):
    lc_n_samples: int
    lc_combos: list
    tiers: list
    policy_seed: int
    batch_size_cpu: int
    batch_size_cuda: int
    corruption_seed: int
    tier_cfg: dict


@dataclass(frozen=True)
class PreparationRunParams:
    seed: int
    cpu: bool


def prepare_dataset(cfg, output, *, force=False, dry_run=False, convert_from=None, weights_dir=None):
    """A preparation command without Run lifecycle, training or experiment logs."""
    values = OmegaConf.to_container(cfg.dataset_build, resolve=True)
    kind = DatasetKind(values['kind'])
    config = (EditGenerationConfig if kind == DatasetKind.EDIT else GenerationConfig)(**values)
    run = PreparationRunParams(**OmegaConf.to_container(cfg.run, resolve=True))
    source_params = SyntheticParams(**config.source)
    limits, split = DatasetLimits(**config.limits), SplitParams(**config.split)
    generation = GenerationParams(kind,config.n_graphs,run.seed,config.fixed_nodes,limits,split)
    if config.fixed_nodes is not None and not source_params.min_nodes <= config.fixed_nodes <= source_params.max_nodes:
        raise ValueError('fixed_nodes conflicts with the synthetic source size range')
    if kind == DatasetKind.EDIT:
        if not config.tiers or config.n_graphs < len(config.tiers) or len(set(config.tiers)) != len(config.tiers):
            raise ValueError('edit generation needs distinct tiers with enough source cities')
        per = config.n_graphs // len(config.tiers)
        labels = [config.tiers[min(i // per, len(config.tiers)-1)] for i in range(config.n_graphs)]
    else:
        labels = ['construction'] * config.n_graphs
    prepare_split(list(range(config.n_graphs)),labels,split)
    source = DynamicCityGraphDataset(**asdict(source_params))
    output = resolve_under_root(output).absolute()
    if output.is_symlink() or output.is_junction():
        raise ValueError('dataset output must not be a directory link')
    output = output.resolve()
    if (output == ROOT_DIR or output in ROOT_DIR.parents or output == MODEL_WEIGHTS_DIR
            or MODEL_WEIGHTS_DIR in output.parents or output in MODEL_WEIGHTS_DIR.parents):
        raise ValueError('choose a named dataset output outside the project root and model weights')
    configuration = OmegaConf.to_container(cfg, resolve=True)
    configuration.pop('paths', None)
    inputs = {}
    checkpoint = None
    if kind == DatasetKind.EDIT and convert_from is None:
        root = weights_dir if weights_dir is not None else cfg.paths.weights_dir
        checkpoint = resolve_weights_path(cfg.model_profile.checkpoint_path, root)
        if output == checkpoint.parent or output in checkpoint.parents:
            raise ValueError('dataset output contains its input checkpoint')
        inputs['checkpoint'] = file_identity(checkpoint)
    if convert_from is not None:
        source_path = resolve_under_root(convert_from).resolve()
        if output == source_path or output in source_path.parents or source_path in output.parents:
            raise ValueError('conversion must write a separate output folder')
        if dry_run:
            from ..training.dataset import load_dataset
            dataset = load_dataset(source_path, kind=kind, legacy_limits=limits, legacy_split=split)
            return dict(status='dry_run', kind=kind, output=str(output), graphs=len(dataset.graphs), conversion=True)
        manifest_path = output / DATASET_MANIFEST
        previous = manifest_path.read_text(encoding='utf-8') if manifest_path.is_file() else None
        dataset = convert_dataset(source_path, output, kind=kind, limits=limits, split=split, commit=current_commit(), force=force)
        status = 'skip' if previous == manifest_path.read_text(encoding='utf-8') else 'converted'
        return dict(status=status, kind=kind, output=str(output), graphs=len(dataset.graphs))
    fingerprint = dataset_fingerprint(configuration, inputs, run.seed)
    existing = existing_dataset(output, fingerprint, force=force)
    if existing is not None:
        return dict(status='skip', kind=kind, output=str(output), graphs=len(existing.graphs), fingerprint=fingerprint)
    device = resolve_device(cpu=run.cpu)
    edit = None
    if kind == DatasetKind.EDIT:
        tiers = {t:CopyTierParams(**v) for t,v in config.tier_cfg.items()}
        if any(t not in tiers for t in config.tiers):
            raise ValueError('edit generation references an absent corruption tier')
        seed_everything(config.policy_seed)
        policy = build_policy(policy_profile(cfg.model_profile.model,cfg.cost,cfg.experiment,device,checkpoint_path=checkpoint))
        prepared = PreparedPolicy.capture(policy,config.policy_seed)
        cost = build_cost(cost_params(cfg.cost),symmetric_routes=cfg.experiment.symmetric_routes,
                          low_memory_mode=cfg.experiment.low_memory_mode)
        env_params = EnvParams(limits.n_routes,limits.min_route_len,limits.max_route_len,True,1.,
                               policy.force_linking_unlinked,policy.max_act_len,None,None,False,False,False)
        edit = EditGeneration(prepared,cost,ConstructionEnv,env_params,device,
                              config.batch_size_cuda if device.type=='cuda' else config.batch_size_cpu,
                              config.lc_n_samples,tuple(tuple(c) for c in config.lc_combos),tuple(config.tiers),
                              {t:copy_tier(tiers[t]) for t in config.tiers},
                              {t:'clean' if not tiers[t].kinds else 'copies' for t in config.tiers},config.corruption_seed)
    if dry_run:
        return dict(status='dry_run',kind=kind,output=str(output),graphs=config.n_graphs,fingerprint=fingerprint)
    dataset = generate_dataset(source,generation,edit=edit)
    save_dataset(dataset,output,configuration=configuration,inputs=inputs,commit=current_commit(),force=force)
    return dict(status='generated',kind=kind,output=str(output),graphs=len(dataset.graphs),fingerprint=fingerprint)

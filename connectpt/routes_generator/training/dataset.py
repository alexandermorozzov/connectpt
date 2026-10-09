"""Dataset storage and legacy adapters; no configuration or policy construction."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import random
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..city.graph import to_batch
from ..common.random import seed_everything
from ..policy.predict import predict
from ..network.corruption import CorruptionCity, CorruptionLimits, corrupt, count_changed_routes, redundancy_stats, uncovered_demand_pct
from ..city.const import STOP_KEY, RAW_GRAPH_FILENAME
from ..city.synthetic import load_graph_collection
from ..network.io import load_routes
from ..network.routes import to_routes
from .const import DatasetKind, DATASET_SCHEMA, DATASET_MANIFEST, DATASET_CONTENT
from .split import DatasetSplit, SplitParams, graph_source_id, prepare_split


@dataclass(frozen=True)
class DatasetLimits:
    n_routes: int | tuple[int, ...]
    min_route_len: int
    max_route_len: int

    def __post_init__(self):
        if isinstance(self.n_routes, list):
            object.__setattr__(self, 'n_routes', tuple(self.n_routes))
        counts = (self.n_routes,) if isinstance(self.n_routes, int) else self.n_routes
        if not counts or any(n < 1 for n in counts):
            raise ValueError('dataset route counts must be positive')
        if not 1 <= self.min_route_len <= self.max_route_len:
            raise ValueError('dataset route length limits are invalid')


@dataclass(frozen=True)
class DatasetParams:
    kind: DatasetKind
    limits: DatasetLimits
    seed: int | None
    source_ids: tuple[str, ...]
    tiers: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    graphs: object
    seeds: torch.Tensor | None
    params: DatasetParams
    split: DatasetSplit
    metadata: tuple[dict, ...]

    def validate(self):
        count = len(self.graphs)
        if not count:
            raise ValueError('dataset must contain graphs')
        if not isinstance(self.params.kind, DatasetKind):
            raise ValueError('dataset kind must be explicit')
        if len(self.params.tiers) != count or (self.metadata and len(self.metadata) != count):
            raise ValueError('dataset metadata is incomplete')
        self.split.validate(self.params.source_ids, count)
        if self.params.kind == DatasetKind.EDIT:
            if self.seeds is None or self.seeds.ndim != 3 or self.seeds.shape[0] != count:
                raise ValueError('edit dataset requires one seed network per graph')
            if not isinstance(self.params.limits.n_routes, int) or self.seeds.shape[1] != self.params.limits.n_routes:
                raise ValueError('seed route count does not match dataset limits')
            if self.seeds.dtype != torch.long:
                raise ValueError('seed routes must have integer node indices')
            for graph, seed in zip(self.graphs, self.seeds):
                if (seed < -1).any() or (seed >= int(graph[STOP_KEY].num_nodes)).any():
                    raise ValueError('seed network references nodes outside its city')
        elif self.seeds is not None:
            raise ValueError('construction dataset must not contain edit seeds')
        return self


def file_identity(path):
    path = Path(path)
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(configuration, inputs, seed):
    """Date and git commit record provenance, never change content identity."""
    body = dict(schema=DATASET_SCHEMA, configuration=configuration, inputs=inputs, seed=seed)
    return sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_manifest(path):
    path = Path(path)
    manifest = json.loads((path / DATASET_MANIFEST).read_text(encoding='utf-8'))
    if manifest['schema'] != DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {manifest['schema']}")
    if manifest['fingerprint'] != dataset_fingerprint(manifest['configuration'], manifest['inputs'], manifest['seed']):
        raise ValueError('dataset manifest fingerprint does not match its parameters')
    if set(manifest['artifacts']) != {DATASET_CONTENT}:
        raise ValueError('dataset manifest must declare the complete dataset artifact')
    content = path / DATASET_CONTENT
    if not content.is_file() or file_identity(content) != manifest['artifacts'][DATASET_CONTENT]:
        raise ValueError('dataset content is incomplete or its checksum does not match')
    return manifest


def load_dataset(path, *, kind: DatasetKind, legacy_limits=None, legacy_split=None) -> Dataset:
    path = Path(path)
    if (path / DATASET_MANIFEST).exists():
        manifest = read_manifest(path)
        actual_kind = DatasetKind(manifest['kind'])
        if kind != actual_kind:
            raise ValueError(f'dataset kind {actual_kind} does not match task {kind}')
        limits = DatasetLimits(**manifest['limits'])
        if legacy_limits is not None and legacy_limits != limits:
            raise ValueError('configured limits conflict with dataset manifest')
        content = torch.load(path / DATASET_CONTENT, weights_only=False, map_location='cpu')
        split = DatasetSplit(**{k: torch.tensor(v, dtype=torch.long) for k,v in manifest['split'].items()})
        result = Dataset(content['graphs'], content['seeds'],
                         DatasetParams(actual_kind, limits, manifest['seed'], tuple(manifest['source_ids']),
                                       tuple(manifest['tiers'])), split, tuple(content['metadata']))
        if manifest['count'] != len(result.graphs):
            raise ValueError('manifest graph count does not match dataset')
        return result.validate()
    if (path / DATASET_CONTENT).exists():
        raise ValueError('dataset is incomplete: manifest is missing')
    if legacy_limits is None or legacy_split is None:
        raise ValueError('legacy dataset needs explicit limits and split parameters')
    graphs = load_graph_collection(path)
    sources = tuple(graph_source_id(g) for g in graphs)
    if kind == DatasetKind.CONSTRUCTION:
        seeds, metadata = None, ()
        tiers = ('construction',) * len(graphs)
    elif kind == DatasetKind.EDIT:
        import pandas as pd
        meta = pd.read_csv(path / 'meta.csv')
        if len(meta) != len(graphs) or set(meta.graph_index) != set(range(len(graphs))):
            raise ValueError('edit metadata must contain every graph exactly once')
        metadata = tuple(meta.sort_values('graph_index').to_dict('records'))
        tiers = tuple(row['tier'] for row in metadata)
        seeds = load_routes(path)
        if seeds.shape[0] != len(graphs):
            raise ValueError('graph count does not match seed network count')
        seeds = to_routes(seeds, width=min(seeds.shape[-1], max(int(g[STOP_KEY].num_nodes) for g in graphs)))
    else:
        raise ValueError(f'unknown dataset kind: {kind}')
    return Dataset(graphs, seeds, DatasetParams(kind, legacy_limits, None, sources, tiers),
                   prepare_split(sources, tiers, legacy_split), metadata).validate()


def existing_dataset(path, fingerprint, *, force=False):
    """Return a matching complete dataset; refuse partial or different outputs."""
    path = Path(path)
    if not path.exists():
        return None
    if force:
        return None
    if not (path / DATASET_MANIFEST).is_file():
        raise ValueError(f'dataset output is incomplete or legacy: {path}; choose a new folder or --force')
    manifest = read_manifest(path)
    if manifest['fingerprint'] != fingerprint:
        raise ValueError('dataset parameters or inputs differ; choose a new folder or --force')
    return load_dataset(path, kind=DatasetKind(manifest['kind']))


def save_dataset(dataset: Dataset, path, *, configuration, inputs, commit, force=False):
    """Write a complete staging folder, then publish it to the explicit output."""
    dataset.validate()
    path = Path(path).absolute()
    if path.is_symlink() or path.is_junction():
        raise ValueError('dataset output must not be a directory link')
    path = path.resolve()
    workspace = Path.cwd().resolve()
    if path == Path(path.anchor) or path == workspace or path in workspace.parents:
        raise ValueError('dataset output must be a named directory, not a filesystem or workspace root')
    fingerprint = dataset_fingerprint(configuration, inputs, dataset.params.seed)
    if existing_dataset(path, fingerprint, force=force) is not None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=path.name + '.prepare-', dir=path.parent))
    try:
        torch.save(dict(graphs=list(dataset.graphs), seeds=dataset.seeds, metadata=list(dataset.metadata)),
                   staging / DATASET_CONTENT)
        manifest = dict(schema=DATASET_SCHEMA, kind=dataset.params.kind, limits=asdict(dataset.params.limits),
                        seed=dataset.params.seed, source_ids=dataset.params.source_ids, tiers=dataset.params.tiers,
                        count=len(dataset.graphs), split={k:v.tolist() for k,v in asdict(dataset.split).items()},
                        configuration=configuration, inputs=inputs, commit=commit,
                        created_at=datetime.now(timezone.utc).isoformat(), fingerprint=fingerprint,
                        artifacts={DATASET_CONTENT: file_identity(staging / DATASET_CONTENT)})
        (staging / DATASET_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')
        load_dataset(staging, kind=dataset.params.kind)
        if path.exists():
            if path.resolve() != path or path.is_symlink() or path.is_junction():
                raise ValueError('dataset output changed while preparing files')
            backup = Path(tempfile.mkdtemp(prefix=path.name + '.previous-', dir=path.parent)).resolve()
            if backup.parent != path.parent or staging.resolve().parent != path.parent:
                raise ValueError('dataset staging directories must remain beside the explicit output')
            backup.rmdir()
            path.rename(backup)
            try:
                staging.rename(path)
            except BaseException:
                backup.rename(path)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return True


def conversion_inputs(source, kind):
    source = Path(source)
    if source.is_file():
        return {source.name:file_identity(source)}
    if (source / DATASET_MANIFEST).is_file():
        manifest = read_manifest(source)
        manifest.pop('created_at')
        manifest.pop('commit')
        return {DATASET_CONTENT:file_identity(source / DATASET_CONTENT),
                DATASET_MANIFEST:sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    collated = source / 'processed' / 'collated.pt'
    paths = [collated if collated.is_file() else source / RAW_GRAPH_FILENAME]
    if kind == DatasetKind.EDIT:
        paths.append(source / 'meta.csv')
        for folder in sorted(source.glob('graph_*')):
            routes = sorted(folder.glob('lc_*_routes_routes.pt'))
            if not routes:
                routes = sorted(folder.glob('lc_*_routes_routes.pkl'))
            if not routes:
                raise ValueError(f'seed route artifact missing: {folder}')
            paths.append(routes[0])
    return {str(p.relative_to(source)):file_identity(p) for p in paths}


def convert_dataset(source, destination, *, kind, limits, split, commit, force=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('conversion must write a separate output folder')
    dataset = load_dataset(source, kind=kind, legacy_limits=limits, legacy_split=split)
    inputs = conversion_inputs(source,kind)
    save_dataset(dataset, destination, configuration=dict(conversion=True, kind=kind, limits=asdict(limits),
                                                        split=asdict(split)), inputs=inputs, commit=commit, force=force)
    return dataset



@dataclass(frozen=True)
class PreparedPolicy:
    policy: object
    seed: int
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    python_rng: tuple
    numpy_rng: tuple

    @classmethod
    def capture(cls, policy, seed):
        """Capture immediately after boundary construction, including initialized-model RNG."""
        return cls(deepcopy(policy), seed, torch.get_rng_state().clone(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   random.getstate(), np.random.get_state())

    def fresh(self):
        seed_everything(self.seed)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        return deepcopy(self.policy)


@dataclass(frozen=True)
class GenerationParams:
    kind: DatasetKind
    n_graphs: int
    seed: int
    fixed_nodes: int | None
    limits: DatasetLimits
    split: SplitParams

    def __post_init__(self):
        if not isinstance(self.kind, DatasetKind) or not isinstance(self.seed, int):
            raise ValueError('generation needs an explicit kind and integer seed')
        if self.n_graphs < 2 or (self.fixed_nodes is not None and self.fixed_nodes < 2):
            raise ValueError('generation needs at least two graphs and valid city sizes')


@dataclass(frozen=True)
class EditGeneration:
    policy: PreparedPolicy
    cost: object
    env_factory: object
    env_params: object
    device: torch.device
    batch_size: int
    samples: int
    combinations: tuple[tuple[float, float, float, str], ...]
    tiers: tuple[str, ...]
    corruption: dict
    tier_kinds: dict[str, str]
    corruption_seed: int


def generate_dataset(source, params: GenerationParams, *, edit: EditGeneration | None = None):
    if (params.kind == DatasetKind.EDIT) != (edit is not None):
        raise ValueError('edit generation requires ready policy, cost and corruption objects')
    seed_everything(params.seed)
    raw = [source.generate_graph(n_nodes=params.fixed_nodes) for _ in range(params.n_graphs)]
    source_ids = tuple(graph_source_id(g) for g in raw)
    if params.kind == DatasetKind.CONSTRUCTION:
        tiers = ('construction',) * len(raw)
        return Dataset(raw, None, DatasetParams(params.kind, params.limits, params.seed, source_ids, tiers),
                       prepare_split(source_ids, tiers, params.split), ()).validate()
    if not edit.tiers or params.n_graphs < len(edit.tiers):
        raise ValueError('generation needs at least one graph per configured tier')
    if not edit.combinations or edit.batch_size < 1 or edit.samples < 1:
        raise ValueError('edit generation needs combinations, a positive batch and sample count')
    per = params.n_graphs // len(edit.tiers)
    tiers = tuple(edit.tiers[min(i // per, len(edit.tiers) - 1)] for i in range(len(raw)))
    split = prepare_split(source_ids, tiers, params.split)
    tensors = [dict(node_locs=g[STOP_KEY].pos.detach().cpu().clone(),
                    street_adj=g.street_adj.detach().cpu().clone(), demand=g.demand.detach().cpu().clone()) for g in raw]
    learned_routes = [None] * len(raw)
    for ci, (d, rt, cn, tag) in enumerate(edit.combinations):
        indices = [i for i in range(len(raw)) if i % len(edit.combinations) == ci]
        cost = edit.cost.with_params(demand_time_weight=d, route_time_weight=rt, median_connectivity_weight=cn)
        for start in range(0, len(indices), edit.batch_size):
            chunk = indices[start:start + edit.batch_size]
            policy = edit.policy.fresh()
            graphs = [to_batch(tensors[i]).to_data_list()[0] for i in chunk]
            data = next(iter(DataLoader(graphs, batch_size=len(chunk)))).to(edit.device)
            routes = predict(policy, data, cost.get_weights(), cost, edit.env_factory, edit.env_params,
                             'sample', edit.samples, sample_batch_size=len(chunk)).routes
            for j, index in enumerate(chunk):
                learned_routes[index] = routes[j].detach().cpu()
    networks, metadata = [], []
    for i, (graph, tier) in enumerate(zip(raw, tiers)):
        routes = to_routes(learned_routes[i], n_routes=params.limits.n_routes, width=params.limits.max_route_len).squeeze(0)
        before = redundancy_stats(routes)
        n_nodes = int(graph[STOP_KEY].num_nodes)
        result = corrupt(routes, random.Random(edit.corruption_seed + i),
                         CorruptionCity(tensors[i]['street_adj'], tensors[i]['demand'], n_nodes),
                         CorruptionLimits(params.limits.min_route_len, params.limits.max_route_len), edit.corruption[tier])
        after = redundancy_stats(result.routes)
        metadata.append(dict(graph_index=i, tier=tier, tier_kind=edit.tier_kinds[tier],
                             lc_combo=edit.combinations[i % len(edit.combinations)][3],
                             applied_events=int(sum(result.applied_events.values())),
                             n_corrupted_routes=int(count_changed_routes(routes, result.routes)),
                             mutation_events=json.dumps(dict(result.applied_events), sort_keys=True),
                             d_un_after_pct=round(uncovered_demand_pct(result.routes, tensors[i]['demand'], n_nodes), 2),
                             redun_before=round(before['redundancy'], 4), redun_after=round(after['redundancy'], 4),
                             max_leg_use_before=before['max_leg_use'], max_leg_use_after=after['max_leg_use']))
        networks.append(result.routes)
    return Dataset(raw, torch.stack(networks), DatasetParams(params.kind, params.limits, params.seed, source_ids, tiers),
                   split, tuple(metadata)).validate()

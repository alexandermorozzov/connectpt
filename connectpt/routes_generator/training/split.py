"""Prepare fixed splits by source city before route variants or augmentation."""
from dataclasses import dataclass
from hashlib import sha256

import torch

from ..city.const import STOP_KEY


def graph_source_id(graph) -> str:
    """Identify physical input cities before features, routes or augmentation."""
    digest = sha256()
    for tensor in (graph[STOP_KEY].pos, graph.street_adj, graph.demand):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitParams:
    strategy: str
    train_fraction: float
    seed: int
    n_val_per_tier: int


@dataclass(frozen=True)
class DatasetSplit:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    monitor: torch.Tensor

    def validate(self, source_ids, count):
        if len(source_ids) != count:
            raise ValueError('source city count does not match dataset')
        ownership = {}
        for name in ('train', 'validation', 'test'):
            values = getattr(self, name)
            if values.dtype != torch.long or values.ndim != 1:
                raise ValueError('split indices must be one-dimensional long tensors')
            indices = values.tolist()
            if len(indices) != len(set(indices)):
                raise ValueError(f'duplicate indices in {name}')
            for index in indices:
                if not 0 <= index < count:
                    raise ValueError(f'{name} index outside dataset: {index}')
                source = source_ids[index]
                if source in ownership and ownership[source] != name:
                    raise ValueError(f'source city leaks between {ownership[source]} and {name}: {source}')
                ownership[source] = name
        all_indices = torch.cat((self.train, self.validation, self.test)).tolist()
        if sorted(all_indices) != list(range(count)):
            raise ValueError('split does not partition the complete dataset')
        if self.monitor.dtype != torch.long or self.monitor.ndim != 1:
            raise ValueError('monitor indices must be a one-dimensional long tensor')
        if not set(self.monitor.tolist()) <= set(self.validation.tolist()):
            raise ValueError('monitor is outside validation split')
        if len(self.monitor) != len(set(self.monitor.tolist())):
            raise ValueError('duplicate monitor indices')


def prepare_split(source_ids, tiers, params: SplitParams) -> DatasetSplit:
    if not 0 < params.train_fraction < 1:
        raise ValueError('train_fraction must be between zero and one')
    if params.strategy not in ('ordered', 'random', 'stratified'):
        raise ValueError(f'unknown split strategy: {params.strategy}')
    if params.n_val_per_tier < 0:
        raise ValueError('n_val_per_tier must not be negative')
    groups = {}
    for index, source in enumerate(source_ids):
        groups.setdefault(source, []).append(index)
    sources = list(groups)
    order = list(range(len(sources))) if params.strategy == 'ordered' else torch.randperm(
        len(sources), generator=torch.Generator().manual_seed(params.seed)).tolist()
    train_groups, val_groups = [], []
    if params.strategy == 'stratified':
        if len(tiers) != len(source_ids):
            raise ValueError('stratified split requires a tier for every graph')
        strata = {}
        for source in sources:
            key = tuple(dict.fromkeys(tiers[i] for i in groups[source]))
            strata.setdefault(key, [])
        for index in order:
            source = sources[index]
            key = tuple(dict.fromkeys(tiers[i] for i in groups[source]))
            strata[key].append(source)
        for key, members in strata.items():
            n_val = max(params.n_val_per_tier, int(round((1 - params.train_fraction) * len(members))))
            n_val = min(n_val, max(0, len(members) - 1))
            if n_val < params.n_val_per_tier or len(members) - n_val < 1:
                raise ValueError(f'tier {key} has too few source cities for the requested split')
            val_groups.extend(members[:n_val])
            train_groups.extend(members[n_val:])
    else:
        n_train = int(len(sources) * params.train_fraction)
        train_groups = [sources[i] for i in order[:n_train]]
        val_groups = [sources[i] for i in order[n_train:]]
    train = torch.tensor([i for source in train_groups for i in groups[source]], dtype=torch.long)
    validation = torch.tensor([i for source in val_groups for i in groups[source]], dtype=torch.long)
    if not len(train) or not len(validation):
        raise ValueError('split requires nonempty train and validation sources')
    if params.strategy == 'stratified':
        monitor = []
        for tier in dict.fromkeys(tiers):
            members = [i for i in validation.tolist() if tiers[i] == tier]
            monitor.extend(members[:params.n_val_per_tier] if params.n_val_per_tier else members)
        monitor = torch.tensor(monitor, dtype=torch.long)
    else:
        monitor = validation.clone()
    result = DatasetSplit(train, validation, torch.empty(0, dtype=torch.long), monitor)
    result.validate(source_ids, len(source_ids))
    return result

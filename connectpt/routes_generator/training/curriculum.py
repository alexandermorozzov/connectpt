"""A prepared train-only curriculum; evaluation membership never depends on progress."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CurriculumStage:
    until: float
    tiers: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[CurriculumStage, ...]
    train_by_tier: dict[str, torch.Tensor]
    train_indices: torch.Tensor

    def get(self, progress):
        if not self.stages:
            return self.train_indices, 'all'
        stage = self.stages[-1]
        for candidate in self.stages:
            if progress < candidate.until:
                stage = candidate
                break
        return torch.cat([self.train_by_tier[t] for t in stage.tiers]), stage.label


def prepare_curriculum(train_indices, tiers, params):
    train_by_tier = {tier: torch.tensor([int(i) for i in train_indices if tiers[int(i)] == tier], dtype=torch.long)
                     for tier in dict.fromkeys(tiers)}
    stages = tuple(CurriculumStage(float(until), tuple(selected), label)
                   for until, selected, label in params.schedule) if params.enabled else ()
    previous = 0.
    for stage in stages:
        if not previous < stage.until <= 1 or not stage.tiers:
            raise ValueError('curriculum stages must have increasing bounds and nonempty tiers')
        if any(t not in train_by_tier or not len(train_by_tier[t]) for t in stage.tiers):
            raise ValueError('curriculum references an absent or empty train tier')
        previous = stage.until
    if stages and stages[-1].until != 1:
        raise ValueError('curriculum must cover the complete training interval')
    return Curriculum(stages, train_by_tier, train_indices.clone())

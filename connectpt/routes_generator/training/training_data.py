"""TrainingDataModule -- load graphs / seed routes, split, batch, curriculum.

Encapsulates the data plumbing the training notebook did inline: loading the
raw graphs + LC seed routes, the tier-stratified train/val split, the
curriculum schedule functions, and the make_batch wrapper. The batching and
loading delegate to the existing ``improvement_learning`` functions so behaviour
is unchanged; only the orchestration moves here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..improvement_learning import (
    load_raw_graphs_and_lc_routes,
    make_improvement_batch,
)


@dataclass
class TrainingDataModule:
    """Owns the training graphs/routes and how they are split, batched, staged."""

    raw_graphs_path: str | Path
    lc_results_dir: str | Path
    device: torch.device | str = "cpu"
    min_route_len: int | None = None
    max_route_len: int | None = None
    target_n_routes: int | None = None

    graphs: list = field(default_factory=list, init=False)
    seed_routes: torch.Tensor | None = field(default=None, init=False)

    # --- loading ------------------------------------------------------------
    def setup(self) -> "TrainingDataModule":
        """Load graphs and LC seed routes into memory."""
        self.graphs, self.seed_routes = load_raw_graphs_and_lc_routes(
            self.raw_graphs_path, self.lc_results_dir
        )
        return self

    def __len__(self) -> int:
        return len(self.graphs)

    # --- batching (delegates to the existing builder) -----------------------
    def make_batch(self, indices, *, training: bool = False, **kwargs):
        """Build a (graph_batch, route_batch) for ``indices``."""
        kwargs.setdefault("target_n_routes", self.target_n_routes)
        return make_improvement_batch(
            self.graphs, self.seed_routes, indices, self.device,
            training=training, **kwargs
        )

    # --- splits -------------------------------------------------------------
    def random_split(self, train_fraction: float = 0.9, seed: int = 0):
        """Seeded permutation split (matches the trainer's fallback split)."""
        n = len(self.graphs)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        train_size = int(train_fraction * n)
        return perm[:train_size], perm[train_size:]

    def stratified_split(self, tier_of: dict, tiers: Sequence[str], *,
                         train_fraction: float = 0.9, n_val_per_tier: int = 4,
                         seed: int = 0):
        """Per-tier stratified split (reproduces the training notebook cell).

        Returns ``(train_indices, val_indices, monitor_val_indices,
        train_by_tier, val_by_tier)`` where the *_by_tier maps hold per-tier
        long tensors (val capped at ``n_val_per_tier`` for the monitor slice).
        """
        n = len(tier_of)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        by_tier: dict[str, list[int]] = {t: [] for t in tiers}
        for gi in perm.tolist():
            by_tier[tier_of[gi]].append(gi)

        train_l, val_l = [], []
        for tier in tiers:
            idxs = by_tier[tier]
            n_val = max(n_val_per_tier, int(round((1 - train_fraction) * len(idxs))))
            n_val = min(n_val, max(0, len(idxs) - 1))
            if len(idxs) - n_val < 1 or n_val < n_val_per_tier:
                raise ValueError(
                    f"tier '{tier}' has only {len(idxs)} graphs -- too few for "
                    f"n_val_per_tier={n_val_per_tier}"
                )
            val_l += idxs[:n_val]
            train_l += idxs[n_val:]

        train_indices = torch.tensor(train_l, dtype=torch.long)
        val_indices = torch.tensor(val_l, dtype=torch.long)

        val_by_tier = {t: [] for t in tiers}
        for gi in val_indices.tolist():
            val_by_tier[tier_of[gi]].append(gi)
        monitor_val = torch.tensor(
            [gi for t in tiers for gi in val_by_tier[t][:n_val_per_tier]],
            dtype=torch.long,
        )

        train_by_tier = {t: [] for t in tiers}
        for gi in train_indices.tolist():
            train_by_tier[tier_of[gi]].append(gi)
        train_by_tier_t = {t: torch.tensor(v, dtype=torch.long)
                           for t, v in train_by_tier.items()}
        val_by_tier_t = {t: torch.tensor(val_by_tier[t][:n_val_per_tier],
                                         dtype=torch.long) for t in tiers}
        return train_indices, val_indices, monitor_val, train_by_tier_t, val_by_tier_t

    # --- curriculum ---------------------------------------------------------
    @staticmethod
    def build_curriculum(curriculum, train_by_tier: dict, val_by_tier: dict
                         ) -> tuple[Callable, Callable]:
        """Build (curriculum_fn, val_curriculum_fn) from a CURRICULUM schedule.

        ``curriculum`` is a list of ``(until_iter, tiers, label)`` rows; the
        returned functions map an iteration to the concatenated train / val
        indices of the currently active tiers (the schedule's last row applies
        once the iteration passes every ``until_iter``).
        """
        def active(iteration):
            for until, tiers, label in curriculum:
                if iteration < until:
                    return tiers, label
            return curriculum[-1][1], curriculum[-1][2]

        def curriculum_fn(iteration):
            tiers, label = active(iteration)
            idx = torch.cat([train_by_tier[t] for t in tiers if len(train_by_tier[t])])
            return idx, label

        def val_curriculum_fn(iteration):
            tiers, _ = active(iteration)
            return torch.cat([val_by_tier[t] for t in tiers if len(val_by_tier[t])])

        return curriculum_fn, val_curriculum_fn

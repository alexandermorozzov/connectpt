"""Runtime helpers: device selection, seeding, and a small RunContext.

Extracts the device/seed setup that ``process_standard_experiment_cfg`` does
inline, so every run (training / search / evaluation) resolves device and seed
the same way without dragging in the full experiment-cfg processing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def resolve_device(cpu: bool = False) -> torch.device:
    """CPU if requested or CUDA is unavailable, else CUDA (matches the trainer)."""
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def seed_everything(seed: int | None) -> None:
    """Seed torch / numpy / random (no-op if ``seed`` is None)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class RunContext:
    """Per-run runtime settings shared by every ExperimentRun."""

    run_name: str
    output_dir: Path
    seed: int | None = 0
    device: torch.device = field(default_factory=lambda: resolve_device())

    @classmethod
    def create(cls, run_name: str, output_dir, *, seed: int | None = 0,
               cpu: bool = False) -> "RunContext":
        seed_everything(seed)
        ctx = cls(run_name=run_name, output_dir=Path(output_dir), seed=seed,
                  device=resolve_device(cpu))
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        return ctx

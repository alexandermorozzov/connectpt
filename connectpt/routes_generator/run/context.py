"""Run names, path resolution, device, seeding and ``RunContext``."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from .const import (
    ROOT_DIR,
    MODEL_WEIGHTS_DIR,
    EDIT_MODEL_WEIGHTS_DIR,
    MODEL_OUTPUTS_DIR,
)


def safe_run_name(run_name: str) -> str:
    """Filesystem/run-name-safe slug (ascii alnum + ._-, collapsed underscores)."""
    safe = "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "_"
                   for c in str(run_name))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe.strip("._-") or "run"


def resolve_under_root(path) -> Path:
    """Resolve a config path against the repo root when it is relative.

    Config output paths (e.g. ``artifacts/runs/...``) are written relative to
    the repo root, so they must NOT be resolved against the process CWD -- the
    notebook runs from ``examples/route_generator``. Absolute paths pass through
    unchanged. Checkpoints resolve against the weights root instead, see
    :func:`resolve_weights_path`.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def resolve_weights_path(path, weights_dir=None) -> Path:
    """Resolve a checkpoint path against the weights root.

    Checkpoint paths in config (``cfg/method/*``, ``paths.checkpoint_path``)
    are written RELATIVE to the weights root -- ``paths.weights_dir`` when the cfg
    sets one, else :data:`MODEL_WEIGHTS_DIR`. Absolute paths pass through
    unchanged, so a checkpoint outside the repo is named by its full path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = resolve_under_root(weights_dir) if weights_dir else MODEL_WEIGHTS_DIR
    return root / p


def ensure_output_dirs() -> None:
    """Create the artifact write targets (inputs under datasets/ must exist)."""
    for d in (MODEL_WEIGHTS_DIR, EDIT_MODEL_WEIGHTS_DIR, MODEL_OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


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

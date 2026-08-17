"""Training run artifacts (lifecycle base lives in core.runs)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.runs import ExperimentRun, RunArtifact


@dataclass
class TrainingArtifact(RunArtifact):
    """Result of a training run: the saved checkpoint + in-memory history."""

    checkpoint_path: Path | None = None
    history: Any = None  # pandas.DataFrame, kept Any to avoid a hard import


__all__ = ["ExperimentRun", "RunArtifact", "TrainingArtifact"]

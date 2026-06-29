"""ExperimentRun lifecycle base + run artifacts.

``ExperimentRun`` is a thin lifecycle interface (setup -> run), NOT a god-object
that fuses training and search. Concrete runs (EditTrainingRun,
ConstructionTrainingRun, BeeColonySearchRun, ...) implement it in their own
layer. The artifacts are plain data carriers returned by ``run()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunArtifact:
    """Base result of a run: identity + output location + free-form metadata."""

    run_name: str
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingArtifact(RunArtifact):
    """Result of a training run: the saved checkpoint + in-memory history."""

    checkpoint_path: Path | None = None
    history: Any = None  # pandas.DataFrame, kept Any to avoid a hard import


class ExperimentRun:
    """Lifecycle interface: build everything in setup(), execute in run()."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self) -> None:
        raise NotImplementedError

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        raise NotImplementedError

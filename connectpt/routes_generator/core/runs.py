"""ExperimentRun lifecycle base + the base run artifact.

Lives in core/ (the shared layer) so both training and search implement it
without importing each other. ``ExperimentRun`` is a thin lifecycle interface
(setup -> run), NOT a god-object fusing training and search.
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


class ExperimentRun:
    """Lifecycle interface: build everything in setup(), execute in run()."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self) -> None:
        raise NotImplementedError

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        raise NotImplementedError

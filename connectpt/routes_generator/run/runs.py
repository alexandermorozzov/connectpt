"""ExperimentRun lifecycle base + the base run artifact.

``ExperimentRun.run()`` owns the lifecycle every run shares: ``setup()`` builds
the objects; a dry run stops there. Otherwise the run directory gets
``run.json`` (status ``running``) and ``config.yaml``, the loggers open, and
``execute()`` does the work. ``run.json`` ends as ``iterations`` (budget done),
``time_budget`` (stopped by time) or ``failed``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .const import ROOT_DIR
from ..common.logging import close
from .logger import make_loggers
from .schemas import logger_names


@dataclass
class RunArtifact:
    """Base result of a run: identity + output location + free-form metadata."""

    run_name: str
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_json(run_dir, *, seed, status: str) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"commit": _commit(), "seed": seed, "argv": sys.argv, "status": status},
        indent=2), encoding="utf-8")


class ExperimentRun:
    """Lifecycle base: ``setup()`` builds the objects and sets ``run_dir``;
    ``execute()`` runs and may set ``stop_reason = "time_budget"``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir: Path | None = None
        self.stop_reason = "iterations"
        self.logger_list = []

    def setup(self) -> None:
        raise NotImplementedError

    def dry_run_artifact(self) -> RunArtifact:
        raise NotImplementedError

    def execute(self) -> RunArtifact:
        raise NotImplementedError

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        self.setup()
        names = logger_names(self.cfg.logger_list)
        if dry_run:
            return self.dry_run_artifact()
        seed = int(self.cfg.run.seed)
        write_run_json(self.run_dir, seed=seed, status="running")
        OmegaConf.save(self.cfg, self.run_dir / "config.yaml", resolve=True)
        self.logger_list = make_loggers(names, self.run_dir)
        try:
            artifact = self.execute()
        except BaseException:
            write_run_json(self.run_dir, seed=seed, status="failed")
            raise
        finally:
            close(self.logger_list)
        write_run_json(self.run_dir, seed=seed, status=self.stop_reason)
        return artifact

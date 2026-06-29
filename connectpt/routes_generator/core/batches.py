"""ExperimentBatch + ExperimentRunFactory -- run a list of ready configs.

The batch is pure orchestration: it iterates ``cfg.batch.runs`` (config names),
composes each one and dispatches it through :class:`ExperimentRunFactory` by
``run.type``. It does NOT build experiment parameters -- every run is a complete
config on disk. The factory imports the concrete run classes lazily so that
``import core`` does not statically pull in training/search/evaluation/reports
(keeping those layers independent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir

from .paths import CFG_DIR
from .runs import RunArtifact


@dataclass
class BatchArtifact:
    name: str
    artifacts: list[RunArtifact] = field(default_factory=list)
    output_dir: Path | None = None


class ExperimentRunFactory:
    """Dispatch a composed run cfg to its ExperimentRun by ``run.type``."""

    @staticmethod
    def from_cfg(cfg):
        run_type = cfg.run.type
        if run_type == "bee_colony_search":
            from ..search import BeeColonySearchRun
            return BeeColonySearchRun(cfg)
        if run_type == "edit_training":
            from ..training import EditTrainingRun
            return EditTrainingRun(cfg)
        if run_type == "evaluation":
            from ..evaluation import ModelEvaluationRun
            return ModelEvaluationRun(cfg)
        if run_type == "report":
            from ..reports import ReportRun
            return ReportRun(cfg)
        raise ValueError(f"Unknown run.type: {run_type!r}")


class ExperimentBatch:
    """Run a batch config: compose + dispatch each listed run config."""

    def __init__(self, cfg, *, cfg_dir: str | Path = CFG_DIR):
        self.cfg = cfg
        self.cfg_dir = Path(cfg_dir)

    def run(self, *, dry_run: bool = False) -> BatchArtifact:
        artifacts: list[RunArtifact] = []
        for config_name in self.cfg.batch.runs:
            with initialize_config_dir(config_dir=str(self.cfg_dir), version_base=None):
                run_cfg = compose(config_name=str(config_name))
            run = ExperimentRunFactory.from_cfg(run_cfg)
            artifacts.append(run.run(dry_run=dry_run))
        out_dir = self.cfg.batch.get("output_dir")
        return BatchArtifact(
            name=self.cfg.batch.name, artifacts=artifacts,
            output_dir=Path(out_dir) if out_dir else None,
        )

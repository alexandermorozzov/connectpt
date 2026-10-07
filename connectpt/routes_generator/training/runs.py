"""Training run artifacts (lifecycle base lives in run.runs)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..run.runs import ExperimentRun, RunArtifact


@dataclass
class TrainingArtifact(RunArtifact):
    """Result of a training run: the saved checkpoint (metrics: ``history.csv``)."""

    checkpoint_path: Path | None = None


class ComparisonRun(ExperimentRun):
    """Compare finished construction runs (``compare_runs``) on their test grid."""

    def setup(self) -> None:
        from ..run.context import resolve_under_root
        from ..run.schemas import ComparisonParams, OutputPaths, RunParams, params

        self.run_params = params(RunParams, self.cfg.run)
        self.comparison = params(ComparisonParams, self.cfg.comparison)
        output_dir = params(OutputPaths, self.cfg.paths).output_dir
        if output_dir is None:
            raise ValueError("set paths.output_dir to the comparison folder")
        self.run_dir = resolve_under_root(output_dir)

    def dry_run_artifact(self) -> RunArtifact:
        return RunArtifact(self.run_params.name, self.run_dir, {"dry_run": True})

    def execute(self) -> RunArtifact:
        from .construction_comparison import compare_runs

        c = self.comparison
        report = compare_runs(self.run_dir, tuple(c.names), checkpoint=c.checkpoint,
                              mode=c.mode, budget_mode=c.budget_mode)
        return RunArtifact(self.run_params.name, self.run_dir, {"summary": report["summary"]})


__all__ = ["ExperimentRun", "RunArtifact", "TrainingArtifact", "ComparisonRun"]

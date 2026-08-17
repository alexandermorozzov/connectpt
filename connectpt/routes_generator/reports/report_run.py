"""ReportRun -- assemble report tables from saved artifacts.

Reads an evaluation result (and any extra tables) from a run's output dir and
returns a dict of named tables ready to display. Plotting helpers live in the
``*_plots`` modules; this run only gathers/derives tabular outputs so it has no
heavy (matplotlib) import and no training/search dependency.
"""
from __future__ import annotations

from ..core.runs import ExperimentRun, RunArtifact
from .artifact_loaders import load_evaluation_result


class ReportRun(ExperimentRun):
    def setup(self) -> None:
        self.output_dir = self.cfg.paths.output_dir
        self.eval_name = self.cfg.get("report", {}).get("evaluation_name",
                                                        self.cfg.run.name)

    def run(self, *, dry_run: bool = False) -> RunArtifact:
        self.setup()
        tables: dict = {}
        if not dry_run:
            result = load_evaluation_result(self.output_dir, self.eval_name)
            tables = {"summary": result.summary, "per_instance": result.per_instance}
        return RunArtifact(
            run_name=self.cfg.run.name, output_dir=self.output_dir,
            metadata={"tables": sorted(tables), "dry_run": dry_run},
        )

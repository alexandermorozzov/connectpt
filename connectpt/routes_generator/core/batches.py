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

from .loaders import build_experiment
from .paths import CFG_DIR
from .runs import RunArtifact


@dataclass
class BatchArtifact:
    name: str
    artifacts: list[RunArtifact] = field(default_factory=list)
    output_dir: Path | None = None
    table: Any = None                # combined per-run comparison table (or None)


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
    """Run a batch config: compose + dispatch each run in the artifact.

    A batch is ONE paper artifact (a comparison table / Pareto figure) realised
    as several runs. Two ways to declare the runs:

    * ``methods:`` -- the common case. The batch config IS a full run config
      (shared data/sweep/metrics) plus a list of methods, each a Hydra-group
      choice ``{label, bee_sets, models, n_bees}``. Every method re-composes THIS
      config with its group override in code (:func:`build_experiment`) -- no
      per-method leaf files. Needs ``base_name`` (the config's own name).
    * ``batch.runs:`` -- a list of *separate* config names, for genuinely
      heterogeneous runs whose stacks differ beyond a group choice (e.g. GA /
      SA / hyper-heuristic baselines with their own ``run.type``).
    """

    def __init__(self, cfg, *, base_name: str | None = None,
                 cfg_dir: str | Path = CFG_DIR):
        self.cfg = cfg
        self.base_name = base_name
        self.cfg_dir = Path(cfg_dir)

    def run(self, *, dry_run: bool = False, **params) -> BatchArtifact:
        """Compose + dispatch each run. ``**params`` (``city``, ``alpha``,
        ``smoke``, ...) are injected into every run via :func:`build_experiment`,
        so one batch config serves any city."""
        artifacts: list[RunArtifact] = []
        methods = self.cfg.get("methods")
        if methods:
            if self.base_name is None:
                raise ValueError(
                    "methods-based batch needs base_name (the config's own name)")
            for m in methods:
                run_cfg = build_experiment(
                    self.base_name, bee_sets=m["bee_sets"], models=m["models"],
                    n_bees=m.get("n_bees"), label=m["label"], **params)
                run = ExperimentRunFactory.from_cfg(run_cfg)
                artifacts.append(run.run(dry_run=dry_run))
        else:
            for config_name in self.cfg.batch.runs:
                run_cfg = build_experiment(str(config_name), **params)
                run = ExperimentRunFactory.from_cfg(run_cfg)
                artifacts.append(run.run(dry_run=dry_run))
        out_dir = self.cfg.batch.get("output_dir")
        return BatchArtifact(
            name=self.cfg.batch.name, artifacts=artifacts,
            output_dir=Path(out_dir) if out_dir else None,
            table=_combine_tables(artifacts),
        )


def _combine_tables(artifacts):
    """Concat each run's sweep table into one comparison table (multi-method).

    Each artifact that carries a ``table`` (a sweep SearchArtifact) contributes
    its rows tagged with the run name; runs without a table (dry-run, from-scratch
    suite) are skipped. Returns ``None`` if no run produced a table.
    """
    import pandas as pd

    frames = []
    for art in artifacts:
        table = getattr(art, "table", None)
        if table is None or len(table) == 0:
            continue
        frame = table.copy()
        frame.insert(0, "run", art.run_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None

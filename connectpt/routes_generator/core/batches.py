"""ExperimentRunFactory -- dispatch a composed config to its run class.

The factory imports the concrete run classes lazily so that ``import core`` does
not statically pull in training/search/evaluation/reports.
"""
from __future__ import annotations

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
        if run_type == "dataset_prepare":
            from ..training import DatasetPreparationRun
            return DatasetPreparationRun(cfg)
        if run_type == "evaluation":
            from ..evaluation import ModelEvaluationRun
            return ModelEvaluationRun(cfg)
        if run_type == "report":
            from ..reports import ReportRun
            return ReportRun(cfg)
        raise ValueError(f"Unknown run.type: {run_type!r}")

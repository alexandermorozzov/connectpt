"""ExperimentRunFactory -- dispatch a composed config to its run class.

The factory imports the concrete run classes lazily so that ``import core`` does
not statically pull in training/search.
"""
from __future__ import annotations

class ExperimentRunFactory:
    """Dispatch a composed run cfg to its ExperimentRun by ``run.type``."""

    @staticmethod
    def from_cfg(cfg):
        run_type = cfg.run.type
        if run_type == "construction_training":
            from ..training.construction_run import ConstructionTrainingRun
            return ConstructionTrainingRun(cfg)
        if run_type == "bee_colony_search":
            from ..search import BeeColonySearchRun
            return BeeColonySearchRun(cfg)
        if run_type == "edit_training":
            from ..training import EditTrainingRun
            return EditTrainingRun(cfg)
        if run_type == "dataset_prepare":
            from ..training import DatasetPreparationRun
            return DatasetPreparationRun(cfg)
        raise ValueError(f"Unknown run.type: {run_type!r}")

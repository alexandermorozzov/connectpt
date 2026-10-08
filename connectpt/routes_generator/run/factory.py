"""ExperimentRunFactory -- dispatch a composed config to its run class.

The factory imports the concrete run classes lazily so that ``import run`` does
not statically pull in training/search.
"""
from __future__ import annotations

class ExperimentRunFactory:
    """Dispatch a composed run cfg to its ExperimentRun by ``run.type``."""

    @staticmethod
    def from_cfg(cfg):
        run_type = cfg.run.type
        if run_type == "training":
            from ..training.runs import TrainingRun
            return TrainingRun(cfg)
        if run_type == "search":
            from ..search.runs import SearchRun
            return SearchRun(cfg)
        if run_type == "comparison":
            from ..training.runs import ComparisonRun
            return ComparisonRun(cfg)
        if run_type == "dataset_prepare":
            from ..training.runs import DatasetRun
            return DatasetRun(cfg)
        raise ValueError(f"Unknown run.type: {run_type!r}")

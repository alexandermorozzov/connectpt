"""ExperimentRunFactory -- dispatch a composed config to its run class.

The run classes are imported on the first call, so that ``import run`` does not
statically pull in training/search.
"""
from __future__ import annotations

class ExperimentRunFactory:
    """Dispatch a composed run cfg to its ExperimentRun by ``run.type``."""

    @staticmethod
    def from_cfg(cfg):
        from ..search.runs import SearchRun
        from ..training.runs import ComparisonRun, DatasetRun, TrainingRun

        runs = {"training": TrainingRun, "search": SearchRun, "comparison": ComparisonRun, "dataset_prepare": DatasetRun}
        run_type = cfg.run.type
        if run_type not in runs:
            raise ValueError(f"Unknown run.type: {run_type!r}")
        return runs[run_type](cfg)

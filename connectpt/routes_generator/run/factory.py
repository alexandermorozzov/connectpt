"""Dispatch a composed configuration through the run registry."""
from __future__ import annotations

from .search import SearchRun
from .training import DatasetRun, TrainingRun


RUNS = {"training": TrainingRun, "search": SearchRun, "dataset_prepare": DatasetRun}


def make_run(cfg):
    """Dispatch a composed config at the run boundary."""
    run_type = cfg.run.type
    if run_type not in RUNS:
        raise ValueError(f"Unknown run.type: {run_type!r}")
    return RUNS[run_type](cfg)

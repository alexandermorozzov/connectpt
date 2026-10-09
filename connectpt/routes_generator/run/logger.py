"""Concrete TensorBoard and CSV sinks implementing common.logging.Logger."""
from __future__ import annotations

import csv
from pathlib import Path
from ..common.logging import Logger


class TensorBoardLogger(Logger):
    """Event files in ``<run_dir>/tensorboard``."""

    def __init__(self, run_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(Path(run_dir) / "tensorboard"))

    def scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class CsvLogger(Logger):
    """``<run_dir>/history.csv``: one row per step, one column per tag."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "history.csv"
        self.rows: dict[int, dict[str, float]] = {}
        self.columns: dict[str, None] = {}  # first-seen order

    def scalar(self, tag, value, step):
        self.rows.setdefault(int(step), {})[tag] = value
        self.columns.setdefault(tag)

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", *self.columns])
            writer.writeheader()
            for step in sorted(self.rows):
                writer.writerow({"step": step, **self.rows[step]})


LOGGERS = {"tensorboard": TensorBoardLogger, "csv": CsvLogger}


def make_loggers(names, run_dir) -> list[Logger]:
    return [LOGGERS[name](run_dir) for name in names]

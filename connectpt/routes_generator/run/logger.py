"""Metric loggers and the one way to write a metric record.

A metric exists only as a field of a dataclass record. ``log(logger_list,
record, step)`` writes every set field (``None`` = not measured) to every logger.
The tag is derived from the record's PHASE and field name. Dictionary fields
append their key with a slash. Records contain scalar measurements only.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path


class Logger(ABC):
    @abstractmethod
    def scalar(self, tag: str, value: float, step: int) -> None:
        ...

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()


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


def record_tags(record, prefix: str | None = None):
    """``(tag, value)`` for every set field of ``record``."""
    prefix = record.PHASE if prefix is None else prefix
    for f in fields(record):
        value = getattr(record, f.name)
        if value is None:
            continue
        tag = f"{prefix}/{f.name}"
        if isinstance(value, dict):
            for key, item in value.items():
                yield f"{tag}/{key}", float(item)
        else:
            yield tag, float(value)


def log(logger_list, record, step: int, prefix: str | None = None) -> None:
    for tag, value in record_tags(record, prefix):
        for logger in logger_list:
            logger.scalar(tag, value, step)


def flush(logger_list) -> None:
    for logger in logger_list:
        logger.flush()


def close(logger_list) -> None:
    for logger in logger_list:
        logger.close()

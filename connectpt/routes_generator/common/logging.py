"""Logger contract and scalar record dispatch for library consumers."""
from abc import ABC, abstractmethod
from dataclasses import fields


class Logger(ABC):
    @abstractmethod
    def scalar(self, tag: str, value: float, step: int) -> None:
        ...

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()


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

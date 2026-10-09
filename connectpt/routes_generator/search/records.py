"""Best network measurements and colony acceptance rate."""
from dataclasses import dataclass
from typing import ClassVar
from .const import MONITOR_FIELDS


@dataclass
class SearchBestRecord:
    PHASE: ClassVar[str] = "search/best"
    cost: float
    objective: dict
    penalty: dict
    monitor: dict


@dataclass
class SearchRecord:
    PHASE: ClassVar[str] = "search"
    accept_rate: float

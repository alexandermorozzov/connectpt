"""Best network measurements and colony acceptance rate."""
from dataclasses import dataclass
from typing import ClassVar


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


MONITOR_FIELDS = {
    "ATT": "ATT",
    "RTT": "RTT",
    "median_connectivity_weighted": "WMC",
    "$d_0$": "d0",
    "$d_1$": "d1",
    "$d_2$": "d2",
    "$d_{un}$": "d_un",
    "# disconnected node pairs": "n_disconnected",
    "# stops out of bounds": "stops_oob",
}

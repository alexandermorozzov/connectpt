"""Search metric records: the best network of the colony after each iteration.

Every current metric keeps its old tag in ``metadata["tag"]`` until Э9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


def _tag(name):
    return field(default=None, metadata={"tag": name})


@dataclass
class SearchBestRecord:
    PHASE: ClassVar[str] = "search"
    cost: float | None = _tag("best cost")
    att: float | None = _tag("best ATT")
    rtt: float | None = _tag("best RTT")
    d0: float | None = _tag("best $d_0$")
    d1: float | None = _tag("best $d_1$")
    d2: float | None = _tag("best $d_2$")
    d_un: float | None = _tag("best $d_{un}$")
    disconnected_pairs: float | None = _tag("best # disconnected node pairs")
    stops_oob: float | None = _tag("best # stops out of bounds")
    wmc: float | None = _tag("best median_connectivity_weighted")
    adjustment_degree: float | None = _tag("best adjustment degree")
    adjustment_penalty: float | None = _tag("best adjustment penalty")


# cost-module metric name -> SearchBestRecord field
METRIC_FIELDS = {
    "cost": "cost", "ATT": "att", "RTT": "rtt", "$d_0$": "d0", "$d_1$": "d1", "$d_2$": "d2",
    "$d_{un}$": "d_un", "# disconnected node pairs": "disconnected_pairs",
    "# stops out of bounds": "stops_oob",
    "median_connectivity_weighted": "wmc",
}

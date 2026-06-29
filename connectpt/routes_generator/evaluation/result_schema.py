"""Structured evaluation result + its on-disk form.

An EvaluationResult is pure data: a per-instance table, a summary table and
metadata. It carries no plotting -- the reports layer reads these back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class EvaluationResult:
    per_instance: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, store, name: str) -> dict[str, Any]:
        """Persist via an ArtifactStore: two CSVs + a json metadata sidecar."""
        paths = {
            "per_instance": store.save_table(self.per_instance, f"{name}_per_instance"),
            "summary": store.save_table(self.summary, f"{name}_summary"),
            "metadata": store.save_json(self.metadata, f"{name}_metadata"),
        }
        return {k: str(v) for k, v in paths.items()}

    @classmethod
    def load(cls, store, name: str) -> "EvaluationResult":
        return cls(
            per_instance=store.load_table(f"{name}_per_instance"),
            summary=store.load_table(f"{name}_summary"),
            metadata=store.load_json(f"{name}_metadata"),
        )

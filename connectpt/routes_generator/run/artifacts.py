"""Write run tables and JSON metadata under an explicit output directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Read/write artifacts under a single output directory."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, suffix: str) -> Path:
        p = self.output_dir / (name if name.endswith(suffix) else f"{name}{suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def append_row(self, row: dict, name: str, *, ndigits: int | None = None) -> Path:
        import pandas as pd
        path = self._path(name, ".csv")
        frame = pd.DataFrame([row])
        if ndigits is not None:
            frame = frame.round(ndigits)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
        return path

    # --- json metadata ------------------------------------------------------
    def save_json(self, obj: dict[str, Any] | list, name: str) -> Path:
        path = self._path(name, ".json")
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

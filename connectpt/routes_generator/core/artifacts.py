"""Load / save run artifacts: tables, json metadata, route dumps.

A small, framework-level sink so the training / search / evaluation layers
persist results the same way and the reports layer can read them back without
importing any of those layers. Mirrors the IO patterns previously inlined in
eval_lib.paper, generalised and decoupled from the paper notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ArtifactStore:
    """Read/write artifacts under a single output directory."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, suffix: str) -> Path:
        p = self.output_dir / (name if name.endswith(suffix) else f"{name}{suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # --- tabular (pandas) ---------------------------------------------------
    def save_table(self, df, name: str, *, index: bool = False) -> Path:
        path = self._path(name, ".csv")
        df.to_csv(path, index=index)
        return path

    def append_row(self, row: dict, name: str, *, ndigits: int | None = None) -> Path:
        import pandas as pd
        path = self._path(name, ".csv")
        frame = pd.DataFrame([row])
        if ndigits is not None:
            frame = frame.round(ndigits)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)
        return path

    def load_table(self, name: str):
        import pandas as pd
        return pd.read_csv(self._path(name, ".csv"))

    # --- json metadata ------------------------------------------------------
    def save_json(self, obj: dict[str, Any], name: str) -> Path:
        path = self._path(name, ".json")
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        return json.loads(self._path(name, ".json").read_text(encoding="utf-8"))

    # --- route tensors ------------------------------------------------------
    def save_routes(self, payload: dict, name: str) -> Path:
        path = self._path(name, ".pt")
        torch.save(payload, path)
        return path

    def load_routes(self, name: str, *, map_location: str = "cpu") -> dict:
        return torch.load(self._path(name, ".pt"), map_location=map_location)

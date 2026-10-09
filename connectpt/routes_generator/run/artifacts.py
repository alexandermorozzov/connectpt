"""Load / save run artifacts: tables, json metadata, route dumps.

A small, framework-level sink so the training / search / evaluation layers
persist results the same way and the reports layer can read them back without
importing any of those layers. Mirrors the IO patterns previously inlined in
eval_lib.paper, generalised and decoupled from the paper notebook.
"""
from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import torch

from ..network.routes import route_lists, to_routes


def save_routes(path, routes, *, coords=None, street_adj=None, meta=None) -> Path:
    """Сохранить сеть или именованные сети с данными для рисунков.

    Новые файлы используют torch; явно указанный .pkl сохраняет формат старого
    edit-датасета (список сетей). Чтение обоих форматов остаётся доступным.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(routes, dict):
        payload = {
            "routes": {k: to_routes(v).detach().cpu() for k, v in routes.items()},
            "coords": coords.detach().cpu() if isinstance(coords, torch.Tensor) else coords,
            "street_adj": (
                street_adj.detach().cpu() if isinstance(street_adj, torch.Tensor) else street_adj
            ),
            "meta": {} if meta is None else meta,
        }
    else:
        payload = to_routes(routes).detach().cpu()
    if path.suffix == ".pkl":
        if isinstance(payload, dict):
            raise ValueError("named route sets must be saved as .pt")
        with path.open("wb") as stream:
            pickle.dump(list(payload), stream)
    else:
        torch.save(payload, path)
    return path


def load_routes(path, device=None):
    """Читать сеть, именованные сети или сети edit-датасета по порядку городов."""
    path = Path(path)
    if path.is_dir():
        networks = []
        for graph_dir in sorted(path.glob("graph_*")):
            matches = sorted(graph_dir.glob("lc_*_routes_routes.pt"))
            if not matches:
                matches = sorted(graph_dir.glob("lc_*_routes_routes.pkl"))
            if not matches:
                raise FileNotFoundError(f"route file missing in {graph_dir}")
            networks.append(route_lists(load_routes(matches[0])[0]))
        if not networks:
            raise FileNotFoundError(f"no graph route sets in {path}")
        return to_routes(networks).to(device)
    if path.suffix == ".pkl" and not zipfile.is_zipfile(path):
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload["routes"] = {k: to_routes(v).to(device) for k, v in payload["routes"].items()}
        return payload
    return to_routes(payload).to(device)


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
    def save_json(self, obj: dict[str, Any], name: str) -> Path:
        path = self._path(name, ".json")
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

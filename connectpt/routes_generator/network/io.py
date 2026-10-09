"""Network tensors and named route artifacts in legacy pickle and torch formats."""
from pathlib import Path
from typing import Any
import pickle
import zipfile

import torch

from .routes import route_lists, to_routes


def _read(path: Path):
    if path.suffix == ".pkl" and not zipfile.is_zipfile(path):
        with path.open("rb") as stream:
            return pickle.load(stream)
    return torch.load(path, map_location="cpu", weights_only=False)


def save_routes(path: Path, routes) -> Path:
    """Save a network tensor; .pkl keeps the legacy list-of-networks representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(routes, dict):
        raise TypeError("use save_route_artifact for named route sets")
    payload = to_routes(routes).detach().cpu()
    if path.suffix == ".pkl":
        with path.open("wb") as stream:
            pickle.dump(list(payload), stream)
    else:
        torch.save(payload, path)
    return path


def load_routes(path: Path, device=None) -> torch.Tensor:
    """Read a network or an edit dataset directory, ordered by graph folder."""
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
    payload = _read(path)
    if isinstance(payload, dict):
        raise TypeError("use load_route_artifact for named route sets")
    return to_routes(payload).to(device)


def save_route_artifact(path: Path, routes: dict, *, coords=None, street_adj=None, meta=None) -> Path:
    """Save named networks and optional plotting metadata in the existing torch format."""
    path = Path(path)
    if path.suffix == ".pkl":
        raise ValueError("named route sets must be saved as .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "routes": {k: to_routes(v).detach().cpu() for k, v in routes.items()},
        "coords": coords.detach().cpu() if isinstance(coords, torch.Tensor) else coords,
        "street_adj": street_adj.detach().cpu() if isinstance(street_adj, torch.Tensor) else street_adj,
        "meta": {} if meta is None else meta,
    }
    torch.save(payload, path)
    return path


def load_route_artifact(path: Path, device=None) -> dict[str, Any]:
    """Read a named network artifact without discarding coordinates or metadata."""
    payload = _read(Path(path))
    if not isinstance(payload, dict) or "routes" not in payload:
        raise TypeError("use load_routes for an unnamed network tensor")
    payload["routes"] = {k: to_routes(v).to(device) for k, v in payload["routes"].items()}
    return payload

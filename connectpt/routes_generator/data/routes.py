"""Small route-tensor utilities shared across the data layer."""
from __future__ import annotations

from pathlib import Path

import torch

from ..torch_utils import get_batch_tensor_from_routes


def as_route_tensor(routes) -> torch.Tensor:
    """Route sets -> a CPU tensor, whatever the caller had (tensor or lists)."""
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()


def pad_routes_to(routes, n_routes, max_route_len):
    """Normalise a route tensor to ``(n_routes, max_route_len)`` with -1 fill."""
    t = as_route_tensor(routes).long()
    if t.ndim == 3:
        t = t[0]
    if t.shape[0] < n_routes:
        pad = torch.full((n_routes - t.shape[0], t.shape[1]), -1, dtype=t.dtype)
        t = torch.cat([t, pad], 0)
    else:
        t = t[:n_routes]
    if t.shape[1] < max_route_len:
        pad = torch.full((t.shape[0], max_route_len - t.shape[1]), -1, dtype=t.dtype)
        t = torch.cat([t, pad], 1)
    return t[:, :max_route_len]


def write_routes_txt(path, routes, header=None) -> int:
    """Write a route tensor as ``Routes.txt``: one route per line, 0-indexed
    node ids, ``-1`` padding dropped. Returns the number of routes written."""
    t = as_route_tensor(routes)
    if t.ndim == 3:
        t = t[0]
    lines = [f"# {header}"] if header else []
    lines.append("# 0-indexed; one route per line")
    for route in t.tolist():
        stops = [s for s in route if s >= 0]
        if stops:
            lines.append(" ".join(str(s) for s in stops))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - (2 if header else 1)

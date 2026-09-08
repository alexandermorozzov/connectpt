"""Result sinks (CSV tables, route dumps, figures) + the results-row builder.

Thin wrappers over :class:`ArtifactStore` plus ``paper_row`` (identity columns +
the full metric set). Every sink takes its output folder EXPLICITLY
(keyword-only ``out_dir``), so where a run writes is decided by the caller
(``--out-dir`` / ``run_experiment(out_dir=...)``) and never by module state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.artifacts import ArtifactStore
from ..data.routes import as_route_tensor
from ..evaluation import full_metric_row


def ravel_hist(h):
    """Flatten a per-sample cost history to a plain list of floats (or [])."""
    if h is None:
        return []
    arr = np.asarray(h.detach().cpu() if hasattr(h, "detach") else h).ravel()
    return [float(v) for v in arr]


def paper_row(city, method, source, m, rt, seed, duration_s=None):
    """One results-table row: identity columns + the full metric set."""
    return {"city": city, "method": method, "source": source,
            "duration_s": (round(float(duration_s), 1)
                           if duration_s is not None else None),
            **full_metric_row(m, rt, seed)}


def save_paper_table(df, name, *, out_dir):
    path = ArtifactStore(out_dir).save_table(df, name)
    print(f"[results] table ({len(df)} rows) -> {path}")
    return path


def save_paper_fig(fig, name, *, out_dir, dpi=220):
    """Persist a figure PNG under ``out_dir``.

    Most figures are NOT persisted -- they rebuild from the saved CSV tables and
    route dumps. This sink is only for panels that go into the manuscript as
    rendered images (the MACSA route grids)."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[results] figure -> {path}")
    return path


def save_paper_routes(name, routes, coords=None, street_adj=None, meta=None, *,
                      out_dir):
    """Dump a {label: route_tensor} mapping (+ coords/street_adj) into ``out_dir``
    so the route figures can be reconstructed later."""
    payload = {
        "routes": {k: as_route_tensor(v).cpu() for k, v in routes.items()},
        "coords": (coords.cpu() if hasattr(coords, "cpu") else coords),
        "street_adj": (street_adj.cpu() if hasattr(street_adj, "cpu") else street_adj),
        "meta": meta or {},
    }
    path = ArtifactStore(out_dir).save_routes(payload, f"{name}_routes")
    print(f"[results] route dump ({len(payload['routes'])} sets) -> {path}")
    return path

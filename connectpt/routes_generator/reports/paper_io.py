"""paper_results output sinks + the results-table row builder.

Library home of the notebook's ``paper_results`` IO (was eval_lib.paper): the
prefix-aware CSV / route-dump sinks (thin wrappers over :class:`ArtifactStore`,
the single tabular/route sink) and ``paper_row`` (identity columns + the full
metric set). Figures are intentionally not persisted -- the underlying data is
(CSV tables + route ``.pt`` dumps), so any figure rebuilds from disk.

Every sink takes the output-filename prefix EXPLICITLY (keyword-only): the caller
threads it from ``RunContext.output_prefix`` ("TEMP_" on smoke runs, "" on full
runs) -- there is no module-global prefix.
"""
from __future__ import annotations

import numpy as np

from ..core.artifacts import ArtifactStore
from ..core.paths import ARTIFACTS_DIR
from ..data.routes import as_route_tensor
from ..evaluation import full_metric_row

PAPER_DIR = ARTIFACTS_DIR / "paper_results"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
_STORE = ArtifactStore(PAPER_DIR)


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


def paper_path(name, *, prefix):
    """Prefix-aware path under paper_results (e.g. to read back a saved dump)."""
    return PAPER_DIR / f"{prefix}{name}"


def save_paper_table(df, name, *, prefix):
    path = _STORE.save_table(df, f"{prefix}{name}")
    print(f"[paper] table ({len(df)} rows) -> {path}")
    return path


def reset_paper_table(name, *, prefix):
    path = PAPER_DIR / f"{prefix}{name}.csv"
    if path.exists():
        path.unlink()
    return path


def append_paper_row(row, name, ndigits=3, *, prefix):
    path = _STORE.append_row(row, f"{prefix}{name}", ndigits=ndigits)
    print(f"[paper] row -> {path}", flush=True)
    return path


def save_paper_fig(fig, name):
    # Figures are intentionally NOT persisted as images -- the underlying data is
    # (CSV tables + route .pt dumps), so any figure can be rebuilt.
    print(f"[paper] figure '{name}' shown inline (rebuild from CSV / route dump)")
    return None


def save_paper_routes(name, routes, coords=None, street_adj=None, meta=None, *,
                      prefix):
    """Dump a {label: route_tensor} mapping (+ coords/street_adj) to paper_results
    so the route figures can be reconstructed later."""
    payload = {
        "routes": {k: as_route_tensor(v).cpu() for k, v in routes.items()},
        "coords": (coords.cpu() if hasattr(coords, "cpu") else coords),
        "street_adj": (street_adj.cpu() if hasattr(street_adj, "cpu") else street_adj),
        "meta": meta or {},
    }
    path = _STORE.save_routes(payload, f"{prefix}{name}_routes")
    print(f"[paper] route dump ({len(payload['routes'])} sets) -> {path}")
    return path

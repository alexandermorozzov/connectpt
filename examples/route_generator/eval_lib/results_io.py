"""Persist experiment results: CSV tables + route data (not figures).

The route figures are *derived* artifacts -- instead of saving PNGs, this
module saves the underlying data (route tensors + RunResult metadata + the
graph coords/street_adj), so any figure can be regenerated later with
:func:`render_route_comparison_figure`.

* ``save_table``        -- a results DataFrame -> ``artifacts/results/<name>.csv``
* ``save_route_results`` -- RunResult route tensors + metadata -> ``<name>_routes.pt``
* ``load_route_results`` -- reload the .pt payload back into RunResult objects
* ``redraw_route_results`` -- reload + render the diff figure in one call
* ``redraw_route_set``    -- reload + render the plain (no-diff) figure in one call
"""
import torch

from .context import ARTIFACTS_DIR
from .helpers import as_route_tensor
from .run import RunResult
from .figures import render_route_comparison_figure, render_route_set_figure

RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RUN_FIELDS = ("label", "kind", "accept_mode", "dataset", "metrics",
               "action_stats", "mutation_stats", "run_name")


def _safe_name(name: str) -> str:
    """Filesystem-safe slug (route/combo labels may carry '/', '(' etc.)."""
    return "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in str(name)).strip("_") or "results"


def save_table(df, name: str):
    """Save a results DataFrame to ``artifacts/results/<name>.csv``."""
    path = RESULTS_DIR / f"{_safe_name(name)}.csv"
    df.to_csv(path, index=False)
    print(f"[results] table ({len(df)} rows) -> {path}")
    return path


def save_route_results(results, name: str, *, coords=None, street_adj=None):
    """Persist a list of :class:`RunResult` (route tensors + metadata) plus the
    graph coords / street_adj, so the route figure can be redrawn from data.

    ``coords`` / ``street_adj`` may be a PyG graph (passed straight back to the
    figure) or explicit arrays; both are stored verbatim.
    """
    runs = []
    for r in results:
        row = {field: getattr(r, field, None) for field in _RUN_FIELDS}
        row["routes"] = as_route_tensor(r.routes) if r.routes is not None else None
        row["seed_routes"] = (as_route_tensor(r.seed_routes)
                              if r.seed_routes is not None else None)
        runs.append(row)
    path = RESULTS_DIR / f"{_safe_name(name)}_routes.pt"
    torch.save({"runs": runs, "coords": coords, "street_adj": street_adj}, path)
    print(f"[results] route data ({len(runs)} runs) -> {path}")
    return path


def load_route_results(name: str):
    """Reload a :func:`save_route_results` payload.

    Returns ``(results, coords, street_adj)`` where ``results`` is a list of
    :class:`RunResult` ready for :func:`render_route_comparison_figure`.
    """
    path = RESULTS_DIR / f"{_safe_name(name)}_routes.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    results = []
    for row in payload["runs"]:
        kwargs = {field: row.get(field) for field in _RUN_FIELDS}
        results.append(RunResult(routes=row.get("routes"),
                                 seed_routes=row.get("seed_routes"), **kwargs))
    return results, payload.get("coords"), payload.get("street_adj")


def redraw_route_results(name: str, **figure_kwargs):
    """Reload a saved route-result payload and re-render its diff figure.

    ``figure_kwargs`` are forwarded to :func:`render_route_comparison_figure`
    (``title``, ``ncols``, ``palette``, ...). The first run is the reference.
    """
    results, coords, street_adj = load_route_results(name)
    if not results:
        raise ValueError(f"no runs stored in results/{name}_routes.pt")
    return render_route_comparison_figure(
        results[0], results[1:], coords, street_adj, **figure_kwargs)


def redraw_route_set(name: str, **figure_kwargs):
    """Reload a saved route-result payload and re-render as plain routes.

    Companion to :func:`redraw_route_results` -- draws every run as a plain
    route set (no diff vs reference, no diff legend); each panel's subtitle
    leads with ``cost=...``. ``figure_kwargs`` forward to
    :func:`render_route_set_figure`.
    """
    results, coords, street_adj = load_route_results(name)
    if not results:
        raise ValueError(f"no runs stored in results/{name}_routes.pt")
    return render_route_set_figure(results, coords, street_adj, **figure_kwargs)

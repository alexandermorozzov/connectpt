"""One renderer for every experiment artifact: table + the right figure.

``render_report`` turns a run/experiment artifact into a styled results table
plus the figure that fits it -- an RTT x WMC Pareto front for a sweep, or a
route-set panel grid for a single network. It is duck-typed on the artifact
attributes (``table`` / ``routes`` / ``instance``), so it renders both the
library ``SearchArtifact`` (sweep mode) and the notebook-side ``ExperimentResult``
through ONE implementation -- neither the notebook nor a run hand-rolls plotting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .figures import (plot_geo_map_panels, plot_pareto, plot_routes_grid,
                      style_table)
from .geo import project_coords_3857, street_underlay_adj


@dataclass
class ReportArtifact:
    table: Any
    figures: dict = field(default_factory=dict)


def _report_kind(result, explicit):
    """Resolve which figure to draw: explicit arg > spec/metadata hint > inferred."""
    if explicit is not None:
        return explicit
    # spec-based (ExperimentResult) or metadata-based (SearchArtifact) hint.
    spec = getattr(result, "spec", None)
    if spec is not None:
        kind = dict(spec.get("report", {}) or {}).get("routes_plot")
        if kind:
            return kind
    meta = getattr(result, "metadata", None) or {}
    if meta.get("report_kind"):
        return meta["report_kind"]
    # a geo instance (real-world coords with a CRS) renders on a street underlay.
    inst = getattr(result, "instance", None)
    if inst is not None and (getattr(inst, "meta", None) or {}).get("crs"):
        return "gis"
    # infer: a multi-row RTT x WMC table is a Pareto front; else route panels.
    table = getattr(result, "table", None)
    if table is not None and len(table) > 1 and {"RTT", "WMC"} <= set(table.columns):
        return "pareto"
    return "network"


def render_report(result, *, kind: str | None = None, max_route_panels: int = 4,
                  title: str | None = None, basemap_zoom: int | None = None
                  ) -> ReportArtifact:
    """Build the results table + the fitting figure for an experiment artifact.

    ``result`` needs a ``table``; a route figure additionally needs ``routes``
    (label -> route tensor) and an ``instance`` with ``coords`` / ``street_adj``.
    ``kind`` forces the figure (``"pareto"`` | ``"network"`` | ``"gis"``); by
    default it is read from a ``report`` hint or inferred from the table.
    ``basemap_zoom`` overrides the map-tile zoom of the ``"gis"`` figures.
    """
    table = result.table
    name = title or getattr(result, "run_name", None) or getattr(result, "name", "experiment")
    resolved = _report_kind(result, kind)
    figures: dict = {}

    if resolved == "pareto":
        figures["pareto"] = plot_pareto(table, title=f"{name}: RTT x WMC front")
    else:
        inst = getattr(result, "instance", None)
        routes = getattr(result, "routes", None) or {}
        if inst is not None and routes:
            label = getattr(inst, "label", str(name))
            if resolved == "gis":
                figures.update(_geo_map_figures(
                    result, inst, routes, label,
                    max_panels=max_route_panels, basemap_zoom=basemap_zoom))
            else:  # "network" -- plain route-set panels
                figures["routes"] = plot_routes_grid(
                    routes, inst.coords, inst.street_adj, title=label,
                    max_panels=max_route_panels)

    return ReportArtifact(table=style_table(table), figures=figures)


def figure_key_slug(text) -> str:
    """Filesystem-safe figure-key suffix for one route-set label.

    Public because a caller that wants to name its output files after the route
    sets it passed in has to rebuild the same ``map_<slug>`` keys.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower() or "candidate"


def _metrics_row(table, label):
    """The results row describing one route-set label, as a dict (or ``None``).

    Sweep keys are ``"<method> a=<alpha> t=<target>"``; the matching table row is
    the one whose method / alpha / adj_target reproduce that key. Falls back to a
    single-row table (one result, one row) so non-sweep runs are captioned too.
    """
    if table is None or len(table) == 0:
        return None
    columns = set(getattr(table, "columns", []))
    if {"method", "alpha", "adj_target"} <= columns:
        for _, row in table.iterrows():
            key = f"{row['method']} a={row['alpha']} t={row['adj_target']}"
            if key == str(label):
                return row.to_dict()
    if "method" in columns:
        hit = table[table["method"].astype(str) == str(label)]
        if len(hit):
            return hit.iloc[0].to_dict()
    if len(table) != 1:
        return None
    # single-result artifact: its one row describes it -- unless that row is the
    # Initial network, which would silently caption a candidate with seed numbers.
    only = table.iloc[0].to_dict()
    return None if "Initial" in str(only.get("method", "")) else only


def _initial_metrics_row(table):
    """The Initial-network row of a results table, if it carries one."""
    if table is None or len(table) == 0 or "method" not in getattr(table, "columns", []):
        return None
    method = table["method"].astype(str)
    hit = table[method.str.contains("Initial", case=False, na=False)]
    return hit.iloc[0].to_dict() if len(hit) else None


def _geo_map_figures(result, inst, routes, label, *, max_panels, basemap_zoom):
    """One map figure per candidate route set (initial network alone if none)."""
    from .basemap import DEFAULT_ZOOM, load_basemap, padded_extent

    crs = (getattr(inst, "meta", None) or {}).get("crs")
    coords = project_coords_3857(inst.coords, crs) if _is_crs(crs) else inst.coords
    underlay = street_underlay_adj(inst.street_adj)
    extent = padded_extent(coords)
    basemap = (load_basemap(extent, basemap_zoom or DEFAULT_ZOOM)
               if _is_crs(crs) else None)

    table = getattr(result, "table", None)
    initial_key = next((k for k in routes if "Initial" in str(k)), None)
    candidates = [k for k in routes if k != initial_key][:max_panels]

    if initial_key is None or not candidates:
        only_key = initial_key or (candidates[0] if candidates else None)
        if only_key is None:
            return {}
        return {"map": plot_geo_map_panels(
            routes[only_key], coords, underlay, basemap=basemap, extent=extent,
            title=f"{label}: {only_key}", reference_title=str(only_key),
            metrics=_metrics_row(table, only_key))}

    initial_metrics = _initial_metrics_row(table)
    figures = {}
    for key in candidates:
        suffix = "" if len(candidates) == 1 else f"_{figure_key_slug(key)}"
        figures[f"map{suffix}"] = plot_geo_map_panels(
            routes[key], coords, underlay, reference_routes=routes[initial_key],
            basemap=basemap, extent=extent, title=f"{label}: {key}", label=str(key),
            metrics=_metrics_row(table, key), reference_metrics=initial_metrics)
    return figures


def _is_crs(value) -> bool:
    """True for a real CRS identifier (``"EPSG:32641"``), not a bare marker."""
    return isinstance(value, str) and ":" in value

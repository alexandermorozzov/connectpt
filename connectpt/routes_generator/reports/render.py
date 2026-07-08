"""One renderer for every experiment artifact: table + the right figure.

``render_report`` turns a run/experiment artifact into a styled results table
plus the figure that fits it -- an RTT x WMC Pareto front for a sweep, or a
route-set panel grid for a single network. It is duck-typed on the artifact
attributes (``table`` / ``routes`` / ``instance``), so it renders both the
library ``SearchArtifact`` (sweep mode) and the notebook-side ``ExperimentResult``
through ONE implementation -- neither the notebook nor a run hand-rolls plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .figures import plot_pareto, plot_routes_grid, style_table
from .geo import street_underlay_adj


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
                  title: str | None = None) -> ReportArtifact:
    """Build the results table + the fitting figure for an experiment artifact.

    ``result`` needs a ``table``; a route figure additionally needs ``routes``
    (label -> route tensor) and an ``instance`` with ``coords`` / ``street_adj``.
    ``kind`` forces the figure (``"pareto"`` | ``"network"`` | ``"gis"``); by
    default it is read from a ``report`` hint or inferred from the table.
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
                # geo panels over the street underlay (any city with real coords),
                # plus a diff-vs-Initial grid. Style knobs are geo defaults.
                underlay = street_underlay_adj(inst.street_adj)
                geo = dict(node_size=8, with_overlap_curves=True,
                           show_node_labels=False, palette="tab20")
                figures["routes"] = plot_routes_grid(
                    routes, inst.coords, underlay, title=f"{label} routes",
                    max_panels=max_route_panels, **geo)
                if "Initial" in routes:
                    figures["diff"] = plot_routes_grid(
                        routes, inst.coords, underlay, diff_against="Initial",
                        title=f"{label}: diff vs Initial", **geo)
            else:  # "network" -- plain route-set panels
                figures["routes"] = plot_routes_grid(
                    routes, inst.coords, inst.street_adj, title=label,
                    max_panels=max_route_panels)

    return ReportArtifact(table=style_table(table), figures=figures)

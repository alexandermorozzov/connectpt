"""One clean call per paper-experiment cell in ``paper_combined.ipynb``.

The notebook names a declarative experiment and hands over the loaded ``suite``;
everything else lives here and in the config:

* the ``_smoke`` variant is selected from ``suite.smoke`` (not spelled in cells);
* the run executes through the config-first factory (``ExperimentRunFactory`` /
  ``ExperimentBatch``);
* the results table + route dump are persisted with the output prefix read from
  ``suite`` and the paper stem read from the experiment cfg (``output.paper_stem``)
  -- no prefix, stem, or path ever appears in the notebook;
* figures are rendered via :func:`render_report`.

Each function returns a :class:`PaperRun` the cell simply displays.  There are no
``if ... in globals()`` guards: a call always re-runs and overwrites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import (ExperimentBatch, ExperimentRunFactory, build_experiment,
                    load_suite)
from ..core.paths import resolve_under_root
from ..reports import render_report, save_paper_table, save_paper_routes


def paper_dir(suite):
    """Resolve the suite's paper output folder against the repo root (or None)."""
    configured = getattr(suite, "paper_output_dir", None)
    return resolve_under_root(configured) if configured else None


@dataclass
class PaperRun:
    """Display bundle for a paper cell: the run artifact + its rendered report."""

    artifact: Any
    table: Any = None
    figures: dict = field(default_factory=dict)

    def display(self) -> None:
        """Show the table then every figure (the whole cell body)."""
        from IPython.display import display as _display
        if self.table is not None:
            _display(self.table)
        for fig in self.figures.values():
            _display(fig)


def _smoke_flag(suite, smoke) -> bool:
    """Resolve the smoke budget: explicit arg wins, else the suite profile."""
    return bool(suite.smoke) if smoke is None else bool(smoke)


def _persist(artifact, cfg, suite) -> None:
    """Save the table (+ route dump) using cfg stem + suite prefix + suite folder."""
    stem = cfg.output.paper_stem
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite)
    if getattr(artifact, "table", None) is not None:
        save_paper_table(artifact.table.round(4), stem, prefix=prefix, out_dir=out_dir)
    routes = getattr(artifact, "routes", None)
    instance = getattr(artifact, "instance", None)
    if routes and instance is not None:
        save_paper_routes(stem, routes, instance.coords, instance.street_adj,
                          meta={"stem": stem}, prefix=prefix, out_dir=out_dir)


def save_paper(suite, stem: str, *, table=None, routes=None, coords=None,
               street_adj=None, meta=None) -> None:
    """Ad-hoc save of a curated table/route set (e.g. the single best EKB
    solution) with the prefix + folder resolved from ``suite`` -- the notebook
    passes neither a prefix nor a path."""
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite)
    if table is not None:
        save_paper_table(table, stem, prefix=prefix, out_dir=out_dir)
    if routes:
        save_paper_routes(stem, routes, coords, street_adj, meta=meta,
                          prefix=prefix, out_dir=out_dir)


def run_experiment(name: str, suite, *, kind: str | None = None,
                   title: str | None = None, smoke: bool | None = None,
                   **params) -> PaperRun:
    """Run a single declarative experiment, persist it, render its report.

    ``**params`` are the procedural config knobs forwarded to
    :func:`build_experiment` (``city``, ``alpha``, ``adj_target``,
    ``n_iterations``, ``route_len``, ...): each defaults to the YAML value and is
    overridden only when passed -- no string overrides in the cell. ``smoke``
    defaults to the suite profile; pass ``smoke=False`` to run a config that
    already carries its own budget (e.g. the MACSA iter-1 paper table).
    """
    cfg = build_experiment(name, smoke=_smoke_flag(suite, smoke), **params)
    artifact = ExperimentRunFactory.from_cfg(cfg).run()
    _persist(artifact, cfg, suite)
    report = render_report(artifact, kind=kind, title=title)
    return PaperRun(artifact=artifact, table=report.table, figures=report.figures)


def run_batch(name: str, suite, *, kind: str | None = None,
              city: str | None = None, smoke: bool | None = None,
              **params) -> PaperRun:
    """Run a declarative batch (multi-method), persist the combined table, render.

    ``city`` / ``**params`` are forwarded to :func:`build_experiment` for every
    run in the batch, so one batch config drives any city (the collapsed E1
    per-method leaves) without a per-city batch file. The paper stem is suffixed
    with the city so per-city outputs never clobber each other.
    """
    cfg = load_suite(name)
    batch = ExperimentBatch(cfg).run(
        city=city, smoke=_smoke_flag(suite, smoke), **params)
    stem = cfg.output.paper_stem
    if city is not None:
        stem = f"{stem}_{city.lower()}"
    prefix = str(suite.output_prefix or "")
    out_dir = paper_dir(suite)
    if batch.table is not None:
        save_paper_table(batch.table.drop(columns=["run"]).round(4), stem,
                         prefix=prefix, out_dir=out_dir)
    report = render_report(batch, kind=kind)
    return PaperRun(artifact=batch, table=report.table, figures=report.figures)

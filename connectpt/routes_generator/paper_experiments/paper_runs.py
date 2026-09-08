"""One clean call per experiment: run it, persist it, render its report.

The caller names a declarative experiment; everything else lives here and in the
config:

* the run executes through the config-first factory (``ExperimentRunFactory`` /
  ``ExperimentBatch``);
* the results table + route dump are persisted under ``out_dir`` (default
  ``artifacts/results/<experiment>``) with the stem read from the experiment cfg
  (``output.paper_stem``) -- no stem or path is spelled by the caller;
* figures are rendered via :func:`render_report`.

Each function returns a :class:`PaperRun` the caller displays or saves. A call
always re-runs and overwrites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

from ..core import (ExperimentBatch, ExperimentRunFactory, build_experiment,
                    load_experiment)
from ..core.paths import RESULTS_DIR, resolve_under_root
from ..reports import render_report, save_paper_table, save_paper_routes


def results_dir(name: str, out_dir=None) -> Path:
    """Where a run's deliverables land: ``out_dir`` when given, else
    ``artifacts/results/<experiment>``. A relative ``out_dir`` resolves against
    the repo root, so it means the same from any working directory."""
    if out_dir is not None:
        return resolve_under_root(out_dir)
    stem = str(name).replace("experiments/", "").replace("/", "_")
    return RESULTS_DIR / stem


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


def _stem_for(stem: str, city=None) -> str:
    """The output stem, suffixed with the city when a run was retargeted -- the
    YAML stem names the city the config itself targets."""
    if city is None:
        return str(stem)
    seg = str(city).lower()
    stem = str(stem)
    return stem if stem.endswith(seg) else f"{stem}_{seg}"


def _persist(artifact, cfg, out_dir: Path, *, city=None) -> None:
    """Save the table (+ route dump) under ``out_dir``, named by the cfg stem.

    A run that declares no ``output.paper_stem`` (e.g. an ad-hoc
    ``run_experiment`` on a method leaf) is not a deliverable -> no persist.
    ``city`` suffixes the stem so per-city outputs never clobber each other."""
    out = cfg.get("output")
    if out is None or out.get("paper_stem") is None:
        return
    stem = _stem_for(out.paper_stem, city)
    if getattr(artifact, "table", None) is not None:
        save_paper_table(artifact.table.round(4), stem, out_dir=out_dir)
    routes = getattr(artifact, "routes", None)
    instance = getattr(artifact, "instance", None)
    if routes and instance is not None:
        save_paper_routes(stem, routes, instance.coords, instance.street_adj,
                          meta={"stem": stem}, out_dir=out_dir)


def save_results(stem: str, *, out_dir, table=None, routes=None, coords=None,
                 street_adj=None, meta=None) -> None:
    """Ad-hoc save of a curated table / route set (e.g. the single best EKB
    solution) into ``out_dir``."""
    out_dir = resolve_under_root(out_dir)
    if table is not None:
        save_paper_table(table, stem, out_dir=out_dir)
    if routes:
        save_paper_routes(stem, routes, coords, street_adj, meta=meta,
                          out_dir=out_dir)


def run_experiment(name: str, *, out_dir=None, kind: str | None = None,
                   title: str | None = None, **params) -> PaperRun:
    """Run a single declarative experiment, persist it, render its report.

    ``**params`` are the procedural config knobs forwarded to
    :func:`build_experiment` (``city``, ``alpha``, ``adj_target``,
    ``n_iterations``, ``route_len``, ``weights_dir``, ...): each defaults to the
    YAML value and is overridden only when passed. ``out_dir`` is where the
    table / route dump land (default ``artifacts/results/<experiment>``).
    """
    cfg = build_experiment(name, **params)
    out_path = results_dir(name, out_dir)
    artifact = ExperimentRunFactory.from_cfg(cfg).run()
    _persist(artifact, cfg, out_path, city=params.get("city"))
    report = render_report(artifact, kind=kind, title=title)
    return PaperRun(artifact=artifact, table=report.table, figures=report.figures)


def run_batch(name: str, *, out_dir=None, kind: str | None = None,
              city: str | None = None, **params) -> PaperRun:
    """Run a declarative batch (multi-method), persist the combined table, render.

    ``city`` / ``**params`` are forwarded to :func:`build_experiment` for every
    run in the batch, so one batch config drives any city without a per-city
    batch file. The stem is suffixed with the city so per-city outputs never
    clobber each other.
    """
    cfg = load_experiment(name)
    out_path = results_dir(name, out_dir)
    batch = ExperimentBatch(cfg, base_name=name).run(city=city, **params)
    if batch.table is not None and bool(
            cfg.batch.get("include_initial_metrics", False)):
        batch.table = _prepend_initial_metrics(
            batch.table, batch.artifacts, cfg, runtime_params=params)
    stem = _stem_for(cfg.output.paper_stem, city)
    if batch.table is not None:
        save_paper_table(batch.table.drop(columns=["run"]).round(4), stem,
                         out_dir=out_path)
    report = render_report(batch, kind=kind)
    return PaperRun(artifact=batch, table=report.table, figures=report.figures)


def _prepend_initial_metrics(table, artifacts, cfg, *, runtime_params=None):
    """Prepend one Initial row per unique sweep point.

    Method artifacts all share the same seeded benchmark network, so scoring
    Initial inside every method run would duplicate it. The paper batch owns
    this comparison row and evaluates it once for each alpha/adjustment point
    using the same objective settings as the searches.
    """
    import pandas as pd

    from ..evaluation import full_metric_row, score_fixed_routes, select_metrics

    artifact = next(
        (art for art in artifacts if getattr(art, "instance", None) is not None),
        None,
    )
    if artifact is None:
        raise ValueError(
            "batch.include_initial_metrics requires a sweep artifact instance")
    instance = artifact.instance
    point_columns = [
        name for name in ("alpha", "adj_target") if name in table.columns
    ]
    points = (
        table[point_columns].drop_duplicates().to_dict("records")
        if point_columns else [{}]
    )

    runtime_params = dict(runtime_params or {})
    sweep = cfg.get("sweep") or {}
    adj_weight = runtime_params.get("adj_weight", sweep.get("adj_weight"))
    keep = list(cfg.get("metrics", []))
    rows = []
    for point in points:
        alpha = point.get("alpha")
        adj_target = point.get("adj_target")
        alpha = None if pd.isna(alpha) else alpha
        adj_target = None if pd.isna(adj_target) else adj_target
        metrics, scored = score_fixed_routes(
            instance.init_routes,
            instance.tensors,
            instance.spec,
            alpha=alpha,
            adj_target=adj_target,
            adj_weight=adj_weight,
            seed_routes=instance.init_routes,
        )
        row = full_metric_row(metrics, scored, instance.init_routes)
        if keep:
            row = select_metrics(row, keep)
        row.update(method="Initial", **point)
        if "run" in table.columns:
            row["run"] = "Initial"
        if "n_iterations" in table.columns:
            row["n_iterations"] = 0
        rows.append(row)

    initial = pd.DataFrame(rows).reindex(columns=table.columns)
    return pd.concat([initial, table], ignore_index=True)

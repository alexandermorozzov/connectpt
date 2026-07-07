"""Unified experiment report -- thin re-export of the library renderer.

The one renderer now lives in the library (``reports.render.render_report``);
it is duck-typed on ``table`` / ``routes`` / ``instance`` so it renders the
notebook-side ``ExperimentResult`` and the library ``SearchArtifact`` alike.
Re-exported here so the existing ``eval_lib.experiment_report`` name keeps
working.
"""
from connectpt.routes_generator.reports import (  # noqa: F401
    render_report, ReportArtifact)

"""Reports application: build tables/plots from saved artifacts.

Reads artifacts (CSV/JSON/route dumps, evaluation results) and renders tables +
figures. It must NOT import the training/ or search/ layers -- it works only off
persisted data + evaluation result schemas, so figures can be rebuilt from disk.
"""

from .artifact_loaders import (
    load_evaluation_result, load_table, load_routes, load_search_summary,
)
from .search_tables import make_search_plan_table, make_search_comparison_table
from .comparison_tables import make_comparison_table
from .figures import plot_routes_grid, plot_pareto, style_table
from .route_plots import plot_plain_route_set, plot_route_diff
from .render import render_report, ReportArtifact
from .paper_io import (PAPER_DIR, paper_path, paper_row, save_paper_table,
                      save_paper_routes, append_paper_row, reset_paper_table,
                      save_paper_fig, ravel_hist)
from .report_run import ReportRun


__all__ = [
    "load_evaluation_result",
    "load_table",
    "load_routes",
    "load_search_summary",
    "make_search_plan_table",
    "make_search_comparison_table",
    "make_comparison_table",
    "plot_routes_grid",
    "plot_pareto",
    "style_table",
    "plot_plain_route_set",
    "plot_route_diff",
    "render_report",
    "ReportArtifact",
    "PAPER_DIR",
    "paper_path",
    "paper_row",
    "save_paper_table",
    "save_paper_routes",
    "append_paper_row",
    "reset_paper_table",
    "save_paper_fig",
    "ravel_hist",
    "ReportRun",
]

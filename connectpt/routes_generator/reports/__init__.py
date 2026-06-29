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
from .report_run import ReportRun


__all__ = [
    "load_evaluation_result",
    "load_table",
    "load_routes",
    "load_search_summary",
    "make_search_plan_table",
    "make_search_comparison_table",
    "make_comparison_table",
    "ReportRun",
]

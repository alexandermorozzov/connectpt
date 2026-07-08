"""Moved to the library: ``connectpt.routes_generator.reports.results_io``.

Re-exported so the eval_lib / notebook callers keep the ``eval_lib.results_io``
names (single implementation now lives in the reports layer).
"""
from connectpt.routes_generator.reports.results_io import (  # noqa: F401
    RESULTS_DIR, save_table)

"""Unified comparison-table builder over :class:`RunResult` objects.

`build_comparison_table` produces the one standard DataFrame used by every
experiment section (LC / NX / MACSA / Mumford-Mandl): one row per run, with
worse / without-worse as separate rows (an ``accept_mode`` column), and deltas
vs the reference (initial-network) row.

Numeric columns are rounded to ``TABLE_DECIMALS`` (default 3) before the
DataFrame is returned, so both the CSV that ``save_table`` writes and any
``display()`` of the DataFrame in notebooks use the same precision.
"""
import pandas as pd

from .params import ENABLED_COST_COMPONENTS
from .helpers import summarize_run, metric_value
from . import plots as _plots


# Number of decimals every table-builder rounds its numeric columns to.
# Tweak here once -- all experiment tables / saved CSVs pick it up.
TABLE_DECIMALS = 3

_METRIC_COLUMNS = [
    "cost", "cost_demand_term", "cost_route_term", "cost_connectivity_term",
    "cost_penalty_term", "ATT", "RTT", "$d_0$", "$d_1$", "$d_2$", "$d_{un}$",
    "median_connectivity", "# disconnected node pairs",
    "# stops out of bounds", "# routes",
]


def run_result_row(result) -> dict:
    """One RunResult -> a flat row dict (summarize_run + d0/d1/d2 + tags)."""
    row = summarize_run(result.label, result.metrics, result.routes)
    row["kind"] = result.kind
    row["accept_mode"] = result.accept_mode
    if result.dataset:
        row["dataset"] = result.dataset
    for key in ("$d_0$", "$d_1$", "$d_2$"):
        row[key] = metric_value(result.metrics, key)
    return row


def build_comparison_table(results, reference_kind="initial") -> pd.DataFrame:
    """Standard comparison DataFrame for a list of RunResult.

    Columns: identifying (dataset / method / kind / accept_mode) + standard
    metrics + ``delta_*_vs_initial``. Disabled cost components are dropped via
    ``filter_component_columns``.
    """
    df = pd.DataFrame([run_result_row(r) for r in results])
    if df.empty:
        return df
    delta_cols = [c for c in ("cost", "ATT", "RTT") if c in df.columns]
    # Deltas are computed against the reference row of the *same* dataset, so a
    # multi-dataset table (e.g. the MACSA scenarios) gets per-scenario deltas.
    groups = (df.groupby("dataset") if "dataset" in df.columns
              else [(None, df)])
    for col in delta_cols:
        df[f"delta_{col}_vs_initial"] = float("nan")
    for _, group in groups:
        ref_rows = group[group["kind"] == reference_kind]
        if ref_rows.empty:
            continue
        ref = ref_rows.iloc[0]
        for col in delta_cols:
            df.loc[group.index, f"delta_{col}_vs_initial"] = (
                group[col] - ref[col])
    lead = [c for c in ("dataset", "method", "kind", "accept_mode")
            if c in df.columns]
    metrics = [c for c in _METRIC_COLUMNS if c in df.columns]
    deltas = [c for c in df.columns if c.startswith("delta_")]
    keep = _plots.filter_component_columns(lead + metrics + deltas,
                                           ENABLED_COST_COMPONENTS)
    # Round numeric columns -- DataFrame.round skips non-numerics.
    return df[keep].round(TABLE_DECIMALS)

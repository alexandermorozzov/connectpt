"""Comparison tables across methods/runs (pure pandas)."""
from __future__ import annotations

import pandas as pd


def make_comparison_table(named_summaries: dict[str, pd.DataFrame],
                          *, label_col: str = "method") -> pd.DataFrame:
    """Stack per-method summary rows into one comparison table.

    ``named_summaries`` maps a method label to its 1-row summary DataFrame.
    """
    frames = []
    for label, summary in named_summaries.items():
        row = summary.copy()
        row.insert(0, label_col, label)
        frames.append(row)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

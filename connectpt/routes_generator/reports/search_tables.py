"""Tables built from search artifacts (pure pandas, no search imports)."""
from __future__ import annotations

import pandas as pd


def make_search_comparison_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Compare bee-type search runs from their saved summaries.

    ``summaries`` maps a run label to its loaded ``*_search.json`` dict
    (mean_cost + metrics + plan). One row per run: cost, key metrics, bee mix.
    """
    rows = []
    for label, summary in summaries.items():
        row = {"run": label, "mean_cost": summary.get("mean_cost")}
        row.update(summary.get("metrics") or {})
        counts = (summary.get("plan") or {}).get("counts", {})
        row["bee_mix"] = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "mean_cost" in df.columns:
        df = df.sort_values("mean_cost").reset_index(drop=True)
    return df


def make_search_plan_table(plan: dict) -> pd.DataFrame:
    """Turn a plan summary dict (BeeColonyRunner.plan_summary) into a table.

    Counts are keyed by bee name (the plan group's name), so the name is the
    label -- no separate type-slot -> label lookup.
    """
    counts = plan.get("counts", {})
    rows = [
        {"bee": name, "count": int(count)}
        for name, count in counts.items() if count
    ]
    df = pd.DataFrame(rows, columns=["bee", "count"])
    return df.sort_values("bee").reset_index(drop=True)

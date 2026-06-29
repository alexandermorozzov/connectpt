"""Tables built from search artifacts (pure pandas, no search imports)."""
from __future__ import annotations

import pandas as pd

# bee_colony type -> human label (kept here so reports don't import search/)
_TYPE_LABELS = {
    "n_type1": "random path combiner",
    "n_type2": "shorten",
    "n_type3": "random path combining",
    "n_type4": "neural construction (extend)",
    "n_type5": "neural edit (extend/+trim)",
    "n_type6": "neural trim-only",
    "n_type7": "compound trim->extend",
}


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
    """Turn a BeeColonyPlan summary dict into a readable bee-mix table."""
    counts = plan.get("counts", {})
    rows = [
        {"bee_type": key, "label": _TYPE_LABELS.get(key, key), "count": int(count)}
        for key, count in counts.items() if count
    ]
    df = pd.DataFrame(rows, columns=["bee_type", "label", "count"])
    return df.sort_values("bee_type").reset_index(drop=True)

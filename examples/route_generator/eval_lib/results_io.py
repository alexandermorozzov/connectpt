"""Persist experiment results as CSV tables under ``artifacts/results``."""
from .context import ARTIFACTS_DIR

RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    """Filesystem-safe slug (route/combo labels may carry '/', '(' etc.)."""
    return "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in str(name)).strip("_") or "results"


def save_table(df, name: str, *, subdir: str | None = None):
    """Save a results DataFrame to ``artifacts/results/[<subdir>/]<name>.csv``.

    ``subdir`` (optional) routes the CSV into a child folder beneath
    ``RESULTS_DIR``; the folder is created if missing. Use this to keep a
    self-contained experiment-series notebook's outputs grouped together
    (e.g. ``subdir="final"`` for the consolidated experiment notebook).
    """
    if subdir:
        out_dir = RESULTS_DIR / _safe_name(subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = RESULTS_DIR
    # Honour the global TEMP/smoke output prefix (single source: eval_lib.paper)
    # so throwaway runs never overwrite real result files.
    from . import paper as _paper
    path = out_dir / f"{_safe_name(_paper.PAPER_PREFIX + name)}.csv"
    df.to_csv(path, index=False)
    print(f"[results] table ({len(df)} rows) -> {path}")
    return path



"""Persist experiment results as CSV tables under ``artifacts/results``."""
from .context import ARTIFACTS_DIR

RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    """Filesystem-safe slug (route/combo labels may carry '/', '(' etc.)."""
    return "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in str(name)).strip("_") or "results"


def save_table(df, name: str, *, prefix: str = "", subdir: str | None = None):
    """Save a results DataFrame to ``artifacts/results/[<subdir>/]<name>.csv``.

    ``prefix`` is the explicit output-filename prefix (thread it from
    ``RunContext.output_prefix``; ``"TEMP_"`` on smoke runs keeps throwaway
    outputs from overwriting real result files). ``subdir`` (optional) routes
    the CSV into a child folder beneath ``RESULTS_DIR``; the folder is created
    if missing.
    """
    if subdir:
        out_dir = RESULTS_DIR / _safe_name(subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = RESULTS_DIR
    path = out_dir / f"{_safe_name(prefix + name)}.csv"
    df.to_csv(path, index=False)
    print(f"[results] table ({len(df)} rows) -> {path}")
    return path



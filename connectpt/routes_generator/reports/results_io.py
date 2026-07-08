"""Persist experiment results as CSV tables under ``artifacts/results``.

Thin prefix/subdir/slug wrapper over the library ``ArtifactStore`` (the single
tabular sink) -- the CSV write itself is not re-implemented here.
"""
from ..core import ArtifactStore
from ..core.paths import ARTIFACTS_DIR

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
    out_dir = RESULTS_DIR / _safe_name(subdir) if subdir else RESULTS_DIR
    path = ArtifactStore(out_dir).save_table(df, _safe_name(prefix + name))
    print(f"[results] table ({len(df)} rows) -> {path}")
    return path



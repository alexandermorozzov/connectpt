"""Shared plumbing for the paper CLI runners (EKB case study, Figure 5).

Deliberately thin: the experiment logic lives in the library
(``paper_experiments.paper_runs`` / ``core`` / ``search``) exactly as the
notebooks call it. These helpers only add the CLI-side concerns -- file+console
logging and writing the rendered figures to disk (the library already persists
the metrics table + route dumps and streams TensorBoard).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    """Log INFO+ to both ``log_path`` and stdout (so a detached run is followable
    via the log file and, live, via the console)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [fh, ch]
    return logging.getLogger("connectpt.cli")


def save_figures(figures: dict, out_dir: Path, stem: str, prefix: str = "") -> list:
    """Write each rendered matplotlib figure to ``<prefix><stem>_<name>.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fig in (figures or {}).items():
        path = out_dir / f"{prefix}{stem}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
    return saved

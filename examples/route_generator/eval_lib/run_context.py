"""Explicit run context for paper_combined.ipynb -- replaces module globals.

Built ONCE from the suite profile in the notebook setup cell and passed
explicitly to everything that needs runtime knobs (edit checkpoint, benchmark
init mode, output-filename prefix). This replaces the old pattern of assigning
``eval_lib.helpers.EDIT_MODEL_WEIGHTS_PATH`` / ``eval_lib.paper.PAPER_PREFIX``
etc. from notebook cells (mutable module state).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .context import EDIT_MODEL_WEIGHTS_DIR


@dataclass(frozen=True)
class RunContext:
    """Per-session runtime knobs (immutable; derive variants via ``replace``).

    * ``edit_weights_path`` / ``edit_adj_cond_feats`` -- the edit-model
      checkpoint the BCO trim/extend bees load, and its adjustment-conditioning
      feature count (0 = unconditioned legacy checkpoints).
    * ``benchmark_init_mode`` -- how benchmark initial route sets are built
      ("rpc" or "nx").
    * ``output_prefix`` -- filename prefix for every results sink (``"TEMP_"``
      on smoke runs so throwaway outputs never overwrite real paper files).
    """

    edit_weights_path: Path
    edit_adj_cond_feats: int = 0
    benchmark_init_mode: str = "rpc"
    output_prefix: str = ""

    @classmethod
    def from_suite(cls, suite) -> "RunContext":
        """Build the context from a loaded suite profile (cfg/experiments/suite*)."""
        return cls(
            edit_weights_path=(EDIT_MODEL_WEIGHTS_DIR
                               / str(suite.model.edit_checkpoint)),
            edit_adj_cond_feats=int(suite.model.edit_adj_cond_feats),
            benchmark_init_mode=str(suite.benchmark_init_mode),
            output_prefix=str(suite.output_prefix or ""),
        )

    def with_edit_model(self, weights_path, adj_cond_feats: int = 0) -> "RunContext":
        """A copy pointing the edit bees at another checkpoint (e.g. just-trained)."""
        return replace(self, edit_weights_path=Path(weights_path),
                       edit_adj_cond_feats=int(adj_cond_feats))

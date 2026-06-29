"""Load artifacts for reporting (no training/ or search/ imports)."""
from __future__ import annotations

from pathlib import Path

from ..core.artifacts import ArtifactStore
from ..evaluation.result_schema import EvaluationResult


def _store(output_dir) -> ArtifactStore:
    return ArtifactStore(Path(output_dir))


def load_evaluation_result(output_dir, name: str) -> EvaluationResult:
    return EvaluationResult.load(_store(output_dir), name)


def load_table(output_dir, name: str):
    return _store(output_dir).load_table(name)


def load_routes(output_dir, name: str):
    return _store(output_dir).load_routes(name)


def load_search_summary(output_dir, run_name: str) -> dict:
    """Load a BeeColonySearchRun's saved summary ({run_name}_search.json)."""
    return _store(output_dir).load_json(f"{run_name.replace('/', '_')}_search")

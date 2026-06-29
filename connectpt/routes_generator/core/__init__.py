"""Shared, framework-level building blocks for route-generation experiments.

This package holds the pieces that the training, search, evaluation and report
layers all depend on but that none of them own: checkpoint IO, the shared batch
state, the action space, paths and runtime helpers.

Modules here must NOT import ``training``/``search``/``reports`` -- the
dependency arrow points the other way (see the refactor plan, section 5).
"""

from .checkpoints import CheckpointStore


__all__ = ["CheckpointStore"]

"""Shared, framework-level building blocks for route-generation experiments.

This package holds the pieces that the training, search, evaluation and report
layers all depend on but that none of them own: checkpoint IO, the shared batch
state, the action space, paths and runtime helpers.

Modules here must NOT import ``training``/``search``/``reports`` -- the
dependency arrow points the other way (see the refactor plan, section 5).
"""

from .checkpoints import CheckpointStore
from .actions import (
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_TRIM_START,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_HALT,
    ACTION_NAME_TO_KIND,
    ACTION_KIND_TO_NAME,
)
from .route_state import RouteGenBatchState
from .runtime import RunContext, resolve_device, seed_everything
from .artifacts import ArtifactStore


__all__ = [
    "CheckpointStore",
    "ROUTE_ACTION_EXTEND",
    "ROUTE_ACTION_TRIM_START",
    "ROUTE_ACTION_TRIM_END",
    "ROUTE_ACTION_HALT",
    "ACTION_NAME_TO_KIND",
    "ACTION_KIND_TO_NAME",
    "RouteGenBatchState",
    "RunContext",
    "resolve_device",
    "seed_everything",
    "ArtifactStore",
]

"""The route action space -- a stable import surface.

Re-exports the ``ROUTE_ACTION_*`` kind constants (defined in
``transit_time_estimator``) plus a name<->kind mapping that the search layer
uses to translate ``allowed_actions`` strings to action kinds. Importing from
here keeps callers decoupled from where the constants physically live.
"""
from __future__ import annotations

from ..transit_time_estimator import (
    ROUTE_ACTION_EXTEND,
    ROUTE_ACTION_TRIM_START,
    ROUTE_ACTION_TRIM_END,
    ROUTE_ACTION_HALT,
)

# action name (search vocabulary) -> integer action kind
ACTION_NAME_TO_KIND: dict[str, int] = {
    "extend": ROUTE_ACTION_EXTEND,
    "trim_start": ROUTE_ACTION_TRIM_START,
    "trim_end": ROUTE_ACTION_TRIM_END,
    "halt": ROUTE_ACTION_HALT,
}
ACTION_KIND_TO_NAME: dict[int, str] = {v: k for k, v in ACTION_NAME_TO_KIND.items()}

__all__ = [
    "ROUTE_ACTION_EXTEND",
    "ROUTE_ACTION_TRIM_START",
    "ROUTE_ACTION_TRIM_END",
    "ROUTE_ACTION_HALT",
    "ACTION_NAME_TO_KIND",
    "ACTION_KIND_TO_NAME",
]

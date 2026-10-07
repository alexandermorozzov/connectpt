"""Which route actions each model role supports.

A single declarative map from model role -> allowed action vocabulary. The
search layer uses it to validate that a bee's ``allowed_actions`` are actually
supported by the policy's model (e.g. a construction model cannot trim), failing fast with a
clear error instead of silently no-op'ing.

Action vocabulary matches the search-side ``allowed_actions`` strings and the
``ROUTE_ACTION_*`` kinds in ``transit_time_estimator``.
"""
from __future__ import annotations

# Role -> the set of action names that role's model can propose.
ROLE_CAPABILITIES: dict[str, set[str]] = {
    # PathCombiningRouteGenerator: builds/extends routes, can stop.
    "construction": {"extend", "halt"},
    # TrimPathCombiningRouteGenerator: the full edit action space.
    "edit": {"extend", "trim_start", "trim_end", "halt"},
}

ALL_ACTIONS: set[str] = set().union(*ROLE_CAPABILITIES.values())


def validate_actions(role: str, allowed_actions) -> None:
    """Raise ValueError if ``allowed_actions`` exceed what ``role`` supports."""
    caps = ROLE_CAPABILITIES.get(role)
    if caps is None:
        raise ValueError(f"unknown model role {role!r}; known: {sorted(ROLE_CAPABILITIES)}")
    unknown = set(allowed_actions) - ALL_ACTIONS
    if unknown:
        raise ValueError(f"unknown action(s): {sorted(unknown)}; known: {sorted(ALL_ACTIONS)}")
    invalid = set(allowed_actions) - caps
    if invalid:
        raise ValueError(
            f"policy role {role!r} does not support action(s): {sorted(invalid)} "
            f"(supports {sorted(caps)})"
        )

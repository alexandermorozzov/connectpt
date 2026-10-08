"""Route action vocabulary and action-space contracts."""
from dataclasses import dataclass
from torch import Tensor

ROUTE_ACTION_EXTEND = 0
ROUTE_ACTION_TRIM_START = 1
ROUTE_ACTION_TRIM_END = 2
ROUTE_ACTION_HALT = 3

ACTION_NAME_TO_KIND = {"extend": 0, "trim_start": 1, "trim_end": 2, "halt": 3}
ACTION_KIND_TO_NAME = {value: name for name, value in ACTION_NAME_TO_KIND.items()}

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
            f"policy role {role!r} does not support action(s): {sorted(invalid)} " f"(supports {sorted(caps)})"
        )


@dataclass(frozen=True)
class ActionSpec:
    role: str

    @property
    def kinds(self):
        return tuple(ACTION_NAME_TO_KIND[name] for name in ACTION_NAME_TO_KIND if name in ROLE_CAPABILITIES[self.role])

    def validate(self, policy_spec):
        if self != policy_spec:
            raise ValueError(f"environment actions {self} differ from policy actions {policy_spec}")


@dataclass(frozen=True)
class RouteAction:
    kind: Tensor
    indices: Tensor

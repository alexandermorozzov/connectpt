"""Route action vocabulary and action-space contracts."""
from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class ActionSpec:
    role: str

    def validate(self, policy_spec):
        if self != policy_spec:
            raise ValueError(f"environment actions {self} differ from policy actions {policy_spec}")


@dataclass(frozen=True)
class RouteAction:
    kind: Tensor
    indices: Tensor

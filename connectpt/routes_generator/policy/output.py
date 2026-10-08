"""One policy decision, including its complete action distribution."""
from dataclasses import dataclass

import torch

from ..env.actions import RouteAction


@dataclass(frozen=True)
class PolicyOut:
    action: RouteAction
    logp: torch.Tensor
    entropy: torch.Tensor
    probs: torch.Tensor

    @property
    def log_probs(self):
        supported = self.probs > 0
        return torch.where(supported, self.probs, 1.0).log().masked_fill(~supported, -torch.inf)

"""Small route-tensor utilities shared across the data layer."""
from __future__ import annotations

import torch

from ..torch_utils import get_batch_tensor_from_routes


def as_route_tensor(routes) -> torch.Tensor:
    """Normalise a route set to a detached CPU tensor.

    Accepts either a tensor (returned as-is on CPU) or a nested route list,
    which is padded into a batch tensor via the library encoder.
    """
    if isinstance(routes, torch.Tensor):
        return routes.detach().cpu()
    return get_batch_tensor_from_routes(routes).detach().cpu()

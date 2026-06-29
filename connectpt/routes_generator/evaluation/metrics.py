"""Compute route metrics from the shared cost machinery.

Reads the normalized cost components from a built cost module + state (the same
numbers training and search optimize), so evaluation reports exactly the
objective. Pure helpers are separated out so they can be unit-tested without a
real graph state.
"""
from __future__ import annotations

import torch

from ..transit_time_estimator import COST_COMPONENT_NAMES


class MetricComputer:
    @staticmethod
    def row_from_components(components: torch.Tensor, names=COST_COMPONENT_NAMES,
                           *, total=None) -> dict[str, float]:
        """Map a (batch, n_components) tensor to {component: mean value}.

        Pure: no state needed, so it is directly unit-testable.
        """
        comp = components.detach().float()
        if comp.ndim == 1:
            comp = comp[None, :]
        row = {f"{name}_cost": float(comp[:, i].mean().item())
               for i, name in enumerate(names)}
        if total is not None:
            t = total.detach().float() if torch.is_tensor(total) else torch.as_tensor(total)
            row["cost"] = float(t.mean().item())
        return row

    @staticmethod
    def compute(state, cost_obj) -> dict[str, float]:
        """Compute the metric row for a built state + cost module."""
        total = cost_obj(state).cost
        # get_cost_components returns all components (columns aligned with
        # COST_COMPONENT_NAMES), regardless of which are enabled/weighted.
        components = cost_obj.get_cost_components(state)
        return MetricComputer.row_from_components(
            components, COST_COMPONENT_NAMES, total=total)

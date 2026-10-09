"""One physical evaluation of a State, shared by RL, search and export."""
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from .components import CostHelperOutput
from .const import GOAL_NAMES, ACTION_PENALTIES
from .penalties import network_adjustment
from .preferences import scalarize, batch_value, named_preferences


@dataclass
class Evaluation(CostHelperOutput):
    goal_names: tuple = ()
    penalty_names: tuple = ()
    normalized: dict = None
    raw_goals: dict = None
    monitors: dict = None
    adjustment: Tensor = None
    reference_routes: Tensor = None
    redundancy: Tensor = None
    _cost: Tensor = None

    @property
    def cost(self):
        return self._cost

    @classmethod
    def concatenate(cls, evaluations):
        """Combine compatible graph batches; named scalar metadata must agree."""
        def combine(values, name=None):
            first = values[0]
            if isinstance(first, Tensor):
                sizes = [max(value.shape[axis] for value in values) for axis in range(1, first.ndim)]
                padding = [[size - value.shape[axis + 1] for axis, size in enumerate(sizes)] for value in values]
                values = [torch.nn.functional.pad(value, tuple(x for amount in reversed(pad) for x in (0, amount)),
                                                 value=-1 if name in ("batch_routes", "reference_routes") else
                                                 float("nan") if name == "connectivity_pair_times" else 0)
                          for value, pad in zip(values, padding)]
                return torch.cat(values)
            if isinstance(first, dict):
                return {key: combine([value[key] for value in values]) for key in first}
            if isinstance(first, list):
                return [combine([value[i] for value in values]) if isinstance(item, Tensor) else item
                        for i, item in enumerate(first)]
            if any(value != first for value in values):
                raise ValueError("Evaluation metadata differs")
            return first
        return cls(**{field.name: combine([getattr(value, field.name) for value in evaluations], field.name) for field in fields(cls)})

    def select(self, indices):
        """Select flattened batch rows, retaining a batch axis and all metadata."""
        indices = torch.as_tensor(indices, device=self.cost.device, dtype=torch.long).reshape(-1)
        def take(value):
            if isinstance(value, Tensor):
                return value[indices].clone()
            if isinstance(value, dict):
                return {key: take(item) for key, item in value.items()}
            if isinstance(value, list):
                return [take(item) if isinstance(item, Tensor) else item for item in value]
            return value
        return replace(self, **{field.name: take(getattr(self, field.name)) for field in fields(self)})

    def updated(self, other, mask):
        """Return selected candidate rows without mutating either evaluation."""
        def merge(left, right):
            if isinstance(left, Tensor):
                if left.shape != right.shape:
                    raise ValueError("Evaluation batches have incompatible shapes")
                selector = mask.reshape((-1,) + (1,) * (left.ndim - 1))
                return torch.where(selector, right, left)
            if isinstance(left, dict):
                return {key: merge(value, right[key]) for key, value in left.items()}
            if isinstance(left, list):
                return [merge(a, b) if isinstance(a, Tensor) else a for a, b in zip(left, right)]
            if left != right:
                raise ValueError("Evaluation metadata differs")
            return left
        return replace(self, **{field.name: merge(getattr(self, field.name), getattr(other, field.name)) for field in fields(self)})


def evaluate(state, cost, *, constraint_weight=None, no_norm=False, return_per_route_riders=False):
    """Pure [B,...] evaluation on the state's device, preserving RNG and inputs.

    Physical values and constraints are computed once. The returned tensors are
    owned snapshots. Goals/preferences use configured goal order; penalties use
    configured order, excluding environment action penalties.
    """
    physical = cost.physical(state, return_per_route_riders)
    inputs = {field.name: getattr(physical, field.name) for field in fields(CostHelperOutput)}
    inputs = {key: value.clone() if isinstance(value, Tensor) else value for key, value in inputs.items()}
    adjustment = (network_adjustment(state.adjustment_routes(), state.adjustment_seed, state.symmetric_routes)
                  if state.adjustment_seed.numel() else state.drive_times.new_zeros(state.batch_size))
    physical_view = Evaluation(**inputs, adjustment=adjustment)
    names = tuple(dict.fromkeys((*GOAL_NAMES, *cost.objectives)))
    raw_goals = {name: cost.goal_criteria[name].compute(state, physical_view, no_norm=True).clone() for name in names}
    normalized = {name: cost.goal_criteria[name].normalize(state, value)
                  if cost.goal_criteria[name].normalize is not None else value
                  for name, value in raw_goals.items()}
    goals = torch.stack([(raw_goals if no_norm else normalized)[name] for name in cost.objectives], -1)
    preferences = named_preferences(state, cost.objectives)
    penalty_names = tuple(name for name in cost.penalty_weights if name not in ACTION_PENALTIES)
    weights, values = [], []
    for name in penalty_names:
        weight = constraint_weight if name == "constraints" and constraint_weight is not None else cost.penalty_weights[name]
        weight = batch_value(weight, state.batch_size, state.device).clone()
        value = (cost.penalty_criteria[name].compute(state, physical_view) if bool((weight != 0).any())
                 else state.drive_times.new_zeros(state.batch_size))
        values.append(value)
        weights.append(weight)
    penalties = torch.stack(values, -1) if values else goals.new_empty((state.batch_size, 0))
    scalar = scalarize(goals, preferences, penalties, weights)
    if not bool(torch.isfinite(scalar).all()) or bool((scalar < 0).any()):
        raise ValueError("invalid cost was computed")
    constraint = penalties[:, penalty_names.index("constraints")] * weights[penalty_names.index("constraints")] if "constraints" in penalty_names else goals.new_zeros(state.batch_size)
    counts = state.total_leg_use_count.float()
    if state.symmetric_routes:
        counts = counts.triu()
    redundancy = (counts - 1).clamp_min(0).sum((1, 2)) / counts.sum((1, 2)).clamp_min(1)
    return Evaluation(
        **{**inputs, "goals": goals, "preferences": preferences, "penalties": penalties,
           "penalty_weights": weights, "components": torch.stack([normalized[name] for name in GOAL_NAMES], -1),
           "constraint_cost": constraint},
        goal_names=tuple(cost.objectives), penalty_names=penalty_names, normalized=normalized, raw_goals=raw_goals,
        monitors={name: raw_goals[name] if name in raw_goals else criterion.compute(state, physical_view, no_norm=True).clone()
                  for name, criterion in cost.monitor_criteria.items()},
        adjustment=adjustment, reference_routes=state.adjustment_seed.clone(),
        redundancy=redundancy, _cost=scalar,
    )

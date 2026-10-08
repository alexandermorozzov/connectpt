"""Rebuild observations for replay of pre-refactor training buffers."""
import torch

from ..env.actions import RouteAction, ROUTE_ACTION_HALT, ROUTE_ACTION_EXTEND
from ..env.features import observe
from ..env.masks import ActionMask


def evaluate_action(policy, state, indices, kinds=None, mask=None):
    if kinds is None:
        kinds = torch.where(indices[:, 0] < 0, ROUTE_ACTION_HALT, ROUTE_ACTION_EXTEND)
    obs = observe(
        state,
        policy.feature_set,
        allow_below_min=policy.allow_trim_below_min if policy.action_spec.role == "edit" else False,
    )
    return policy.evaluate(obs, ActionMask() if mask is None else mask, RouteAction(kinds, indices))

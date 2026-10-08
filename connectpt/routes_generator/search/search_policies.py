"""Search policy adapters -- how a bee turns state into a route action.

A policy adapter wraps a route model and exposes ``propose_action``. It does NOT
own the model's ``state_dict`` (the model is the trained class, loaded straight
from a checkpoint); it only routes calls to the model's step methods and gates
them by ``allowed_actions``. The capability check uses the central
``model_capabilities`` map, so a construction policy asked to trim fails loudly.
"""
from __future__ import annotations

from ..env.masks import ActionMask
from ..env.actions import ROLE_CAPABILITIES, validate_actions
from ..env.actions import ACTION_NAME_TO_KIND


class SearchPolicyAdapter:
    """Base adapter: a named role + a model, with action validation."""

    role: str = ""

    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or self.role

    @property
    def capabilities(self) -> set[str]:
        return ROLE_CAPABILITIES[self.role]

    def validate_actions(self, allowed_actions) -> None:
        validate_actions(self.role, allowed_actions)

    def propose_action(self, state, *, allowed_actions, greedy=False):
        raise NotImplementedError


class ConstructionSearchPolicy(SearchPolicyAdapter):
    """PathCombiningRouteGenerator: extend / halt only."""

    role = "construction"

    def propose_action(self, state, *, allowed_actions, greedy=False):
        self.validate_actions(allowed_actions)
        # construction model exposes step() (extend/halt); no trim support
        return self.model.step(state, greedy=greedy)


class EditSearchPolicy(SearchPolicyAdapter):
    """TrimPathCombiningRouteGenerator: extend / trim_start / trim_end / halt."""

    role = "edit"

    def propose_action(self, state, *, allowed_actions, greedy=False):
        self.validate_actions(allowed_actions)
        return self.model.step_route_action(
            state, greedy=greedy, mask=ActionMask.from_names(allowed_actions)
        )


_ADAPTERS = {
    "construction_search_policy": ConstructionSearchPolicy,
    "edit_search_policy": EditSearchPolicy,
    "construction": ConstructionSearchPolicy,
    "edit": EditSearchPolicy,
}


def build_policies(models: dict) -> dict[str, SearchPolicyAdapter]:
    """Build the canonical adapter for each loaded model role."""
    return {role: _ADAPTERS[role](model, name=role)
            for role, model in models.items()}

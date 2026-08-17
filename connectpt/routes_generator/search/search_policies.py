"""Search policy adapters -- how a bee turns state into a route action.

A policy adapter wraps a route model and exposes ``propose_action``. It does NOT
own the model's ``state_dict`` (the model is the trained class, loaded straight
from a checkpoint); it only routes calls to the model's step methods and gates
them by ``allowed_actions``. The capability check uses the central
``model_capabilities`` map, so a construction policy asked to trim fails loudly.
"""
from __future__ import annotations

from ..model_capabilities import ROLE_CAPABILITIES, validate_actions
from ..core.actions import ACTION_NAME_TO_KIND


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

    def _allow_flags(self, allowed_actions) -> dict[str, bool]:
        allowed = set(allowed_actions)
        return {
            "allow_extend": "extend" in allowed,
            "allow_halt": "halt" in allowed,
            "allow_trim_start": "trim_start" in allowed,
            "allow_trim_end": "trim_end" in allowed,
        }

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
            state, greedy=greedy, **self._allow_flags(allowed_actions)
        )


_ADAPTERS = {
    "construction_search_policy": ConstructionSearchPolicy,
    "edit_search_policy": EditSearchPolicy,
    "construction": ConstructionSearchPolicy,
    "edit": EditSearchPolicy,
}


def build_policies(policies_cfg, models: dict) -> dict[str, SearchPolicyAdapter]:
    """Build named policy adapters from cfg.policies + a {name: model} map.

    Each policy entry has ``model_ref`` (key into ``models``) and ``adapter``
    (one of the registered adapter names). Returns {policy_name: adapter}.
    """
    out: dict[str, SearchPolicyAdapter] = {}
    for name, spec in dict(policies_cfg).items():
        spec = dict(spec)
        model = models[spec["model_ref"]]
        adapter_cls = _ADAPTERS[spec["adapter"]]
        out[name] = adapter_cls(model, name=name)
    return out

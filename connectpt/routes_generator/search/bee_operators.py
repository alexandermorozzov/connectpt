"""Bee operators -- what a bee does, decoupled from which model it uses.

``NeuralRouteActionBee`` is the key piece: it pairs ANY policy adapter with an
``allowed_actions`` set, a route selector and an acceptance rule. The same class
works with a construction policy + [extend, halt], an edit policy + [extend,
halt], or an edit policy + [trim_start, trim_end] -- the only difference is the
injected policy and action set. Construction validates the action set against
the policy's capabilities, so an invalid combination fails immediately.
"""
from __future__ import annotations

from dataclasses import dataclass

from .acceptance import get_acceptance
from .route_selectors import get_route_selector
from .search_policies import SearchPolicyAdapter


class NeuralRouteActionBee:
    """A neural mutation operator over a policy adapter and an action set."""

    def __init__(self, policy: SearchPolicyAdapter, allowed_actions,
                 route_selector, acceptance_policy):
        self.policy = policy
        self.allowed_actions = list(allowed_actions)
        self.route_selector = route_selector
        self.acceptance_policy = acceptance_policy
        # Fail fast: the policy's model must support every requested action.
        self.policy.validate_actions(self.allowed_actions)

    def propose(self, state, context=None):
        return self.policy.propose_action(
            state, allowed_actions=self.allowed_actions,
            greedy=(context or {}).get("greedy", False),
        )


class NeuralRebuildBee:
    """Full-route rebuild via a construction policy (bee_colony type-1 neural).

    Unlike ``NeuralRouteActionBee`` (one extend/trim/halt step), this replaces the
    chosen route wholesale using the construction model's rollout -- the paper's
    neural-BCO rebuild operator. There is no per-action gating, so no
    ``allowed_actions`` validation is needed.
    """

    def __init__(self, policy, route_selector, acceptance_policy):
        self.policy = policy
        self.route_selector = route_selector
        self.acceptance_policy = acceptance_policy


class HeuristicMutationBee:
    """A model-free mutation (e.g. shorten, random path combine)."""

    def __init__(self, kind: str, route_selector, acceptance_policy):
        self.kind = kind
        self.route_selector = route_selector
        self.acceptance_policy = acceptance_policy


@dataclass
class _CompoundStep:
    bee: NeuralRouteActionBee


class CompoundBee:
    """A bee that applies several operator steps before one acceptance check."""

    def __init__(self, steps: list[NeuralRouteActionBee], acceptance_policy):
        self.steps = list(steps)
        self.acceptance_policy = acceptance_policy


def build_bee(spec, policies: dict):
    """Build a concrete bee from a BeeSpec + the {name: policy} map."""
    acceptance = get_acceptance(spec.acceptance)

    if spec.operator == "neural_route_action":
        policy = policies[spec.policy]
        return NeuralRouteActionBee(
            policy=policy,
            allowed_actions=spec.allowed_actions or [],
            route_selector=get_route_selector(spec.route_selection),
            acceptance_policy=acceptance,
        )

    if spec.operator == "neural_rebuild":
        return NeuralRebuildBee(
            policy=policies[spec.policy],
            route_selector=get_route_selector(spec.route_selection),
            acceptance_policy=acceptance,
        )

    if spec.operator == "heuristic_mutation":
        return HeuristicMutationBee(
            kind=spec.mutation_kind or spec.route_selection,
            route_selector=get_route_selector(spec.route_selection),
            acceptance_policy=acceptance,
        )

    if spec.operator == "compound":
        steps = []
        for step in spec.steps or []:
            policy = policies[step["policy"]]
            steps.append(NeuralRouteActionBee(
                policy=policy,
                allowed_actions=list(step.get("allowed_actions", [])),
                route_selector=get_route_selector(step.get("route_selection", "uniform")),
                acceptance_policy=acceptance,
            ))
        return CompoundBee(steps=steps, acceptance_policy=acceptance)

    raise ValueError(f"unknown bee operator {spec.operator!r}")

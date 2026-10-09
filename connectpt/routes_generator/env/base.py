"""Environment contract: shared episode mechanics, task rules in the subclasses."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .actions import ActionSpec, RouteAction
from .const import RouteActionKind
from .features import FeatureParams, observe, select_features, NODE_COLUMNS
from ..city.graph import STOP_KEY
from ..objectives.const import ACTION_PENALTIES


@dataclass(frozen=True)
class EnvParams:
    n_routes: int
    min_route_len: int
    max_route_len: int | None
    diff_reward: bool
    reward_scale: float
    force_linking_unlinked: bool
    max_act_len: int | None
    max_route_edit_steps: int | None
    max_trim_actions_per_route: int | None
    force_nonhalt_first_step: bool
    allow_trim_below_min: bool
    forbid_halt_when_overlong: bool


def normalize_nodes(state, normalizer, feature_set):
    """Normalize once at reset; the neural layer retains its checkpoint keys."""
    nodes = select_features(state.graph_data[STOP_KEY].x, NODE_COLUMNS, feature_set.node)
    state.set_normalized_features(normalizer(nodes))
    state.set_redundancy_feature_mode(feature_set.redundancy)


class Env(ABC):
    """``reset``, ``step``, ``observe``. A subclass sets ``action_spec`` and the task rules."""

    action_spec: ActionSpec

    def __init__(self, cost, features: FeatureParams, params: EnvParams, policy_spec: ActionSpec, *, node_normalizer):
        self.action_spec.validate(policy_spec)
        self.cost = cost
        self.feature_set = features
        self.params = params
        self.node_normalizer = node_normalizer
        self.state = None
        self.steps = 0
        self.result = None
        self.mask_override = None

    @abstractmethod
    def reset(self, graphs, preferences, seed_networks=None, *, normalize=True):
        """Build a State for the graphs and bind it; returns ``obs, mask``."""

    @abstractmethod
    def action_mask(self, state):
        """Allowed action families for the next step."""

    @abstractmethod
    def forced_halt(self):
        """True when the next action is replaced by halt."""

    @abstractmethod
    def _apply(self, action: RouteAction):
        """Change the State by one action."""

    @abstractmethod
    def action_penalties(self, action: RouteAction, active, forced):
        """Unweighted penalties of the action, ``[B, len(ACTION_PENALTIES)]``."""

    def bind(self, state, *, normalize=True, single_route=False, evaluation=None):
        self.state = state
        self.state.set_redundancy_feature_mode(self.feature_set.redundancy)
        if normalize:
            normalize_nodes(state, self.node_normalizer, self.feature_set)
        self.steps = 0
        self.single_route = single_route
        self.ended = torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)
        self.result = self.cost(state) if evaluation is None else evaluation
        return self.observe(state)

    def is_done(self):
        return self.ended if self.single_route else self.state.is_done()

    def observe(self, state=None):
        state = self.state if state is None else state
        obs = observe(state, self.feature_set, allow_below_min=self.params.allow_trim_below_min)
        return obs, self.action_mask(state)

    def step(self, action: RouteAction):
        active = ~self.is_done()
        forced = self.forced_halt()
        if forced:
            action = RouteAction(torch.full_like(action.kind, RouteActionKind.HALT), torch.full_like(action.indices, -1))
        previous = self.result
        self._apply(action)
        self.ended |= action.kind == RouteActionKind.HALT
        self.steps += 1
        self.result = self.cost(self.state)
        done = self.is_done()
        if self.params.diff_reward:
            goals = previous.goals - self.result.goals
            penalties = previous.penalties - self.result.penalties
        else:
            goals = -self.result.goals * done[:, None]
            penalties = -self.result.penalties * done[:, None]
        goals = goals * self.params.reward_scale * active[:, None]
        penalties = penalties * self.params.reward_scale * active[:, None]
        # Action penalties are episodic costs, not potential differences.
        actions = self.action_penalties(action, active, forced).to(penalties.dtype)
        penalties = torch.cat((penalties, torch.zeros_like(actions) - actions), dim=-1)
        obs, mask = self.observe()
        info = {
            "preferences": self.result.preferences,
            "penalty_weights": [
                *self.result.penalty_weights,
                *(self.cost.penalty_weights[name] for name in ACTION_PENALTIES),
            ],
            "forced_halt": forced,
        }
        return obs, mask, (goals, penalties), done, info

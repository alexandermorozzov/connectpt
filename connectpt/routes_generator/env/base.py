"""Environment contract: shared episode mechanics, task rules in the subclasses."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch

from .actions import ActionSpec, RouteAction
from .const import RouteActionKind
from .features import FeatureParams, observe, select_features, NODE_COLUMNS
from .masks import prepare_action_mask
from .rewards import potential_reward
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


@dataclass(frozen=True)
class EditEnvParams(EnvParams):
    max_route_edit_steps: int | None
    max_trim_actions_per_route: int | None
    force_nonhalt_first_step: bool
    allow_trim_below_min: bool
    forbid_halt_when_overlong: bool


@dataclass(frozen=True)
class EnvironmentBinding:
    factory: object
    params: EnvParams

    def create(self, policy, cost, *, mask=None):
        return self.factory(cost, policy.feature_set, self.params, policy.action_spec,
                            node_normalizer=policy.node_norm, mask=mask)

    def bind(self, state, policy, cost, *, normalize=True, single_route=False, mask=None):
        params=replace(self.params,n_routes=state.n_routes_to_plan,
                       min_route_len=state.min_route_len,max_route_len=state.max_route_len)
        env=EnvironmentBinding(self.factory,params).create(policy,cost,mask=mask)
        env.bind(state,normalize=normalize,single_route=single_route)
        return env


def normalize_nodes(state, normalizer, feature_set):
    """Normalize once at reset; the neural layer retains its checkpoint keys."""
    nodes = select_features(state.graph_data[STOP_KEY].x, NODE_COLUMNS, feature_set.node)
    state.set_normalized_features(normalizer(nodes))


class Env(ABC):
    """``reset``, ``step``, ``observe``. A subclass sets ``action_spec`` and the task rules."""

    action_spec: ActionSpec
    params_type = EnvParams

    def __init__(self, cost, features: FeatureParams, params: EnvParams, policy_spec: ActionSpec, *, node_normalizer, mask=None):
        self.action_spec.validate(policy_spec)
        if type(params) is not self.params_type:
            raise TypeError(f'{self.action_spec.role} requires {self.params_type.__name__}')
        self.cost = cost
        self.feature_set = features
        self.params = params
        self.node_normalizer = node_normalizer
        self.state = None
        self.steps = 0
        self.result = None
        self.mask_override = mask

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
        if normalize:
            normalize_nodes(state, self.node_normalizer, self.feature_set)
        self.steps = 0
        self.single_route = single_route
        self.ended = torch.zeros(state.batch_size, dtype=torch.bool, device=state.device)
        self.result = self.cost(state) if evaluation is None else evaluation
        return self.observe(state)

    def is_done(self):
        return self.ended if self.single_route else self.state.is_done()

    def set_route_count(self, count):
        """Configure the next reset; existing State keeps its own episode limits."""
        self.params = replace(self.params, n_routes=count)

    def prepare_iteration(self, iteration, force_nonhalt_until):
        """Construction has no edit first-step restriction."""

    @property
    def allow_trim_below_min(self):
        return False

    def observation(self, state=None):
        """Compute features without consulting the current episode's action limits."""
        state = self.state if state is None else state
        return observe(state, self.feature_set, allow_below_min=self.allow_trim_below_min)

    def observe(self, state=None):
        state = self.state if state is None else state
        obs = self.observation(state)
        return obs, prepare_action_mask(obs, self.params, self.action_spec.role, self.action_mask(state))

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
        goals = potential_reward(previous.goals, self.result.goals, active, done,
                                 diff=self.params.diff_reward, scale=self.params.reward_scale)
        penalties = potential_reward(previous.penalties, self.result.penalties, active, done,
                                     diff=self.params.diff_reward, scale=self.params.reward_scale)
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

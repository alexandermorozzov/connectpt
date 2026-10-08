"""Batched route construction, observation and vector rewards."""
from dataclasses import dataclass

import torch

from .actions import ActionSpec, RouteAction
from .features import FeatureParams, observe, select_features, NODE_COLUMNS
from .masks import ActionMask
from .state import RouteGenBatchState
from ..city.graph import STOP_KEY, to_batch
from ..network.routes import to_routes


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


class ConstructionEnv:
    action_spec = ActionSpec("construction")

    def __init__(self, cost, features: FeatureParams, params: EnvParams, policy_spec: ActionSpec, *, node_normalizer):
        self.action_spec.validate(policy_spec)
        self.cost = cost
        self.feature_set = features
        self.params = params
        self.node_normalizer = node_normalizer
        self.state = None
        self.steps = 0
        self.result = None

    def reset(self, graphs, preferences, seed_networks=None):
        self.state = RouteGenBatchState(
            to_batch(graphs),
            self.cost.constants,
            self.params.n_routes,
            self.params.min_route_len,
            self.params.max_route_len,
            preferences=preferences,
        )
        if seed_networks is not None:
            self.state.add_new_routes(to_routes(seed_networks).to(self.state.device))
        return self.bind(self.state)

    def bind(self, state, *, normalize=True):
        self.state = state
        self.state.set_redundancy_feature_mode(self.feature_set.redundancy)
        if normalize:
            normalize_nodes(state, self.node_normalizer, self.feature_set)
        self.steps = 0
        self.result = self.cost(state)
        return self.observe(state)

    def observe(self, state=None):
        state = self.state if state is None else state
        obs = observe(state, self.feature_set, allow_below_min=self.params.allow_trim_below_min)
        return obs, self.action_mask(state)

    def action_mask(self, state):
        return ActionMask(extend=True, trim_start=False, trim_end=False, halt=True)

    def forced_halt(self):
        return False

    def _apply(self, action: RouteAction):
        self.state.shortest_path_action(action.indices)

    def _penalties(self, result):
        zeros = result.goals.new_zeros(result.goals.shape[0], 2)
        return torch.cat((result.penalties[:, :2], zeros), dim=-1)

    def step(self, action: RouteAction):
        active = ~self.state.is_done()
        previous = self.result
        self._apply(action)
        self.steps += 1
        self.result = self.cost(self.state)
        done = self.state.is_done()
        if self.params.diff_reward:
            goals = previous.goals - self.result.goals
            penalties = self._penalties(previous) - self._penalties(self.result)
        else:
            goals = -self.result.goals * done[:, None]
            penalties = -self._penalties(self.result) * done[:, None]
        goals = goals * self.params.reward_scale * active[:, None]
        penalties = penalties * self.params.reward_scale * active[:, None]
        obs, mask = self.observe()
        info = {
            "preferences": self.result.preferences,
            "penalty_weights": [
                *self.result.penalty_weights[:2],
                self.cost.penalty_weights["edit_step"],
                self.cost.penalty_weights["forced_halt"],
            ],
            "forced_halt": False,
        }
        return obs, mask, (goals, penalties), done, info

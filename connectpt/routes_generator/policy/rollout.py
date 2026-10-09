"""The shared policy-to-environment loop."""
from dataclasses import dataclass, replace

import torch

from ..env.actions import RouteAction
from ..env.const import RouteActionKind
from ..env.base import EnvParams
from ..env.construction import ConstructionEnv
from ..env.edit import EditEnv
from .output import PolicyOut


ENVS = {"construction": ConstructionEnv, "edit": EditEnv}


def bind_env(
    state,
    policy,
    cost,
    *,
    normalize=True,
    single_route=False,
    max_route_edit_steps=None,
    max_trim_actions=None,
    force_nonhalt_first_step=False,
    mask=None,
):
    """Bind a pre-existing state at integration boundaries."""
    role = policy.action_spec.role
    trim = role == "edit"
    params = EnvParams(
        state.n_routes_to_plan,
        state.min_route_len,
        state.max_route_len,
        True,
        1.0,
        policy.force_linking_unlinked,
        policy.max_act_len,
        max_route_edit_steps,
        max_trim_actions,
        force_nonhalt_first_step,
        policy.allow_trim_below_min if trim else False,
        policy.forbid_halt_when_overlong if trim else False,
    )
    env = ENVS[role](cost, policy.feature_set, params, policy.action_spec, node_normalizer=policy.node_norm)
    env.mask_override = mask
    env.bind(state, normalize=normalize, single_route=single_route)
    return env


@dataclass
class Trajectory:
    states: list
    outputs: list[PolicyOut]
    rewards: list
    dones: list
    infos: list
    final_state: object

    @property
    def actions(self):
        return torch.stack([out.action.indices for out in self.outputs], dim=1)

    @property
    def action_kinds(self):
        return torch.stack([out.action.kind for out in self.outputs], dim=1)

    @property
    def logp(self):
        return torch.stack([out.logp for out in self.outputs], dim=1)

    @property
    def entropy(self):
        return torch.stack([out.entropy for out in self.outputs], dim=1)

    @property
    def total_logp(self):
        total = 0
        for out in self.outputs:
            total = total + out.logp
        return total

    @property
    def total_entropy(self):
        total = 0.0
        for out in self.outputs:
            total = total + out.entropy
        return total


def rollout(env, policy, greedy=False, max_steps=None, *, actions=None, record_states=True):
    """Collect steps without treating a horizon boundary as episode termination.

    Replay uses the same masks and state transitions as sampling.
    """
    trajectory = Trajectory([], [], [], [], [], env.state)
    obs, mask = env.observe()
    encoding = None
    if env.single_route and policy.action_spec.role == "construction":
        encoding = policy._encode_graph(obs)
    previous_finished = int(env.state.n_finished_routes[0].item())
    route_steps, current_steps, first_done = [], 0, False
    while not env.is_done().all():
        if max_steps is not None and len(trajectory.outputs) >= max_steps:
            break
        if record_states:
            trajectory.states.append(env.state.snapshot_for_buffer(device=torch.device("cpu")))
        active = ~env.is_done().clone()
        if encoding is not None:
            obs = replace(obs, encoding=encoding)
        if env.forced_halt() and actions is None:
            kind = torch.full((obs.batch_size,), RouteActionKind.HALT, dtype=torch.long, device=obs.device)
            indices = torch.full((obs.batch_size, 2), -1, dtype=torch.long, device=obs.device)
            zeros = torch.zeros(obs.batch_size, device=obs.device)
            out = PolicyOut(RouteAction(kind, indices), zeros, zeros.clone(), zeros[:, None])
        elif actions is None:
            out = policy.act(obs, mask, greedy)
        else:
            out = policy.evaluate(obs, mask, actions[len(trajectory.outputs)])
        if env.single_route:
            kind = out.action.kind.clone()
            indices = out.action.indices.clone()
            kind[~active] = RouteActionKind.HALT
            indices[~active | (kind == RouteActionKind.HALT)] = -1
            out = replace(out, action=RouteAction(kind, indices), logp=out.logp * active, entropy=out.entropy * active)
        obs, mask, reward, done, info = env.step(out.action)
        # Full distributions are needed by replay losses, not by the buffer.
        trajectory.outputs.append(replace(out, probs=out.probs.new_empty(0)))
        trajectory.rewards.append(reward)
        trajectory.dones.append(done.clone())
        trajectory.infos.append(info)
        current_steps += 1
        if env.single_route:
            if not first_done and done[0]:
                route_steps.append(current_steps)
                first_done = True
        else:
            finished = int(env.state.n_finished_routes[0].item())
            while finished > previous_finished:
                route_steps.append(current_steps)
                current_steps = 0
                previous_finished += 1
    if actions is None:
        policy._log_step_counts(route_steps)
    return trajectory

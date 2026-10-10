"""Algorithm contract and shared return calculations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AlgorithmParams:
    name: str
    gamma: float
    epsilon: float
    use_gae: bool
    gae_lambda: float
    n_epochs: int
    minibatch_size: int
    entropy_weight: float
    normalize_advantages: bool
    diversity_weight: float
    diversity_alpha: float
    preference_noise_sigma: float
    max_grad_norm: float


class Algorithm(ABC):
    def __init__(self, params, cost):
        self.params, self.cost = params, cost

    @property
    @abstractmethod
    def scalar_values(self):
        """True for [B] values; False for [B,K], including a single goal."""

    @abstractmethod
    def potential(self, result):
        """Scalarize before differencing, preserving floating point order."""

    @abstractmethod
    def surrogate(self, ratios, clipped, advantages, preferences):
        """Clipped policy objective in scalar or goal space."""

    @abstractmethod
    def diversity(self, policy, env, state, output, mask):
        """Extra algorithm objective, from the replay's full distribution."""

    @property
    @abstractmethod
    def value_dim(self):
        """Number of value predictions per state."""

    @property
    def value_names(self):
        """Stable head identifiers in the Evaluation's configured goal order."""
        return ('cost',) if self.scalar_values else tuple(self.cost.objectives)

    def returns(self, rewards, values, dones, final_values):
        if dones.ndim != 2 or dones.dtype != torch.bool:
            raise ValueError('terminal masks must be boolean [T,B]')
        shape = tuple(dones.shape) + (() if self.scalar_values else (self.value_dim,))
        if tuple(rewards.shape) != shape or tuple(values.shape) != shape or tuple(final_values.shape) != shape[1:]:
            raise ValueError('rewards, values and bootstrap must match the algorithm value contract')
        p = self.params
        return returns_and_advantages(rewards, values, dones, final_values, p.gamma, p.use_gae, p.gae_lambda)

    def bootstrap(self, values, unfinished):
        """Retain horizon values; terminal episodes contribute zero bootstrap."""
        shape = tuple(unfinished.shape) + (() if self.scalar_values else (self.value_dim,))
        if unfinished.ndim != 1 or unfinished.dtype != torch.bool or tuple(values.shape) != shape:
            raise ValueError('bootstrap requires [B] terminal mask and algorithm-shaped values')
        return values * expand_mask(unfinished, values)


@dataclass
class Diversity:
    loss: torch.Tensor
    kl: torch.Tensor
    target: torch.Tensor


def expand_mask(mask, value):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def normalize_advantages(advantages, *, per_objective=False):
    """Holliday's minibatch mean/sample-std, with a safe singleton fallback."""
    dim = 0 if per_objective else None
    centered = advantages - advantages.mean(dim=dim, keepdim=per_objective)
    count = advantages.shape[0] if per_objective else advantages.numel()
    if count < 2:
        return centered
    return centered / (advantages.std(dim=dim, keepdim=per_objective) + 1e-8)


def returns_and_advantages(rewards, value_estimates, dones, final_value_estimates, gamma, use_gae, gae_lambda):
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    if use_gae:
        lastgaelam = torch.zeros_like(final_value_estimates)
        for step_idx in reversed(range(rewards.shape[0])):
            if step_idx == rewards.shape[0] - 1:
                next_values = final_value_estimates
            else:
                next_values = value_estimates[step_idx + 1]
            next_nonterminal = (~dones[step_idx]).to(torch.float32)
            next_nonterminal = expand_mask(next_nonterminal, next_values)
            delta = rewards[step_idx] + gamma * next_values * next_nonterminal - value_estimates[step_idx]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            advantages[step_idx] = lastgaelam
        returns = advantages + value_estimates
    else:
        next_return = final_value_estimates
        for step_idx in reversed(range(rewards.shape[0])):
            next_nonterminal = (~dones[step_idx]).to(torch.float32)
            next_nonterminal = expand_mask(next_nonterminal, next_return)
            next_return = rewards[step_idx] + gamma * next_nonterminal * next_return
            returns[step_idx] = next_return
        advantages = returns - value_estimates

    return returns, advantages

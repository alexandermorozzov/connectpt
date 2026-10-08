"""Scalar PPO: scalarize the potential before return estimation."""
import torch

from .base import Algorithm, Diversity


class PPO(Algorithm):
    value_dim = 1
    scalar_values = True

    def potential(self, result):
        return result.cost

    def surrogate(self, ratios, clipped, advantages, preferences):
        return torch.minimum(ratios * advantages, clipped * advantages)

    def diversity(self, policy, state, output, mask):
        zero = output.logp.new_zeros(())
        return Diversity(zero, zero, zero)

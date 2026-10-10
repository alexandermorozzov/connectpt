"""Potential rewards for scalar and goal-wise consumers of one Evaluation."""
import torch


def potential_reward(previous, current, active, done, *, diff, scale):
    """Difference already-selected potentials, then scale and mask [B] rows.

    PPO supplies Evaluation.cost; vector consumers supply their own goal-space
    potentials. Scalarization happens before this function, never afterwards.
    """
    if previous.shape != current.shape or current.ndim not in (1, 2):
        raise ValueError('potential tensors must have matching [B] or [B,K] shapes')
    if active.dtype != torch.bool or done.dtype != torch.bool or active.shape != current.shape[:1] or done.shape != active.shape:
        raise ValueError('active/done masks must be boolean [B] tensors')
    row_shape = (-1,) + (1,) * (current.ndim - 1)
    if diff:
        reward = (previous - current) * scale
    else:
        reward = -current * scale * done.reshape(row_shape)
    return reward * active.reshape(row_shape)

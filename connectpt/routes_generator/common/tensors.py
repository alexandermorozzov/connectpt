# Copyright 2023 Andrew Holliday
#
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Transit Learning is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.


"""Tensor helpers that more than one package needs."""
from typing import Tuple

import torch


# currently unused, but might be useful in the future
def batch_2combinations(sequences):
    dev = sequences.device
    max_seq_len = sequences.shape[-1]
    triu_idxs = torch.triu_indices(max_seq_len, max_seq_len, 1, device=dev)
    # flatten all the non_sequence dimensions into one
    flat_seqs = sequences.reshape(-1, max_seq_len)
    from_grid = flat_seqs[..., None].expand(-1, -1, max_seq_len)
    from_idxs = from_grid[..., triu_idxs[0], triu_idxs[1]]
    to_grid = flat_seqs[..., None, :].expand(-1, max_seq_len, -1)
    to_idxs = to_grid[..., triu_idxs[0], triu_idxs[1]]
    flat_combinations = torch.stack([from_idxs, to_idxs], dim=-1)
    combinations = flat_combinations.reshape(*sequences.shape[:-1], -1, 2)
    return combinations


def get_update_at_mask(tensor, mask, new_value=0):
    """Pytorch's backprop doesn't like in-place updates to tensors in the
    computation graph, so we often need to get a new tensor with the
    same values at unmasked locations and a new value at masked locations.
    The way of doing it is a little counter-intuitive, so we've defined
    a helper function to hide the details.

    new_value: a scalar or a tensor matching the shape of tensor and mask.
        Defaults to 0.
    """
    src_elem_is_inf = tensor.isinf()
    src_has_inf = src_elem_is_inf.any()
    if src_has_inf:
        # multiplying infinity by 0 gives nan, so clone and assign instead
        tensor = tensor.clone()
        tensor[mask] = 0

    if type(new_value) is torch.Tensor:
        new_has_inf = new_value.isinf().any()
        if new_has_inf:
            new_value = new_value.clone()
    else:
        # new value is a scalar
        new_has_inf = abs(new_value) == float("inf")
        if new_has_inf:
            new_value = torch.full_like(tensor, new_value)

    if new_has_inf:
        # multiplying infinity by 0 gives nan, so clone and assign instead
        new_value = new_value.clone()
        new_value[~mask] = 0

    # add singleton dimensions to the mask as needed for the below arithmetic
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    updated = tensor * ~mask + new_value * mask

    return updated


def square_pdist(vectors):
    flat_dists = torch.pdist(vectors)
    num_vecs = vectors.shape[0]
    square = torch.zeros((num_vecs, num_vecs))
    end_i = 0
    for ii in range(num_vecs - 1):
        start_i = end_i
        end_i = start_i + num_vecs - (ii + 1)
        square[ii, ii + 1 :] = flat_dists[start_i:end_i]
    # make it diagonally symmetrical
    square += square.T
    return square




def cat_var_size_tensors(tensors, dim=0):
    """tensors: a list of tensors with the same ndim but variable sizes."""
    shapes = torch.tensor([tt.shape for tt in tensors])
    max_shape, _ = shapes.max(dim=0)
    padded_tensors = []
    for tensor in tensors:
        # wierdly, padding is a tuple of 2 * ndim ints, where element 0 is the
        # number of elements to pad at the start of the *last* dim, element 1
        # is the number to pad at the end of the last dim, elements 2 and 3
        # are for the start and end of the 2nd-to-last dim, and so on.
        padding = [(0, (ps - ss).item()) for ps, ss in zip(max_shape, tensor.shape)]
        padding[dim] = (0, 0)
        padding = sum(reversed(padding), ())
        padded = torch.nn.functional.pad(tensor, padding)
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim)


def get_variable_slice_mask(tensor, dim=0, froms=None, tos=None):
    """Suppose we want to slice along some dimension, but want different slices
    along that dimension at each location.  This function produces a mask
    that is True for all the included elements."""
    assert froms is not None or tos is not None, "Must provide indices"
    # 'while' just in case dim < -ndim, but that's doubtful
    assert tensor.ndim > 0, "Can't slice a scalar"
    while dim < 0:
        dim += tensor.ndim

    froms = _make_all_indices_positive(tensor.shape[dim], froms)
    tos = _make_all_indices_positive(tensor.shape[dim], tos)

    rng = torch.arange(tensor.shape[dim], device=tensor.device)
    # expand range to have the same shape as the input tensor
    dims = [1] * tensor.ndim
    dims[dim] = -1
    rng = rng.reshape(*dims).expand_as(tensor)

    mask = torch.ones_like(tensor, dtype=bool)
    if froms is not None:
        mask &= _get_forward_slice_mask(dim, rng, froms)
    if tos is not None:
        mask &= ~_get_forward_slice_mask(dim, rng, tos)
    return mask


# functions only used in get_variable_slice_mask().


def _make_all_indices_positive(dim_size, indices):
    """Assumes no -ve index is < -dim_size."""
    if indices is not None:
        is_neg = indices < 0
        indices = indices + is_neg * dim_size
    return indices


def _get_forward_slice_mask(dim, rng, idxs):
    is_start_idx = rng == idxs.unsqueeze(dim)
    return is_start_idx.cumsum(dim).to(bool)


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        # indices = indices // dim
        # use this form instead to silence the warning about deprecation
        indices = torch.div(indices, dim, rounding_mode="floor")

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

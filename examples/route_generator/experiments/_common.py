"""Small shared helpers for the one-off experiment modules (macsa / ekb /
training_lc). Extracted to remove copy-paste duplication found in the audit.
"""
from __future__ import annotations


def route_2d(routes):
    """Squeeze a leading batch dim: ``(1, R, L) -> (R, L)`` (else unchanged).

    The ``x[0] if x.ndim == 3 else x`` pattern was repeated across macsa / ekb /
    training_lc; this is the single source.
    """
    from eval_lib import as_route_tensor

    t = as_route_tensor(routes)
    return t[0] if t.ndim == 3 else t


def pad_routes_to(routes, n_routes, max_route_len, *, strict=False):
    """Normalise a route tensor to ``(n_routes, max_route_len)`` with -1 fill.

    ``strict=False`` (training / dataset generation): silently pad *or clip* both
    the route count and the width. ``strict=True`` (MACSA scoring): validate
    instead -- raise if the route count differs or the width exceeds
    ``max_route_len`` -- and only pad a short width. The two behaviours are kept
    byte-identical to the originals; the flag just selects which.
    """
    import torch

    from eval_lib import as_route_tensor

    t = as_route_tensor(routes).long()
    if t.ndim == 3:
        t = t[0]
    if strict:
        if t.shape[0] != int(n_routes):
            raise ValueError(f"Expected {n_routes} routes, got {tuple(t.shape)}")
        if t.shape[-1] > int(max_route_len):
            raise ValueError(f"Route width {t.shape[-1]} exceeds {max_route_len}")
    else:
        if t.shape[0] < n_routes:
            t = torch.cat([t, torch.full((n_routes - t.shape[0], t.shape[1]), -1, dtype=t.dtype)], 0)
        else:
            t = t[:n_routes]
    if t.shape[1] < max_route_len:
        t = torch.cat([t, torch.full((t.shape[0], max_route_len - t.shape[1]), -1, dtype=t.dtype)], 1)
    elif not strict and t.shape[1] > max_route_len:
        t = t[:, :max_route_len]
    return t

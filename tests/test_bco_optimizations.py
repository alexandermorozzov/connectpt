"""Bit-exactness tests for the BCO performance optimizations.

The optimized fast paths (count_path_nodes, the Needleman-Wunsch identical-
pair shortcut, deferred route-data updates) must produce EXACTLY the same
numbers as the original implementations -- these tests pin that.
"""
import random

import torch

from connectpt.routes_generator import torch_utils as tu
from connectpt.routes_generator.bee_colony import (
    _get_alignment_scores, _reverse_padded_routes, needleman_wunsch)


def test_count_path_nodes_equals_reconstruct_all_paths():
    torch.manual_seed(0)
    for _ in range(15):
        B = int(torch.randint(1, 8, (1,)))
        N = int(torch.randint(2, 35, (1,)))
        m = torch.rand(B, N, N) * 100
        m[torch.rand(B, N, N) > 0.65] = float("inf")
        m[:, range(N), range(N)] = 0
        nexts, _ = tu.floyd_warshall(m)
        assert torch.equal(tu.reconstruct_all_paths(nexts)[1],
                           tu.count_path_nodes(nexts))
    # unbatched form
    m = torch.rand(15, 15) * 10
    m[torch.rand(15, 15) > 0.5] = float("inf")
    m[range(15), range(15)] = 0
    nexts, _ = tu.floyd_warshall(m)
    assert torch.equal(tu.reconstruct_all_paths(nexts)[1],
                       tu.count_path_nodes(nexts))


def _alignment_reference(cand, ref, symmetric, gap):
    """The pre-optimization _get_alignment_scores (no identical-pair shortcut)."""
    L = cand.shape[-1]
    fc, fr = cand.reshape(-1, L), ref.reshape(-1, L)
    cv, rv = fc > -1, fr > -1
    m = (fc[:, :, None] == fr[:, None, :]) & cv[:, :, None] & rv[:, None, :]
    s = needleman_wunsch(m.to(torch.float32), gap=gap)
    if symmetric:
        rr = _reverse_padded_routes(fr)
        rrv = rr > -1
        m2 = (fc[:, :, None] == rr[:, None, :]) & cv[:, :, None] & rrv[:, None, :]
        s = torch.maximum(s, needleman_wunsch(m2.to(torch.float32), gap=gap))
    return s, cv.sum(-1), rv.sum(-1)


def test_alignment_shortcut_bit_identical():
    rng = random.Random(0)
    for trial in range(20):
        n_pairs, L, n_nodes = rng.randint(1, 25), rng.randint(2, 14), 40
        cand = torch.full((n_pairs, L), -1, dtype=torch.long)
        ref = torch.full((n_pairs, L), -1, dtype=torch.long)
        for i in range(n_pairs):
            rl = rng.randint(0, L)
            seq = rng.sample(range(n_nodes), rl)
            ref[i, :rl] = torch.tensor(seq, dtype=torch.long)
            mode = rng.random()
            if mode < 0.5:            # identical pair -> the fast path
                cand[i] = ref[i]
            elif mode < 0.65 and rl:  # reversed route
                cand[i, :rl] = torch.tensor(seq[::-1], dtype=torch.long)
            else:                     # unrelated route
                cl = rng.randint(0, L)
                cand[i, :cl] = torch.tensor(
                    rng.sample(range(n_nodes), cl), dtype=torch.long)
        for symmetric in (True, False):
            for gap in (0.0, 0.1):
                got = _get_alignment_scores(cand, ref, symmetric, gap=gap)
                want = _alignment_reference(cand, ref, symmetric, gap=gap)
                for g, w in zip(got, want):
                    assert torch.equal(g, w), \
                        f"trial={trial} symmetric={symmetric} gap={gap}"

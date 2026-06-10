import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

from eval_lib.route_copies import (  # noqa: E402
    REALISTIC_TIER_CFG, inject_realistic_tier, redundancy_fraction,
    repair_street_legs, street_legs_valid, try_detour_mutation,
    try_drop_cover_mutation, uncovered_demand_pct,
)

# The default max_redundancy_gain (5%) is tuned for benchmark-size networks;
# on the tiny test grid (~40 edge traversals) a single shared leg already
# exceeds it, so the unit tests relax the cap to exercise the mechanism.
_TEST_TIER_CFG = {
    **REALISTIC_TIER_CFG,
    "detour": {**REALISTIC_TIER_CFG["detour"], "max_redundancy_gain": 0.5},
}


def _grid_street_adj(side=5, t=60.0):
    """side x side grid street graph; nodes numbered row-major."""
    n = side * side
    adj = torch.full((n, n), float("inf"))
    adj.fill_diagonal_(0.0)
    for r in range(side):
        for c in range(side):
            u = r * side + c
            if c + 1 < side:
                adj[u, u + 1] = adj[u + 1, u] = t
            if r + 1 < side:
                adj[u, u + side] = adj[u + side, u] = t
    return adj


def _snake_routes(side=5, max_len=12, n_column_routes=2):
    """Row-snake routes covering every grid node, plus a couple of vertical
    column routes so part of the grid is doubly covered (street-valid legs).
    The double coverage is what lets a coverage-preserving detour exist."""
    routes = torch.full((side + n_column_routes, max_len), -1, dtype=torch.long)
    for r in range(side):
        cols = range(side) if r % 2 == 0 else range(side - 1, -1, -1)
        nodes = [r * side + c for c in cols]
        # extend into the next row so consecutive routes overlap a little
        if r + 1 < side:
            nodes.append((r + 1) * side + (nodes[-1] % side))
        routes[r, :len(nodes)] = torch.tensor(nodes)
    for k in range(n_column_routes):
        col = 1 + 2 * k
        nodes = [r * side + col for r in range(side)]
        routes[side + k, :len(nodes)] = torch.tensor(nodes)
    return routes


def _uniform_demand(n):
    d = torch.ones(n, n)
    d.fill_diagonal_(0.0)
    return d


def test_repair_street_legs_fixes_phantom_junction():
    adj = _grid_street_adj()
    # 0 -> 7 is not a street edge (0=(0,0), 7=(1,2)); repair must splice a path
    repaired = repair_street_legs([0, 7, 8], adj, 2, 12)
    assert repaired is not None
    assert repaired[0] == 0 and repaired[-1] == 8
    assert street_legs_valid(repaired, adj)


def test_detour_stretches_segment_and_stays_street_valid():
    adj = _grid_street_adj()
    routes = _snake_routes()
    res = try_detour_mutation(routes, random.Random(0), adj, 2, 12,
                              min_stretch=1.3, max_stretch=2.5,
                              max_redundancy_gain=0.5)
    assert res is not None
    ri, candidate, old_t, new_t = res
    assert 1.3 * old_t <= new_t <= 2.5 * old_t
    assert street_legs_valid(candidate, adj)
    assert len(candidate) == len(set(candidate))


def test_drop_cover_uncovers_a_singly_covered_stop():
    adj = _grid_street_adj()
    routes = _snake_routes()
    demand = _uniform_demand(25)
    res = try_drop_cover_mutation(routes, random.Random(0), demand, adj, 2, 12)
    assert res is not None
    ri, candidate, node = res
    assert node not in candidate
    assert street_legs_valid(candidate, adj)


def test_inject_realistic_tier_targets():
    adj = _grid_street_adj()
    routes = _snake_routes()
    demand = _uniform_demand(25)
    base_redun = redundancy_fraction(routes)
    out, applied = inject_realistic_tier(
        routes, random.Random(0), 2, 12,
        street_adj=adj, demand=demand, n_nodes=25, tier_cfg=_TEST_TIER_CFG)

    # something was injected, and no phantom legs anywhere
    assert sum(applied.values()) >= 3
    for row in out:
        ns = [int(x) for x in row.tolist() if int(x) >= 0]
        assert street_legs_valid(ns, adj)
        assert len(ns) == len(set(ns))

    # moderate redundancy growth (not the wholesale cloning of covered_dup)
    delta = redundancy_fraction(out) - base_redun
    assert 0.0 <= delta <= 0.20

    # a few honestly-uncovered points, bounded by the coverage cap
    d_un = uncovered_demand_pct(out, demand, 25)
    assert 0.0 < d_un <= 10.0

    # the seed network itself is untouched (clone semantics)
    assert redundancy_fraction(routes) == base_redun

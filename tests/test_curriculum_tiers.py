"""Contract tests for the graded training-curriculum tiers.

Every tier must produce street-valid simple routes within the length
contract, and each defect must be fixable by construction (the property the
curriculum is built on: a known reward direction exists).
"""
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTE_EXAMPLES = REPO_ROOT / "examples" / "route_generator"
if str(ROUTE_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(ROUTE_EXAMPLES))

from eval_lib.route_copies import (  # noqa: E402
    CURRICULUM_TIER_CFG, inject_curriculum_tier, redundancy_fraction,
    route_nodes, segment_time, street_legs_valid, truncate_route_tails,
    uncovered_demand_pct,
)

MIN_LEN, MAX_LEN = 2, 12
SIDE, N_NODES = 5, 25


def _grid_street_adj(side=SIDE, t=60.0):
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


def _base_routes(side=SIDE, max_len=MAX_LEN, n_column_routes=2):
    routes = torch.full((side + n_column_routes, max_len), -1, dtype=torch.long)
    for r in range(side):
        cols = range(side) if r % 2 == 0 else range(side - 1, -1, -1)
        nodes = [r * side + c for c in cols]
        if r + 1 < side:
            nodes.append((r + 1) * side + (nodes[-1] % side))
        routes[r, :len(nodes)] = torch.tensor(nodes)
    for k in range(n_column_routes):
        col = 1 + 2 * k
        nodes = [r * side + col for r in range(side)]
        routes[side + k, :len(nodes)] = torch.tensor(nodes)
    return routes


def _uniform_demand(n=N_NODES):
    d = torch.ones(n, n)
    d.fill_diagonal_(0.0)
    return d


def _assert_contract(routes, adj):
    for row in routes:
        ns = route_nodes(row)
        if not ns:
            continue
        assert len(ns) <= MAX_LEN and len(ns) >= MIN_LEN
        assert len(ns) == len(set(ns))
        assert street_legs_valid(ns, adj)


def _run_tier(name, seed=0):
    adj = _grid_street_adj()
    routes = _base_routes()
    demand = _uniform_demand()
    out, applied = inject_curriculum_tier(
        routes, CURRICULUM_TIER_CFG[name], random.Random(seed),
        MIN_LEN, MAX_LEN, street_adj=adj, demand=demand, n_nodes=N_NODES)
    _assert_contract(out, adj)
    return routes, out, applied, adj, demand


def test_all_tiers_respect_contract_and_run():
    for name in CURRICULUM_TIER_CFG:
        _run_tier(name)


def test_corrupt_all_routes_damages_every_route():
    from eval_lib.route_copies import count_changed_routes

    adj = _grid_street_adj()
    demand = _uniform_demand()
    for name in CURRICULUM_TIER_CFG:
        base = _base_routes()
        n_routes = int(base.shape[0])
        out, applied = inject_curriculum_tier(
            base, CURRICULUM_TIER_CFG[name], random.Random(0),
            MIN_LEN, MAX_LEN, street_adj=adj, demand=demand, n_nodes=N_NODES,
            target_corrupt_routes=n_routes)
        _assert_contract(out, adj)
        # every route slot differs from the clean seed, even on lc_clean.
        assert count_changed_routes(base, out) == n_routes, name


def test_target_corrupt_routes_respects_minimum_count():
    from eval_lib.route_copies import count_changed_routes

    adj = _grid_street_adj()
    demand = _uniform_demand()
    base = _base_routes()
    out, _ = inject_curriculum_tier(
        base, CURRICULUM_TIER_CFG["lc_clean"], random.Random(0),
        MIN_LEN, MAX_LEN, street_adj=adj, demand=demand, n_nodes=N_NODES,
        target_corrupt_routes=3)
    assert count_changed_routes(base, out) >= 3
    # default (no target) leaves the clean tier untouched.
    base2 = _base_routes()
    out2, _ = inject_curriculum_tier(
        base2, CURRICULUM_TIER_CFG["lc_clean"], random.Random(0),
        MIN_LEN, MAX_LEN, street_adj=adj, demand=demand, n_nodes=N_NODES)
    assert count_changed_routes(base2, out2) == 0


def test_dup_gross_increases_redundancy():
    base, out, applied, _, _ = _run_tier("dup_gross")
    assert sum(v for k, v in applied.items() if k.startswith("dup_")) >= 1
    assert redundancy_fraction(out) > redundancy_fraction(base)


def test_stub_routes_are_prefixes_of_originals():
    base, out, applied, _, _ = _run_tier("stub_routes")
    assert applied["stub_cut_stops"] > 0
    # fixability proof: every stub is a strict prefix of the original route,
    # so re-extension back to the original is always available.
    for orig_row, new_row in zip(base, out):
        orig, new = route_nodes(orig_row), route_nodes(new_row)
        assert len(new) >= MIN_LEN
        assert new == orig[:len(new)]


def test_detour_gross_stretches_street_time_without_uncovering():
    base, out, applied, adj, demand = _run_tier("detour_gross")
    assert applied["detour"] >= 1
    base_time = sum(segment_time(route_nodes(r), adj) for r in base)
    out_time = sum(segment_time(route_nodes(r), adj) for r in out)
    assert out_time > base_time  # the detour really is suboptimal
    assert uncovered_demand_pct(out, demand, N_NODES) <= \
        uncovered_demand_pct(base, demand, N_NODES) + 1e-9


def test_drop_cover_uncovers_demand():
    base, out, applied, _, demand = _run_tier("drop_cover")
    assert applied["drop_cover"] >= 1
    assert uncovered_demand_pct(out, demand, N_NODES) > 0.0


def test_lc_clean_is_a_noop():
    base, out, applied, _, _ = _run_tier("lc_clean")
    assert torch.equal(base, out)
    assert sum(applied.values()) == 0


def test_truncate_respects_min_len():
    routes = _base_routes()
    out, cut = truncate_route_tails(routes, random.Random(0), MIN_LEN, MAX_LEN,
                                    frac_range=(0.9, 0.95))
    assert cut > 0
    for row in out:
        ns = route_nodes(row)
        if ns:
            assert len(ns) >= MIN_LEN

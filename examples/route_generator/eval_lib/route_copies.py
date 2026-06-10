"""Copy-redundancy injection shared by paper_combined.ipynb PART 1 and PART 2.

One implementation of the route-copy corruption machinery that used to exist
twice in the notebook: the training-dataset tier generator (PART 1) and the
experiment init-network builder (PART 2) both inject duplicate routes /
sub-routes into a clean network so the edit model has removable redundancy to
trim. The tier presets (``COPY_TIER_CFG``) are the single source for both.

A "tier" config is a dict with:

* ``events``           -- how many injection events to attempt,
* ``max_multiplicity`` -- max copies of one donor per event (donor + N-1 targets),
* ``kinds``            -- allowed mutation kinds (subset of COPY_MUTATION_KINDS),
* ``covered``          -- if True, only accept copies that keep demand coverage
  intact (``uncovered_demand_pct <= d_un_cap``), so the redundancy is purely
  removable and trimming is the only way to cut cost,
* ``d_un_cap``         -- the coverage cap (%) for ``covered`` tiers.
"""
from collections import Counter

import torch

COPY_MUTATION_KINDS = ("full_copy", "prefix_copy", "suffix_copy", "middle_copy")

# Cap on unserved demand (%) for the coverage-preserving duplicate tier: a copy
# is only injected if the network still leaves <= this much demand disconnected.
D_UN_CAP_PCT = 10.0

# Tier presets used both for the training curriculum (PART 1, incl. lc_clean)
# and the experiment init corruption (PART 2 uses "covered_dup").
COPY_TIER_CFG = {
    "copy_full":     {"events": 3, "max_multiplicity": 3, "kinds": ("full_copy",)},
    "copy_boundary": {"events": 4, "max_multiplicity": 4,
                      "kinds": ("prefix_copy", "suffix_copy")},
    "copy_mixed":    {"events": 6, "max_multiplicity": 5, "kinds": COPY_MUTATION_KINDS},
    "covered_dup":   {"events": 6, "max_multiplicity": 4, "kinds": COPY_MUTATION_KINDS,
                      "covered": True, "d_un_cap": D_UN_CAP_PCT},
    "lc_clean":      {"events": 0, "max_multiplicity": 1, "kinds": ()},
}


def route_nodes(route):
    """Node sequence of one padded route row (drops the -1 padding)."""
    return [int(x) for x in route.tolist() if int(x) >= 0]


def is_simple_sequence(nodes, min_len, max_len):
    """True if the node sequence is a valid simple route within the contract."""
    return (min_len <= len(nodes) <= max_len and
            len(nodes) == len(set(nodes)) and
            all(a != b for a, b in zip(nodes[:-1], nodes[1:])))


def leg_counts(routes):
    """Counter of undirected edge traversals over all routes."""
    counts = Counter()
    for route in routes:
        ns = route_nodes(route)
        for a, b in zip(ns[:-1], ns[1:]):
            counts[(min(a, b), max(a, b))] += 1
    return counts


def redundancy_stats(routes):
    """Edge-redundancy summary: fraction of traversals that re-cover an edge."""
    counts = leg_counts(routes)
    traversals = sum(counts.values())
    redundancy = 0.0 if traversals == 0 else (traversals - len(counts)) / traversals
    return {
        "redundancy": float(redundancy),
        "max_leg_use": max(counts.values(), default=0),
        "edge_traversals": traversals,
        "unique_edges": len(counts),
    }


def redundancy_fraction(routes):
    return redundancy_stats(routes)["redundancy"]


def pad_routes(routes, n_routes, max_len):
    """Pad/trim a (n, len) or (1, n, len) route tensor to exactly (n_routes, max_len)."""
    t = torch.as_tensor(routes).long()
    if t.ndim == 3:
        t = t[0]
    if t.shape[0] < n_routes:
        t = torch.cat([t, torch.full((n_routes - t.shape[0], t.shape[1]), -1,
                                     dtype=t.dtype)], 0)
    else:
        t = t[:n_routes]
    if t.shape[1] < max_len:
        t = torch.cat([t, torch.full((t.shape[0], max_len - t.shape[1]), -1,
                                     dtype=t.dtype)], 1)
    elif t.shape[1] > max_len:
        t = t[:, :max_len]
    return t


def uncovered_demand_pct(routes, demand, n_nodes):
    """Demand-weighted % of OD pairs not connected by any route (union-find)."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for route in routes:
        ns = route_nodes(route)
        for a, b in zip(ns[:-1], ns[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    D = demand if torch.is_tensor(demand) else torch.as_tensor(demand)
    total = uncovered = 0.0
    for a, b in (D > 0).nonzero(as_tuple=False).tolist():
        if a == b:
            continue
        w = float(D[a, b])
        total += w
        if find(a) != find(b):
            uncovered += w
    return 0.0 if total == 0 else 100.0 * uncovered / total


def copy_candidate(routes, donor_idx, target_idx, kind, rng, min_len, max_len):
    """Propose a copy of (part of) the donor route into the target slot."""
    donor = route_nodes(routes[donor_idx])
    target = route_nodes(routes[target_idx])
    if not donor or not target:
        return None
    if kind == "full_copy":
        candidate = donor
    else:
        max_seg_len = min(len(donor), len(target), 6)
        if kind == "middle_copy":
            max_seg_len = min(max_seg_len, len(target) - 2)
        if max_seg_len < 2:
            return None
        seg_len = rng.randint(2, max_seg_len)
        if kind == "prefix_copy":
            candidate = donor[:seg_len] + target[seg_len:]
        elif kind == "suffix_copy":
            candidate = target[:-seg_len] + donor[-seg_len:]
        elif kind == "middle_copy":
            donor_start = rng.randint(0, len(donor) - seg_len)
            target_start = rng.randint(1, len(target) - seg_len - 1)
            candidate = (target[:target_start] +
                         donor[donor_start:donor_start + seg_len] +
                         target[target_start + seg_len:])
        else:
            raise ValueError(f"unknown mutation kind: {kind}")
    if candidate == target or not is_simple_sequence(candidate, min_len, max_len):
        return None
    return candidate


def replace_route(routes, route_idx, nodes):
    routes[route_idx] = -1
    routes[route_idx, :len(nodes)] = torch.as_tensor(nodes, dtype=routes.dtype)


def try_copy_mutation(routes, kind, max_multiplicity, rng, min_len, max_len,
                      demand=None, n_nodes=None, d_un_cap=None, attempts=120):
    """Copy one donor route/subroute into 1..max_multiplicity-1 recipients.

    Each accepted copy must strictly increase edge redundancy; when
    ``d_un_cap`` is set, copies that push uncovered demand above the cap are
    rejected (coverage-preserving "covered" tiers). Returns
    ``(mutated_routes, n_routes_touched)``; touched == 0 means no-op.
    """
    base = routes.clone()
    for _ in range(attempts):
        donor_idx = rng.randrange(routes.shape[0])
        target_idxs = [i for i in range(routes.shape[0]) if i != donor_idx]
        rng.shuffle(target_idxs)
        wanted = rng.randint(1, min(max_multiplicity - 1, len(target_idxs)))
        mutated = base.clone()
        current_redundancy = redundancy_fraction(mutated)
        touched = 0
        for target_idx in target_idxs:
            candidate = copy_candidate(mutated, donor_idx, target_idx, kind, rng,
                                       min_len, max_len)
            if candidate is None:
                continue
            proposal = mutated.clone()
            replace_route(proposal, target_idx, candidate)
            if redundancy_fraction(proposal) <= current_redundancy + 1e-12:
                continue
            if d_un_cap is not None and \
                    uncovered_demand_pct(proposal, demand, n_nodes) > d_un_cap:
                continue  # would break coverage -> reject
            mutated = proposal
            current_redundancy = redundancy_fraction(mutated)
            touched += 1
            if touched >= wanted:
                return mutated, touched
    return base, 0


def inject_route_copies(routes, tier_cfg, rng, min_len, max_len,
                        demand=None, n_nodes=None):
    """Apply one tier's worth of copy-injection events to a route tensor.

    Returns ``(routes, applied_events, mutated_routes)`` where the two
    Counters tally events / touched routes per mutation kind (used by the
    training-dataset meta table; callers that only need the routes can ignore
    them).
    """
    routes = routes.clone()
    applied_events = Counter()
    mutated_routes = Counter()
    covered = bool(tier_cfg.get("covered", False))
    d_un_cap = float(tier_cfg.get("d_un_cap", 100.0)) if covered else None
    for _ in range(int(tier_cfg["events"])):
        kinds = list(tier_cfg["kinds"])
        rng.shuffle(kinds)
        for kind in kinds:
            proposal, touched = try_copy_mutation(
                routes, kind, int(tier_cfg["max_multiplicity"]), rng,
                min_len, max_len, demand=demand, n_nodes=n_nodes, d_un_cap=d_un_cap)
            if touched:
                routes = proposal
                applied_events[kind] += 1
                mutated_routes[kind] += touched
                break
    return routes, applied_events, mutated_routes

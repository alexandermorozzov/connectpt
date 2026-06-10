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
import heapq
import math
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


# --- street-graph utilities (phantom-edge-free candidates) ------------------

def street_legs_valid(nodes, street_adj):
    """True if every consecutive leg of the node sequence is a real street edge."""
    return all(math.isfinite(float(street_adj[a, b]))
               for a, b in zip(nodes[:-1], nodes[1:]))


def street_path(street_adj, src, dst, rng=None, noise=0.0, forbidden=frozenset()):
    """Dijkstra over the street graph; optional multiplicative weight noise
    (``noise`` > 0 with ``rng``) randomizes the chosen path. ``forbidden``
    nodes are not traversed. Returns the node list src..dst or None."""
    n = street_adj.shape[0]
    dist = {src: 0.0}
    prev = {}
    heap = [(0.0, src)]
    seen = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v in range(n):
            if v == u or v in seen or (v != dst and v in forbidden):
                continue
            t = float(street_adj[u, v])
            if not math.isfinite(t):
                continue
            w = t * (1.0 + noise * rng.random()) if (rng is not None and noise > 0) else t
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in prev and dst != src:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def repair_street_legs(nodes, street_adj, min_len, max_len):
    """Splice shortest street paths into non-street junctions of a candidate
    route (prefix/suffix/middle copies can glue two segments at a node pair
    with no street edge -> a "teleport" leg). Returns the repaired simple
    sequence or None if it cannot be fixed within the contract."""
    repaired = [nodes[0]]
    for b in nodes[1:]:
        a = repaired[-1]
        if math.isfinite(float(street_adj[a, b])):
            repaired.append(b)
            continue
        fill = street_path(street_adj, a, b, forbidden=set(repaired) - {a})
        if fill is None:
            return None
        repaired.extend(fill[1:])
    if not is_simple_sequence(repaired, min_len, max_len):
        return None
    return repaired


def segment_time(nodes, street_adj):
    return sum(float(street_adj[a, b]) for a, b in zip(nodes[:-1], nodes[1:]))


def copy_candidate(routes, donor_idx, target_idx, kind, rng, min_len, max_len,
                   street_adj=None):
    """Propose a copy of (part of) the donor route into the target slot.

    When ``street_adj`` is given, glue junctions that don't exist in the
    street graph are repaired with the shortest street path (or the candidate
    is rejected) so no phantom legs enter the network."""
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
    if street_adj is not None and not street_legs_valid(candidate, street_adj):
        candidate = repair_street_legs(candidate, street_adj, min_len, max_len)
        if candidate is None or candidate == target:
            return None
    return candidate


def replace_route(routes, route_idx, nodes):
    routes[route_idx] = -1
    routes[route_idx, :len(nodes)] = torch.as_tensor(nodes, dtype=routes.dtype)


def try_copy_mutation(routes, kind, max_multiplicity, rng, min_len, max_len,
                      demand=None, n_nodes=None, d_un_cap=None, attempts=120,
                      street_adj=None):
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
                                       min_len, max_len, street_adj=street_adj)
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


# --- realistic experiment-init tier ------------------------------------------
# The "covered_dup" tier above over-corrupts a network when used as the
# EXPERIMENT init: 6 events x up to 3 recipients clone half the network,
# prefix/suffix glue creates phantom (non-street) legs, and the d_un cap keeps
# the network unrealistically connected. The realistic tier instead injects
#   * a couple of street-valid (partial) duplicates -- a real-world shared
#     trunk corridor, not wholesale clones,
#   * 1-2 "detour" routes -- a deliberately suboptimal segment the agent
#     should straighten (segment time stretched by min_stretch..max_stretch),
#   * a few dropped low-demand singly-covered stops -- honestly uncovered
#     points (targets d_un around d_un_target_pct).

REALISTIC_TIER_CFG = {
    "dup":        {"events": 1, "max_multiplicity": 2, "kinds": COPY_MUTATION_KINDS,
                   "covered": True, "d_un_cap": D_UN_CAP_PCT},
    "detour":     {"events": 2, "min_stretch": 1.3, "max_stretch": 2.5,
                   "max_redundancy_gain": 0.05},
    "drop_cover": {"events": 3, "d_un_target_pct": 5.0},
}


def try_detour_mutation(routes, rng, street_adj, min_len, max_len,
                        min_stretch=1.3, max_stretch=2.5, attempts=120,
                        demand=None, n_nodes=None, max_redundancy_gain=0.05):
    """Replace a route segment with a longer street path (a "detour").

    Picks two anchor stops >= 2 legs apart and reroutes between them via a
    noise-randomized Dijkstra; accepts when the segment time grows by
    min_stretch..max_stretch and the route stays simple/street-valid. The
    detour models a *suboptimal* route, not coverage loss or duplication:
    when ``demand``/``n_nodes`` are given, candidates that disconnect served
    demand are rejected, and the network redundancy may grow by at most
    ``max_redundancy_gain``. Returns ``(route_idx, new_nodes, old_time,
    new_time)`` or None.
    """
    n_routes = routes.shape[0]
    base_d_un = (uncovered_demand_pct(routes, demand, n_nodes)
                 if demand is not None else None)
    base_redundancy = redundancy_fraction(routes)
    for _ in range(attempts):
        ri = rng.randrange(n_routes)
        ns = route_nodes(routes[ri])
        if len(ns) < 4 or not street_legs_valid(ns, street_adj):
            continue
        i = rng.randrange(0, len(ns) - 2)
        j = rng.randrange(i + 2, len(ns))
        old_seg = ns[i:j + 1]
        old_t = segment_time(old_seg, street_adj)
        if old_t <= 0:
            continue
        # Block a random interior stop of the segment ("closed street") so the
        # path MUST go around -- a plain noised Dijkstra almost always returns
        # the direct segment again, since fewer hops usually beats the noise.
        blocked = rng.choice(old_seg[1:-1])
        forbidden = ((set(ns[:i]) | set(ns[j + 1:])) - {ns[i], ns[j]}) | {blocked}
        path = street_path(street_adj, ns[i], ns[j], rng=rng, noise=1.0,
                           forbidden=forbidden)
        if path is None or path == old_seg:
            continue
        new_t = segment_time(path, street_adj)
        if not (min_stretch * old_t <= new_t <= max_stretch * old_t):
            continue
        candidate = ns[:i] + path + ns[j + 1:]
        if not is_simple_sequence(candidate, min_len, max_len):
            continue
        proposal = routes.clone()
        replace_route(proposal, ri, candidate)
        if base_d_un is not None and \
                uncovered_demand_pct(proposal, demand, n_nodes) > base_d_un + 1e-9:
            continue  # a detour must not uncover demand (drop_cover does that)
        if redundancy_fraction(proposal) > base_redundancy + max_redundancy_gain:
            continue  # nor turn into wholesale corridor duplication
        return ri, candidate, old_t, new_t
    return None


def try_drop_cover_mutation(routes, rng, demand, street_adj, min_len, max_len,
                            attempts=60):
    """Uncover one low-demand stop that is served by exactly one route.

    Endpoint stops are trimmed; interior stops are bypassed when their
    neighbors share a street edge. Returns ``(route_idx, new_nodes, node)``
    or None.
    """
    cover = {}
    for ri in range(routes.shape[0]):
        for node in route_nodes(routes[ri]):
            cover.setdefault(node, []).append(ri)
    D = demand if torch.is_tensor(demand) else torch.as_tensor(demand)
    node_demand = (D.sum(0) + D.sum(1))
    singly = sorted((n for n, c in cover.items() if len(c) == 1),
                    key=lambda n: float(node_demand[n]))
    for node in singly[:attempts]:
        ri = cover[node][0]
        ns = route_nodes(routes[ri])
        pos = ns.index(node)
        if pos in (0, len(ns) - 1):
            candidate = ns[1:] if pos == 0 else ns[:-1]
        else:
            prev_n, next_n = ns[pos - 1], ns[pos + 1]
            if not math.isfinite(float(street_adj[prev_n, next_n])):
                continue
            candidate = ns[:pos] + ns[pos + 1:]
        if not is_simple_sequence(candidate, min_len, max_len):
            continue
        return ri, candidate, node
    return None


def inject_realistic_tier(routes, rng, min_len, max_len, *, street_adj,
                          demand, n_nodes, tier_cfg=None):
    """Apply the realistic corruption tier to a clean network.

    Returns ``(routes, applied)`` where ``applied`` counts the injected
    events per kind (``dup_<kind>`` / ``detour`` / ``drop_cover``).
    """
    cfg = tier_cfg or REALISTIC_TIER_CFG
    routes = routes.clone()
    applied = Counter()

    dup = cfg["dup"]
    d_un_cap = float(dup.get("d_un_cap", 100.0)) if dup.get("covered") else None
    for _ in range(int(dup["events"])):
        kinds = list(dup["kinds"])
        rng.shuffle(kinds)
        for kind in kinds:
            proposal, touched = try_copy_mutation(
                routes, kind, int(dup["max_multiplicity"]), rng, min_len, max_len,
                demand=demand, n_nodes=n_nodes, d_un_cap=d_un_cap,
                street_adj=street_adj)
            if touched:
                routes = proposal
                applied[f"dup_{kind}"] += 1
                break

    det = cfg["detour"]
    for _ in range(int(det["events"])):
        res = try_detour_mutation(routes, rng, street_adj, min_len, max_len,
                                  min_stretch=float(det["min_stretch"]),
                                  max_stretch=float(det["max_stretch"]),
                                  demand=demand, n_nodes=n_nodes,
                                  max_redundancy_gain=float(
                                      det.get("max_redundancy_gain", 0.05)))
        if res is not None:
            ri, candidate, _, _ = res
            replace_route(routes, ri, candidate)
            applied["detour"] += 1

    dc = cfg["drop_cover"]
    for _ in range(int(dc["events"])):
        if uncovered_demand_pct(routes, demand, n_nodes) >= float(dc["d_un_target_pct"]):
            break
        res = try_drop_cover_mutation(routes, rng, demand, street_adj,
                                      min_len, max_len)
        if res is None:
            break
        ri, candidate, _ = res
        replace_route(routes, ri, candidate)
        applied["drop_cover"] += 1

    return routes, applied

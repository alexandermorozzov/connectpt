"""Per-city two-objective Pareto metrics with a fixed minimization reference."""
from collections import defaultdict

import numpy as np


def nondominated(points):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    keep = [not np.any(np.all(points <= p, axis=1) & np.any(points < p, axis=1)) for p in points]
    return points[np.asarray(keep, dtype=bool)]


def sparsity(front):
    """D3PO SP (Def. B.2): mean L2 gap between consecutive points of the front ordered by goal 0."""
    ordered = front[np.argsort(front[:, 0], kind="stable")]
    return float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).mean())


def hypervolume_2d(points, reference):
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all():
        raise ValueError("HV reference must contain two finite normalized costs")
    front = nondominated(points)
    front = front[np.all(front < reference, axis=1)]
    front = front[np.argsort(front[:, 0])]
    volume, y = 0.0, float(reference[1])
    for x, new_y in front:
        volume += (float(reference[0]) - x) * max(y - new_y, 0.0)
        y = min(y, new_y)
    return float(volume)


def canonical_network(routes, symmetric_routes=True):
    """Route order and bidirectional reversal do not create a new network.

    Preserve duplicate routes: their multiplicity affects RTT.
    """
    canonical = []
    for route in routes:
        route = tuple(int(node) for node in route if int(node) >= 0)
        if route:
            canonical.append(min(route, route[::-1]) if symmetric_routes else route)
    return tuple(sorted(canonical))


def front_metrics(rows, reference, eu_preference_count):
    """One city and one independent preference-grid draw; never pool cities."""
    if not rows:
        raise ValueError("Cannot evaluate an empty preference grid")
    if eu_preference_count < 2:
        raise ValueError("EU needs at least two query preferences")
    points = np.array([[r["penalized_0"], r["penalized_1"]] for r in rows])
    pure = np.array([[r["objective_0"], r["objective_1"]] for r in rows])
    mask = np.array([r["feasible"] for r in rows], dtype=bool)
    feasible = pure[mask]
    alphas = np.linspace(0, 1, eu_preference_count)
    weights = np.stack((alphas, 1 - alphas), -1)
    metrics = dict(
        hv_penalized=hypervolume_2d(points, reference),
        hv_feasible=hypervolume_2d(feasible, reference),
        eu_conditioned_penalized=float(
            np.mean([-r["preference"] * r["penalized_0"] - (1 - r["preference"]) * r["penalized_1"] for r in rows])
        ),
        eu_archive_penalized=float((-np.min(weights @ points.T, axis=1)).mean()),
        eu_archive_feasible=float((-np.min(weights @ feasible.T, axis=1)).mean()) if len(feasible) else None,
        pareto_points_penalized=len(nondominated(points)),
        pareto_points_feasible=len(nondominated(feasible)),
        feasible_fraction=float(mask.mean()),
        mean_unserved_fraction=float(np.mean([r["unserved_fraction"] for r in rows])),
        hv_reference_coverage=float(np.all(pure < reference, axis=1)[mask].mean()) if mask.any() else None,
        ATT=float(np.mean([r["ATT"] for r in rows])) if "ATT" in rows[0] else None,
        RTT=float(np.mean([r["RTT"] for r in rows])) if "RTT" in rows[0] else None,
    )
    if all("network_signature" in r for r in rows):
        signatures = [tuple(tuple(route) for route in r["network_signature"]) for r in rows]
        metrics["distinct_networks"] = len(set(signatures))
        metrics["distinct_feasible_networks"] = len({s for s, ok in zip(signatures, mask) if ok})
        pareto = nondominated(feasible)
        metrics["distinct_pareto_networks"] = len(
            {s for s, p, ok in zip(signatures, pure, mask) if ok and np.any(np.all(p == pareto, axis=1))}
        )
    if len(feasible) > 1:
        front = nondominated(feasible)
        if len(front) > 1:
            distance = np.abs(front[:, None] - front[None]).sum(-1)
            np.fill_diagonal(distance, np.inf)
            metrics["spacing_feasible"] = float(np.std(distance.min(1), ddof=1))
            metrics["sparsity_feasible"] = sparsity(front)
        else:
            metrics["spacing_feasible"] = None
            metrics["sparsity_feasible"] = None
    else:
        metrics["spacing_feasible"] = None
        metrics["sparsity_feasible"] = None
    return metrics


def summarize_evaluation(rows, reference, eu_preference_count):
    groups = defaultdict(list)
    for row in rows:
        groups[row["graph"], row.get("sample", 0)].append(row)
    per_graph = [
        dict(graph=g, sample=s, metrics=front_metrics(rs, reference, eu_preference_count))
        for (g, s), rs in sorted(groups.items())
    ]
    if not per_graph:
        raise ValueError("Cannot summarize empty evaluation")
    fields = set().union(*(g["metrics"] for g in per_graph))
    summary = {}
    for key in sorted(fields):
        values = [g["metrics"].get(key) for g in per_graph]
        values = [v for v in values if v is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["graph_count"] = len({g["graph"] for g in per_graph})
    summary["outcome_count"] = len(rows)
    if all("network_signature" in r for r in rows):
        unique = defaultdict(set)
        unique_feasible = defaultdict(set)
        for row in rows:
            signature = tuple(tuple(route) for route in row["network_signature"])
            unique[row["graph"]].add(signature)
            if row["feasible"]:
                unique_feasible[row["graph"]].add(signature)
        summary["distinct_networks_total"] = sum(map(len, unique.values()))
        summary["distinct_feasible_networks_total"] = sum(map(len, unique_feasible.values()))
    return dict(
        summary=summary,
        per_graph=per_graph,
        reference=list(reference),
        direction="minimize configured objectives; maximize HV and EU (negative cost)",
        stochastic_rule="Independent fronts per city/draw, then mean; no pooling/best-of-N",
        network_rule="Route permutation and symmetric reversal ignored; duplicate routes retained",
    )

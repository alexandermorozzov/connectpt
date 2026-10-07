"""BeeSpec -- the declarative description of one bee population.

A BeeSpec is pure config: it names the operator, the policy it uses, the actions
it may take, the route selector and the acceptance rule. The plan
(``executable_plan``) turns specs into operators; this module is just the
parsed schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BeeSpec:
    name: str
    count: int
    operator: str
    model_ref: str | None = None
    allowed_actions: list[str] | None = None
    route_selection: str = "uniform"
    acceptance: str = "greedy"
    steps: list[dict[str, Any]] | None = None
    # heuristic_mutation kind (e.g. "classic_bco", "shorten"); ignored otherwise.
    mutation_kind: str | None = None


def parse_bee_specs(bees_cfg) -> list[BeeSpec]:
    """Parse the named ``cfg.bees`` mapping into :class:`BeeSpec` objects."""
    specs: list[BeeSpec] = []
    for name, raw in dict(bees_cfg).items():
        d = dict(raw)
        specs.append(BeeSpec(
            name=name,
            count=int(d["count"]),
            operator=d["operator"],
            model_ref=d.get("model_ref"),
            allowed_actions=list(d["allowed_actions"]) if d.get("allowed_actions") else None,
            route_selection=d.get("route_selection", "uniform"),
            acceptance=d.get("acceptance", "greedy"),
            steps=[dict(s) for s in d["steps"]] if d.get("steps") else None,
            mutation_kind=d.get("mutation_kind"),
        ))
    return specs

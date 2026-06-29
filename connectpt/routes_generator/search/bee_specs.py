"""BeeSpec -- the declarative description of one bee population.

A BeeSpec is pure config: it names the operator, the policy it uses, the actions
it may take, the route selector and the acceptance rule. Building a concrete bee
(and validating the action set against the policy's model) happens in
``bee_operators``; this module is just the parsed schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BeeSpec:
    name: str
    count: int
    operator: str
    policy: str | None = None
    allowed_actions: list[str] | None = None
    route_selection: str = "uniform"
    acceptance: str = "greedy"
    steps: list[dict[str, Any]] | None = None


def parse_bee_specs(bees_cfg) -> list[BeeSpec]:
    """Parse a list of bee config nodes (from cfg.bees) into BeeSpec objects."""
    specs: list[BeeSpec] = []
    for raw in bees_cfg:
        d = dict(raw)
        specs.append(BeeSpec(
            name=d["name"],
            count=int(d["count"]),
            operator=d["operator"],
            policy=d.get("policy"),
            allowed_actions=list(d["allowed_actions"]) if d.get("allowed_actions") else None,
            route_selection=d.get("route_selection", "uniform"),
            acceptance=d.get("acceptance", "greedy"),
            steps=[dict(s) for s in d["steps"]] if d.get("steps") else None,
        ))
    return specs

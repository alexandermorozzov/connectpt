"""Translate flexible BeeSpecs into the existing bee_colony dispatch.

The current ``bee_colony`` picks bees by per-type counts (n_type1..n_type7) and
fixed model roles. Rather than rewrite that algorithm now, this plan maps the
declarative BeeSpecs onto those counts + the models bee_colony needs, so the new
config drives the old executor unchanged. The bee_colony type meanings:

    type1  random path combiner (heuristic)        -- no model
    type2  shorten mutation (heuristic)            -- no model
    type4  neural construction extend              -- construction model
    type5  neural edit (extend [+ trim] + halt)    -- edit model
    type6  neural trim-only                        -- edit model
    type7  compound trim-then-extend               -- edit (+ construction) model
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bee_operators import build_bee
from .bee_specs import BeeSpec


@dataclass
class BeeColonyPlan:
    counts: dict[str, int] = field(default_factory=dict)
    needs_construction: bool = False
    needs_edit: bool = False
    bees: list = field(default_factory=list)

    @classmethod
    def from_specs(cls, specs: list[BeeSpec], policies: dict) -> "BeeColonyPlan":
        counts = {f"n_type{i}": 0 for i in range(1, 8)}
        needs_construction = needs_edit = False
        bees = []

        for spec in specs:
            # building the bee validates allowed_actions against the policy model
            bees.append(build_bee(spec, policies))
            type_key = cls._classify(spec, policies)
            counts[type_key] += int(spec.count)

            if spec.operator == "neural_rebuild":
                # full-route neural rebuild (type-1) drives the construction model
                needs_construction = True
            elif type_key == "n_type4":
                needs_construction = True
            elif type_key in ("n_type5", "n_type6"):
                needs_edit = True
            elif type_key == "n_type7":
                needs_edit = True
                # compound may also extend via the construction policy
                for step in spec.steps or []:
                    if policies[step["policy"]].role == "construction":
                        needs_construction = True

        return cls(counts=counts, needs_construction=needs_construction,
                   needs_edit=needs_edit, bees=bees)

    @staticmethod
    def _classify(spec: BeeSpec, policies: dict) -> str:
        if spec.operator == "compound":
            return "n_type7"
        if spec.operator == "neural_rebuild":
            # full-route rebuild via the construction model (bee_colony type-1)
            return "n_type1"
        if spec.operator == "heuristic_mutation":
            kind = (spec.mutation_kind or spec.route_selection or "").lower()
            return "n_type2" if "shorten" in kind else "n_type1"
        if spec.operator == "neural_route_action":
            role = policies[spec.policy].role
            if role == "construction":
                return "n_type4"
            actions = set(spec.allowed_actions or [])
            if actions and actions <= {"trim_start", "trim_end"}:
                return "n_type6"
            return "n_type5"
        raise ValueError(f"cannot classify bee operator {spec.operator!r}")

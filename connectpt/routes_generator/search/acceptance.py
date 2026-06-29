"""Acceptance policies -- how a proposed mutation is accepted.

Named strategy objects matching the search config vocabulary (``acceptance``).
The trim-grace variant mirrors the existing bee_colony behaviour of shielding a
trim setup move from immediate cost-based rejection for a few iterations.
"""
from __future__ import annotations


class AcceptancePolicy:
    name = "base"


class GreedyAcceptance(AcceptancePolicy):
    """Accept only strictly improving mutations."""

    name = "greedy"


class TrimGraceAcceptance(AcceptancePolicy):
    """Temporarily accept non-improving trim setup moves (grace period)."""

    name = "trim_grace"


_ACCEPTANCE = {c.name: c for c in (GreedyAcceptance, TrimGraceAcceptance)}


def get_acceptance(name: str) -> AcceptancePolicy:
    try:
        return _ACCEPTANCE[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown acceptance {name!r}; known: {sorted(_ACCEPTANCE)}"
        ) from exc

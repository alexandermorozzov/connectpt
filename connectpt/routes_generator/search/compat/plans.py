"""Legacy taxonomy compatibility surface (``n_type1..n_type7``).

Everything here maps the historical per-type bee counts onto an
:class:`~connectpt.routes_generator.search.executable_plan.ExecutablePlan`.
It is isolated in this package so the engine and the declarative bee-set path
(``cfg/search/bee_sets`` + :meth:`ExecutablePlan.from_specs`) carry no
``n_type`` taxonomy of their own.

* :func:`plan_from_counts` -- build a plan from explicit ``n_type*`` counts +
  model roles (the old ``bee_colony`` defaulting, bit-for-bit).
* :func:`plan_from_flat_cfg` -- adapt the flat ``n_type*_bees`` config format
  (the captured experiment presets / ``compat.bco_config`` factory).
* :func:`bee_colony` -- the old-signature engine entry the frozen notebooks
  (evaluation.ipynb / experiment.ipynb) and legacy callers use.

New bee sets are declarative specs; nothing new should reach for this module.
"""
from __future__ import annotations

from ..executable_plan import (
    ConstructionExtendOp,
    EditOp,
    ExecutablePlan,
    HeuristicRebuildOp,
    NeuralRebuildOp,
    PlanGroup,
    ShortenOp,
    TrimOp,
    TrimThenExtendOp,
)


def plan_from_counts(*, n_bees, n_type1=None, n_type2=None, n_type4=0,
                     n_type5=0, n_type6=0, n_type7=0, bee_model=None,
                     edit_model=None, type4_allow_halt=True,
                     type5_allow_halt=True, type6_allow_halt=True,
                     type7_allow_halt=True, ignore_type4_max_route_len=False,
                     ignore_type5_max_route_len=False,
                     ignore_type6_max_route_len=False,
                     ignore_type7_max_route_len=False):
    """Build a plan from the legacy per-type bee counts + model roles.

    Reproduces the old ``bee_colony`` defaulting exactly: ``n_type1`` omitted ->
    half the bees; ``n_type2`` omitted -> the remainder after the explicit
    counts; any bees left over become random-path-combiner rebuilds (the legacy
    type-3 remainder); a trim-capable construction model doubles as the edit
    model when no explicit edit model is given.
    """
    n_bees = int(n_bees)
    n_type4 = int(n_type4)
    n_type5 = int(n_type5)
    n_type6 = int(n_type6)
    n_type7 = int(n_type7)
    if n_type1 is None:
        n_type1 = n_bees // 2
    n_type1 = int(n_type1)
    if n_type2 is None:
        n_type2 = (n_bees - n_type1 - n_type4 - n_type5 - n_type6
                   - n_type7)
    n_type2 = int(n_type2)
    n_type3 = (n_bees - n_type1 - n_type2 - n_type4 - n_type5 - n_type6
               - n_type7)
    if n_type3 < 0:
        raise ValueError(
            "Sum of n_type1/2/4/5/6/7 bees exceeds n_bees: "
            f"{n_type1}+{n_type2}+{n_type4}+{n_type5}+{n_type6}+{n_type7}"
            f" > {n_bees}")

    # legacy aliasing: a trim-capable construction model doubles as the
    # edit model when no explicit edit model is given.
    if edit_model is None and getattr(bee_model, "supports_trim_actions",
                                      False):
        edit_model = bee_model

    rebuild_op = (NeuralRebuildOp(bee_model, name="type1")
                  if bee_model is not None else
                  HeuristicRebuildOp(name="type1"))
    extend_model = bee_model if bee_model is not None else edit_model
    groups = [
        PlanGroup(rebuild_op, n_type1),
        PlanGroup(ShortenOp(name="type2"), n_type2),
        PlanGroup(NeuralRebuildOp(None, name="type3", lazy_rpc=True),
                  n_type3),
        PlanGroup(ConstructionExtendOp(
            bee_model, name="type4", allow_halt=bool(type4_allow_halt),
            ignore_max_route_len=bool(ignore_type4_max_route_len),
        ), n_type4),
        PlanGroup(EditOp(
            edit_model, name="type5", allow_halt=bool(type5_allow_halt),
            ignore_max_route_len=bool(ignore_type5_max_route_len),
        ), n_type5),
        PlanGroup(TrimOp(
            edit_model, name="type6", allow_halt=bool(type6_allow_halt),
            ignore_max_route_len=bool(ignore_type6_max_route_len),
        ), n_type6),
        PlanGroup(TrimThenExtendOp(
            edit_model, extend_model, name="type7",
            allow_halt=bool(type7_allow_halt),
            ignore_max_route_len=bool(ignore_type7_max_route_len),
        ), n_type7),
    ]
    return ExecutablePlan(groups, construction_model=bee_model,
                          edit_model=edit_model)

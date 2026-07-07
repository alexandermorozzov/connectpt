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


def plan_from_flat_cfg(cfg, *, bee_model=None, edit_model=None):
    """Adapt the flat ``n_type*_bees`` config format (captured presets).

    Thin wrapper over :func:`plan_from_counts` that reads the flat cfg keys with
    their historical defaults. The flat format is a compatibility surface -- new
    bee sets are written as declarative specs (:meth:`ExecutablePlan.from_specs`).
    """
    return plan_from_counts(
        n_bees=cfg.n_bees,
        n_type1=cfg.get("n_type1_bees", None),
        n_type2=cfg.get("n_type2_bees", None),
        n_type4=cfg.get("n_type4_bees", 0),
        n_type5=cfg.get("n_type5_bees", 0),
        n_type6=cfg.get("n_type6_bees", 0),
        n_type7=cfg.get("n_type7_bees", 0),
        bee_model=bee_model, edit_model=edit_model,
        type4_allow_halt=cfg.get("type4_allow_halt", True),
        type5_allow_halt=cfg.get("type5_allow_halt", True),
        type6_allow_halt=cfg.get("type6_allow_halt", True),
        type7_allow_halt=cfg.get("type7_allow_halt", True),
        ignore_type4_max_route_len=cfg.get(
            "ignore_type4_max_route_len", False),
        ignore_type5_max_route_len=cfg.get(
            "ignore_type5_max_route_len", False),
        ignore_type6_max_route_len=cfg.get(
            "ignore_type6_max_route_len", False),
        ignore_type7_max_route_len=cfg.get(
            "ignore_type7_max_route_len", False),
    )


def bee_colony(state, cost_obj, init_network, n_bees=10, passes_per_it=5,
               mod_steps_per_pass=2, shorten_prob=0.2, n_iterations=400,
               n_type1_bees=None, n_type2_bees=None, n_type4_bees=0,
               n_type5_bees=0, n_type6_bees=0, n_type7_bees=0,
               silent=False, iteration_callback=None,
               force_linking_unlinked=False,
               bee_model=None, edit_model=None,
               sum_writer=None, mutation_counts_out=None,
               adjustment_degree_weight=0.0,
               adjustment_degree_gap=0.1,
               adjustment_degree_mode='current',
               adjustment_degree_objective='raw',
               adjustment_degree_target=0.2,
               ignore_type4_max_route_len=False,
               ignore_type5_max_route_len=False,
               ignore_type6_max_route_len=False,
               ignore_type7_max_route_len=False,
               type4_allow_halt=True,
               type5_allow_halt=True,
               type6_allow_halt=True,
               type7_allow_halt=True,
               use_demand_weighted_route_selection=False,
               worse_accept_temperature=0.0,
               worse_accept_decay=0.995,
               worse_accept_min_temperature=0.001,
               worse_selection_temperature=0.0,
               worse_selection_decay=0.995,
               worse_selection_min_temperature=0.001,
               worse_selection_uniform_mix=0.05,
               worse_selection_elite_count=1,
               trim_grace_period=0,
               process_neural_bees_sequentially=False,
               early_stop_patience=None,
               early_stop_min_delta=0.0):
    """Backward-compatible bee_colony entry using the legacy per-type counts.

    The engine (:func:`~connectpt.routes_generator.bee_colony.run_bee_colony_plan`)
    is driven by an ``ExecutablePlan``; this wrapper is the compatibility surface
    that maps the historical ``n_type1..n_type7`` bee counts +
    ``bee_model``/``edit_model`` roles + per-type halt/max-len flags onto a plan,
    then runs it. It preserves the exact signature the frozen notebooks
    (evaluation.ipynb / experiment.ipynb) and legacy callers use, so their
    behaviour is byte-identical.
    """
    from ...bee_colony import run_bee_colony_plan

    plan = plan_from_counts(
        n_bees=n_bees, n_type1=n_type1_bees, n_type2=n_type2_bees,
        n_type4=n_type4_bees, n_type5=n_type5_bees, n_type6=n_type6_bees,
        n_type7=n_type7_bees, bee_model=bee_model, edit_model=edit_model,
        type4_allow_halt=type4_allow_halt, type5_allow_halt=type5_allow_halt,
        type6_allow_halt=type6_allow_halt, type7_allow_halt=type7_allow_halt,
        ignore_type4_max_route_len=ignore_type4_max_route_len,
        ignore_type5_max_route_len=ignore_type5_max_route_len,
        ignore_type6_max_route_len=ignore_type6_max_route_len,
        ignore_type7_max_route_len=ignore_type7_max_route_len)
    return run_bee_colony_plan(
        state, cost_obj, init_network, n_bees=n_bees,
        passes_per_it=passes_per_it, mod_steps_per_pass=mod_steps_per_pass,
        shorten_prob=shorten_prob, n_iterations=n_iterations, plan=plan,
        silent=silent, iteration_callback=iteration_callback,
        force_linking_unlinked=force_linking_unlinked,
        sum_writer=sum_writer, mutation_counts_out=mutation_counts_out,
        adjustment_degree_weight=adjustment_degree_weight,
        adjustment_degree_gap=adjustment_degree_gap,
        adjustment_degree_mode=adjustment_degree_mode,
        adjustment_degree_objective=adjustment_degree_objective,
        adjustment_degree_target=adjustment_degree_target,
        use_demand_weighted_route_selection=use_demand_weighted_route_selection,
        worse_accept_temperature=worse_accept_temperature,
        worse_accept_decay=worse_accept_decay,
        worse_accept_min_temperature=worse_accept_min_temperature,
        worse_selection_temperature=worse_selection_temperature,
        worse_selection_decay=worse_selection_decay,
        worse_selection_min_temperature=worse_selection_min_temperature,
        worse_selection_uniform_mix=worse_selection_uniform_mix,
        worse_selection_elite_count=worse_selection_elite_count,
        trim_grace_period=trim_grace_period,
        process_neural_bees_sequentially=process_neural_bees_sequentially,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta)

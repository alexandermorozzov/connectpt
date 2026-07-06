"""Executable bee-colony plan: ordered mutation-operator groups.

The engine (``bee_colony``) iterates these groups instead of branching on
``n_type1..n_type7``. Bit-for-bit parity with the legacy typed dispatch is
preserved by construction:

* groups execute in the canonical (legacy type) order;
* the slot permutation is sliced by group counts exactly like the old
  ``type1_idxs..type7_idxs`` slices;
* the neural operators keep the legacy "run the model over ALL bees, then
  gather the group's slots" pattern (RNG-relevant);
* the random-path-combiner model is created lazily at the same point of the
  engine (``materialize()``) where the legacy code built it.

Builders: :meth:`ExecutablePlan.from_flat_cfg` adapts the flat
``n_type*_bees`` config format (the captured experiment presets);
:meth:`ExecutablePlan.from_specs` builds a plan from declarative BeeSpecs
(``cfg/search/bee_sets``). The flat format is a compatibility surface -- new
bee sets are written as specs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import bee_colony as _bc
from .. import utils as _lrnu


# ---------------------------------------------------------------------------
# context handed to every operator at each mutation step


@dataclass
class MutationContext:
    """Everything a mutation operator may need at one mod step."""

    bee_networks: Any          # (batch, n_bees, n_routes, max_n_nodes)
    chosen_route_idxs: Any     # (batch, n_bees)
    modified_routes: Any       # (batch, n_bees, max_n_nodes), empties refilled
    env_state: Any             # bee-expanded RouteGenBatchState
    seq_state: Any             # single-bee state (sequential mode) or env_state
    remaining_state: Any       # state without the chosen routes, or None
    process_sequentially: bool
    direct_sat_dmd: Any
    shorten_prob: float
    street_node_neighbours: Any
    shortest_paths: Any
    force_linking_unlinked: bool
    adj_condition_target: Any
    adj_condition_weight: Any
    max_n_nodes: int


def _gather_group_routes(all_networks, ctx: MutationContext, idxs):
    """Legacy per-type gather: pick the group's slots' chosen routes."""
    gather = ctx.chosen_route_idxs[:, idxs, None, None].expand(
        -1, -1, -1, ctx.max_n_nodes)
    return all_networks[:, idxs].gather(2, gather).squeeze(2)


# ---------------------------------------------------------------------------
# operators (one per legacy mutation type)


class HeuristicRebuildOp:
    """Legacy type-1 heuristic: rebuild the chosen route from shortest paths."""

    kind = "heuristic_rebuild"
    uses_remaining_state = True

    def __init__(self, name="type1"):
        self.name = name

    def mutate(self, ctx: MutationContext, idxs):
        variants = _bc.get_bee_1_variants(
            ctx.remaining_state, ctx.modified_routes, ctx.direct_sat_dmd,
            ctx.shortest_paths, ctx.force_linking_unlinked)
        return variants[:, idxs]


class NeuralRebuildOp:
    """Legacy type-1 neural / type-3 RPC: full-route rebuild via a model.

    ``model=None`` marks a lazily-materialized random path combiner (the
    legacy engine built it at run start; see :meth:`ExecutablePlan.materialize`).
    """

    kind = "neural_rebuild"
    uses_remaining_state = False

    def __init__(self, model, name="type1", lazy_rpc=False):
        self.model = model
        self.name = name
        self.lazy_rpc = lazy_rpc

    def mutate(self, ctx: MutationContext, idxs):
        if ctx.process_sequentially:
            return _bc._run_rebuild_variants_for_bees(
                _bc.get_neural_variants, self.model, ctx.seq_state,
                ctx.bee_networks, ctx.chosen_route_idxs, idxs)
        networks = _bc.get_neural_variants(
            self.model, ctx.env_state, ctx.bee_networks, ctx.chosen_route_idxs)
        return networks[:, idxs, -1]


class ShortenOp:
    """Legacy type-2: heuristic shorten/lengthen of the chosen route."""

    kind = "shorten"
    uses_remaining_state = False

    def __init__(self, name="type2"):
        self.name = name

    def mutate(self, ctx: MutationContext, idxs):
        new_routes = _bc.get_bee_2_variants(
            ctx.modified_routes[:, idxs], ctx.shorten_prob,
            ctx.street_node_neighbours)
        assert ((new_routes > -1).sum(dim=-1) > 0).all()
        return new_routes


class _OneStepNeuralOp:
    """Shared plumbing of the one-step (type-4/5/6) neural operators."""

    uses_remaining_state = False
    variant_fn_name = ""            # overridden per subclass
    passes_adj_condition = False

    def __init__(self, model, *, name, allow_halt=True,
                 ignore_max_route_len=False):
        self.model = model
        self.name = name
        self.allow_halt = bool(allow_halt)
        self.ignore_max_route_len = bool(ignore_max_route_len)

    def _variant_kwargs(self, ctx: MutationContext) -> dict:
        kwargs = dict(ignore_max_route_len=self.ignore_max_route_len,
                      allow_halt=self.allow_halt)
        if self.passes_adj_condition:
            kwargs.update(adj_condition_target=ctx.adj_condition_target,
                          adj_condition_weight=ctx.adj_condition_weight)
        return kwargs

    def mutate(self, ctx: MutationContext, idxs):
        variant_fn = getattr(_bc, self.variant_fn_name)
        if ctx.process_sequentially:
            return _bc._run_selected_route_variants_for_bees(
                variant_fn, self.model, ctx.seq_state,
                ctx.bee_networks, ctx.chosen_route_idxs, idxs,
                **self._variant_kwargs(ctx))
        all_networks = variant_fn(
            self.model, ctx.env_state, ctx.bee_networks, ctx.chosen_route_idxs,
            **self._variant_kwargs(ctx))
        return _gather_group_routes(all_networks, ctx, idxs)


class ConstructionExtendOp(_OneStepNeuralOp):
    """Legacy type-4: one construction extend/halt step on the chosen route."""

    kind = "construction_extend"
    variant_fn_name = "get_neural_extend_variants"


class EditOp(_OneStepNeuralOp):
    """Legacy type-5: one extend/trim/halt edit step (trim-capable model)."""

    kind = "edit"
    variant_fn_name = "get_neural_edit_variants"
    passes_adj_condition = True


class TrimOp(_OneStepNeuralOp):
    """Legacy type-6: one trim/halt step (extend masked out)."""

    kind = "trim"
    variant_fn_name = "get_neural_trim_variants"


class TrimThenExtendOp:
    """Legacy type-7: a trim step then a construction-style extend step."""

    kind = "trim_then_extend"
    uses_remaining_state = False

    def __init__(self, trim_model, extend_model, *, name="type7",
                 allow_halt=True, ignore_max_route_len=False):
        self.trim_model = trim_model
        self.extend_model = extend_model
        self.name = name
        self.allow_halt = bool(allow_halt)
        self.ignore_max_route_len = bool(ignore_max_route_len)

    def mutate(self, ctx: MutationContext, idxs):
        if ctx.process_sequentially:
            return _bc._run_selected_route_variants_for_bees(
                _bc.get_neural_trim_then_extend_variants,
                self.trim_model, self.extend_model, ctx.seq_state,
                ctx.bee_networks, ctx.chosen_route_idxs, idxs,
                ignore_max_route_len=self.ignore_max_route_len,
                allow_halt=self.allow_halt)
        all_networks = _bc.get_neural_trim_then_extend_variants(
            self.trim_model, self.extend_model, ctx.env_state,
            ctx.bee_networks, ctx.chosen_route_idxs,
            ignore_max_route_len=self.ignore_max_route_len,
            allow_halt=self.allow_halt)
        return _gather_group_routes(all_networks, ctx, idxs)


# ---------------------------------------------------------------------------
# the plan


@dataclass
class PlanGroup:
    op: Any
    count: int


class ExecutablePlan:
    """Ordered mutation-operator groups + the models the engine must know."""

    def __init__(self, groups: list[PlanGroup], *, construction_model=None,
                 edit_model=None):
        self.groups = list(groups)
        # the construction ("bee") model: drives setup_planning on the bee
        # states, neural rebuilds, construction-extend steps and the extend
        # half of compound bees. May be None (heuristic-only plans).
        self.construction_model = construction_model
        self.edit_model = edit_model

    # -- properties the engine reads ----------------------------------------

    @property
    def total_bees(self) -> int:
        return sum(g.count for g in self.groups)

    @property
    def planning_model(self):
        """Model whose ``setup_planning`` prepares the bee states (legacy
        semantics: the construction model when present)."""
        return self.construction_model

    def needs_remaining_state(self) -> bool:
        """Legacy condition: heuristic rebuild present, or no construction
        model at all (the old ``bee_model is None`` branch)."""
        if self.construction_model is None:
            return True
        return any(g.op.uses_remaining_state and g.count > 0
                   for g in self.groups)

    def group_names(self) -> list[str]:
        return [g.op.name for g in self.groups]

    def summary(self) -> dict:
        return {g.op.name: int(g.count) for g in self.groups}

    # -- lifecycle -----------------------------------------------------------

    def materialize(self):
        """Create lazily-built models (the RPC combiner) at engine start --
        the exact point the legacy engine built them, so RNG order and model
        state match bit-for-bit."""
        for g in self.groups:
            if isinstance(g.op, NeuralRebuildOp) and g.op.lazy_rpc \
                    and g.op.model is None and g.count > 0:
                g.op.model = _lrnu.get_random_path_combiner()
        return self

    def validate(self, n_bees: int) -> None:
        total = self.total_bees
        if total != int(n_bees):
            raise ValueError(
                f"plan bee counts sum to {total}, but n_bees={n_bees}")
        trim_count = sum(g.count for g in self.groups
                         if g.op.kind in ("edit", "trim", "trim_then_extend"))
        if trim_count > 0:
            if self.edit_model is None:
                raise ValueError(
                    "edit/trim/compound bees require an edit_model that "
                    "supports trim actions")
            if not getattr(self.edit_model, "supports_trim_actions", False):
                raise ValueError(
                    "edit_model must have supports_trim_actions=True to "
                    "drive edit/trim/compound mutations")

    # -- builders ------------------------------------------------------------

    @classmethod
    def from_flat_cfg(cls, cfg, *, bee_model=None, edit_model=None):
        """Adapt the flat ``n_type*_bees`` config format (captured presets).

        Reproduces the legacy engine's defaulting exactly: ``n_type1_bees``
        omitted -> half the bees; ``n_type2_bees`` omitted -> the remainder
        after the explicit counts; any bees left over become random-path-
        combiner rebuilds (the legacy type-3 remainder). This builder is the
        flat-format compatibility boundary -- new bee sets are declarative
        specs (``from_specs``).
        """
        n_bees = int(cfg.n_bees)
        n_type4 = int(cfg.get("n_type4_bees", 0))
        n_type5 = int(cfg.get("n_type5_bees", 0))
        n_type6 = int(cfg.get("n_type6_bees", 0))
        n_type7 = int(cfg.get("n_type7_bees", 0))
        n_type1 = cfg.get("n_type1_bees", None)
        if n_type1 is None:
            n_type1 = n_bees // 2
        n_type1 = int(n_type1)
        n_type2 = cfg.get("n_type2_bees", None)
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

        def flag(key, default):
            return bool(cfg.get(key, default))

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
                bee_model, name="type4",
                allow_halt=flag("type4_allow_halt", True),
                ignore_max_route_len=flag("ignore_type4_max_route_len", False),
            ), n_type4),
            PlanGroup(EditOp(
                edit_model, name="type5",
                allow_halt=flag("type5_allow_halt", True),
                ignore_max_route_len=flag("ignore_type5_max_route_len", False),
            ), n_type5),
            PlanGroup(TrimOp(
                edit_model, name="type6",
                allow_halt=flag("type6_allow_halt", True),
                ignore_max_route_len=flag("ignore_type6_max_route_len", False),
            ), n_type6),
            PlanGroup(TrimThenExtendOp(
                edit_model, extend_model, name="type7",
                allow_halt=flag("type7_allow_halt", True),
                ignore_max_route_len=flag("ignore_type7_max_route_len", False),
            ), n_type7),
        ]
        return cls(groups, construction_model=bee_model,
                   edit_model=edit_model)

    @classmethod
    def from_specs(cls, specs, policies: dict, *, models: dict | None = None,
                   algo_cfg=None):
        """Build a plan from declarative BeeSpecs (``cfg/search/bee_sets``).

        Specs mapping to the same canonical operator slot are MERGED into one
        group (the legacy engine executes each slot as one vectorized block;
        splitting them would change the RNG stream). A merged group's stats
        name joins the spec names with '+'. Per-slot halt permission is the
        permissive OR of the merged specs' ``allowed_actions`` (matching the
        old BeeColonyPlan bridge).
        """
        models = models or {}
        construction_model = models.get("construction")
        edit_model = models.get("edit")
        extend_model = (construction_model if construction_model is not None
                        else edit_model)
        ignore = {}
        if algo_cfg is not None:
            for i in (4, 5, 6, 7):
                ignore[i] = bool(
                    getattr(algo_cfg, f"ignore_type{i}_max_route_len"))

        slots: dict[int, dict] = {}

        def add(slot: int, spec, halt: bool | None):
            entry = slots.setdefault(slot, {"count": 0, "names": [],
                                            "halt": None})
            entry["count"] += int(spec.count)
            entry["names"].append(spec.name)
            if halt is not None:
                entry["halt"] = halt if entry["halt"] is None \
                    else (entry["halt"] or halt)

        for spec in specs:
            slot = _canonical_slot(spec, policies)
            # fail fast: the policy's model must support every requested action
            if spec.operator == "neural_route_action":
                policies[spec.policy].validate_actions(spec.allowed_actions or [])
            elif spec.operator == "compound":
                for step in spec.steps or []:
                    policies[step["policy"]].validate_actions(
                        step.get("allowed_actions", []))
            halt = None
            if 4 <= slot <= 7:
                halt = "halt" in set(spec.allowed_actions or [])
            add(slot, spec, halt)

        def name_of(slot, default):
            return "+".join(slots[slot]["names"]) if slot in slots else default

        def count_of(slot):
            return slots[slot]["count"] if slot in slots else 0

        def halt_of(slot):
            h = slots.get(slot, {}).get("halt")
            return True if h is None else h

        rebuild_uses_model = any(
            s.operator == "neural_rebuild"
            and _canonical_slot(s, policies) == 1 for s in specs)
        rebuild_op = (NeuralRebuildOp(construction_model, name=name_of(1, "type1"))
                      if rebuild_uses_model else
                      HeuristicRebuildOp(name=name_of(1, "type1")))
        groups = [
            PlanGroup(rebuild_op, count_of(1)),
            PlanGroup(ShortenOp(name=name_of(2, "type2")), count_of(2)),
            PlanGroup(NeuralRebuildOp(None, name=name_of(3, "type3"),
                                      lazy_rpc=True), count_of(3)),
            PlanGroup(ConstructionExtendOp(
                construction_model, name=name_of(4, "type4"),
                allow_halt=halt_of(4),
                ignore_max_route_len=ignore.get(4, False)), count_of(4)),
            PlanGroup(EditOp(
                edit_model, name=name_of(5, "type5"), allow_halt=halt_of(5),
                ignore_max_route_len=ignore.get(5, False)), count_of(5)),
            PlanGroup(TrimOp(
                edit_model, name=name_of(6, "type6"), allow_halt=halt_of(6),
                ignore_max_route_len=ignore.get(6, False)), count_of(6)),
            PlanGroup(TrimThenExtendOp(
                edit_model, extend_model, name=name_of(7, "type7"),
                allow_halt=halt_of(7),
                ignore_max_route_len=ignore.get(7, False)), count_of(7)),
        ]
        return cls(groups, construction_model=construction_model,
                   edit_model=edit_model)


def _canonical_slot(spec, policies: dict) -> int:
    """Map a BeeSpec to its canonical execution slot (legacy type number)."""
    if spec.operator == "compound":
        return 7
    if spec.operator == "neural_rebuild":
        return 1
    if spec.operator == "heuristic_mutation":
        kind = (spec.mutation_kind or spec.route_selection or "").lower()
        if "shorten" in kind:
            return 2
        if "path_mix" in kind or "path_combin" in kind or "rpc" in kind:
            return 3
        return 1
    if spec.operator == "neural_route_action":
        if policies[spec.policy].role == "construction":
            return 4
        # edit role: the 5/6 split is EXTEND presence, not halt.
        actions = set(spec.allowed_actions or [])
        return 5 if "extend" in actions else 6
    raise ValueError(f"cannot classify bee operator {spec.operator!r}")

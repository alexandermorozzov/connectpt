"""Declarative ordered bee groups; every specification executes separately."""
from dataclasses import dataclass
from typing import Any
import torch
from ..network.routes import get_unselected_routes
from .operators import MutationContext, HeuristicRebuildOp, NeuralRebuildOp, ShortenOp, EditOp, get_new_route_variants


@dataclass(frozen=True)
class BeeSpec:
    count: int
    operator: str
    model_ref: str | None
    allowed_actions: list[str]
    ignore_max_route_len: bool


@dataclass(frozen=True)
class MethodParams:
    bees: dict[str, BeeSpec]
    passes_per_it: int
    mod_steps_per_pass: int
    shorten_prob: float
    force_linking_unlinked: bool
    process_neural_bees_sequentially: bool

    @classmethod
    def from_values(cls, values):
        return cls(**{**values, "bees": {name: BeeSpec(**spec) for name, spec in values["bees"].items()}})


@dataclass
class PlanGroup:
    op: Any
    count: int


class ExecutablePlan:
    def __init__(self, groups: list[PlanGroup], *, construction_model=None, edit_model=None):
        self.groups = list(groups)
        # A construction policy normalizes bee states before mutation.
        self.construction_model = construction_model
        self.edit_model = edit_model

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
        return any(g.op.uses_remaining_state and g.count > 0 for g in self.groups)

    def group_names(self) -> list[str]:
        return [g.op.name for g in self.groups]

    def summary(self) -> dict:
        return {g.op.name: int(g.count) for g in self.groups}

    def get_mutants(
        self,
        bee_networks,
        chosen_route_idxs,
        *,
        direct_sat_dmd,
        shorten_prob,
        street_node_neighbours,
        shortest_paths,
        force_linking_unlinked,
        env_state,
        cost,
        single_bee_env_state=None,
        adj_condition_target=None,
        adj_condition_weight=None,
        process_neural_bees_sequentially=False,
    ):
        """Apply one mutation step to every bee, dispatched by plan group.

        Reproduces the legacy ``bee_colony.get_mutants`` bit-for-bit without the
        legacy per-type branching: draw one bee permutation, slice it by
        group counts in canonical (slot) order, refill empty routes, then run
        each group's operator over its slots. Returns the mutated networks and a
        per-bee 1-based group slot tensor for mutation stats.
        """
        bee_networks = bee_networks.clone()
        seq_state = single_bee_env_state if single_bee_env_state is not None else env_state
        max_n_nodes = bee_networks.shape[3]
        gather_idx = chosen_route_idxs[..., None, None].expand(-1, -1, -1, max_n_nodes)
        modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)
        empty_routes = (modified_routes > -1).sum(-1) == 0
        n_bees = bee_networks.shape[1]
        dev = bee_networks.device

        # one bee permutation, sliced by group counts in canonical slot order
        scen_idxs = torch.randperm(n_bees, device=dev)
        mutation_types = torch.zeros(n_bees, device=dev, dtype=torch.long)
        group_idxs = []
        off = 0
        for slot, g in enumerate(self.groups, start=1):
            idxs = scen_idxs[off : off + g.count]
            off += g.count
            group_idxs.append(idxs)
            mutation_types[idxs] = slot

        # remaining-network state: the heuristic rebuild (no construction model)
        # and force-linked empty-route construction both need the routes NOT
        # under mutation. Matches the legacy condition exactly.
        remaining_state = None
        if self.construction_model is None or (empty_routes.any() and force_linking_unlinked):
            unsel_routes = get_unselected_routes(bee_networks, chosen_route_idxs)
            remaining_state = env_state.clone()
            remaining_state.replace_routes(unsel_routes.flatten(0, 1))

        if empty_routes.any():
            new_empty = get_new_route_variants(
                modified_routes,
                direct_sat_dmd,
                shortest_paths,
                force_linking_unlinked=force_linking_unlinked,
                remaining_state=remaining_state,
            )
            modified_routes[empty_routes] = new_empty[empty_routes]

        ctx = MutationContext(
            bee_networks=bee_networks,
            chosen_route_idxs=chosen_route_idxs,
            modified_routes=modified_routes,
            env_state=env_state,
            seq_state=seq_state,
            remaining_state=remaining_state,
            process_sequentially=process_neural_bees_sequentially,
            direct_sat_dmd=direct_sat_dmd,
            shorten_prob=shorten_prob,
            street_node_neighbours=street_node_neighbours,
            shortest_paths=shortest_paths,
            force_linking_unlinked=force_linking_unlinked,
            adj_condition_target=adj_condition_target,
            adj_condition_weight=adj_condition_weight,
            max_n_nodes=max_n_nodes,
            cost=cost,
        )

        # Each slot is a disjoint block of bees; run in canonical order so the
        # RNG stream (empty-route refill, then each operator) matches the legacy
        # typed dispatch. Empty groups consume no RNG and are skipped.
        new_routes = modified_routes.clone()
        for g, idxs in zip(self.groups, group_idxs):
            if g.count == 0:
                continue
            new_routes[:, idxs] = g.op.mutate(ctx, idxs)

        bee_networks.scatter_(2, gather_idx, new_routes[..., None, :])
        return bee_networks, mutation_types

    @property
    def needs_edit(self):
        return any(g.op.kind == "neural_edit" and g.count > 0 for g in self.groups)

    @property
    def needs_construction(self):
        return any(g.op.kind == "neural_rebuild" and g.count > 0 for g in self.groups)

    def validate(self, n_bees):
        if self.total_bees != n_bees or n_bees <= 0:
            raise ValueError("positive bee counts must sum to n_bees")
        if any(g.count < 0 for g in self.groups):
            raise ValueError("bee counts cannot be negative")
        if self.needs_edit and (self.edit_model is None or not self.edit_model.supports_trim_actions):
            raise ValueError("edit bees require a policy with trim actions")

    @classmethod
    def from_method(cls, method, policies,bindings):
        factories = {
            "shortest_path_rebuild": HeuristicRebuildOp,
            "local_edit": ShortenOp,
            "neural_rebuild": NeuralRebuildOp,
            "neural_edit": EditOp,
        }
        groups = []
        for name, spec in method.bees.items():
            factory = factories[spec.operator]
            if spec.model_ref is None:
                if spec.allowed_actions or spec.ignore_max_route_len:
                    raise ValueError("heuristic bees do not take action masks or neural bounds")
                op = factory(name=name)
            else:
                model = policies[spec.model_ref]
                if spec.operator == "neural_rebuild":
                    if model.action_spec.role != "construction" or spec.allowed_actions != ["extend", "halt"]:
                        raise ValueError("rebuild requires construction extend/halt actions")
                    if spec.ignore_max_route_len:
                        raise ValueError("full rebuild respects route bounds")
                    op = factory(model, name=name,binding=bindings[spec.model_ref])
                elif spec.operator == "neural_edit":
                    valid = {"extend", "trim_start", "trim_end", "halt"}
                    if (
                        model.action_spec.role != "edit"
                        or not spec.allowed_actions
                        or not set(spec.allowed_actions) <= valid
                    ):
                        raise ValueError("edit bee has an incompatible policy or action mask")
                    op = factory(
                        model,
                        binding=bindings[spec.model_ref],
                        name=name,
                        allowed_actions=spec.allowed_actions,
                        ignore_max_route_len=spec.ignore_max_route_len,
                    )
                else:
                    raise ValueError("heuristic operator cannot reference a policy")
            groups.append(PlanGroup(op, spec.count))
        plan = cls(
            groups,
            construction_model=policies["construction"] if "construction" in policies else None,
            edit_model=policies["edit"] if "edit" in policies else None,
        )
        plan.validate(plan.total_bees)
        if method.passes_per_it <= 0 or method.mod_steps_per_pass <= 0 or not 0 <= method.shorten_prob <= 1:
            raise ValueError("invalid colony pass count or shortening probability")
        return plan

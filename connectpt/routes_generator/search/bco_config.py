"""Parametric BCO-config factory (flat Holliday-BCO schema).

``compose_bco_cfg`` assembles the flat bee-colony eval config the notebook /
golden / MACSA path consumes (``n_bees`` + ``n_type*_bees`` + ``eval`` + ``model``
+ ``experiment.cost_function``), by composing a base Hydra config
(``bco_mumford`` / ``neural_bco_mumford``) with per-call overrides. Unlike a
static captured YAML, it is parametric: bee splits, weights, worse-accept
schedule and route bounds are function arguments.

This is the library home of what used to be ``eval_lib.helpers.build_bco_cfg``;
the composition is identical. Unset search knobs default from the BCO algorithm
config (``cfg/bco_mumford.yaml``) and unset weights from the unified objective
(``cfg/objective/rtt_wmc_no_demand.yaml``) -- no module-level constant pile.
"""
from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ..core.paths import CFG_DIR, CONSTRUCTION_MODEL_WEIGHTS_PATH
from ..objectives import load_bco_algo_config, load_unified_objective

_OBJ = load_unified_objective()


def safe_run_name(run_name: str) -> str:
    run_name = str(run_name)
    safe_chars = []
    for char in run_name:
        if char.isascii() and (char.isalnum() or char in "._-"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "_".join(part for part in "".join(safe_chars).split("_") if part)
    return safe.strip("._-") or "run"


def apply_disabled_components(cfg):
    """Inject the objective's disabled cost components into the cost-function
    kwargs so every cost module built from this cfg drops those components from
    the weighted cost (and renormalizes the survivors)."""
    disabled = list(_OBJ.disabled_components)
    if not disabled:
        return cfg
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.cost_function.kwargs.disabled_components = disabled
    OmegaConf.set_struct(cfg, True)
    return cfg


def compose_bco_cfg(
    run_name: str,
    n_routes: int,
    min_route_len: int,
    max_route_len: int,
    use_neural_bees: bool = False,
    n_bees: int | None = None,
    n_type1_bees: int | None = None,
    n_type2_bees: int | None = None,
    n_type4_bees: int = 0,
    n_type5_bees: int = 0,
    n_type6_bees: int = 0,
    n_type7_bees: int = 0,
    ignore_type4_max_route_len: bool | None = None,
    ignore_type5_max_route_len: bool | None = None,
    ignore_type6_max_route_len: bool | None = None,
    ignore_type7_max_route_len: bool | None = None,
    type4_allow_halt: bool = True,
    type5_allow_halt: bool = True,
    type6_allow_halt: bool = True,
    type7_allow_halt: bool = True,
    use_demand_weighted_route_selection: bool = False,
    demand_time_weight: float = _OBJ.demand_time_weight,
    route_time_weight: float = _OBJ.route_time_weight,
    median_connectivity_weight: float = _OBJ.median_connectivity_weight,
    connectivity_mode: str = "median_weighted",
    worse_accept_temperature: float | None = None,
    worse_accept_decay: float | None = None,
    worse_accept_min_temperature: float | None = None,
    worse_selection_temperature: float | None = None,
    worse_selection_decay: float | None = None,
    worse_selection_min_temperature: float | None = None,
    worse_selection_uniform_mix: float | None = None,
    worse_selection_elite_count: int | None = None,
    trim_grace_period: int = 0,
    process_neural_bees_sequentially: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    bee_model_arch: str | None = None,
    bee_model_weights=None,
    force_cpu: bool = False,
):
    """Build a flat BCO config.

    ``bee_model_arch`` / ``bee_model_weights`` (optional) override the neural
    *rebuild/construction* bee model. By default the type-1 rebuild bee uses the
    construction model (``bestsofar_feb2023``); set
    ``bee_model_arch="bestsofar_feb2023_trim"`` + ``bee_model_weights=<edit ckpt>``
    to drive the rebuild bee with the trim/edit model instead (rebuild via the
    edit model's RL rollout).

    User-facing mutation names:
      heuristic_rebuild      - rebuild the chosen route
      local_endpoint_edit    - local extend/shorten by one stop
      path_mix_rebuild       - random path-combiner rebuild
      construction_extend    - one construction-model extension/halt step
      extend_trim_edit       - one edit-model extend/trim/halt step
      trim_only              - one edit-model trim/halt step
      trim_then_extend       - trim/halt, then construction extend/halt
    """
    # Resolve unset args from the BCO algorithm config (reader + factory: the
    # builder reads cfg/bco_mumford.yaml directly instead of module constants).
    bco = load_bco_algo_config()
    if n_bees is None:
        n_bees = int(bco.n_bees)
    if n_type1_bees is None:
        n_type1_bees = int(bco.n_type1_bees)
    if ignore_type4_max_route_len is None:
        ignore_type4_max_route_len = bool(bco.ignore_type4_max_route_len)
    if ignore_type5_max_route_len is None:
        ignore_type5_max_route_len = bool(bco.ignore_type5_max_route_len)
    if ignore_type6_max_route_len is None:
        ignore_type6_max_route_len = bool(bco.ignore_type6_max_route_len)
    if ignore_type7_max_route_len is None:
        ignore_type7_max_route_len = bool(bco.ignore_type7_max_route_len)
    if worse_accept_temperature is None:
        worse_accept_temperature = float(bco.worse_accept_temperature)
    if worse_accept_decay is None:
        worse_accept_decay = float(bco.worse_accept_decay)
    if worse_accept_min_temperature is None:
        worse_accept_min_temperature = float(bco.worse_accept_min_temperature)
    if worse_selection_temperature is None:
        worse_selection_temperature = float(bco.worse_selection_temperature)
    if worse_selection_decay is None:
        worse_selection_decay = float(bco.worse_selection_decay)
    if worse_selection_min_temperature is None:
        worse_selection_min_temperature = float(bco.worse_selection_min_temperature)
    if worse_selection_uniform_mix is None:
        worse_selection_uniform_mix = float(bco.worse_selection_uniform_mix)
    if worse_selection_elite_count is None:
        worse_selection_elite_count = int(bco.worse_selection_elite_count)

    run_name = safe_run_name(run_name)
    base_cfg_name = "neural_bco_mumford" if use_neural_bees else "bco_mumford"
    effective_type2 = (
        n_bees - n_type1_bees - n_type4_bees - n_type5_bees - n_type6_bees - n_type7_bees
        if n_type2_bees is None
        else n_type2_bees
    )
    overrides = [
        "+eval=mumford0",
        "++eval.dataset.type=tensor",
        "++experiment.logdir=null",  # no empty TensorBoard run dir for eval runs
        f"++experiment.cpu={str(force_cpu).lower()}",
        f"++eval.n_routes={n_routes}",
        f"++eval.min_route_len={min_route_len}",
        f"++eval.max_route_len={max_route_len}",
        f"++experiment.cost_function.kwargs.demand_time_weight={demand_time_weight}",
        f"++experiment.cost_function.kwargs.route_time_weight={route_time_weight}",
        f"++experiment.cost_function.kwargs.median_connectivity_weight={median_connectivity_weight}",
        f"++experiment.cost_function.kwargs.connectivity_mode={connectivity_mode}",
        f"++run_name={run_name}",
        f"++n_bees={n_bees}",
        f"++n_iterations={int(bco.n_iterations)}",
        f"++n_type1_bees={n_type1_bees}",
        f"++n_type2_bees={effective_type2}",
        f"++n_type4_bees={n_type4_bees}",
        f"++n_type5_bees={n_type5_bees}",
        f"++n_type6_bees={n_type6_bees}",
        f"++n_type7_bees={n_type7_bees}",
        f"++ignore_type4_max_route_len={str(ignore_type4_max_route_len).lower()}",
        f"++ignore_type5_max_route_len={str(ignore_type5_max_route_len).lower()}",
        f"++ignore_type6_max_route_len={str(ignore_type6_max_route_len).lower()}",
        f"++ignore_type7_max_route_len={str(ignore_type7_max_route_len).lower()}",
        f"++type4_allow_halt={str(type4_allow_halt).lower()}",
        f"++type5_allow_halt={str(type5_allow_halt).lower()}",
        f"++type6_allow_halt={str(type6_allow_halt).lower()}",
        f"++type7_allow_halt={str(type7_allow_halt).lower()}",
        f"++use_demand_weighted_route_selection={str(use_demand_weighted_route_selection).lower()}",
        f"++worse_accept_temperature={worse_accept_temperature}",
        f"++worse_accept_decay={worse_accept_decay}",
        f"++worse_accept_min_temperature={worse_accept_min_temperature}",
        f"++worse_selection_temperature={worse_selection_temperature}",
        f"++worse_selection_decay={worse_selection_decay}",
        f"++worse_selection_min_temperature={worse_selection_min_temperature}",
        f"++worse_selection_uniform_mix={worse_selection_uniform_mix}",
        f"++worse_selection_elite_count={worse_selection_elite_count}",
        f"++trim_grace_period={trim_grace_period}",
        "++process_neural_bees_sequentially="
        f"{str(process_neural_bees_sequentially).lower()}",
        f"++early_stop_min_delta={float(early_stop_min_delta)}",
    ]
    if bee_model_arch is not None:
        # rebuild/construction bee uses a custom model arch (e.g. our trim edit
        # model) -- compose-time defaults-group override + serial halting.
        overrides = [f"model={bee_model_arch}",
                     "model.route_generator.kwargs.serial_halting=True"] + overrides
    if early_stop_patience is not None:
        overrides.append(f"++early_stop_patience={int(early_stop_patience)}")
    if use_neural_bees:
        _bee_w = bee_model_weights if bee_model_weights is not None else CONSTRUCTION_MODEL_WEIGHTS_PATH
        overrides.append(f"+model.weights='{_bee_w}'")
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name=base_cfg_name, overrides=overrides)
    cfg.batch_size = 1
    n_type3 = (n_bees - n_type1_bees - effective_type2 - n_type4_bees -
               n_type5_bees - n_type6_bees - n_type7_bees)
    route_selection_label = "weighted" if use_demand_weighted_route_selection else "uniform-random"
    _rebuild_name = "neural_rebuild" if use_neural_bees else "heuristic_rebuild"
    print(
        f"[{run_name}] route_selection={route_selection_label} | "
        f"bee split: {_rebuild_name}={n_type1_bees} "
        f"local_endpoint_edit={effective_type2} path_mix_rebuild={n_type3} "
        f"construction_extend={n_type4_bees} extend_trim_edit={n_type5_bees} "
        f"trim_only={n_type6_bees} trim_then_extend={n_type7_bees} "
        f"(total={n_bees})"
    )
    return apply_disabled_components(cfg)

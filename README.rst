ConnectPT
=========

.. logo-start

.. figure:: https://psv4.userapi.com/s/v1/d2/aE8kEC2MYzxzxQgGbG4SIXKijfv-ouCe9jNMag7ONZ8TdctZo5IKBe-MR2OTRxVEWMIaq7yxqnPSpyKEK4HMDw_yf5_XLgYa-7MQxgABQBIUzMCtFT7G5FsrWZN7GbfnTQUP-1X-NSqK/connectpt_logo_gen_gpt_v4.png
   :alt: ConnectPT

.. logo-end

|PythonVersion|

.. readme-start

Overview
--------

ConnectPT is a research toolkit for public-transport route generation. It
contains graph preprocessing utilities, cost and metric evaluation, classical
route improvement methods, and neural route generators based on the
``transit_learning`` project.

The current route-generation stack supports two neural training modes:

``LC construction``
   The model builds a route network from scratch. It starts with an empty
   ``RouteGenBatchState`` and plans routes one by one.

``LC improvement``
   The model starts from an existing route network, usually LC routes saved on
   disk, and learns how to improve each route with typed edit actions:
   extend, trim start, trim end, or halt.

The main notebook for construction is
``examples/route_generator/experiment.ipynb``. The main notebook for
improvement is ``examples/route_generator/lc_improvement_training.ipynb``.


Installation
------------

Install the package from GitHub:

.. code-block:: bash

   pip install git+https://github.com/alexandermorozzov/connectpt@main

For local development, create a virtual environment and install the repository
in editable mode:

.. code-block:: bash

   pip install -e .


Project Layout
--------------

Important files for route generation:

``connectpt/routes_generator/models.py``
   Neural route-generator models. ``TrimPathCombiningRouteGenerator`` adds
   typed route-edit actions on top of the path-combining route generator.

``connectpt/routes_generator/transit_time_estimator.py``
   ``RouteGenBatchState`` and route-network state updates. This is where
   current routes, finished routes, trim actions, route matrices, transit
   times, and cost-state data are maintained.

``connectpt/routes_generator/improvement_learning.py``
   LC-improvement training pipeline. It loads graphs and seed routes, creates
   improvement batches, samples edit actions, replays them for policy-gradient
   training, evaluates before/after costs, and returns generated routes.

``connectpt/routes_generator/cfg/ppo_50nodes.yaml``
   PPO-style base config used by the improvement notebook. The notebook
   overrides the model with ``model=bestsofar_feb2023_trim``.

``connectpt/routes_generator/cfg/model/bestsofar_feb2023_trim.yaml``
   Model config that selects the trim-capable route generator.

``connectpt/routes_generator/cfg/model/route_generator/biased_trim.yaml``
   Route-generator config for ``TrimPathCombiningRouteGenerator``.

``examples/data/raw_graphs_1000.pkl``
   Pickled graph data used by the LC-improvement notebook.

``examples/lc_results/``
   Existing LC route sets used as seed routes for improvement training.


Preprocessing Example
---------------------

Preprocess public-transport data for selected modes:

.. code-block:: python

   import geopandas as gpd

   from connectpt.preprocess import Modality, preprocess

   blocks = gpd.read_file("path/to/blocks.geojson")
   result, graph = preprocess(blocks, [Modality.BUS])
   stops_gdf, time_matrix, stop_graph = result[Modality.BUS]


Route Generation Before LC Improvement
--------------------------------------

Originally the neural route generator was used mostly as a constructor:

1. Start from an empty ``RouteGenBatchState``.
2. Plan route 0, then route 1, and so on.
3. At every step, select a shortest-path terminal pair ``[u, v]`` to extend
   the current route.
4. Select halt when the current route is finished.
5. Add the finished route to the route network and continue with the next one.

The legacy action API represented only:

``extend``
   A terminal pair ``[u, v]``. The state converts it into the corresponding
   shortest-path segment and prepends/appends it to the current route.

``halt``
   A terminal pair ``[-1, -1]``. The current route is committed to the finished
   route list.

This works for construction from scratch, but it is awkward for improvement:
the model can add route pieces, but it cannot remove bad pieces from an
existing route.


LC Improvement: What Changed
----------------------------

LC improvement trains the same family of model as a route editor. Instead of
starting with no routes, it receives a complete seed route network and revisits
each route slot.

The added pieces are:

``TrimPathCombiningRouteGenerator``
   A trim-capable route generator. It exposes
   ``supports_trim_actions = True`` and uses ``step_route_action`` instead of
   only the legacy ``step`` method.

``RouteGenBatchState.set_current_routes``
   Seeds an already existing route as the active route being edited.

``RouteGenBatchState.apply_route_actions``
   Applies typed route actions to the active route and rebuilds route tensors
   after trim edits so stale route edges do not remain in the network.

``rollout_lc_improvement``
   Runs the improvement policy over seed routes and returns the original cost,
   improved cost, route logits, entropies, and optionally the sampled actions.

``train_lc_improvement``
   Trains the improvement policy with a policy-gradient style objective based
   on ``seed_cost - final_cost``.

``train_lc_improvement_ppo``
    A separate PPO-style improvement trainer. It keeps the same seeded-route
    environment, but records old per-step action log-probabilities and updates
    the policy with a clipped ratio objective. This is separate from the
    construction PPO loop in ``inductive_route_learning.py`` and separate from
    the simpler ``train_lc_improvement`` trainer.

``train_lc_improvement_cfg_ppo``
   A fuller PPO adaptation for improvement. It keeps the old construction PPO
   mechanics but swaps the construction environment for seeded-route editing:
   rollout horizon, PPO epochs, minibatch size, clipping epsilon, GAE,
   discounting, reward scaling, entropy weight, neural value baseline,
   optimizer type, learning rate, decay, and ``diff_reward`` all come from the
   selected PPO config such as ``ppo_20nodes.yaml`` or ``ppo_50nodes.yaml``.


How Seeded Routes Are Handled
-----------------------------

Seed routes are loaded as a tensor with shape:

.. code-block:: text

   batch_size x n_routes x max_route_len

Padding uses ``-1``.

For every route index ``i`` in the seed network, LC improvement creates a
temporary planning state:

1. Create a fresh ``RouteGenBatchState`` with ``n_routes_to_plan = n_routes``.
2. Add all seed routes except route ``i`` as finished routes.
3. Set seed route ``i`` as ``current_routes``.
4. Let the model choose typed actions for the current route.
5. Store the resulting edited route for slot ``i``.

After all route slots are processed, the edited routes are assembled back in
the original route order and added to a final ``RouteGenBatchState``. Final
metrics are computed on that complete improved network.

This is important: the model now sees the whole seed route network while
editing each route. Earlier improvement rollout behaved more like sequential
construction: route 0 did not see future routes, route 1 only saw route 0, and
so on. That was conceptually wrong for LC improvement because the seed network
already exists.


Typed Route Actions
-------------------

Typed actions are defined in ``transit_time_estimator.py``:

.. code-block:: python

   ROUTE_ACTION_EXTEND = 0
   ROUTE_ACTION_TRIM_START = 1
   ROUTE_ACTION_TRIM_END = 2
   ROUTE_ACTION_HALT = 3

Each step has two tensors:

``action_kinds``
   Shape ``batch_size``. Stores one of the constants above.

``path_indices``
   Shape ``batch_size x 2``. Stores the selected node pair or ``[-1, -1]`` for
   halt.

Action behavior:

``ROUTE_ACTION_EXTEND``
   ``path_indices = [u, v]``. The state reconstructs the shortest path between
   ``u`` and ``v`` and attaches it to the current route. If the route is empty,
   this starts a new route. If the route is non-empty, the segment must attach
   to the current start or current end.

``ROUTE_ACTION_TRIM_START``
   Removes the prefix of the current route. The new start is read from
   ``path_indices[:, 1]``. The resulting route must still satisfy
   ``min_route_len``.

``ROUTE_ACTION_TRIM_END``
   Removes the suffix of the current route. The new end is read from
   ``path_indices[:, 0]``. The resulting route must still satisfy
   ``min_route_len``.

``ROUTE_ACTION_HALT``
   Commits the current route as finished. The corresponding ``path_indices``
   are set to ``[-1, -1]``.


How Actions Are Scored and Encoded
----------------------------------

The trim-capable model scores three route-edit blocks:

.. code-block:: text

   extend scores      -> n_nodes * n_nodes candidates
   trim-start scores  -> n_nodes * n_nodes candidates
   trim-end scores    -> n_nodes * n_nodes candidates

Internally, route-action selections are encoded as flat indices:

.. code-block:: text

   extend(u, v)      = u * n_nodes + v
   trim_start(u, v)  = 1 * n_nodes * n_nodes + u * n_nodes + v
   trim_end(u, v)    = 2 * n_nodes * n_nodes + u * n_nodes + v

If ``serial_halting`` is disabled, halt is appended as one more candidate in
the same action distribution. If ``serial_halting`` is enabled, halt is chosen
by a separate binary continue-or-halt decision before choosing a route-edit
action.

Halt is normally allowed when the current route length is at least
``min_route_len``. It is forced when the route is already done or when no valid
route-edit action exists. ``force_nonhalt_first_step`` exists as a debugging
option, but for LC improvement it should usually stay ``False``: if a seed
route is already good, immediate halt is a valid decision.

Because trim actions can shorten a route and make further extend actions valid
again, the improvement rollout also has ``max_route_edit_steps``. If the model
does not halt by that limit, the current route is forced to halt. The default
is ``2 * max_route_len``.


Training Pipeline
-----------------

The improvement notebook follows the shape of
``examples/route_generator/experiment.ipynb``:

1. Load graphs from ``examples/data/raw_graphs_1000.pkl``.
2. Load seed LC routes from ``examples/lc_results``.
3. Build the model from ``ppo_50nodes.yaml`` with
   ``model=bestsofar_feb2023_trim``.
4. Warm up feature normalization on improvement batches.
5. Run ``train_lc_improvement_cfg_ppo`` by default.
6. Collect edit-action rollout buffers using the PPO config:

.. code-block:: text

   cfg.ppo.horizon
   cfg.ppo.n_epochs
   cfg.ppo.minibatch_size
   cfg.ppo.epsilon
   cfg.ppo.use_gae
   cfg.ppo.gae_lambda
   cfg.diff_reward
   cfg.discount_rate
   cfg.reward_scale

7. If ``cfg.diff_reward`` is true, compute the per-edit-step reward as:

.. code-block:: text

   reward_t = cost_before_action - cost_after_action

8. If ``cfg.diff_reward`` is false, use the terminal negative-cost reward from
   the original PPO code path.
9. Compute returns and advantages with GAE or simple discounted returns,
   depending on ``cfg.ppo.use_gae``.
10. Replay sampled fixed actions, update the neural value baseline, and update
    the policy with the PPO clipped objective.
11. Evaluate on validation graphs every ``cfg.ppo.val_period`` iterations and
    save the best checkpoint.

The older improvement trainers are still available. ``train_lc_improvement``
and ``train_lc_improvement_ppo`` use a simpler graph-level improvement signal:

.. code-block:: text

   advantage = seed_cost - final_cost

The notebook exposes ``TRAINING_ALGORITHM``:

``cfg_ppo_improvement``
   Default. Uses ``train_lc_improvement_cfg_ppo`` and reads PPO behavior from
   the selected config file, including ``diff_reward``.
   The notebook intentionally overrides ``BATCH_SIZE`` to a much smaller value
   than ``cfg.batch_size`` for ``ppo_50nodes.yaml``. The improvement rollout
   stores full ``RouteGenBatchState`` snapshots, so using ``batch_size: 512``
   with ``horizon: 200`` can exhaust notebook memory before one iteration
   finishes.

``ppo_improvement``
   Uses the older ``train_lc_improvement_ppo`` with PPO clipping. It reads
   ``cfg.ppo.n_epochs`` and ``cfg.ppo.epsilon`` for the number of PPO update
   passes and clip range, but does not use the full PPO horizon/value/GAE loop.

``policy_gradient``
   Uses the original ``train_lc_improvement`` loop. This keeps the first
   implementation available for comparison.

The replay step exists because route planning mutates ``RouteGenBatchState``.
Backpropagating through a graph that still references mutable route-index
tensors can trigger PyTorch in-place version-counter errors. Replaying sampled
actions and calling ``backward`` before each state mutation avoids that class
of error.


Evaluation and Diagnostics
--------------------------

``evaluate_lc_improvement`` reports:

``seed_cost``
   Mean cost of the original LC route networks.

``final_cost``
   Mean cost after the improvement model edits routes.

``delta``
   ``seed_cost - final_cost``. Positive values mean the improvement policy made
   the route network cheaper under the current cost.

``win_rate``
   Fraction of validation graphs where improved cost is lower than seed cost.

``changed_route_rate``
   Fraction of route slots whose tensor differs from the seed route.

``changed_graph_rate``
   Fraction of graphs where at least one route changed.

During training, ``train_lc_improvement`` also stores action statistics in
``history`` with ``train_action_*`` fields. The most useful ones are
``train_action_extend_count``, ``train_action_trim_start_count``,
``train_action_trim_end_count``, ``train_action_halt_count``,
``train_action_total_count``, and ``train_action_avg_actions_per_route``.
Counts include only actions executed before the first halt for each planned
route, so post-halt batch padding is ignored.

The improvement notebook also contains before/after metric comparison and
visualization cells for metrics such as cost, ATT, RTT, disconnected demand
pairs, out-of-bounds stops, and connectivity.


Bee Colony Example
------------------

Classical Bee Colony Optimization is still available:

.. code-block:: python

   from omegaconf import OmegaConf

   from connectpt.routes_generator import (
       RouteGenBatchState,
       bee_colony,
       get_cost_module_from_cfg,
       get_dataset_from_config,
       init_from_cfg,
   )

   cfg = OmegaConf.load("connectpt/routes_generator/cfg/bco_mumford.yaml")
   dataset = get_dataset_from_config(cfg.data)
   batch = dataset[:1]
   cost = get_cost_module_from_cfg(cfg.cost)
   state = RouteGenBatchState(batch, cost, n_routes_to_plan=cfg.experiment.n_routes)
   init_routes = init_from_cfg(state, cfg.init)
   best = bee_colony(state, cost, init_routes, n_bees=cfg.experiment.n_bees)

   print(best.shape)  # batch_size x n_routes x max_nodes


Notes for Development
---------------------

When working on LC improvement, the most useful sanity checks are:

``changed_route_rate``
   If this is always zero, the policy is halting immediately or all sampled
   edits are being discarded.

``serial_halting``
   With ``True``, halt is decided before route editing. With ``False``, halt is
   part of the same action distribution as extend and trim.

``min_route_len`` and ``max_route_len``
   Trim cannot make a route shorter than ``min_route_len``. Extend cannot make
   it longer than ``max_route_len``.

``max_route_edit_steps``
   Caps the number of edit actions for one route. This prevents greedy eval
   from hanging in a trim/extend loop.

``n_routes``
   The improvement pipeline preserves the number of route slots from the seed
   route tensor. It edits each slot rather than inventing an additional route
   count.


Data and Examples
-----------------

Benchmark and generated data live under ``data/`` and ``examples/data/``.
Route-generation notebooks live under ``examples/route_generator/``.


Contacts
--------

- Alexander Morozov: alexandermorozzov@gmail.com
- Ruslan Kozlyak: rkozliak@gmail.com


Acknowledgments
---------------

This work is supported by the Ministry of Economic Development of the Russian
Federation (IGK 000000C313925P4C0002), agreement No. 139-15-2025-010.

.. readme-end

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.11+-blue
   :target: https://www.python.org/

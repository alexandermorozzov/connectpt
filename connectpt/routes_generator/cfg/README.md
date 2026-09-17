# Configuration layout

The configuration is assembled from small Hydra presets. Files under `runs/`
are the public entry points; they select reusable pieces and contain only
run-level settings.

```text
cfg/
├── runs/            # search, train, prepare_dataset
├── instance/        # benchmark path and route constraints
├── method/          # named bee-colony variants
├── bee/             # one atomic bee/operator type
├── bee_set/         # mixtures of bee presets and their counts
├── cost/            # base objective
├── adjustment/      # cap, target, or disabled adjustment term
├── model/           # model architecture
├── model_profile/   # model version: architecture preset + checkpoint
├── dataset/         # dataset identity and split
├── dataset_build/   # dataset generation parameters
├── curriculum/      # training schedule
├── trainer/         # optimizer/PPO loop
├── sweep/           # search grid
└── stacks/          # internal compatibility composition
```

`model/` answers *what architecture is built*. `model_profile/` answers *which
version of that architecture is used*, including its checkpoint. A method
selects profiles by semantic role (`construction` or `edit`); policy adapters
are derived from those roles in code, so there are no separate `model_set` or
`policy_set` layers.

Examples:

```powershell
python scripts/run_search.py --dry-run --methods ea
python scripts/run_search.py --methods nea_edit --instances mandl
python scripts/run_training.py --dry-run
python scripts/prepare_data.py
```

Both search and training compose the same `cost=rtt_wmc` preset. Adjustment
behaviour is orthogonal: use `adjustment=cap`, `adjustment=target`, or
`adjustment=disabled`. Current public runs select `cap`; `target` is available
for an explicit override.

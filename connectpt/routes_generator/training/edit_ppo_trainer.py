"""EditPPOTrainer -- a thin wrapper over the existing PPO training function.

``fit()`` delegates to ``train_lc_improvement_cfg`` (the ppo/d3po dispatcher);
the PPO loop itself is NOT reimplemented here. The wrapper only gathers the
arguments from cfg + the data module so the run layer stays small.
"""
from __future__ import annotations

import pandas as pd

from ..improvement_learning import train_lc_improvement_cfg


class EditPPOTrainer:
    def __init__(self, cfg, model, cost_obj, data_module, *, run_name, output_dir,
                 device, best_model_path=None):
        self.cfg = cfg
        self.model = model
        self.cost_obj = cost_obj
        self.data = data_module
        self.run_name = run_name
        self.output_dir = output_dir
        self.device = device
        self.best_model_path = best_model_path

    def fit(self, *, train_indices=None, val_indices=None,
            curriculum_fn=None, val_curriculum_fn=None) -> pd.DataFrame:
        """Run training and return the history as a DataFrame."""
        data_cfg = self.cfg.data
        result = train_lc_improvement_cfg(
            model=self.model,
            cost_obj=self.cost_obj,
            graphs=self.data.graphs,
            seed_routes=self.data.seed_routes,
            device=self.device,
            cfg=self.cfg,
            output_dir=self.output_dir,
            run_name=self.run_name,
            train_fraction=float(data_cfg.train_fraction),
            min_route_len=int(data_cfg.min_route_len),
            max_route_len=int(data_cfg.max_route_len),
            target_n_routes=data_cfg.get("target_n_routes"),
            seed=int(self.cfg.run.seed),
            train_indices=train_indices,
            val_indices=val_indices,
            best_model_path=self.best_model_path,
            curriculum_fn=curriculum_fn,
            val_curriculum_fn=val_curriculum_fn,
        )
        return pd.DataFrame(result["history"])

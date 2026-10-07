"""EditPPOTrainer -- a thin wrapper over the existing PPO training function.

``fit()`` delegates to ``train_lc_improvement_cfg`` (the ppo/d3po dispatcher);
the PPO loop itself is NOT reimplemented here. The wrapper only gathers the
arguments from cfg + the data module so the run layer stays small.
"""
from __future__ import annotations

import torch

from ..improvement_learning import train_lc_improvement_cfg


class EditPPOTrainer:
    def __init__(self, cfg, model, cost_obj, data_module, *, dataset, seed, run_name,
                 output_dir, device, best_model_path=None):
        self.cfg = cfg
        self.model = model
        self.cost_obj = cost_obj
        self.data = data_module
        self.dataset = dataset
        self.seed = seed
        self.run_name = run_name
        self.output_dir = output_dir
        self.device = device
        self.best_model_path = best_model_path

    def fit(self, *, logger_list, train_indices=None, val_indices=None,
            curriculum_fn=None, val_curriculum_fn=None) -> None:
        """Run training; metrics go to ``logger_list``."""
        data = self.dataset
        trainer_cfg = self.cfg.get("train_loop", {}) or {}
        # gpu-aware graph batch size (was the notebook's BATCH_SIZE = 16 if cuda
        # else 4); n_iterations from the trainer block (was N_ITERATIONS).
        batch_size = int(trainer_cfg.get("batch_size_cuda", 16)
                         if torch.cuda.is_available()
                         else trainer_cfg.get("batch_size_cpu", 4))
        n_iterations = int(trainer_cfg.get("n_iterations", self.cfg.ppo.n_iterations))
        train_lc_improvement_cfg(
            batch_size=batch_size,
            n_iterations=n_iterations,
            model=self.model,
            cost_obj=self.cost_obj,
            graphs=self.data.graphs,
            seed_routes=self.data.seed_routes,
            device=self.device,
            cfg=self.cfg,
            output_dir=self.output_dir,
            run_name=self.run_name,
            train_fraction=data.train_fraction,
            min_route_len=data.min_route_len,
            max_route_len=data.max_route_len,
            target_n_routes=data.target_n_routes,
            seed=self.seed,
            train_indices=train_indices,
            val_indices=val_indices,
            best_model_path=self.best_model_path,
            curriculum_fn=curriculum_fn,
            val_curriculum_fn=val_curriculum_fn,
            logger_list=logger_list,
        )

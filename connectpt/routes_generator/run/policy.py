"""Configuration boundary for standalone pretrained policy use."""
from ..objectives.cost import build_cost
from .config import compose_policy_model, cost_params, policy_profile
from .const import MODEL_WEIGHTS_DIR
from ..common.random import seed_everything
from .context import resolve_weights_path
from .runs import ExperimentRun
from ..policy.build import ModelProfileParams, load_policy


class PolicyRun(ExperimentRun):
    def __init__(self, cfg, device, *, weights_root=MODEL_WEIGHTS_DIR):
        super().__init__(cfg)
        self.device = device
        self.weights_root = weights_root

    def setup(self):
        cfg = self.cfg
        profile = ModelProfileParams(**cfg.model_profile)
        self.seed = cfg.experiment.seed
        seed_everything(self.seed)
        self.cost = build_cost(
            cost_params(cfg.cost), symmetric_routes=cfg.experiment.symmetric_routes, low_memory_mode=cfg.experiment.low_memory_mode
        )
        self.cost.to(self.device)
        self.profile = policy_profile(
            compose_policy_model(profile.config),
            cfg.cost,
            cfg.experiment,
            self.device,
            checkpoint_path=None
            if profile.checkpoint_path is None
            else resolve_weights_path(profile.checkpoint_path, self.weights_root),
        )
        self.policy = load_policy(self.profile)

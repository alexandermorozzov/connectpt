"""A scalar or multi-goal value head with checkpoint-compatible model keys."""
from dataclasses import dataclass
import torch
from torch_geometric.data import Batch
from ..policy.models import FeatureNorm, get_mlp
from ..env.features import critic_inputs, critic_features, PREFERENCE_KEYS, dims


@dataclass(frozen=True)
class CriticParams:
    learning_rate: float
    decay: float
    shared: bool
    unbiased: bool
    normalize_returns: bool
    huber: bool
    huber_delta: float
    value_clip: float | None
    return_norm_momentum: float
    feature_norm_momentum: float
    n_layers: int
    width_multiplier: int
    dropout: float


class ValueHead:
    """Only the value head trains; shared actor features are detached."""

    def __init__(self, k, features, params, device, *, scalar, actor_model=None, observation=None):
        self.n_objectives = k
        self.scalar = scalar
        self.params = params
        self.learning_rate = params.learning_rate
        self.decay = params.decay
        self.device = device
        self.preference_goals = features.preference_goals
        self.weight_input_keys = tuple(PREFERENCE_KEYS[g] for g in ("ATT", "RTT", "WMC") if g in self.preference_goals)
        self.holliday_features = params.unbiased
        self.feature_set = features
        self.optim = self.model = self._curr_estimate = None
        self.loss_fn = torch.nn.MSELoss()
        self.actor_model = actor_model
        # A shared critic reads the environment's observation of a State.
        self.observation = observation
        self._shared_critic = actor_model is not None
        self.normalize_returns = params.normalize_returns
        self.huber = params.huber
        self.huber_delta = params.huber_delta
        self.value_clip = params.value_clip
        self._ret_momentum = params.return_norm_momentum
        self._ret_mean = torch.zeros((), device=device)
        self._ret_var = self._ret_std = torch.ones((), device=device)
        self._ret_initialized = False
        if not self._shared_critic:
            self._build_model(dims(features)["critic"])

    def _build_model(self, input_dim):
        self.model = torch.nn.Sequential(
            FeatureNorm(input_dim, self.params.feature_norm_momentum, unbiased=self.holliday_features),
            get_mlp(
                self.params.n_layers,
                input_dim * self.params.width_multiplier,
                in_dim=input_dim,
                out_dim=self.n_objectives,
                dropout=self.params.dropout,
            ),
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)

    @property
    def input_dim(self):
        return dims(self.feature_set)["critic"]

    def _value_loss(self, pred, target):
        if self.huber:
            return torch.nn.functional.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="none")
        return (pred - target) ** 2

    def _update_return_stats(self, costs):
        batch_mean = costs.mean()
        batch_var = costs.var(unbiased=False)
        if not self._ret_initialized:
            self._ret_mean = batch_mean.detach().clone()
            self._ret_var = batch_var.detach().clone()
            self._ret_initialized = True
        else:
            m = self._ret_momentum
            self._ret_mean = (1 - m) * self._ret_mean + m * batch_mean.detach()
            self._ret_var = (1 - m) * self._ret_var + m * batch_var.detach()
        self._ret_std = (self._ret_var + 1e-6).sqrt()

    def update(self, costs, old_values=None):
        """Run a critic update step.

        Returns a dict with detached snapshots so the trainer can log
        critic-quality metrics (MSE, value-vs-return calibration, residuals)
        without needing to re-run the model. ``old_values`` (collection-time
        value estimates, raw units) enables PPO value-clipping when
        ``value_clip`` is set.
        """
        self.optim.zero_grad()
        pred = self._curr_estimate
        costs = costs.to(pred.dtype)

        if self.normalize_returns:
            mean, std = self._ret_mean, self._ret_std
            target = (costs - mean) / std
            old_n = None if old_values is None else (old_values.to(pred.dtype) - mean) / std
        else:
            target = costs
            old_n = None if old_values is None else old_values.to(pred.dtype)

        if self.value_clip is not None and old_n is not None:
            pred_clipped = old_n + torch.clamp(pred - old_n, -self.value_clip, self.value_clip)
            loss = torch.maximum(self._value_loss(pred, target), self._value_loss(pred_clipped, target)).mean()
        else:
            loss = self._value_loss(pred, target).mean() if self.scalar or self.huber else self.loss_fn(pred, target)

        # Snapshots in RAW units (de-normalized) so explained-variance / MSE
        # diagnostics stay comparable across modes.
        if self.normalize_returns:
            raw_values = pred.detach() * self._ret_std + self._ret_mean
        else:
            raw_values = pred.detach()
        values_snapshot = raw_values.clone()
        targets_snapshot = costs.detach().clone()

        loss.backward()
        self.optim.step()
        self._curr_estimate = None
        # update the feature normalization statistics
        self.model[0].update()
        # update running return statistics AFTER using them for this step
        if self.normalize_returns:
            self._update_return_stats(costs.detach())
        return {
            "loss": float(loss.detach().item()),
            "per_obj_mse": (values_snapshot - targets_snapshot).pow(2).mean(dim=0),
            "values": values_snapshot,
            "targets": targets_snapshot,
        }

    def inputs_from_data(self, graph_data, cost_weights):
        return critic_inputs(
            graph_data,
            cost_weights,
            input_dim=self.input_dim,
            holliday_features=self.holliday_features,
            weight_input_keys=self.weight_input_keys,
        )

    def set_input_norm(self, all_graph_data):
        if not isinstance(all_graph_data, Batch):
            all_graph_data = Batch.from_data_list(all_graph_data)
        input_data = self.inputs_from_data(all_graph_data, {})
        self.model[0].running_mean[...] = input_data.mean(dim=0)
        self.model[0].running_var[...] = input_data.var(dim=0)

        self.model[0].running_mean[-15:] = 0
        self.model[0].running_var[-15:] = 1

    def _features_from_state(self, state):
        if self._shared_critic:
            return self.actor_model.get_critic_features(self.observation(state))
        return critic_features(state, self.feature_set)

    def from_state(self, state):
        input_data = self._features_from_state(state)
        if self.model is None:
            self._build_model(input_data.shape[-1])
        pred = self.model(input_data)
        if self.scalar:
            pred = pred.squeeze(-1)
        # ``_curr_estimate`` is what update() regresses: the normalized
        # prediction when return-normalization is on, else the raw value.
        self._curr_estimate = pred
        if self.normalize_returns:
            baseline = pred * self._ret_std + self._ret_mean
        else:
            baseline = pred

        assert baseline.isfinite().all()

        return baseline.detach()

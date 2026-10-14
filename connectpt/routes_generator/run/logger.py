"""Concrete TensorBoard and CSV sinks implementing common.logging.Logger."""
from __future__ import annotations

import csv
from pathlib import Path
from ..common.logging import Logger


class TensorBoardLogger(Logger):
    """Event files in ``<run_dir>/tensorboard``."""

    _DISPLAY_TAGS = frozenset({
        'eval/cost', 'eval/feasible_fraction',
        'rollout/cost', 'rollout/feasible_fraction', 'rollout/ep_rew_mean', 'rollout/ep_len_mean',
        'rollout/att_minutes', 'rollout/rtt_minutes',
        'train/value_loss', 'train/policy_gradient_loss', 'train/entropy_loss',
        'train/approx_kl', 'train/clip_fraction', 'train/explained_variance',
        'train/diversity_loss', 'train/diversity_kl', 'train/diversity_target',
        'time/total_timesteps', 'time/fps',
        'train/diagnostics/policy_gradient_norm', 'train/diagnostics/weighted_diversity_gradient_norm',
        'train/diagnostics/diversity_gradient_ratio', 'train/diagnostics/gradient_cosine',
        'train/diagnostics/diversity_kl_free_mean', 'train/diagnostics/diversity_target_free_mean',
        'train/diagnostics/diversity_free_fraction',
    })
    _PARETO_FIELDS = frozenset({
        'hv', 'eu_conditioned', 'eu_archive', 'feasible_fraction', 'distinct_networks',
        'distinct_pareto_networks', 'pareto_points_feasible', 'sparsity_feasible',
    })

    @classmethod
    def _display_scalar(cls, tag):
        """Filter display-only redundancy; complete training records remain in CSV."""
        if tag in cls._DISPLAY_TAGS or tag.startswith('eval/objective/'):
            return True
        if tag.startswith('pareto/'):
            return tag.rsplit('/', 1)[-1] in cls._PARETO_FIELDS
        if tag.startswith('eval/test/') or tag.startswith('eval/final_validation/'):
            field = tag.rsplit('/', 1)[-1]
            return ('/objective/' in tag or '/penalty/' in tag or field in cls._PARETO_FIELDS or field == 'cost')
        # Search and unrelated record families retain their existing display tags.
        return not tag.startswith(('eval/', 'rollout/', 'train/', 'time/'))

    def __init__(self, run_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(Path(run_dir) / "tensorboard"))

    def scalar(self, tag, value, step):
        if self._display_scalar(tag):
            self.writer.add_scalar(tag, value, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class CsvLogger(Logger):
    """``<run_dir>/history.csv``: one row per step, one column per tag."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "history.csv"
        self.rows: dict[int, dict[str, float]] = {}
        self.columns: dict[str, None] = {}  # first-seen order

    def scalar(self, tag, value, step):
        self.rows.setdefault(int(step), {})[tag] = value
        self.columns.setdefault(tag)

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", *self.columns])
            writer.writeheader()
            for step in sorted(self.rows):
                writer.writerow({"step": step, **self.rows[step]})


LOGGERS = {"tensorboard": TensorBoardLogger, "csv": CsvLogger}


def make_loggers(names, run_dir) -> list[Logger]:
    return [LOGGERS[name](run_dir) for name in names]

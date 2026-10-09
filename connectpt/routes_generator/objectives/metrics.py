"""Named metric projections of an already computed Evaluation."""
import numpy as np
import torch
from .const import METRIC_NAMES


def metric_value(metrics, key, default=np.nan):
    """Read one metric as a plain float (mean over the batch), or ``default``."""
    if key not in metrics or metrics[key] is None:
        return default
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 0:
            return default
        return float(value.float().mean().item())
    return float(value)


def select_metrics(row: dict, keep) -> dict:
    """Filter a metric row to the ``keep`` keys (preserving present ones)."""
    keep = list(keep)
    return {k: row[k] for k in keep if k in row}


class Metrics(dict):
    """Named network measurements, in the units of the result tables."""


def evaluate_network(result):
    """Per-network tensor metrics from a prepared Evaluation; no rescoring."""
    percent = (result.trips_at_transfers / result.total_demand[:, None]) * 100
    return Metrics(
        zip(
            METRIC_NAMES,
            (
                result.cost,
                result.mean_demand_time / 60,
                result.total_route_time / 60,
                percent[:, 0],
                percent[:, 1],
                percent[:, 2],
                percent[:, 3],
                result.n_disconnected_demand_edges.float(),
                result.n_stops_oob.float(),
                result.median_connectivity_weighted / 60,
            ),
        )
    )

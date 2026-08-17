"""Training history plots (matplotlib imported lazily so import stays cheap)."""
from __future__ import annotations


def plot_training_history(history, columns=None, ax=None):
    """Plot training-history curves from a history DataFrame.

    ``columns`` selects which metric columns to draw (default: all numeric).
    Returns the matplotlib Axes so callers can further annotate / save it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    cols = columns or [c for c in history.columns
                       if history[c].dtype.kind in "fi"]
    for col in cols:
        ax.plot(history.index, history[col], label=col)
    ax.set_xlabel("iteration")
    ax.legend(loc="best", fontsize="small")
    return ax

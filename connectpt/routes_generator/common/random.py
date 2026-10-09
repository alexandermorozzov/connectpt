"""Explicit seeding shared by run setup and seeded library operations."""
import random
import numpy as np
import torch


def seed_everything(seed: int | None) -> None:
    """Seed torch / numpy / random (no-op if ``seed`` is None)."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

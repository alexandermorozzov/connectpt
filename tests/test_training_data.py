"""TrainingDataModule split/curriculum reproduce the notebook logic exactly."""
import torch

from connectpt.routes_generator.training import TrainingDataModule


def _reference_notebook_split(tier_of, tiers, train_fraction, n_val_per_tier, seed):
    """Verbatim re-implementation of the paper_combined cell-9 stratified split."""
    n = len(tier_of)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    by_tier = {t: [] for t in tiers}
    for gi in perm.tolist():
        by_tier[tier_of[gi]].append(gi)
    train_l, val_l = [], []
    for tier in tiers:
        idxs = by_tier[tier]
        n_val = max(n_val_per_tier, int(round((1 - train_fraction) * len(idxs))))
        n_val = min(n_val, max(0, len(idxs) - 1))
        val_l += idxs[:n_val]
        train_l += idxs[n_val:]
    return torch.tensor(train_l), torch.tensor(val_l)


def _make_module(n=40):
    dm = TrainingDataModule(raw_graphs_path="x", lc_results_dir="y")
    dm.graphs = list(range(n))  # only len() is used by the split helpers
    return dm


def test_stratified_split_matches_notebook():
    tiers = ["a", "b", "c", "d"]
    tier_of = {i: tiers[i % 4] for i in range(40)}
    dm = _make_module(40)

    train, val, monitor, train_by_tier, val_by_tier = dm.stratified_split(
        tier_of, tiers, train_fraction=0.9, n_val_per_tier=4, seed=0
    )
    ref_train, ref_val = _reference_notebook_split(tier_of, tiers, 0.9, 4, 0)

    assert torch.equal(train, ref_train)
    assert torch.equal(val, ref_val)
    # every tier present in the monitor slice, capped at n_val_per_tier
    assert len(monitor) == 4 * 4
    assert sum(len(v) for v in val_by_tier.values()) == len(monitor)


def test_random_split_is_deterministic_and_sized():
    dm = _make_module(100)
    a1, b1 = dm.random_split(train_fraction=0.9, seed=0)
    a2, b2 = dm.random_split(train_fraction=0.9, seed=0)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert len(a1) == 90 and len(b1) == 10
    # different seed -> different order
    a3, _ = dm.random_split(train_fraction=0.9, seed=1)
    assert not torch.equal(a1, a3)


def test_build_curriculum_activates_tiers_by_iteration():
    train_by_tier = {t: torch.tensor([i]) for i, t in enumerate(["a", "b", "c"])}
    val_by_tier = {t: torch.tensor([10 + i]) for i, t in enumerate(["a", "b", "c"])}
    curriculum = [(5, ["a"], "s1"), (10, ["a", "b"], "s2"), (15, ["a", "b", "c"], "s3")]

    cf, vcf = TrainingDataModule.build_curriculum(curriculum, train_by_tier, val_by_tier)

    idx, label = cf(0)
    assert label == "s1" and idx.tolist() == [0]
    idx, label = cf(7)
    assert label == "s2" and idx.tolist() == [0, 1]
    idx, label = cf(99)  # past the schedule -> last stage
    assert label == "s3" and idx.tolist() == [0, 1, 2]
    assert vcf(7).tolist() == [10, 11]

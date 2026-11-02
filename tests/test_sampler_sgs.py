"""Success-guided sampling (arXiv 2610.12465) weights and per-step survival labels (CPU)."""

import torch

from supermimic.envs.mdp.sampler_math import cap_group_mass, sgs_weights, survival_labels

PAPER = dict(target=0.66, kappa=5.0, temperature=2.0, eps=1e-8)  # the paper's locomotion values


def _mix(hard: torch.Tensor, uniform_ratio: float = 0.5, hard_ratio: float = 0.4, memory_ratio: float = 0.1):
    """The reset distribution of commands._sample_frames with the memory term equal to the hard term."""
    n = hard.numel()
    p = uniform_ratio / n + (hard_ratio + memory_ratio) * hard
    return p / p.sum()


def test_weights_peak_at_the_target_and_fall_to_the_floor_at_0_and_1():
    s = torch.linspace(0, 1, 101, dtype=torch.float64)
    w = sgs_weights(s, **PAPER)
    assert abs(float(s[w.argmax()]) - 0.66) < 0.011
    assert (w[: int(s.numel() * 0.66)].diff() > 0).all()  # rises up to the mode
    assert (w[int(s.numel() * 0.66) + 1 :].diff() < 0).all()  # falls after it
    assert w[0] < 1e-3 * w.max() and w[-1] < 1e-3 * w.max()
    # Solved and hopeless bins both end at the eps floor: (eps)^(1/T) relative to the unnormalized peak.
    assert w[0] > 0 and w[-1] > 0


def test_medium_beats_easy_and_hard_and_ordering_is_paper_shaped():
    s = torch.tensor([0.0, 0.1, 0.5, 0.66, 0.9, 0.99, 1.0], dtype=torch.float64)
    w = sgs_weights(s, **PAPER)
    assert w[3] == w.max()
    assert w[2] > w[1] > w[0]
    assert w[4] > w[5] > w[6]


def test_weights_normalize_over_many_bins_including_extremes():
    g = torch.Generator().manual_seed(0)
    s = torch.rand(200_000, generator=g)
    s[:1000] = 0.0
    s[1000:2000] = 1.0
    w = sgs_weights(s, **PAPER)
    assert torch.isfinite(w).all() and (w > 0).all()
    assert abs(float(w.double().sum()) - 1.0) < 1e-5
    # All-equal success rates give the uniform distribution.
    u = sgs_weights(torch.full((1000,), 1.0), **PAPER)
    assert torch.allclose(u, torch.full((1000,), 1e-3), rtol=1e-5)


def test_uniform_floor_survives_a_collapsed_frontier():
    # Late-training loco: almost every bin solved, a few hundred at the frontier.
    n, k = 200_000, 400
    s = torch.ones(n, dtype=torch.float64)
    s[:k] = 0.66
    h = sgs_weights(s, **PAPER)
    # A frontier bin outweighs a solved one by ~2000x, so k frontier bins hold k*2000 / (k*2000 + n) of the SGS mass.
    ratio = float(h[0] / h[-1])
    assert 1500 < ratio < 2500
    assert abs(float(h[:k].sum()) - k * ratio / (k * ratio + n - k)) < 1e-9
    assert float(h[:k].sum()) > 0.75
    p = _mix(h)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert float(p.min()) >= 0.5 / n - 1e-15  # every bin keeps its uniform share
    assert float(p[k:].sum()) > 0.5 - 1e-9


def test_cap_still_spreads_a_one_clip_frontier():
    # 2,000 clips x 5 bins; only clip 0 is at the frontier: the cap keeps it at 1% of the SGS mass.
    group = torch.arange(2000).repeat_interleave(5)
    s = torch.ones(group.numel(), dtype=torch.float64)
    s[:5] = 0.66
    h, before, after = cap_group_mass(sgs_weights(s, **PAPER), group, 2000, 0.01)
    assert before.max() > 0.4
    assert after.max() <= 0.01 + 1e-9
    assert abs(float(h.sum()) - 1.0) < 1e-9


def test_survival_labels():
    T, H = 12, 3
    lens = torch.tensor([10, 10, 10, 4])
    terminated = torch.tensor([True, False, False, True])
    censored = torch.tensor([False, False, True, False])  # row 1 reached the clip end, row 2 hit the time limit
    s, v = survival_labels(lens, terminated, censored, H, T)
    # Failure at the end of row 0: the last H steps fail, earlier steps succeed.
    assert s[0, :7].eq(1).all() and s[0, 7:10].eq(0).all()
    assert v[0, :10].all() and not v[0, 10:].any()
    # Clip end: every step succeeds, even the last ones.
    assert s[1, :10].eq(1).all() and v[1, :10].all()
    # Time limit: the last H-1 steps have no outcome yet.
    assert v[2, :8].all() and not v[2, 8:].any()
    # Short failing episode: all its steps are within H of the failure except the first.
    assert s[3, :4].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_eps_sets_the_floor_inside_the_sgs_mass():
    s = torch.tensor([0.66, 1.0], dtype=torch.float64)
    for eps, lo, hi in ((1e-8, 3e-4, 7e-4), (1e-4, 3e-2, 7e-2)):
        w = sgs_weights(s, target=0.66, kappa=5.0, temperature=2.0, eps=eps)
        assert lo < float(w[1] / w[0]) < hi

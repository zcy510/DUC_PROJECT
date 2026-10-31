"""
T0 correctness tests for Phase 2 (run with:  pytest -v tests/test_dp.py).

These check that the DP mechanisms do what the theory says, BEFORE we trust
any experiment results:
  - central DP: noise has the right scale (Delta / eps_i) and Laplace shape,
    the budget is split correctly, contribution bounding really limits users
  - local DP:   k-RR probabilities are right, and the estimator is unbiased

They use PipelineDP's LocalBackend (plain Python, no Spark) so they run fast;
the DP logic is identical to the Spark backend.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pipeline_dp
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import central_dp  # noqa: E402
import local_dp  # noqa: E402

CATS = ["a", "b", "c", "d", "e"]


def run_local(rows, eps, **bounds):
    """Run the central-DP computation on PipelineDP's LocalBackend."""
    lazy, _, finalize = central_dp.dp_aggregate(rows, pipeline_dp.LocalBackend(), eps,
                                                categories=CATS, **bounds)
    finalize()
    return central_dp.to_frames(list(lazy["per_category"]), list(lazy["per_category_hour"]))


# ---------------------------------------------------------------- central DP

def test_budget_split_and_noise_scale_follow_theory():
    """Total eps is split: aggregation 1 (Q1+Q2) and aggregation 2 (Q3) get
    eps/2 each; inside aggregation 1, COUNT and PRIVACY_ID_COUNT get eps/4.
    Laplace std = sqrt(2) * Delta / eps_i, with Delta = L0 * Linf."""
    rows = [(u, CATS[u % 5], u % 24) for u in range(100)]
    eps, l0, linf, l0h, linfh = 2.0, 3, 5, 7, 2
    out = run_local(rows, eps, l0=l0, linf=linf, l0_hour=l0h, linf_hour=linfh)

    expect_q1 = math.sqrt(2) * (l0 * linf) / (eps / 4)
    expect_q2 = math.sqrt(2) * l0 / (eps / 4)          # each user counts once per category
    expect_q3 = math.sqrt(2) * (l0h * linfh) / (eps / 2)
    assert out["events_per_category"]["events_noise_std"].iloc[0] == pytest.approx(expect_q1)
    assert out["users_per_category"]["users_noise_std"].iloc[0] == pytest.approx(expect_q2)
    assert out["events_per_category_hour"]["noise_std"].iloc[0] == pytest.approx(expect_q3)


def test_noise_is_laplace_with_expected_scale():
    """Run the same release many times; the noise (DP - true) must be
    centred at 0 with a Laplace distribution of scale b = Delta/eps_i."""
    rows = [(u, "a", 0) for u in range(50)]          # true count for 'a' = 50
    eps, l0, linf = 4.0, 1, 1
    b = (l0 * linf) / (eps / 4)                      # Laplace scale for Q1
    noise = np.array([run_local(rows, eps, l0=l0, linf=linf, l0_hour=1, linf_hour=1)
                      ["events_per_category"].set_index("category").loc["a", "n_events"] - 50
                      for _ in range(400)])
    assert abs(noise.mean()) < 4 * noise.std() / math.sqrt(len(noise))  # unbiased
    assert noise.std() == pytest.approx(math.sqrt(2) * b, rel=0.15)
    assert stats.kstest(noise, "laplace", args=(0, b)).pvalue > 0.001  # Laplace shape


def test_contribution_bounding_limits_a_power_user():
    """One user with 1,000 'a' events must count as at most Linf events."""
    rows = [(0, "a", 0)] * 1000
    out = run_local(rows, 1e4, l0=1, linf=10, l0_hour=1, linf_hour=1)  # huge eps ~ no noise
    dp_count = out["events_per_category"].set_index("category").loc["a", "n_events"]
    assert dp_count == pytest.approx(10, abs=0.5)


def test_cross_partition_bounding_limits_categories():
    """One user active in 5 categories, L0=2 -> only 2 categories keep data."""
    rows = [(0, c, 0) for c in CATS for _ in range(3)]
    out = run_local(rows, 1e4, l0=2, linf=3, l0_hour=1, linf_hour=1)
    counts = out["events_per_category"]["n_events"].round()
    assert (counts > 0.5).sum() == 2 and counts.sum() == pytest.approx(6, abs=1)


def test_all_public_partitions_are_released():
    """Categories nobody used still appear (with pure noise), so the
    presence of a category in the output leaks nothing."""
    rows = [(u, "a", 0) for u in range(20)]
    out = run_local(rows, 1.0)
    assert sorted(out["events_per_category"]["category"]) == CATS
    assert len(out["events_per_category_hour"]) == len(CATS) * 24


# ---------------------------------------------------------------- local DP

@pytest.mark.parametrize("eps,k", [(0.5, 50), (1.0, 50), (4.0, 5)])
def test_krr_probabilities(eps, k):
    p, q = local_dp.krr_probabilities(eps, k)
    assert p + (k - 1) * q == pytest.approx(1.0)
    assert p / q == pytest.approx(math.exp(eps))  # the eps-LDP guarantee itself


def test_krr_keeps_true_value_with_probability_p():
    rng = np.random.default_rng(1)
    eps, k, n = 1.0, 50, 200_000
    values = rng.integers(0, k, n)
    noisy = local_dp.krr_perturb(values, eps, k, rng)
    p, _ = local_dp.krr_probabilities(eps, k)
    assert np.mean(noisy == values) == pytest.approx(p, abs=0.003)
    assert noisy.min() >= 0 and noisy.max() < k


def test_krr_estimator_is_unbiased():
    """Average of many estimates == true counts (within statistical error)."""
    rng = np.random.default_rng(2)
    eps, k = 1.0, 5
    true = np.array([5000, 2000, 1000, 500, 100])
    values = np.repeat(np.arange(k), true)
    est = np.array([local_dp.krr_estimate(np.bincount(local_dp.krr_perturb(values, eps, k, rng),
                                                      minlength=k), eps, k)
                    for _ in range(300)])
    se = est.std(axis=0) / math.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - true) < 4 * se)


def test_devices_send_at_most_m_reports():
    """User-level privacy: no user sends more than m reports."""
    events = pd.DataFrame({"user_id": [0] * 100 + [1] * 2 + [2],
                           "category": ["messaging"] * 103})
    for m in (1, 3):
        reports, true_counts = local_dp.simulate_devices(events, 1.0, m, seed=0)
        assert len(reports) == min(100, m) + min(2, m) + 1
        assert true_counts.sum() == len(reports)

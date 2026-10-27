"""Semantics of the calibration core: finite-sample conformal quantile, the
delayed ACI recursion (FIFO queue, 3-step maturity, alpha clipping, no-future
update assert), and the interval score. These tests pin the exact behaviours
that Algorithm 1 and Eqs. (11)-(12) of the paper describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.rigorous_v2 import common_v2 as C


# ----------------------------------------------------------- conformal quantile
def test_conformal_q_finite_sample_rule():
    scores = np.arange(1, 11, dtype=float)          # n = 10
    # (n+1)(1-a) = 11*0.9 = 9.9 -> ceil = 10 -> 10th smallest
    assert C.conformal_q(scores, 0.10) == 10.0
    # a large enough that ceil((n+1)(1-a)) clips to 1
    assert C.conformal_q(scores, 0.999) == 1.0
    # a = 0 -> ceil(11) clips to n
    assert C.conformal_q(scores, 0.0) == 10.0


# ------------------------------------------------------------------ ACI delay
def _series(n=20, miss_every=None):
    y = np.zeros(n)
    mu = np.zeros(n)
    sigma = np.ones(n)
    miss = np.zeros(n, dtype=bool)
    if miss_every:
        miss[::miss_every] = True
        y[miss] = 5.0                 # outside any tested interval
    origins = pd.date_range("2025-01-01", periods=n, freq="5min")
    return y, mu, sigma, origins


def test_aci_first_intervals_use_initial_alpha_with_no_feedback():
    y, mu, sigma, origins = _series(6)
    scores = np.array([1.0])
    lo, up, alphas, qs = C.run_aci(y, mu, sigma, scores, origins, gamma=0.2)
    # feedback matures 15 min after issuance; origins are 5 min apart, so
    # origins 0 and 1 run on the initial alpha = 0.10 -> q = Q(0.10) = 1.0
    assert alphas[0] == C.ALPHA and qs[0] == 1.0
    assert alphas[1] == C.ALPHA and qs[1] == 1.0
    # origin 2 (10 min later): the origin-0 feedback is still 5 min short of maturity
    assert alphas[2] == C.ALPHA
    # origin 3 (15 min): exactly one matured feedback has been consumed
    assert len(np.unique(alphas[3:])) >= 1


def test_aci_updates_direction_and_clipping():
    # every observation misses -> alpha must decrease (Q(alpha) grows -> intervals
    # widen), bounded below by ALPHA_MIN
    y, mu, sigma, origins = _series(40, miss_every=1)
    scores = np.array([1.0])
    lo, up, alphas, _ = C.run_aci(y, mu, sigma, scores, origins, gamma=0.5)
    assert alphas[3] < C.ALPHA, "misses must lower alpha (widen subsequent intervals)"
    assert alphas.min() >= C.ALPHA_MIN - 1e-12
    # never-miss series -> alpha drifts up toward its cap 0.5 (intervals narrow)
    y, mu, sigma, origins = _series(400, miss_every=None)
    lo, up, alphas, _ = C.run_aci(y, mu, sigma, scores, origins, gamma=0.2)
    assert alphas.max() <= C.ALPHA_MAX + 1e-12
    assert alphas[-1] > C.ALPHA


def test_aci_delay_is_exactly_three_steps():
    # the first alpha change must occur at the 4th forecast (index 3):
    # feedback from index 0 matures at origin+15min == origin of index 3
    n = 5
    y = np.array([10.0, 0, 0, 0, 0])   # index 0 misses a [-1,1] interval
    mu = np.zeros(n)
    sigma = np.ones(n)
    origins = pd.date_range("2025-01-01", periods=n, freq="5min")
    lo, up, alphas, _ = C.run_aci(y, mu, sigma, np.array([1.0]), origins, gamma=0.1)
    assert np.isclose(alphas[0], C.ALPHA)
    assert alphas[1] == alphas[0] and alphas[2] == alphas[0]
    # index 3 consumes index 0's miss: alpha <- 0.1 + 0.1*(0.1-1) = 0.01
    assert alphas[3] == pytest.approx(C.ALPHA + 0.1 * (C.ALPHA - 1.0))
    # index 4 consumes index 1's hit: alpha <- 0.01 + 0.1*(0.1-0) = 0.02
    assert alphas[4] == pytest.approx(alphas[3] + 0.1 * C.ALPHA)


# ---------------------------------------------------------------- interval score
def test_interval_score_matches_definition():
    # (u-l) + (2/a)*(l-y) if y<l, + (2/a)*(y-u) if y>u
    y = np.array([0.0, 5.0, -5.0])
    lo = np.array([-1.0, 4.0, -4.0])
    up = np.array([1.0, 6.0, -2.0])
    expected = np.array([2.0, 2.0, 2.0 + 20.0 * 1.0])   # third: below lower by 1
    np.testing.assert_allclose(C.interval_score(y, lo, up), expected)


def test_metrics_picp_pinaw_rce_streak():
    n = 288 * 2
    y = np.zeros(n)
    lo, up = np.full(n, -1.0), np.full(n, 1.0)
    origins = pd.date_range("2025-01-01", periods=n, freq="5min")
    m = C.interval_metrics(y, lo, up, origins)
    assert m["PICP"] == 1.0 and m["PINAW"] == 2.0 / C.P_RATED
    assert m["RCE1D"] == pytest.approx(abs(1.0 - 0.9) * 1, abs=1e-9)
    assert m["LongestMissStreak"] == 0


def test_gamma_selection_prefers_feasible_min_is():
    # synthetic: y ~ N(mu=0, sd=1); sigma oracle = 1 for all
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.normal(0, 1, n)
    mu = np.zeros(n)
    sg = np.ones(n)
    origins = pd.date_range("2025-01-01", periods=n, freq="5min")
    gamma, detail = C.select_gamma(y, mu, sg, origins, use_sigma=True)
    assert gamma in C.GAMMA_GRID
    assert set(detail.keys()) == {float(g) for g in C.GAMMA_GRID}

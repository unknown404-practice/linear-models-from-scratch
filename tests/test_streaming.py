"""
Unit tests for Recursive Least Squares (RLS) and Kalman Filter streaming models.
"""

import numpy as np
import pytest
from src.streaming import RecursiveLeastSquares, KalmanFilterRegression
from src.solvers import ClosedFormLinearRegression


def test_rls_asymptotic_ols_equivalence():
    rng = np.random.default_rng(42)
    N, D = 500, 2
    X = rng.normal(0, 1, size=(N, D))
    true_w = np.array([3.2, -1.8])
    true_b = 2.5
    y = X @ true_w + true_b + rng.normal(0, 0.1, size=N)

    # 1. Closed-form batch OLS
    ols = ClosedFormLinearRegression(method="svd")
    ols.fit(X, y)

    # 2. RLS with infinite memory (lambda_ = 1.0)
    rls = RecursiveLeastSquares(lambda_=1.0, delta=1000.0)
    rls.fit(X, y)

    # RLS should asymptotically match batch OLS
    np.testing.assert_allclose(rls.weights, ols.weights, atol=0.05)
    assert abs(rls.bias - ols.bias) < 0.05
    assert rls.score(X, y) > 0.95


def test_rls_forgetting_factor_regime_shift():
    rng = np.random.default_rng(42)
    N = 200
    X = rng.normal(0, 1, size=(N, 1))

    # Regime 1: slope +10.0 for first 100 samples
    y1 = 10.0 * X[:100, 0]
    # Regime 2: sudden shift to slope -10.0 for next 100 samples
    y2 = -10.0 * X[100:, 0]
    y = np.concatenate([y1, y2])

    # Model with exponential forgetting (lambda_ = 0.85)
    rls_adaptive = RecursiveLeastSquares(lambda_=0.85, delta=100.0, fit_intercept=False)
    rls_adaptive.fit(X, y)

    # Adaptive model should rapidly track the NEW slope (-10.0)
    assert rls_adaptive.weights[0] < -7.0


def test_rls_minibatch_partial_fit():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))
    y = 2.0 * X[:, 0] + X[:, 1] + 1.0

    # Sequential mini-batch streaming
    rls = RecursiveLeastSquares(lambda_=1.0)
    for i in range(0, 100, 25):
        rls.partial_fit(X[i : i + 25], y[i : i + 25])

    assert rls.n_samples_seen_ == 100
    np.testing.assert_allclose(rls.weights, [2.0, 1.0], atol=0.1)


def test_kalman_filter_regression_tracking():
    rng = np.random.default_rng(42)
    N = 300
    X = rng.normal(0, 1, size=(N, 2))
    y = 1.5 * X[:, 0] - 2.5 * X[:, 1] + 0.8 + rng.normal(0, 0.2, size=N)

    kf = KalmanFilterRegression(q=1e-4, r=1.0)
    kf.fit(X, y)

    assert kf.weights is not None
    np.testing.assert_allclose(kf.weights, [1.5, -2.5], atol=0.2)
    assert kf.score(X, y) > 0.90

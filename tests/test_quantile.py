"""
Unit tests for QuantileRegressorScratch from scratch.
"""

import numpy as np
import pytest
from src.quantile import QuantileRegressorScratch


def test_quantile_median_regression():
    rng = np.random.default_rng(42)
    N = 300
    X = rng.normal(0, 1, size=(N, 2))
    true_w = np.array([2.0, -1.0])
    true_b = 3.0
    # Add Laplace noise (median is 0)
    y = X @ true_w + true_b + rng.laplace(0, 0.5, size=N)

    qreg = QuantileRegressorScratch(quantile=0.5, max_iter=200)
    qreg.fit(X, y)

    assert qreg.weights is not None
    np.testing.assert_allclose(qreg.weights, true_w, atol=0.25)
    assert abs(qreg.bias - true_b) < 0.25

    # Loss must be lower at end than beginning
    assert qreg.loss_history_[-1] <= qreg.loss_history_[0]


def test_quantile_monotonicity_corridor():
    rng = np.random.default_rng(42)
    N = 400
    X = rng.uniform(-2, 2, size=(N, 1))
    y = 2.0 * X[:, 0] + rng.normal(0, 1.0, size=N)

    # Fit 10th, 50th, and 90th percentiles
    q10 = QuantileRegressorScratch(quantile=0.10).fit(X, y)
    q50 = QuantileRegressorScratch(quantile=0.50).fit(X, y)
    q90 = QuantileRegressorScratch(quantile=0.90).fit(X, y)

    X_test = np.linspace(-2, 2, 50).reshape(-1, 1)
    y_10 = q10.predict(X_test)
    y_50 = q50.predict(X_test)
    y_90 = q90.predict(X_test)

    # Monotonicity ordering for quantile predictions
    assert np.mean(y_10) < np.mean(y_50) < np.mean(y_90)
    assert np.all(y_10 <= y_90 + 1e-4)


def test_quantile_residual_fraction():
    rng = np.random.default_rng(42)
    N = 500
    X = rng.normal(0, 1, size=(N, 1))
    y = 1.5 * X[:, 0] + rng.normal(0, 1.0, size=N)

    tau = 0.25
    qreg = QuantileRegressorScratch(quantile=tau, max_iter=200).fit(X, y)
    residuals = y - qreg.predict(X)

    # Fraction of points below the quantile estimate should be close to tau
    fraction_below = float(np.mean(residuals < 0.0))
    assert abs(fraction_below - tau) < 0.08

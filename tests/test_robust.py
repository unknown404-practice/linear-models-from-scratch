"""
Unit tests for Robust Regression Suite: HuberRegressorScratch and RANSACRegressorScratch.
"""

import numpy as np
import pytest

from src.robust import HuberRegressorScratch, RANSACRegressorScratch
from src.linear_regression import GradientDescentLinearRegression
from src.solvers import ClosedFormLinearRegression


def test_huber_loss_piecewise_gradient():
    """Verify analytical Huber derivative behavior for inliers and outliers."""
    delta = 1.35
    errors = np.array([-3.0, -1.0, 0.0, 0.5, 2.5])
    # Expected psi(e):
    # -3.0: -1.35
    # -1.0: -1.0
    #  0.0:  0.0
    #  0.5:  0.5
    #  2.5:  1.35
    expected_psi = np.array([-1.35, -1.0, 0.0, 0.5, 1.35])
    psi = np.where(np.abs(errors) <= delta, errors, delta * np.sign(errors))
    np.testing.assert_allclose(psi, expected_psi, rtol=1e-7)


def test_huber_regressor_outlier_resilience():
    """Verify HuberRegressorScratch is immune to severe target outliers that collapse OLS."""
    rng = np.random.default_rng(42)
    n_samples = 300
    X = rng.uniform(-3, 3, size=(n_samples, 2))
    true_w = np.array([3.0, -2.0])
    true_b = 1.5
    y = X @ true_w + true_b + rng.normal(0, 0.2, n_samples)

    # Corrupt 20% of samples with massive +50.0 noise (extreme outliers)
    n_outliers = int(0.20 * n_samples)
    outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
    y_corrupted = y.copy()
    y_corrupted[outlier_idx] += 50.0

    # 1. Standard OLS fails on corrupted data
    ols = GradientDescentLinearRegression(lr=0.01, n_iters=800, seed=42)
    ols.fit(X, y_corrupted)
    r2_ols = ols.score(X, y)  # evaluate on ground truth y

    # 2. Huber regressor bounds outlier gradients
    huber = HuberRegressorScratch(epsilon=1.35, lr=0.03, n_iters=1000, seed=42)
    huber.fit(X, y_corrupted)
    r2_huber = huber.score(X, y)

    # Huber must drastically outperform OLS under heavy-tailed contamination
    assert r2_huber > 0.85
    assert r2_huber > r2_ols + 0.40
    assert np.allclose(huber.weights, true_w, atol=0.5)


def test_ransac_exact_inlier_recovery():
    """Verify RANSACRegressorScratch identifies inlier mask amidst 30% corrupted points."""
    rng = np.random.default_rng(101)
    n_samples = 200
    X = rng.uniform(-5, 5, size=(n_samples, 1))
    true_slope = 2.5
    true_intercept = 1.0
    y = true_slope * X[:, 0] + true_intercept + rng.normal(0, 0.1, n_samples)

    # Corrupt 30% of points with leverage anomalies
    n_outliers = int(0.30 * n_samples)
    outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
    y[outlier_idx] = rng.uniform(-20, 20, size=n_outliers)

    ransac = RANSACRegressorScratch(
        min_samples=5,
        residual_threshold=0.5,
        max_trials=100,
        random_state=42,
    )
    ransac.fit(X, y)

    # Inlier recovery accuracy
    ground_truth_inliers = np.ones(n_samples, dtype=bool)
    ground_truth_inliers[outlier_idx] = False

    detected_inliers = ransac.inlier_mask_
    # Overlap between detected and true inliers
    accuracy = np.mean(detected_inliers == ground_truth_inliers)
    assert accuracy > 0.90

    # Recovered model slope must be very accurate
    fitted_estimator = ransac.estimator_
    assert np.isclose(fitted_estimator.weights[0], true_slope, atol=0.15)
    assert np.isclose(fitted_estimator.bias, true_intercept, atol=0.2)


def test_ransac_custom_estimator():
    """Verify RANSACRegressorScratch works with custom estimators and has predict/score."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.5 + rng.normal(0, 0.05, 100)

    # Inject 10 outliers
    y[:10] += 30.0

    custom_est = ClosedFormLinearRegression(method="svd")
    ransac = RANSACRegressorScratch(
        estimator=custom_est,
        min_samples=10,
        residual_threshold=0.3,
        random_state=42,
    )
    ransac.fit(X, y)

    y_pred = ransac.predict(X[10:])
    score = ransac.score(X[10:], y[10:])
    assert score > 0.95

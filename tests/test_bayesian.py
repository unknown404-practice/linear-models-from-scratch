"""
Unit tests for BayesianLinearRegression from scratch.
"""

import numpy as np
import pytest
from src.bayesian import BayesianLinearRegression


def test_bayesian_linear_regression_recovery():
    rng = np.random.default_rng(42)
    N, D = 500, 2
    X = rng.normal(0, 1, size=(N, D))
    true_w = np.array([2.5, -3.0])
    true_b = 1.5
    y = X @ true_w + true_b + rng.normal(0, 0.2, size=N)

    model = BayesianLinearRegression(alpha=1.0, beta=25.0, fit_intercept=True)
    model.fit(X, y)

    assert model.weights is not None
    assert model.weights.shape == (2,)
    np.testing.assert_allclose(model.weights, true_w, atol=0.2)
    assert abs(model.bias - true_b) < 0.2

    # Check covariance is symmetric positive definite
    assert model.cov_.shape == (2, 2)
    np.testing.assert_allclose(model.cov_, model.cov_.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(model.cov_)
    assert np.all(eigvals > 0)

    # R^2 score
    assert model.score(X, y) > 0.95


def test_bayesian_epistemic_uncertainty_extrapolation():
    rng = np.random.default_rng(42)
    N = 100
    X = rng.uniform(-1.0, 1.0, size=(N, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0, 0.1, size=N)

    model = BayesianLinearRegression(alpha=1.0, beta=100.0)
    model.fit(X, y)

    # Inlier near center vs far extrapolation point
    X_inlier = np.array([[0.0, 0.0]])
    X_outlier = np.array([[8.0, 8.0]])

    _, std_in = model.predict(X_inlier, return_std=True)
    _, std_out = model.predict(X_outlier, return_std=True)

    # Epistemic uncertainty must flare under extrapolation
    assert float(std_out[0]) > 2.0 * float(std_in[0])


def test_bayesian_ard_feature_pruning():
    rng = np.random.default_rng(42)
    N = 300
    X = rng.normal(0, 1, size=(N, 3))
    # Feature 0 and 1 are predictive, Feature 2 is pure uninformative noise
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.0 * X[:, 2] + rng.normal(0, 0.1, size=N)

    model = BayesianLinearRegression(alpha=1.0, beta=1.0, max_iter=200)
    model.fit_ard(X, y)

    assert len(model.scores_) > 0
    assert abs(model.weights[2]) < 0.1
    assert abs(model.weights[0] - 3.0) < 0.2
    assert abs(model.weights[1] - (-2.0)) < 0.2


def test_bayesian_posterior_sample_weights():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))
    y = 1.5 * X[:, 0] + 0.5 * X[:, 1] + 1.0

    model = BayesianLinearRegression(alpha=1.0, beta=10.0)
    model.fit(X, y)

    samples = model.sample_weights(n_samples=50, seed=42)
    assert samples.shape == (50, 2)
    sample_mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(sample_mean, model.weights, atol=0.2)

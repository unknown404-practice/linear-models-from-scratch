"""
Unit Tests for Regularization (Ridge L2, Lasso L1, ElasticNet) and Optimizers Integration.
"""

import numpy as np
import pytest
from src.linear_regression import (
    GradientDescentLinearRegression,
    soft_threshold,
    compute_mse,
)
from src.schedulers import CosineAnnealingLR


def test_soft_threshold_operator():
    """Verify mathematical properties of the soft-thresholding operator."""
    # Values inside [-gamma, gamma] shrink to exact zero
    assert soft_threshold(0.5, gamma=1.0) == 0.0
    assert soft_threshold(-0.8, gamma=1.0) == 0.0

    # Values outside [-gamma, gamma] shrink by gamma towards zero
    assert pytest.approx(soft_threshold(2.5, gamma=1.0)) == 1.5
    assert pytest.approx(soft_threshold(-3.0, gamma=1.0)) == -2.0


def test_adam_optimizer_linear_regression():
    """Verify Linear Regression trains accurately using Adam."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((800, 5))
    true_w = np.array([2.0, -1.5, 0.8, -0.5, 1.2])
    y = X @ true_w + 3.0 + rng.normal(0, 0.05, 800)

    model = GradientDescentLinearRegression(lr=0.05, n_iters=350, optimizer="adam", seed=42)
    model.fit(X, y)

    assert model.score(X, y) > 0.99
    assert np.allclose(model.weights, true_w, atol=0.1)
    assert np.isclose(model.bias, 3.0, atol=0.1)


def test_cosine_scheduler_integration():
    """Verify learning rate scheduler integration with gradient descent."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 4))
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + 1.0

    scheduler = CosineAnnealingLR(initial_lr=0.05, T_max=200, eta_min=0.001)
    model = GradientDescentLinearRegression(
        lr=0.05,
        n_iters=200,
        scheduler=scheduler,
        seed=42,
    )
    _, history = model.fit_with_history(X, y)

    assert len(history) == 200
    assert history[-1] < history[0] * 0.05
    assert model.score(X, y) > 0.98


def test_ridge_l2_weight_shrinkage():
    """Verify Ridge L2 penalty shrinks weights monotonically as alpha increases."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 10
    X = rng.standard_normal((n_samples, n_features))
    y = X @ np.ones(n_features) + rng.normal(0, 0.1, n_samples)

    # Train OLS (alpha=0)
    m_ols = GradientDescentLinearRegression(lr=0.01, n_iters=500, penalty="none", seed=42)
    m_ols.fit(X, y)

    # Train Ridge with moderate penalty
    m_ridge_mod = GradientDescentLinearRegression(
        lr=0.01, n_iters=500, penalty="l2", alpha=1.0, seed=42
    )
    m_ridge_mod.fit(X, y)

    # Train Ridge with heavy penalty
    m_ridge_heavy = GradientDescentLinearRegression(
        lr=0.01, n_iters=500, penalty="l2", alpha=10.0, seed=42
    )
    m_ridge_heavy.fit(X, y)

    norm_ols = float(np.linalg.norm(m_ols.weights))
    norm_mod = float(np.linalg.norm(m_ridge_mod.weights))
    norm_heavy = float(np.linalg.norm(m_ridge_heavy.weights))

    # Weight norms must shrink monotonically: norm_ols > norm_mod > norm_heavy
    assert norm_ols > norm_mod > norm_heavy


def test_lasso_coordinate_descent_exact_sparsity():
    """Verify Lasso coordinate descent sets non-predictive features to EXACT zero."""
    rng = np.random.default_rng(101)
    n_samples, n_features = 300, 10
    X = rng.standard_normal((n_samples, n_features))
    # Only features 0, 1, 2 are signal. Features 3..9 are pure noise
    y = 4.0 * X[:, 0] - 3.0 * X[:, 1] + 2.0 * X[:, 2] + 1.5 + rng.normal(0, 0.1, n_samples)

    model = GradientDescentLinearRegression(penalty="l1", alpha=0.15, n_iters=400, seed=42)
    model.fit_coordinate_descent(X, y)

    # Features 3 to 9 must be precisely 0.0
    zero_weights = model.weights[3:]
    assert np.all(zero_weights == 0.0), f"Expected exact zeros, got {zero_weights}"

    # Predictive features should remain non-zero
    assert np.abs(model.weights[0]) > 3.0
    assert np.abs(model.weights[1]) > 2.0
    assert np.abs(model.weights[2]) > 1.0
    assert model.score(X, y) > 0.95


def test_elasticnet_coordinate_descent_sparsity_and_shrinkage():
    """Verify ElasticNet coordinate descent balances L1 sparsity and L2 shrinkage."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 250, 12
    X = rng.standard_normal((n_samples, n_features))
    # Features 0, 1 are strong signal; others are noise
    y = 3.5 * X[:, 0] - 2.5 * X[:, 1] + 1.0 + rng.normal(0, 0.1, n_samples)

    enet = GradientDescentLinearRegression(
        penalty="elasticnet", alpha=0.1, l1_ratio=0.5, n_iters=300, seed=42
    )
    enet.fit_coordinate_descent(X, y)

    # Must be sparse on noise features (at least half of noise features exactly 0)
    noise_zeros = np.sum(enet.weights[2:] == 0.0)
    assert noise_zeros >= 5, f"Expected sparse noise weights, got {noise_zeros} zeros"

    # Signal features should be well-recovered
    assert enet.weights[0] > 2.5
    assert enet.weights[1] < -1.8
    assert enet.score(X, y) > 0.95


def test_elasticnet_extremes():
    """Verify ElasticNet with l1_ratio=1.0 matches pure Lasso and l1_ratio=0.0 matches coordinate Ridge."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((150, 6))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0]) + rng.normal(0, 0.1, 150)

    # 1. Pure Lasso comparison
    lasso = GradientDescentLinearRegression(penalty="l1", alpha=0.08, n_iters=300, seed=42)
    lasso.fit_coordinate_descent(X, y)

    enet_lasso = GradientDescentLinearRegression(
        penalty="elasticnet", alpha=0.08, l1_ratio=1.0, n_iters=300, seed=42
    )
    enet_lasso.fit_coordinate_descent(X, y)

    np.testing.assert_allclose(lasso.weights, enet_lasso.weights, atol=1e-5)
    np.testing.assert_allclose(lasso.bias, enet_lasso.bias, atol=1e-5)

    # 2. Pure Ridge comparison
    ridge_cd = GradientDescentLinearRegression(
        penalty="elasticnet", alpha=0.5, l1_ratio=0.0, n_iters=300, seed=42
    )
    ridge_cd.fit_coordinate_descent(X, y)

    # None of the weights should be exactly zero under pure Ridge
    assert not np.any(ridge_cd.weights == 0.0)

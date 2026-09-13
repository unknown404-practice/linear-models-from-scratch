"""
Unit tests for Second-Order Newton-Raphson Optimization: NewtonLinearRegression.
"""

import numpy as np
import pytest

from src.solvers import NewtonLinearRegression, ClosedFormLinearRegression


def test_newton_raphson_exact_one_step_convergence():
    """Verify Newton-Raphson achieves analytical global optimum in EXACTLY one step."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 5
    X = rng.standard_normal((n_samples, n_features))
    true_w = np.array([2.5, -1.8, 0.9, -0.4, 1.1])
    true_b = 3.2
    y = X @ true_w + true_b + rng.normal(0, 0.05, n_samples)

    newton = NewtonLinearRegression(damping=0.0, tol=1e-10)
    newton.fit(X, y)

    # Fundamental theorem: on a purely quadratic convex loss bowl,
    # Newton's method converges in exactly 1 iteration!
    assert newton.n_iter_ == 1, f"Expected 1 Newton step, took {newton.n_iter_}"

    # Gradient norm at the minimum should be near machine epsilon
    assert newton.grad_norm_ < 1e-8

    # Compare against SVD closed-form tri-solver
    svd = ClosedFormLinearRegression(method="svd")
    svd.fit(X, y)

    np.testing.assert_allclose(newton.weights, svd.weights, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(newton.bias, svd.bias, rtol=1e-6, atol=1e-6)
    assert newton.score(X, y) > 0.98


def test_newton_raphson_regularized():
    """Verify damped Newton-Raphson solves ill-conditioned collinear systems stably."""
    rng = np.random.default_rng(101)
    X_base = rng.standard_normal((150, 4))
    # Inject exact collinear duplicate column
    X_collinear = np.column_stack([X_base, X_base[:, 0] * 3.0])
    y = X_base @ np.array([1.0, -2.0, 0.5, 1.5]) + 2.0 + rng.normal(0, 0.1, 150)

    # Without damping or L2, (X^T X) would be singular; with damping, it solves smoothly
    newton_reg = NewtonLinearRegression(alpha=0.5, damping=1e-4)
    newton_reg.fit(X_collinear, y)

    assert newton_reg.weights is not None
    assert np.all(np.isfinite(newton_reg.weights))
    assert newton_reg.score(X_collinear, y) > 0.85


def test_newton_raphson_get_set_params():
    """Verify scikit-learn parameter inspection and modification interface."""
    model = NewtonLinearRegression(alpha=0.1, damping=1e-5, fit_intercept=True)
    params = model.get_params()

    assert params["alpha"] == 0.1
    assert params["damping"] == 1e-5
    assert params["fit_intercept"] is True

    model.set_params(alpha=2.5, damping=1e-3)
    assert model.alpha == 2.5
    assert model.damping == 1e-3

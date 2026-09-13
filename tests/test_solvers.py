"""
Unit and Numerical Stability Tests for Closed-Form Analytical Solvers.
"""

import numpy as np
import pytest
from src.solvers import ClosedFormLinearRegression


def test_closed_form_solvers_exact_recovery():
    """Verify SVD, Cholesky, QR, and Normal Equation all recover true parameters on clean data."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 4
    X = rng.standard_normal((n_samples, n_features))
    true_w = np.array([3.0, -1.5, 2.2, -0.7])
    true_b = 4.5
    y = X @ true_w + true_b

    methods = ["svd", "cholesky", "qr", "normal_equation"]
    for method in methods:
        model = ClosedFormLinearRegression(method=method)
        model.fit(X, y)
        assert np.allclose(model.weights, true_w, atol=1e-5), f"Failed for method: {method}"
        assert np.isclose(model.bias, true_b, atol=1e-5), f"Failed for method: {method}"
        assert model.score(X, y) > 0.9999


def test_svd_handles_rank_deficient_collinear_matrix():
    """Verify SVD pseudoinverse solver handles singular, perfectly collinear matrices where normal equation fails."""
    rng = np.random.default_rng(42)
    n_samples = 150
    x1 = rng.standard_normal(n_samples)
    x2 = 2.0 * x1  # Perfectly collinear!
    x3 = rng.standard_normal(n_samples)
    X = np.column_stack([x1, x2, x3])
    y = 3.0 * x1 + 1.5 * x3 + 2.0

    # Normal equation with naive inversion should fail or raise LinAlgError
    model_naive = ClosedFormLinearRegression(method="normal_equation")
    with pytest.raises((np.linalg.LinAlgError, ValueError)):
        model_naive.fit(X, y)

    # SVD pseudoinverse handles rank deficiency seamlessly
    model_svd = ClosedFormLinearRegression(method="svd")
    model_svd.fit(X, y)
    preds = model_svd.predict(X)
    assert model_svd.score(X, y) > 0.99
    assert model_svd.rank_ == 2  # Discovered true rank


def test_condition_number_computation():
    """Verify condition number accurately reflects matrix conditioning."""
    rng = np.random.default_rng(42)
    # Well conditioned orthogonal matrix
    Q, _ = np.linalg.qr(rng.standard_normal((100, 5)))
    model_well = ClosedFormLinearRegression().fit(Q, rng.standard_normal(100))
    assert model_well.condition_number_ < 5.0

    # Severely ill-conditioned matrix
    X_ill = rng.standard_normal((100, 5))
    X_ill[:, 4] = X_ill[:, 0] + 1e-8 * rng.standard_normal(100)
    model_ill = ClosedFormLinearRegression(method="svd").fit(X_ill, rng.standard_normal(100))
    assert model_ill.condition_number_ > 1e6

"""
Unit tests for ConformalLinearRegression from scratch.
"""

import numpy as np
import pytest
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression, NewtonLinearRegression


def test_conformal_coverage_guarantee():
    rng = np.random.default_rng(42)
    N = 1000
    X = rng.normal(0, 1, size=(N, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.laplace(0, 0.5, size=N)

    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]

    conf_model = ConformalLinearRegression(
        estimator=ClosedFormLinearRegression(method="svd"),
        confidence_level=0.90,
    )
    conf_model.fit(X_train, y_train, cal_size=0.25, random_state=42)

    assert conf_model.q_hat_ > 0.0
    coverage = conf_model.coverage_score(X_test, y_test)
    # Conformal theory guarantees empirical coverage >= 1 - alpha (up to finite sample error)
    assert coverage >= 0.87


def test_conformal_interval_properties():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))
    y = 1.0 * X[:, 0] + rng.normal(0, 0.1, size=100)

    model = ConformalLinearRegression(confidence_level=0.95)
    model.fit(X, y, cal_size=0.2)

    y_pred, y_lower, y_upper = model.predict(X, return_intervals=True)

    assert np.all(y_lower <= y_pred)
    assert np.all(y_pred <= y_upper)
    np.testing.assert_allclose(y_upper - y_lower, model.interval_width, atol=1e-8)


def test_conformal_custom_estimator():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.2, size=200)

    newton_base = NewtonLinearRegression(damping=1e-5)
    model = ConformalLinearRegression(estimator=newton_base, confidence_level=0.90)
    model.fit(X, y)

    y_pred, y_low, y_high = model.predict(X, return_intervals=True)
    assert len(y_pred) == 200
    assert model.score(X, y) > 0.90


def test_conformal_get_set_params_and_score():
    model = ConformalLinearRegression(confidence_level=0.90)
    params = model.get_params()
    assert params["confidence_level"] == 0.90

    model.set_params(confidence_level=0.99)
    assert model.confidence_level == 0.99

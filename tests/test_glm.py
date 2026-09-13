"""
Unit tests for Generalized Linear Models (GLM) from scratch.
"""

import numpy as np
import pytest
from src.glm import (
    GeneralizedLinearModel,
    LogisticRegressionScratch,
    PoissonRegressionScratch,
    GammaRegressionScratch,
)


def test_logistic_regression_binary_accuracy():
    rng = np.random.default_rng(42)
    N = 400
    X = rng.normal(0, 1, size=(N, 2))
    logits = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(np.float64)

    clf = LogisticRegressionScratch(max_iter=100)
    clf.fit(X, y)

    assert clf.weights is not None
    assert clf.weights.shape == (2,)
    # Accuracy should be high
    accuracy = clf.score(X, y)
    assert accuracy > 0.88

    # Check probabilities
    prob_preds = clf.predict_proba(X)
    assert prob_preds.shape == (N, 2)
    np.testing.assert_allclose(np.sum(prob_preds, axis=1), 1.0, atol=1e-8)
    assert np.all(prob_preds >= 0.0) and np.all(prob_preds <= 1.0)


def test_poisson_regression_counts():
    rng = np.random.default_rng(42)
    N = 500
    X = rng.normal(0, 0.5, size=(N, 2))
    true_eta = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.2
    mu = np.exp(true_eta)
    y = rng.poisson(mu).astype(np.float64)

    reg = PoissonRegressionScratch(max_iter=100)
    reg.fit(X, y)

    assert reg.weights is not None
    # Estimated coefficients should match true sign and magnitude
    np.testing.assert_allclose(reg.weights, [0.8, -0.5], atol=0.25)
    assert abs(reg.bias - 1.2) < 0.25

    # Predictions must be strictly positive
    preds = reg.predict(X)
    assert np.all(preds > 0.0)

    # Deviance must decrease
    assert reg.deviance_history_[-1] <= reg.deviance_history_[0]


def test_gamma_regression_positive_continuous():
    rng = np.random.default_rng(42)
    N = 400
    X = rng.normal(0, 0.5, size=(N, 2))
    true_eta = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 2.0
    mu = np.exp(true_eta)
    # Generate Gamma-distributed values with mean mu
    shape_param = 5.0
    y = rng.gamma(shape=shape_param, scale=mu / shape_param)

    reg = GammaRegressionScratch(max_iter=100)
    reg.fit(X, y)

    assert reg.weights is not None
    preds = reg.predict(X)
    assert np.all(preds > 0.0)
    assert reg.score(X, y) > 0.25


def test_glm_l2_regularization():
    rng = np.random.default_rng(42)
    N = 100
    x0 = rng.normal(0, 1, size=N)
    # Perfectly collinear columns
    X = np.column_stack([x0, x0, x0])
    y = 2.0 * x0 + 1.0

    # L2 regularization prevents singular matrix error in IRLS
    glm = GeneralizedLinearModel(family="gaussian", l2_reg=1.0)
    glm.fit(X, y)

    assert glm.weights is not None
    assert np.all(np.isfinite(glm.weights))
    assert np.isfinite(glm.bias)

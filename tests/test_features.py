"""
Unit tests for PolynomialFeaturesScratch transformer.
"""

import numpy as np
import pytest

from src.features import PolynomialFeaturesScratch
from src.pipeline import PipelineScratch
from src.linear_regression import GradientDescentLinearRegression


def test_polynomial_features_degree_2():
    X = np.array(
        [
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    poly = PolynomialFeaturesScratch(degree=2, include_bias=True, interaction_only=False)
    X_poly = poly.fit_transform(X)

    # Expected features: 1, x0, x1, x0^2, x0*x1, x1^2
    # Row 0: 1, 2, 3, 4, 6, 9
    # Row 1: 1, 4, 5, 16, 20, 25
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 4.0, 5.0, 16.0, 20.0, 25.0],
        ]
    )
    np.testing.assert_allclose(X_poly, expected, rtol=1e-7)

    feat_names = poly.get_feature_names_out(["a", "b"])
    assert feat_names == ["1", "a", "b", "a^2", "a b", "b^2"]


def test_polynomial_features_interaction_only():
    X = np.array(
        [
            [2.0, 3.0, 4.0],
            [1.0, 5.0, 2.0],
        ]
    )
    poly = PolynomialFeaturesScratch(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X)

    # Degree 1: x0, x1, x2
    # Degree 2 interaction: x0*x1, x0*x2, x1*x2
    # Row 0: 2, 3, 4, 6, 8, 12
    # Row 1: 1, 5, 2, 5, 2, 10
    expected = np.array(
        [
            [2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
            [1.0, 5.0, 2.0, 5.0, 2.0, 10.0],
        ]
    )
    np.testing.assert_allclose(X_poly, expected, rtol=1e-7)

    feat_names = poly.get_feature_names_out(["x", "y", "z"])
    assert feat_names == ["x", "y", "z", "x y", "x z", "y z"]


def test_polynomial_features_pipeline_integration():
    rng = np.random.RandomState(42)
    x = np.linspace(-3, 3, 100)
    y = 2.0 * (x**2) - 1.5 * x + 0.5 + rng.normal(0, 0.1, size=len(x))
    X = x.reshape(-1, 1)

    pipe = PipelineScratch(
        [
            ("poly", PolynomialFeaturesScratch(degree=2, include_bias=False)),
            ("regressor", GradientDescentLinearRegression(lr=0.01, n_iters=1000, seed=42)),
        ]
    )
    pipe.fit(X, y)
    r2 = pipe.score(X, y)
    assert r2 > 0.95

"""
Unit Tests for Model Persistence & JSON Serialization Engine (src/serialization.py).
"""

import numpy as np
import pytest

from src.serialization import save_model, load_model
from src.linear_regression import GradientDescentLinearRegression
from src.solvers import ClosedFormLinearRegression
from src.pipeline import PipelineScratch, StandardScalerScratch


def test_serialize_and_load_gradient_descent_model(tmp_path):
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + 1.2

    model = GradientDescentLinearRegression(lr=0.05, n_iters=200, seed=42)
    model.fit(X, y)

    filepath = tmp_path / "gd_model.json"
    save_model(model, filepath)

    assert filepath.exists()

    loaded_model = load_model(filepath)
    assert isinstance(loaded_model, GradientDescentLinearRegression)
    assert np.allclose(loaded_model.weights, model.weights)
    assert np.isclose(loaded_model.bias, model.bias)

    # Predictions match identically
    X_test = np.random.randn(20, 3)
    assert np.allclose(loaded_model.predict(X_test), model.predict(X_test))


def test_serialize_and_load_closed_form_model(tmp_path):
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = X @ np.array([1.0, 3.0, -2.0, 0.5]) + 0.8

    model = ClosedFormLinearRegression(method="svd")
    model.fit(X, y)

    filepath = tmp_path / "svd_model.json"
    save_model(model, filepath)

    loaded_model = load_model(filepath)
    assert isinstance(loaded_model, ClosedFormLinearRegression)
    assert loaded_model.method == "svd"
    assert np.allclose(loaded_model.weights, model.weights)
    assert np.isclose(loaded_model.bias, model.bias)

    X_test = np.random.randn(15, 4)
    assert np.allclose(loaded_model.predict(X_test), model.predict(X_test))


def test_serialize_and_load_pipeline(tmp_path):
    np.random.seed(42)
    X = np.random.randn(100, 3) * 5.0 + 12.0
    y = X @ np.array([1.5, -2.0, 0.5]) + 2.0

    pipe = PipelineScratch(
        [
            ("scaler", StandardScalerScratch()),
            ("regressor", ClosedFormLinearRegression(method="svd")),
        ]
    )
    pipe.fit(X, y)

    filepath = tmp_path / "pipe_model.json"
    save_model(pipe, filepath)

    loaded_pipe = load_model(filepath)
    assert isinstance(loaded_pipe, PipelineScratch)
    assert "scaler" in loaded_pipe.named_steps
    assert "regressor" in loaded_pipe.named_steps

    X_test = np.random.randn(25, 3) * 5.0 + 12.0
    assert np.allclose(loaded_pipe.predict(X_test), pipe.predict(X_test))


def test_serialize_and_load_newton_model(tmp_path):
    from src.solvers import NewtonLinearRegression

    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1.2, -0.8, 0.4]) + 0.5

    model = NewtonLinearRegression(alpha=0.1, damping=1e-5)
    model.fit(X, y)

    filepath = tmp_path / "newton_model.json"
    save_model(model, filepath)

    loaded = load_model(filepath)
    assert isinstance(loaded, NewtonLinearRegression)
    assert np.allclose(loaded.weights, model.weights)
    assert np.isclose(loaded.bias, model.bias)
    assert loaded.n_iter_ == model.n_iter_

    X_test = np.random.randn(10, 3)
    assert np.allclose(loaded.predict(X_test), model.predict(X_test))


def test_serialize_and_load_huber_model(tmp_path):
    from src.robust import HuberRegressorScratch

    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = X @ np.array([2.0, -1.5]) + 1.0

    model = HuberRegressorScratch(epsilon=1.35, lr=0.02, n_iters=200)
    model.fit(X, y)

    filepath = tmp_path / "huber_model.json"
    save_model(model, filepath)

    loaded = load_model(filepath)
    assert isinstance(loaded, HuberRegressorScratch)
    assert np.allclose(loaded.weights, model.weights)
    assert np.isclose(loaded.bias, model.bias)
    assert np.isclose(loaded.scale_, model.scale_)

    X_test = np.random.randn(12, 2)
    assert np.allclose(loaded.predict(X_test), model.predict(X_test))


def test_serialize_and_load_polynomial_pipeline(tmp_path):
    from src.features import PolynomialFeaturesScratch

    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 1.5 * (X[:, 0] ** 2) - 0.8 * X[:, 1] + 1.0

    pipe = PipelineScratch(
        [
            ("poly", PolynomialFeaturesScratch(degree=2, include_bias=False)),
            ("regressor", ClosedFormLinearRegression(method="svd")),
        ]
    )
    pipe.fit(X, y)

    filepath = tmp_path / "poly_pipe.json"
    save_model(pipe, filepath)

    loaded = load_model(filepath)
    assert isinstance(loaded, PipelineScratch)
    assert "poly" in loaded.named_steps
    assert isinstance(loaded.named_steps["poly"], PolynomialFeaturesScratch)

    X_test = np.random.randn(15, 2)
    assert np.allclose(loaded.predict(X_test), pipe.predict(X_test))


def test_serialize_and_load_bayesian_model(tmp_path):
    from src.bayesian import BayesianLinearRegression

    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5

    model = BayesianLinearRegression(alpha=2.0, beta=10.0)
    model.fit(X, y)

    filepath = tmp_path / "bayesian_model.json"
    save_model(model, filepath)

    loaded = load_model(filepath)
    assert isinstance(loaded, BayesianLinearRegression)
    assert np.allclose(loaded.weights, model.weights)
    assert np.isclose(loaded.bias, model.bias)
    assert np.allclose(loaded.cov_, model.cov_)

    X_test = np.random.randn(10, 2)
    pred_orig, std_orig = model.predict(X_test, return_std=True)
    pred_load, std_load = loaded.predict(X_test, return_std=True)
    assert np.allclose(pred_orig, pred_load)
    assert np.allclose(std_orig, std_load)


def test_serialize_and_load_conformal_and_glm(tmp_path):
    from src.conformal import ConformalLinearRegression
    from src.glm import LogisticRegressionScratch

    np.random.seed(42)
    X = np.random.randn(80, 2)
    y_reg = 1.5 * X[:, 0] + 2.0
    conf = ConformalLinearRegression(confidence_level=0.90)
    conf.fit(X, y_reg, cal_size=0.25)

    path_conf = tmp_path / "conf.json"
    save_model(conf, path_conf)
    loaded_conf = load_model(path_conf)
    assert isinstance(loaded_conf, ConformalLinearRegression)
    assert np.isclose(loaded_conf.q_hat_, conf.q_hat_)

    # Logistic Regression
    y_bin = (X[:, 0] > 0).astype(np.float64)
    clf = LogisticRegressionScratch(max_iter=50)
    clf.fit(X, y_bin)

    path_clf = tmp_path / "clf.json"
    save_model(clf, path_clf)
    loaded_clf = load_model(path_clf)
    assert isinstance(loaded_clf, LogisticRegressionScratch)
    assert np.allclose(loaded_clf.predict(X), clf.predict(X))


def test_serialize_and_load_streaming_and_quantile(tmp_path):
    from src.streaming import RecursiveLeastSquares
    from src.quantile import QuantileRegressorScratch

    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 1.0

    rls = RecursiveLeastSquares(lambda_=0.98)
    rls.fit(X, y)

    path_rls = tmp_path / "rls.json"
    save_model(rls, path_rls)
    loaded_rls = load_model(path_rls)
    assert isinstance(loaded_rls, RecursiveLeastSquares)
    assert np.allclose(loaded_rls.weights, rls.weights)
    assert np.isclose(loaded_rls.bias, rls.bias)

    qreg = QuantileRegressorScratch(quantile=0.75)
    qreg.fit(X, y)

    path_qreg = tmp_path / "qreg.json"
    save_model(qreg, path_qreg)
    loaded_qreg = load_model(path_qreg)
    assert isinstance(loaded_qreg, QuantileRegressorScratch)
    assert np.allclose(loaded_qreg.predict(X), qreg.predict(X))

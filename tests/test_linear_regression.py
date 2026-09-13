"""
Comprehensive Unit and Integration Tests for Linear Regression from Scratch.

Tests cover:
1. Mathematical correctness on noise-free synthetic data (R^2 ~ 1.0).
2. Convergence diagnostics (monotonic or steady decrease of MSE loss).
3. Mini-batch vs full-batch gradient descent consistency.
4. Determinism and reproducibility across runs with identical seeds.
5. Large-scale dataset execution (50,000+ samples, 20+ features) with zero memory errors.
6. Feature standardization and scaling utilities.
7. Data loader interfaces on real-world datasets.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.linear_regression import (
    GradientDescentLinearRegression,
    standardize_features,
    apply_scaler,
    compute_mse,
    compute_r2,
)
from src.data_loader import (
    load_california_housing,
    load_house_prices_kaggle,
    clean_and_split_regression_data,
)


def test_standardize_features_and_apply_scaler():
    """Verify standardization achieves mean 0 and std 1, and apply_scaler is consistent."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=10.0, scale=5.0, size=(1000, 5))
    # Add a constant column to test zero-division prevention
    X[:, 4] = 7.0

    X_std, scaler = standardize_features(X, return_scaler=True)

    # First 4 columns should have mean ~ 0 and std ~ 1
    assert np.allclose(np.mean(X_std[:, :4], axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.std(X_std[:, :4], axis=0), 1.0, atol=1e-7)

    # Constant column should be zeros (scaled safely with std=1.0)
    assert np.allclose(X_std[:, 4], 0.0, atol=1e-7)

    # Apply scaler to a new test matrix
    X_test = rng.normal(loc=10.0, scale=5.0, size=(100, 5))
    X_test[:, 4] = 7.0
    X_test_scaled = apply_scaler(X_test, scaler)
    assert X_test_scaled.shape == (100, 5)
    assert not np.isnan(X_test_scaled).any()


def test_linear_regression_r2_noise_free():
    """Verify exact analytical recovery of ground truth linear function on noise-free data."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 1000, 4
    X = rng.standard_normal(size=(n_samples, n_features))

    # True coefficients: y = 3.5*x0 - 2.0*x1 + 1.2*x2 - 0.8*x3 + 4.0
    true_w = np.array([3.5, -2.0, 1.2, -0.8])
    true_b = 4.0
    y = X @ true_w + true_b

    # Train with full-batch GD
    model = GradientDescentLinearRegression(lr=0.05, n_iters=1500, seed=42)
    model.fit(X, y)

    r2 = model.score(X, y)
    mse = compute_mse(y, model.predict(X))

    assert r2 >= 0.999, f"R^2 is {r2}, expected >= 0.999 on noise-free data"
    assert mse < 1e-4, f"MSE is {mse}, expected < 1e-4"
    assert np.allclose(model.weights, true_w, atol=0.05)
    assert np.isclose(model.bias, true_b, atol=0.05)


def test_linear_regression_synthetic_convergence():
    """Verify MSE decreases over training epochs."""
    rng = np.random.default_rng(123)
    n_samples, n_features = 2000, 5
    X = rng.standard_normal(size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 3.0 + rng.normal(0, 0.2, n_samples)

    model = GradientDescentLinearRegression(lr=0.05, n_iters=200, seed=42)
    _, history = model.fit_with_history(X, y)

    assert len(history) == 200
    # Final MSE must be substantially lower than initial MSE
    assert history[-1] < history[0] * 0.1, (
        "Final MSE should be at least an order of magnitude smaller than initial"
    )
    # General downward trajectory
    assert history[50] < history[5]
    assert history[-1] <= history[50]


def test_linear_regression_mini_batch_vs_full_batch():
    """Verify both mini-batch and full-batch reach comparable low MSE."""
    rng = np.random.default_rng(999)
    n_samples, n_features = 5000, 6
    X = rng.standard_normal(size=(n_samples, n_features))
    y = 1.8 * X[:, 0] - 2.5 * X[:, 1] + 0.9 * X[:, 2] - 1.1 * X[:, 3] + 2.0

    # Full batch
    fb_model = GradientDescentLinearRegression(lr=0.05, n_iters=500, seed=42, batch_size=None)
    fb_model.fit(X, y)
    fb_mse = compute_mse(y, fb_model.predict(X))

    # Mini-batch (batch_size=128)
    mb_model = GradientDescentLinearRegression(lr=0.02, n_iters=150, seed=42, batch_size=128)
    mb_model.fit(X, y)
    mb_mse = compute_mse(y, mb_model.predict(X))

    assert fb_mse < 0.01, f"Full batch MSE was {fb_mse}"
    assert mb_mse < 0.01, f"Mini batch MSE was {mb_mse}"
    # Both should have R^2 > 0.99
    assert fb_model.score(X, y) > 0.99
    assert mb_model.score(X, y) > 0.99


def test_linear_regression_determinism_seed():
    """Verify identical seeds produce bitwise identical model parameters."""
    rng = np.random.default_rng(77)
    X = rng.standard_normal(size=(1000, 8))
    y = X @ np.ones(8) + rng.normal(0, 0.1, 1000)

    m1 = GradientDescentLinearRegression(lr=0.02, n_iters=100, seed=42, batch_size=64)
    m1.fit(X, y)

    m2 = GradientDescentLinearRegression(lr=0.02, n_iters=100, seed=42, batch_size=64)
    m2.fit(X, y)

    np.testing.assert_array_equal(m1.weights, m2.weights)
    assert m1.bias == m2.bias


def test_linear_regression_large_synthetic_scale():
    """
    Stress test model on large dataset: 50,000 samples x 20 features.
    Ensures memory efficiency, vectorized speed, and no memory exhaustion.
    """
    rng = np.random.default_rng(2024)
    n_samples, n_features = 50000, 20

    X = rng.standard_normal(size=(n_samples, n_features), dtype=np.float64)
    true_w = rng.uniform(-2.0, 2.0, size=n_features)
    y = X @ true_w + 1.5 + rng.normal(0, 0.1, size=n_samples)

    # Mini-batch training for large scale
    model = GradientDescentLinearRegression(lr=0.01, n_iters=30, batch_size=256, seed=42)
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (n_samples,)
    r2 = model.score(X, y)
    assert r2 > 0.95, f"Expected R^2 > 0.95 on large synthetic dataset, got {r2}"


def test_linear_regression_pandas_compatibility():
    """Verify model accepts pandas DataFrames and Series directly."""
    df_X = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat_b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        }
    )
    s_y = pd.Series([3.0, 3.0, 7.0, 7.0, 11.0, 11.0])

    model = GradientDescentLinearRegression(lr=0.01, n_iters=100, seed=42)
    model.fit(df_X, s_y)

    preds = model.predict(df_X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (6,)


def test_data_loader_california_housing():
    """Test loading California Housing dataset."""
    X_tr, X_te, y_tr, y_te, feats = load_california_housing(test_size=0.2, random_state=42)

    assert len(X_tr) == 16512
    assert len(X_te) == 4128
    assert len(y_tr) == 16512
    assert len(y_te) == 4128
    assert X_tr.shape[1] == 8
    assert len(feats) == 8
    # Zero missing values after pipeline
    assert not X_tr.isna().any().any()
    assert not X_te.isna().any().any()


def test_data_loader_house_prices_kaggle():
    """Test loading Kaggle House Prices dataset."""
    X_tr, X_te, y_tr, y_te, feats = load_house_prices_kaggle(
        data_dir="data/house_prices",
        test_size=0.2,
        random_state=42,
    )

    assert len(X_tr) == int(1460 * 0.8)
    assert len(X_te) == 1460 - int(1460 * 0.8)
    assert X_tr.shape[1] == X_te.shape[1]
    assert len(feats) == X_tr.shape[1]
    # No NaNs
    assert not X_tr.isna().any().any()
    assert not X_te.isna().any().any()
    # Log transform check: target should be log1p scale (~10 to ~14)
    assert y_tr.min() > 5.0 and y_tr.max() < 16.0


def test_data_loader_missing_file_error():
    """Test clear error is raised when train.csv does not exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_house_prices_kaggle(data_dir="data/non_existent_folder")

    err_msg = str(exc_info.value)
    assert "kaggle competitions download" in err_msg
    assert "download_data.py" in err_msg

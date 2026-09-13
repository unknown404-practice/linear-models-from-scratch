"""
Unit Tests for Custom Transformers & Pipeline Engine (src/pipeline.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    BaseTransformer,
    StandardScalerScratch,
    SimpleImputerScratch,
    OneHotEncoderScratch,
    ColumnTransformerScratch,
    PipelineScratch,
)
from src.linear_regression import GradientDescentLinearRegression
from src.solvers import ClosedFormLinearRegression


def test_standard_scaler_scratch():
    np.random.seed(42)
    X = np.random.randn(100, 3) * np.array([10.0, 0.5, 2.0]) + np.array([50.0, 1.0, -5.0])
    # Add a constant column to test zero-variance safety
    X[:, 1] = 7.0

    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape
    # Feature 0 and 2 should have mean ~ 0 and std ~ 1
    assert np.isclose(np.mean(X_scaled[:, 0]), 0.0, atol=1e-7)
    assert np.isclose(np.std(X_scaled[:, 0]), 1.0, atol=1e-7)
    assert np.isclose(np.mean(X_scaled[:, 2]), 0.0, atol=1e-7)
    assert np.isclose(np.std(X_scaled[:, 2]), 1.0, atol=1e-7)
    # Constant column should not be NaN
    assert not np.isnan(X_scaled[:, 1]).any()
    assert np.all(X_scaled[:, 1] == 0.0)

    # Inverse transform recovers original data
    X_restored = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_restored)

    # Test with DataFrame and feature names
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    scaler_df = StandardScalerScratch()
    scaler_df.fit(df)
    assert scaler_df.feature_names_in_ == ["feat_a", "feat_b", "feat_c"]


def test_simple_imputer_scratch():
    X = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, np.nan, 6.0],
            [np.nan, 4.0, 9.0],
            [7.0, 8.0, 15.0],
        ]
    )

    # Median imputation
    imputer_median = SimpleImputerScratch(strategy="median")
    X_imp_med = imputer_median.fit_transform(X)
    assert not np.isnan(X_imp_med).any()
    # Expected medians: col 0 -> median([1, 3, 7]) = 3; col 1 -> median([2, 4, 8]) = 4; col 2 -> median([6, 9, 15]) = 9
    assert np.allclose(imputer_median.statistics_, [3.0, 4.0, 9.0])
    assert X_imp_med[0, 2] == 9.0
    assert X_imp_med[2, 0] == 3.0

    # Mean imputation
    imputer_mean = SimpleImputerScratch(strategy="mean")
    X_imp_mean = imputer_mean.fit_transform(X)
    assert not np.isnan(X_imp_mean).any()
    # Expected means: col 0 -> 11/3; col 1 -> 14/3; col 2 -> 30/3 = 10
    assert np.isclose(imputer_mean.statistics_[2], 10.0)

    # Constant imputation
    imputer_const = SimpleImputerScratch(strategy="constant", fill_value=-999.0)
    X_imp_const = imputer_const.fit_transform(X)
    assert X_imp_const[0, 2] == -999.0


def test_one_hot_encoder_scratch():
    train_cats = np.array(
        [
            ["Red", "Small"],
            ["Blue", "Medium"],
            ["Green", "Large"],
            ["Red", "Large"],
        ]
    )

    encoder = OneHotEncoderScratch(handle_unknown="ignore")
    X_enc = encoder.fit_transform(train_cats)

    assert X_enc.shape[0] == 4
    # Column 0 has 3 categories: Blue, Green, Red; Column 1 has 3: Large, Medium, Small -> total 6
    assert X_enc.shape[1] == 6

    # Test unknown categories in test data
    test_cats = np.array(
        [
            ["Yellow", "Small"],  # "Yellow" unseen
            ["Blue", "ExtraLarge"],  # "ExtraLarge" unseen
        ]
    )
    X_test_enc = encoder.transform(test_cats)
    assert X_test_enc.shape == (2, 6)
    # First row: Blue=0, Green=0, Red=0, Large=0, Medium=0, Small=1
    assert np.all(X_test_enc[0, :3] == 0.0)
    assert X_test_enc[0, -1] == 1.0

    # Feature names
    names = encoder.get_feature_names_out(["color", "size"])
    assert len(names) == 6
    assert any("color_Red" in n or "color_red" in n for n in names)


def test_column_transformer_scratch():
    df = pd.DataFrame(
        {
            "num1": [10.0, 20.0, 30.0, 40.0],
            "num2": [1.0, 2.0, 3.0, 4.0],
            "cat": ["A", "B", "A", "C"],
        }
    )

    ct = ColumnTransformerScratch(
        [
            ("num", StandardScalerScratch(), ["num1", "num2"]),
            ("cat", OneHotEncoderScratch(), ["cat"]),
        ]
    )

    X_trans = ct.fit_transform(df)
    # 2 numeric standardized + 3 one-hot categories for 'cat' = 5 columns
    assert X_trans.shape == (4, 5)
    names = ct.get_feature_names_out()
    assert len(names) == 5


def test_pipeline_scratch_fit_predict_score():
    np.random.seed(42)
    X = np.random.randn(150, 4) * 5.0 + 10.0
    true_w = np.array([2.0, -3.0, 1.5, 0.5])
    y = X @ true_w + 3.0 + 0.2 * np.random.randn(150)

    # 1. Pipeline with Gradient Descent
    pipe_gd = PipelineScratch(
        [
            ("scaler", StandardScalerScratch()),
            ("regressor", GradientDescentLinearRegression(lr=0.05, n_iters=350, seed=42)),
        ]
    )

    pipe_gd.fit(X, y)
    r2_gd = pipe_gd.score(X, y)
    assert r2_gd > 0.95
    assert "scaler" in pipe_gd.named_steps
    assert "regressor" in pipe_gd.named_steps

    # 2. Pipeline with Closed-Form SVD
    pipe_svd = PipelineScratch(
        [
            ("scaler", StandardScalerScratch()),
            ("regressor", ClosedFormLinearRegression(method="svd")),
        ]
    )
    pipe_svd.fit(X, y)
    r2_svd = pipe_svd.score(X, y)
    assert r2_svd > 0.99

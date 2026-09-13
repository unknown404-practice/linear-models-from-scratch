"""
Unit Tests for Model Selection & Cross-Validation Engine (src/model_selection.py).
"""

import numpy as np
import pytest

from src.model_selection import (
    KFoldScratch,
    cross_val_score_scratch,
    GridSearchCVScratch,
)
from src.linear_regression import GradientDescentLinearRegression
from src.solvers import ClosedFormLinearRegression


def test_kfold_scratch_partitions():
    n_samples = 100
    n_splits = 5
    kf = KFoldScratch(n_splits=n_splits, shuffle=True, random_state=42)

    val_indices_collected = []
    splits = list(kf.split(np.zeros(n_samples)))

    assert len(splits) == n_splits

    for train_idx, val_idx in splits:
        assert len(train_idx) + len(val_idx) == n_samples
        # Check disjointness
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        val_indices_collected.extend(val_idx)

    # Union of all validation indices covers all samples exactly once
    assert sorted(val_indices_collected) == list(range(n_samples))


def test_cross_val_score_scratch():
    np.random.seed(42)
    X = np.random.randn(120, 3)
    y = X @ np.array([2.0, -1.0, 3.0]) + 0.5 + 0.1 * np.random.randn(120)

    model = ClosedFormLinearRegression(method="svd")

    # 1. R^2 scoring
    r2_scores = cross_val_score_scratch(model, X, y, cv=4, scoring="r2")
    assert len(r2_scores) == 4
    assert np.all(r2_scores > 0.95)

    # 2. Neg MSE scoring
    neg_mse_scores = cross_val_score_scratch(model, X, y, cv=4, scoring="neg_mean_squared_error")
    assert len(neg_mse_scores) == 4
    assert np.all(neg_mse_scores < 0.0)  # negative MSE
    assert np.all(neg_mse_scores > -0.1)


def test_grid_search_cv_scratch_finds_best_alpha():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X @ np.array([1.5, -2.0, 0.5, 3.0]) + 0.2 * np.random.randn(100)

    base_model = GradientDescentLinearRegression(lr=0.05, n_iters=250, penalty="l2", seed=42)
    param_grid = {
        "alpha": [0.0, 0.01, 1.0],
    }

    grid = GridSearchCVScratch(base_model, param_grid, cv=3, scoring="r2")
    grid.fit(X, y)

    assert "alpha" in grid.best_params_
    assert grid.best_estimator_ is not None
    assert grid.best_score_ > 0.90
    assert len(grid.cv_results_["params"]) == 3
    assert len(grid.cv_results_["mean_test_score"]) == 3

    # Prediction and scoring via best_estimator delegation
    y_pred = grid.predict(X)
    assert len(y_pred) == 100
    assert grid.score(X, y) > 0.90

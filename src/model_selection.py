"""
Model Selection & Cross-Validation Engine from Scratch.

Provides KFold cross-validation partitioning, cross_val_score_scratch,
and GridSearchCVScratch matching Scikit-Learn's API.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Generator, Sequence
import numpy as np
import pandas as pd


def _index_slice(data: Any, indices: np.ndarray) -> Any:
    """Safely slice arrays or DataFrames by index array."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices]
    arr = np.asarray(data)
    return arr[indices]


def clone_estimator(estimator: Any) -> Any:
    """Safely clone an unfitted or fitted estimator."""
    if hasattr(estimator, "get_params"):
        # For PipelineScratch, clone each step
        if hasattr(estimator, "steps"):
            cloned_steps = [(name, clone_estimator(step)) for name, step in estimator.steps]
            return estimator.__class__(cloned_steps)
        params = estimator.get_params(deep=False)
        return estimator.__class__(**params)
    return copy.deepcopy(estimator)


class KFoldScratch:
    """
    K-Fold cross-validator partitioner.

    Provides train/val indices to split data in train/val sets.
    Splits dataset into k consecutive or shuffled folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.
    random_state : int or None, default=42
        Random seed for shuffling.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Any, y: Any = None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have {self.n_splits} folds with only {n_samples} samples.")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current : current + fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            current += fold_size
            yield train_idx, val_idx


def cross_val_score_scratch(
    estimator: Any,
    X: Any,
    y: Any,
    cv: int | KFoldScratch = 5,
    scoring: str = "r2",
) -> np.ndarray:
    """
    Evaluate a score by cross-validation.

    Parameters
    ----------
    estimator : object
        Estimator object implementing 'fit' and 'predict' / 'score'.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,)
        The target variable to try to predict.
    cv : int or KFoldScratch, default=5
        Cross-validation generator or number of folds.
    scoring : str, default='r2'
        Scoring metric: 'r2', 'neg_mean_squared_error', or 'neg_root_mean_squared_error'.

    Returns
    -------
    scores : np.ndarray of float
        Array of scores of the estimator for each run of the cross-validation.
    """
    if isinstance(cv, int):
        cv_gen = KFoldScratch(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_gen = cv

    scores: list[float] = []

    for train_idx, val_idx in cv_gen.split(X):
        est = clone_estimator(estimator)
        X_train, y_train = _index_slice(X, train_idx), _index_slice(y, train_idx)
        X_val, y_val = _index_slice(X, val_idx), _index_slice(y, val_idx)

        est.fit(X_train, y_train)
        y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
        y_pred = est.predict(X_val)

        if scoring == "r2":
            ss_res = float(np.sum((y_val_arr - y_pred) ** 2))
            ss_tot = float(np.sum((y_val_arr - np.mean(y_val_arr)) ** 2))
            score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        elif scoring in ("neg_mean_squared_error", "neg_mse"):
            score = -float(np.mean((y_val_arr - y_pred) ** 2))
        elif scoring in ("neg_root_mean_squared_error", "neg_rmse"):
            score = -float(np.sqrt(np.mean((y_val_arr - y_pred) ** 2)))
        else:
            raise ValueError(
                f"Unsupported scoring metric: '{scoring}'. Use 'r2' or 'neg_mean_squared_error'."
            )

        scores.append(score)

    return np.array(scores, dtype=np.float64)


class GridSearchCVScratch:
    """
    Exhaustive search over specified parameter values for an estimator.

    Parameters
    ----------
    estimator : estimator object
        An object of that type is instantiated for each grid point.
    param_grid : dict or list of dicts
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values.
    cv : int or KFoldScratch, default=5
        Determines the cross-validation splitting strategy.
    scoring : str, default='r2'
        Strategy to evaluate the performance of the cross-validated model on the test set.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: dict[str, Sequence[Any]],
        cv: int | KFoldScratch = 5,
        scoring: str = "r2",
        refit: bool = True,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = -float("inf")
        self.best_estimator_: Any | None = None
        self.cv_results_: dict[str, Any] = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
        }

    def fit(self, X: Any, y: Any) -> GridSearchCVScratch:
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        # Generate all combinations
        all_combinations = [
            dict(zip(param_keys, combo)) for combo in itertools.product(*param_values)
        ]

        self.cv_results_ = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
        }
        for k in param_keys:
            self.cv_results_[f"param_{k}"] = []
        best_score = -float("inf")
        best_params = {}

        for combo in all_combinations:
            est = clone_estimator(self.estimator)
            est.set_params(**combo)
            scores = cross_val_score_scratch(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            self.cv_results_["params"].append(combo)
            for k in param_keys:
                self.cv_results_[f"param_{k}"].append(combo[k])
            self.cv_results_["mean_test_score"].append(mean_score)
            self.cv_results_["std_test_score"].append(std_score)

            if mean_score > best_score:
                best_score = mean_score
                best_params = combo

        self.best_score_ = best_score
        self.best_params_ = best_params

        if self.refit:
            self.best_estimator_ = clone_estimator(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV is not fitted yet.")
        return self.best_estimator_.predict(X)

    def score(self, X: Any, y: Any) -> float:
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV is not fitted yet.")
        if hasattr(self.best_estimator_, "score"):
            return float(self.best_estimator_.score(X, y))
        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def __repr__(self) -> str:
        status = (
            f"best_params={self.best_params_}, best_score={self.best_score_:.4f}"
            if self.best_params_
            else "Unfitted"
        )
        return f"GridSearchCVScratch(estimator={self.estimator.__class__.__name__}, scoring='{self.scoring}') [{status}]"

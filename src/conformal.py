"""
Distribution-Free Conformal Prediction Engine from Scratch.

Provides split conformal predictive intervals with guaranteed finite-sample
coverage (1 - alpha) without requiring Gaussian, linearity, or homoscedasticity
assumptions.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from src.solvers import ClosedFormLinearRegression


class ConformalLinearRegression:
    """
    Split Conformal Prediction for Linear Regression.

    Computes predictive intervals [y_hat - q_hat, y_hat + q_hat] that satisfy:
        P(y_* in C(x_*)) >= 1 - alpha

    Parameters
    ----------
    estimator : object, optional
        Base regression estimator implementing fit() and predict().
        Defaults to ClosedFormLinearRegression(method="svd").
    confidence_level : float, default=0.95
        Desired target marginal coverage probability (1 - alpha).
    """

    def __init__(
        self,
        estimator: Any | None = None,
        confidence_level: float = 0.95,
        alpha: float | None = None,
        cal_size: float | None = None,
        random_state: int | None = None,
    ) -> None:
        if alpha is not None:
            confidence_level = 1.0 - alpha
        self.estimator = (
            estimator if estimator is not None else ClosedFormLinearRegression(method="svd")
        )
        self.confidence_level = float(confidence_level)
        self.default_cal_size = cal_size
        self.default_random_state = random_state

        # Fitted attributes
        self.estimator_: Any = None
        self.q_hat_: float = 0.0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None
        self.cal_residuals_: np.ndarray | None = None

    def _prepare_data(self, X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        elif isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            else:
                y_arr = np.asarray(y, dtype=np.float64).ravel()

        return X_arr, y_arr

    def fit(
        self,
        X: Any,
        y: Any,
        X_cal: Any | None = None,
        y_cal: Any | None = None,
        cal_size: float = 0.2,
        random_state: int | None = 42,
    ) -> ConformalLinearRegression:
        """
        Fit base estimator and calibrate conformal nonconformity scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        X_cal : array-like, optional
            Explicit held-out calibration features. If None, split from (X, y).
        y_cal : array-like, optional
            Explicit held-out calibration targets.
        cal_size : float, default=0.2
            Fraction of samples held out for calibration if X_cal is None.
        random_state : int, optional
            Random seed for calibration split shuffling.
        """
        if cal_size == 0.2 and self.default_cal_size is not None:
            cal_size = self.default_cal_size
        if random_state == 42 and self.default_random_state is not None:
            random_state = self.default_random_state
        X_arr, y_arr = self._prepare_data(X, y)
        self.n_features_in_ = X_arr.shape[1]

        if X_cal is not None and y_cal is not None:
            X_train = X_arr
            y_train = y_arr
            X_c, y_c = self._prepare_data(X_cal, y_cal)
        else:
            n_samples = len(X_arr)
            n_cal = max(1, int(n_samples * cal_size))
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(n_samples)
            cal_idx = indices[:n_cal]
            train_idx = indices[n_cal:]

            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]
            X_c = X_arr[cal_idx]
            y_c = y_arr[cal_idx]

        # 1. Fit base model on training split
        self.estimator_ = self.estimator
        self.estimator_.fit(X_train, y_train)

        # 2. Compute nonconformity scores s_i = |y_i - y_hat_i| on calibration split
        y_c_pred = np.asarray(self.estimator_.predict(X_c)).ravel()
        scores = np.abs(y_c - y_c_pred)
        self.cal_residuals_ = scores

        # 3. Conformal quantile level: ceil((n + 1)(1 - alpha)) / n
        n = len(scores)
        alpha = 1.0 - self.confidence_level
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)

        # Compute empirical quantile (method='higher' matches conformal theory)
        self.q_hat_ = float(np.quantile(scores, q_level, method="higher"))

        return self

    def predict(
        self, X: Any, return_intervals: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point predictions and guaranteed conformal prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.
        return_intervals : bool, default=False
            If True, returns (y_pred, y_lower, y_upper).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Point predictions.
        y_lower : ndarray of shape (n_samples,), optional
            Conformal lower bounds.
        y_upper : ndarray of shape (n_samples,), optional
            Conformal upper bounds.
        """
        if self.estimator_ is None:
            raise RuntimeError("Conformal estimator is not fitted. Call fit() first.")

        X_arr, _ = self._prepare_data(X)
        y_pred = np.asarray(self.estimator_.predict(X_arr)).ravel()

        if not return_intervals:
            return y_pred

        y_lower = y_pred - self.q_hat_
        y_upper = y_pred + self.q_hat_
        return y_pred, y_lower, y_upper

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method returning (y_pred, y_lower, y_upper)."""
        res = self.predict(X, return_intervals=True)
        return res  # type: ignore

    def coverage_score(self, X: Any, y: Any) -> float:
        """
        Compute the empirical coverage percentage on test set.

        Returns fraction of test samples where y in [y_lower, y_upper].
        """
        _, y_arr = self._prepare_data(X, y)
        _, y_lower, y_upper = self.predict(X, return_intervals=True)
        covered = (y_arr >= y_lower) & (y_arr <= y_upper)
        return float(np.mean(covered))

    # Alias for coverage_score
    score_coverage = coverage_score

    @property
    def interval_width(self) -> float:
        """Total width of the symmetric prediction interval (2 * q_hat)."""
        return 2.0 * self.q_hat_

    def score(self, X: Any, y: Any) -> float:
        """Compute base estimator R^2 score."""
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return float(self.estimator_.score(X, y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "confidence_level": self.confidence_level,
        }

    def set_params(self, **params: Any) -> ConformalLinearRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self

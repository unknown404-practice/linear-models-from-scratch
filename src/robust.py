"""
Robust Linear Regression Suite: Huber Regressor & RANSAC Consensus Engine.

Implements outlier-immune linear regression algorithms in pure NumPy:
- HuberRegressorScratch: Piecewise Huber loss with adaptive derivative bounding
- RANSACRegressorScratch: Random Sample Consensus for extreme leverage anomalies
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.linear_regression import compute_r2
from src.solvers import ClosedFormLinearRegression
from src.model_selection import clone_estimator


class HuberRegressorScratch:
    """
    Linear regression model robust to outliers using Huber Loss.

    Parameters
    ----------
    epsilon : float, default=1.35
        The parameter epsilon controls the number of samples should be classified as outliers.
        The smaller the epsilon, the more robust it is to outliers.
    alpha : float, default=0.0
        L2 regularization parameter.
    lr : float, default=0.01
        Learning rate for optimization.
    n_iters : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence tolerance on weight updates.
    seed : int or None, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 1.35,
        alpha: float = 0.0,
        lr: float = 0.01,
        n_iters: int = 1000,
        tol: float = 1e-5,
        seed: int | None = 42,
        max_iter: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        if max_iter is not None:
            n_iters = max_iter
        if learning_rate is not None:
            lr = learning_rate
        if epsilon <= 1.0:
            raise ValueError(f"epsilon must be > 1.0, got {epsilon}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.tol = float(tol)
        self.seed = seed

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.scale_: float = 1.0
        self.outliers_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> HuberRegressorScratch:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # Initialize weights with median-based heuristics
        w = np.zeros(n_features, dtype=np.float64)
        b = float(np.median(y_arr))

        # Momentum buffers for smooth descent
        v_w = np.zeros_like(w)
        v_b = 0.0
        momentum = 0.9

        for iteration in range(self.n_iters):
            preds = X_arr @ w + b
            errors = preds - y_arr

            # Robust scale estimation via MAD (Median Absolute Deviation)
            abs_err = np.abs(errors)
            med_abs_err = float(np.median(abs_err))
            scale = med_abs_err / 0.6745 if med_abs_err > 1e-7 else 1.0
            self.scale_ = scale

            delta = self.epsilon * scale

            # Piecewise Huber derivative: psi(e)
            mask = abs_err <= delta
            psi = np.empty_like(errors)
            psi[mask] = errors[mask]
            psi[~mask] = delta * np.sign(errors[~mask])

            # Gradient computation
            grad_w = (1.0 / n_samples) * (X_arr.T @ psi) + 2.0 * self.alpha * w
            grad_b = float(np.mean(psi))

            # Momentum update
            v_w = momentum * v_w + self.lr * grad_w
            v_b = momentum * v_b + self.lr * grad_b

            w -= v_w
            b -= v_b

            if np.max(np.abs(v_w)) < self.tol and abs(v_b) < self.tol:
                break

        self.weights = w
        self.bias = b

        final_err = np.abs((X_arr @ w + b) - y_arr)
        self.outliers_ = final_err > (self.epsilon * self.scale_)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("HuberRegressorScratch is not fitted yet.")
        X_arr = (
            X.values.astype(np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )
        return X_arr @ self.weights + self.bias

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        return compute_r2(y_arr, y_pred)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "lr": self.lr,
            "n_iters": self.n_iters,
            "tol": self.tol,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> HuberRegressorScratch:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        status = "Fitted" if self.weights is not None else "Unfitted"
        return f"HuberRegressorScratch(epsilon={self.epsilon}, alpha={self.alpha}, lr={self.lr}) [{status}]"


class RANSACRegressorScratch:
    """
    RANSAC (RANdom SAmple Consensus) robust linear regression algorithm.

    Parameters
    ----------
    estimator : object or None, default=None
        Base estimator. If None, defaults to ClosedFormLinearRegression(method="svd").
    min_samples : int or None, default=None
        Minimum number of samples chosen randomly from original data.
        If None, min_samples = n_features + 1.
    residual_threshold : float or None, default=None
        Maximum residual for a data sample to be classified as an inlier.
        If None, threshold is estimated as MAD of target y.
    max_trials : int, default=100
        Maximum number of iterations for random sample selection.
    stop_n_inliers : int or float or None, default=None
        Stop iteration if at least this number of inliers are found.
    random_state : int or None, default=42
        Random seed for sampling.
    """

    def __init__(
        self,
        estimator: Any = None,
        min_samples: int | None = None,
        residual_threshold: float | None = None,
        max_trials: int = 100,
        stop_n_inliers: int | float | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = int(max_trials)
        self.stop_n_inliers = stop_n_inliers
        self.random_state = random_state

        self.estimator_: Any = None
        self.inlier_mask_: np.ndarray | None = None
        self.n_trials_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> RANSACRegressorScratch:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)

        base_estimator = (
            self.estimator
            if self.estimator is not None
            else ClosedFormLinearRegression(method="svd")
        )

        min_samples = self.min_samples
        if min_samples is None:
            min_samples = min(n_features + 1, n_samples)
        min_samples = max(1, min(min_samples, n_samples))

        threshold = self.residual_threshold
        if threshold is None:
            med = float(np.median(y_arr))
            mad = float(np.median(np.abs(y_arr - med)))
            threshold = mad if mad > 1e-6 else 1.0

        best_inliers = np.zeros(n_samples, dtype=bool)
        best_inlier_count = -1

        for trial in range(self.max_trials):
            # Sample random subset
            subset_idx = rng.choice(n_samples, size=min_samples, replace=False)
            X_subset = X_arr[subset_idx]
            y_subset = y_arr[subset_idx]

            try:
                candidate_est = clone_estimator(base_estimator)
                candidate_est.fit(X_subset, y_subset)
                preds = candidate_est.predict(X_arr)
                residuals = np.abs(y_arr - preds)
                inliers = residuals <= threshold
                inlier_count = int(np.sum(inliers))

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_inliers = inliers

                if self.stop_n_inliers is not None:
                    stop_count = (
                        int(self.stop_n_inliers * n_samples)
                        if isinstance(self.stop_n_inliers, float)
                        else int(self.stop_n_inliers)
                    )
                    if inlier_count >= stop_count:
                        break
            except Exception:
                continue

        self.n_trials_ = trial + 1
        self.inlier_mask_ = best_inliers

        # Refit final estimator on all identified inliers
        if np.sum(best_inliers) >= min_samples:
            final_est = clone_estimator(base_estimator)
            final_est.fit(X_arr[best_inliers], y_arr[best_inliers])
            self.estimator_ = final_est
        else:
            # Fallback to fitting on full data if inliers degenerate
            final_est = clone_estimator(base_estimator)
            final_est.fit(X_arr, y_arr)
            self.estimator_ = final_est

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("RANSACRegressorScratch is not fitted yet.")
        return self.estimator_.predict(X)

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        if self.estimator_ is None:
            raise RuntimeError("RANSACRegressorScratch is not fitted yet.")
        return self.estimator_.score(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "estimator": self.estimator,
            "min_samples": self.min_samples,
            "residual_threshold": self.residual_threshold,
            "max_trials": self.max_trials,
            "stop_n_inliers": self.stop_n_inliers,
            "random_state": self.random_state,
        }
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params: Any) -> RANSACRegressorScratch:
        for k, v in params.items():
            if "__" in k and hasattr(self.estimator, "set_params"):
                est_param = k.split("__", 1)[1]
                self.estimator.set_params(**{est_param: v})
            else:
                setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        status = "Fitted" if self.estimator_ is not None else "Unfitted"
        return f"RANSACRegressorScratch(max_trials={self.max_trials}, threshold={self.residual_threshold}) [{status}]"

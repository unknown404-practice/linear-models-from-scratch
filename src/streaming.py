"""
Streaming & Real-Time Linear Regression Engine from Scratch.

Provides infinite-scale online parameter updating in O(D^2) time/memory per
sample without retaining historical samples in memory:
- Recursive Least Squares (RLS) with Sherman-Morrison rank-1 updates and
  exponential forgetting factors (lambda)
- Kalman Filter Regression for tracking dynamically drifting parameters
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class RecursiveLeastSquares:
    """
    Recursive Least Squares (RLS) Online Streaming Regressor.

    Parameters
    ----------
    lambda_ : float, default=1.0
        Exponential forgetting factor in (0, 1].
        - lambda_ = 1.0: Infinite memory; asymptotically identical to batch OLS.
        - lambda_ < 1.0: Exponential forgetting for non-stationary environments.
    delta : float, default=1000.0
        Initial inverse covariance regularization scale (P_0 = delta * I).
    fit_intercept : bool, default=True
        Whether to calculate an intercept for this model.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        delta: float = 1000.0,
        fit_intercept: bool = True,
        forgetting_factor: float | None = None,
    ) -> None:
        if forgetting_factor is not None:
            lambda_ = forgetting_factor
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.fit_intercept = bool(fit_intercept)

        # Fitted attributes
        self.theta_: np.ndarray | None = None  # Augmented parameter vector [w, b]
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.P_: np.ndarray | None = None  # Inverse covariance matrix
        self.n_samples_seen_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def _prepare_data(self, X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        elif isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            if (
                y is not None
                and (np.ndim(y) == 0 or len(np.asarray(y).ravel()) == 1)
                and len(X_arr) > 1
            ):
                X_arr = X_arr.reshape(1, -1)
            elif self.n_features_in_ > 0 and len(X_arr) == self.n_features_in_:
                X_arr = X_arr.reshape(1, -1)
            else:
                X_arr = X_arr.reshape(-1, 1)

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            else:
                y_arr = np.asarray(y, dtype=np.float64).ravel()

        return X_arr, y_arr

    def _init_state(self, n_features: int) -> None:
        """Initialize parameters and covariance matrix."""
        self.n_features_in_ = n_features
        dim = n_features + (1 if self.fit_intercept else 0)
        self.theta_ = np.zeros(dim, dtype=np.float64)
        self.P_ = self.delta * np.eye(dim, dtype=np.float64)
        self.n_samples_seen_ = 0

    def partial_fit(self, X: Any, y: Any) -> RecursiveLeastSquares:
        """
        Incrementally update parameters with incoming stream observations in O(D^2) time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
            Incoming feature observations.
        y : array-like of shape (n_samples,) or scalar
            Incoming target values.
        """
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape

        if self.theta_ is None or self.P_ is None:
            self._init_state(n_features)

        # Vectorize incoming mini-batch sample by sample
        for i in range(n_samples):
            x_i = X_arr[i]
            y_i = float(y_arr[i])

            if self.fit_intercept:
                x_aug = np.append(x_i, 1.0)
            else:
                x_aug = x_i

            # 1. Compute a priori prediction and error
            y_pred = float(x_aug @ self.theta_)
            error = y_i - y_pred

            # 2. Gain vector: k = (P * x) / (lambda + x^T * P * x)
            Px = self.P_ @ x_aug
            denom = self.lambda_ + float(x_aug @ Px)
            k = Px / denom

            # 3. Update parameter vector
            self.theta_ += k * error

            # 4. Update inverse covariance: P = (P - k * x^T * P) / lambda
            # Outer product rank-1 update
            self.P_ = (self.P_ - np.outer(k, Px)) / self.lambda_

            # Maintain numerical symmetry
            self.P_ = 0.5 * (self.P_ + self.P_.T)
            self.n_samples_seen_ += 1

        if self.fit_intercept:
            self.weights = self.theta_[:-1]
            self.bias = float(self.theta_[-1])
        else:
            self.weights = self.theta_
            self.bias = 0.0

        return self

    def fit(self, X: Any, y: Any) -> RecursiveLeastSquares:
        """Fit on full initial dataset (resets state)."""
        X_arr, _ = self._prepare_data(X)
        self._init_state(X_arr.shape[1])
        return self.partial_fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for query samples."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() or partial_fit() first.")
        X_arr, _ = self._prepare_data(X)
        return X_arr @ self.weights + self.bias

    def score(self, X: Any, y: Any) -> float:
        """Compute R^2 score."""
        _, y_arr = self._prepare_data(X, y)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "lambda_": self.lambda_,
            "delta": self.delta,
            "fit_intercept": self.fit_intercept,
        }

    def set_params(self, **params: Any) -> RecursiveLeastSquares:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class KalmanFilterRegression:
    """
    Dynamic State-Space Linear Regression via Kalman Filtering.

    Models time-varying coefficients:
        w_t = w_{t-1} + nu_t,   nu_t ~ N(0, Q)
        y_t = x_t^T w_t + eps_t, eps_t ~ N(0, R)
    """

    def __init__(
        self,
        q: float = 1e-4,
        r: float = 1.0,
        fit_intercept: bool = True,
        q_cov: float | None = None,
        r_cov: float | None = None,
    ) -> None:
        if q_cov is not None:
            q = q_cov
        if r_cov is not None:
            r = r_cov
        self.q = float(q)
        self.r = float(r)
        self.fit_intercept = bool(fit_intercept)

        self.theta_: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.P_: np.ndarray | None = None
        self.n_samples_seen_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def _prepare_data(self, X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        elif isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            if (
                y is not None
                and (np.ndim(y) == 0 or len(np.asarray(y).ravel()) == 1)
                and len(X_arr) > 1
            ):
                X_arr = X_arr.reshape(1, -1)
            elif self.n_features_in_ > 0 and len(X_arr) == self.n_features_in_:
                X_arr = X_arr.reshape(1, -1)
            else:
                X_arr = X_arr.reshape(-1, 1)

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            else:
                y_arr = np.asarray(y, dtype=np.float64).ravel()

        return X_arr, y_arr

    def _init_state(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        dim = n_features + (1 if self.fit_intercept else 0)
        self.theta_ = np.zeros(dim, dtype=np.float64)
        self.P_ = 1000.0 * np.eye(dim, dtype=np.float64)
        self.n_samples_seen_ = 0

    def partial_fit(self, X: Any, y: Any) -> KalmanFilterRegression:
        """Online state-space update via Kalman filter recursion."""
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape

        if self.theta_ is None or self.P_ is None:
            self._init_state(n_features)

        dim = len(self.theta_)
        Q = self.q * np.eye(dim, dtype=np.float64)

        for i in range(n_samples):
            x_i = X_arr[i]
            y_i = float(y_arr[i])

            if self.fit_intercept:
                x_aug = np.append(x_i, 1.0)
            else:
                x_aug = x_i

            # 1. Predict step
            P_pred = self.P_ + Q

            # 2. Innovation and residual covariance S
            y_pred = float(x_aug @ self.theta_)
            error = y_i - y_pred
            Px = P_pred @ x_aug
            S = float(x_aug @ Px) + self.r

            # 3. Kalman gain K
            K = Px / max(S, 1e-12)

            # 4. State update
            self.theta_ += K * error

            # 5. Covariance update
            self.P_ = P_pred - np.outer(K, x_aug) @ P_pred
            self.P_ = 0.5 * (self.P_ + self.P_.T)
            self.n_samples_seen_ += 1

        if self.fit_intercept:
            self.weights = self.theta_[:-1]
            self.bias = float(self.theta_[-1])
        else:
            self.weights = self.theta_
            self.bias = 0.0

        return self

    def fit(self, X: Any, y: Any) -> KalmanFilterRegression:
        X_arr, _ = self._prepare_data(X)
        self._init_state(X_arr.shape[1])
        return self.partial_fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        X_arr, _ = self._prepare_data(X)
        return X_arr @ self.weights + self.bias

    def score(self, X: Any, y: Any) -> float:
        _, y_arr = self._prepare_data(X, y)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"q": self.q, "r": self.r, "fit_intercept": self.fit_intercept}

    def set_params(self, **params: Any) -> KalmanFilterRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self

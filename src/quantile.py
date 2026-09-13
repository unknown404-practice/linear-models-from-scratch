"""
Quantile Regression Engine from Scratch.

Provides non-parametric conditional quantile modeling across arbitrary quantiles
tau in (0, 1) by minimizing asymmetric check / pinball loss via smoothed
Iteratively Reweighted Least Squares (IRLS).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class QuantileRegressorScratch:
    """
    Quantile Linear Regression via Smoothed Iteratively Reweighted Least Squares.

    Minimizes:
        J(w, b) = (1 / N) * sum_i rho_tau(y_i - (x_i^T w + b))
    where:
        rho_tau(u) = u * (tau - I(u < 0))

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile level tau in (0, 1).
        - tau = 0.5: Conditional median regression (L1 loss).
        - tau = 0.1 / 0.9: 10th / 90th percentile risk envelope.
    fit_intercept : bool, default=True
        Whether to calculate an intercept for this model.
    max_iter : int, default=200
        Maximum number of IRLS iterations.
    tol : float, default=1e-5
        Convergence tolerance for parameter changes.
    l2_reg : float, default=1e-5
        L2 regularization applied to feature weights (excluding intercept).
    epsilon : float, default=1e-4
        Zero-residual smoothing denominator to prevent division-by-zero.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 200,
        tol: float = 1e-5,
        l2_reg: float = 1e-5,
        epsilon: float = 1e-4,
    ) -> None:
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        self.quantile = float(quantile)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.l2_reg = float(l2_reg)
        self.epsilon = float(epsilon)

        # Fitted attributes
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []
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
            X_arr = X_arr.reshape(-1, 1)

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float64).ravel()
            else:
                y_arr = np.asarray(y, dtype=np.float64).ravel()

        return X_arr, y_arr

    def _pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean pinball / check loss."""
        diff = y_true - y_pred
        loss = np.where(diff >= 0, self.quantile * diff, (self.quantile - 1.0) * diff)
        return float(np.mean(loss))

    def fit(self, X: Any, y: Any) -> QuantileRegressorScratch:
        """
        Fit quantile regression parameters via smoothed IRLS.
        """
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            X_aug = np.hstack([X_arr, np.ones((n_samples, 1), dtype=np.float64)])
            n_params = n_features + 1
        else:
            X_aug = X_arr
            n_params = n_features

        reg_matrix = self.l2_reg * np.eye(n_params, dtype=np.float64)
        if self.fit_intercept:
            reg_matrix[-1, -1] = 0.0

        # Initialize theta with unweighted OLS
        try:
            theta = np.linalg.solve(X_aug.T @ X_aug + reg_matrix, X_aug.T @ y_arr)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_aug) @ y_arr

        self.loss_history_ = []

        for iteration in range(1, self.max_iter + 1):
            y_pred = X_aug @ theta
            residuals = y_arr - y_pred

            loss = self._pinball_loss(y_arr, y_pred)
            self.loss_history_.append(loss)

            # Smoothed IRLS weights:
            # W_ii = |tau - I(r_i < 0)| / max(eps, |r_i|)
            indicator = (residuals < 0.0).astype(np.float64)
            numer = np.abs(self.quantile - indicator)
            denom = np.maximum(self.epsilon, np.abs(residuals))
            weights = numer / denom

            # Ensure minimum positive weight
            weights = np.maximum(weights, 1e-12)

            # Weighted least squares: (X_aug^T W X_aug + reg) theta_new = X_aug^T W y
            WX = weights[:, np.newaxis] * X_aug
            XtWX = X_aug.T @ WX + reg_matrix
            XtWy = X_aug.T @ (weights * y_arr)

            try:
                theta_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                theta_new = np.linalg.pinv(XtWX) @ XtWy

            diff = float(np.linalg.norm(theta_new - theta))
            theta = theta_new

            if diff < self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        if self.fit_intercept:
            self.weights = theta[:-1]
            self.bias = float(theta[-1])
        else:
            self.weights = theta
            self.bias = 0.0

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict conditional quantile values."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_arr, _ = self._prepare_data(X)
        return X_arr @ self.weights + self.bias

    def score(self, X: Any, y: Any) -> float:
        """
        Compute negative pinball loss (higher is better for scikit-learn convention).
        """
        _, y_arr = self._prepare_data(X, y)
        y_pred = self.predict(X)
        return -float(self._pinball_loss(y_arr, y_pred))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "l2_reg": self.l2_reg,
            "epsilon": self.epsilon,
        }

    def set_params(self, **params: Any) -> QuantileRegressorScratch:
        for key, value in params.items():
            setattr(self, key, value)
        return self

"""
Generalized Linear Models (GLM) Engine from Scratch.

Provides a unified Exponential Dispersion Family solver using pure NumPy
Iteratively Reweighted Least Squares (IRLS) with support for:
- Logistic Regression (Bernoulli family / Logit link)
- Poisson Count Regression (Poisson family / Log link)
- Gamma Regression (Gamma family / Log link for positive skewed continuous targets)
- Gaussian Linear Regression (Gaussian family / Identity link)
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class GeneralizedLinearModel:
    """
    Generalized Linear Model (GLM) fitted via Iteratively Reweighted Least Squares (IRLS).

    Parameters
    ----------
    family : str, default="gaussian"
        Exponential dispersion family: "gaussian", "binomial", "poisson", or "gamma".
    fit_intercept : bool, default=True
        Whether to fit an intercept scalar b.
    max_iter : int, default=100
        Maximum number of IRLS iterations.
    tol : float, default=1e-6
        Convergence tolerance for parameter changes.
    l2_reg : float, default=1e-5
        L2 regularization parameter (applied to weights, not intercept)
        to guarantee numerical stability in ill-conditioned Hessians.
    """

    def __init__(
        self,
        family: str = "gaussian",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        l2_reg: float = 1e-5,
    ) -> None:
        self.family = str(family).lower()
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.l2_reg = float(l2_reg)

        # Fitted attributes
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_iter_: int = 0
        self.deviance_: float = 0.0
        self.deviance_history_: list[float] = []
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

    def _link(self, mu: np.ndarray) -> np.ndarray:
        """Compute linear predictor eta = g(mu)."""
        eps = 1e-12
        if self.family == "binomial":
            mu_safe = np.clip(mu, eps, 1.0 - eps)
            return np.log(mu_safe / (1.0 - mu_safe))
        elif self.family in ("poisson", "gamma"):
            return np.log(np.maximum(mu, eps))
        else:  # gaussian
            return mu

    def _link_inv(self, eta: np.ndarray) -> np.ndarray:
        """Compute mean response mu = g^-1(eta)."""
        if self.family == "binomial":
            # Numerically stable sigmoid
            return np.where(
                eta >= 0,
                1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0))),
                np.exp(np.clip(eta, -50.0, 50.0)) / (1.0 + np.exp(np.clip(eta, -50.0, 50.0))),
            )
        elif self.family in ("poisson", "gamma"):
            return np.exp(np.clip(eta, -30.0, 30.0))
        else:  # gaussian
            return eta

    def _compute_weights_and_working_response(
        self, y: np.ndarray, eta: np.ndarray, mu: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute IRLS diagonal weights W_ii and working response z_i.
        """
        eps = 1e-12
        if self.family == "binomial":
            mu_safe = np.clip(mu, eps, 1.0 - eps)
            # W_ii = mu * (1 - mu)
            w = mu_safe * (1.0 - mu_safe)
            # z_i = eta + (y - mu) / (mu * (1 - mu))
            z = eta + (y - mu_safe) / (w + eps)
            return w, z

        elif self.family == "poisson":
            mu_safe = np.maximum(mu, eps)
            # W_ii = mu
            w = mu_safe
            # z_i = eta + (y - mu) / mu
            z = eta + (y - mu_safe) / mu_safe
            return w, z

        elif self.family == "gamma":
            mu_safe = np.maximum(mu, eps)
            # For log link, V(mu) = mu^2, g'(mu) = 1/mu => W_ii = 1.0
            w = np.ones_like(y, dtype=np.float64)
            z = eta + (y - mu_safe) / mu_safe
            return w, z

        else:  # gaussian
            w = np.ones_like(y, dtype=np.float64)
            z = y
            return w, z

    def _compute_deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Compute unit deviance for the chosen family."""
        eps = 1e-12
        if self.family == "binomial":
            mu_safe = np.clip(mu, eps, 1.0 - eps)
            dev = 2.0 * np.sum(
                np.where(y > 0, y * np.log(np.maximum(y, eps) / mu_safe), 0.0)
                + np.where(
                    y < 1, (1.0 - y) * np.log(np.maximum(1.0 - y, eps) / (1.0 - mu_safe)), 0.0
                )
            )
            return float(dev)

        elif self.family == "poisson":
            mu_safe = np.maximum(mu, eps)
            dev = 2.0 * np.sum(
                np.where(y > 0, y * np.log(np.maximum(y, eps) / mu_safe) - (y - mu_safe), mu_safe)
            )
            return float(dev)

        elif self.family == "gamma":
            mu_safe = np.maximum(mu, eps)
            y_safe = np.maximum(y, eps)
            dev = 2.0 * np.sum(-np.log(y_safe / mu_safe) + (y_safe - mu_safe) / mu_safe)
            return float(dev)

        else:  # gaussian
            return float(np.sum((y - mu) ** 2))

    def fit(self, X: Any, y: Any) -> GeneralizedLinearModel:
        """
        Fit the Generalized Linear Model using Iteratively Reweighted Least Squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # Build augmented design matrix if fit_intercept is True
        if self.fit_intercept:
            X_aug = np.hstack([X_arr, np.ones((n_samples, 1), dtype=np.float64)])
            n_params = n_features + 1
        else:
            X_aug = X_arr
            n_params = n_features

        # Regularization matrix (do not penalize intercept)
        reg_matrix = self.l2_reg * np.eye(n_params, dtype=np.float64)
        if self.fit_intercept:
            reg_matrix[-1, -1] = 0.0

        # Initialize mu with null model mean
        mean_y = float(np.mean(y_arr))
        if self.family == "binomial":
            mu = np.full(n_samples, np.clip(mean_y, 0.05, 0.95), dtype=np.float64)
        elif self.family in ("poisson", "gamma"):
            mu = np.full(n_samples, max(mean_y, 0.1), dtype=np.float64)
        else:
            mu = np.full(n_samples, mean_y, dtype=np.float64)

        eta = self._link(mu)
        theta = np.zeros(n_params, dtype=np.float64)

        self.deviance_history_ = []
        dev_prev = self._compute_deviance(y_arr, mu)
        self.deviance_history_.append(dev_prev)

        for iteration in range(1, self.max_iter + 1):
            w, z = self._compute_weights_and_working_response(y_arr, eta, mu)

            # Prevent zero or negative weights
            w = np.maximum(w, 1e-12)

            # Weighted regularized normal equations: (X^T W X + lambda I) theta = X^T W z
            # Vectorized computation: X^T (w * X)
            WX = w[:, np.newaxis] * X_aug
            XtWX = X_aug.T @ WX + reg_matrix
            XtWz = X_aug.T @ (w * z)

            try:
                theta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                theta_new = np.linalg.pinv(XtWX) @ XtWz

            eta = X_aug @ theta_new
            mu = self._link_inv(eta)

            dev_new = self._compute_deviance(y_arr, mu)
            self.deviance_history_.append(dev_new)

            param_change = float(np.linalg.norm(theta_new - theta))
            theta = theta_new

            if param_change < self.tol or abs(dev_prev - dev_new) < self.tol:
                self.n_iter_ = iteration
                break
            dev_prev = dev_new
        else:
            self.n_iter_ = self.max_iter

        if self.fit_intercept:
            self.weights = theta[:-1]
            self.bias = float(theta[-1])
        else:
            self.weights = theta
            self.bias = 0.0

        self.deviance_ = dev_prev
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict conditional expected mean response mu = g^-1(X w + b)."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_arr, _ = self._prepare_data(X)
        eta = X_arr @ self.weights + self.bias
        return self._link_inv(eta)

    def score(self, X: Any, y: Any) -> float:
        """Compute deviance explained or R^2 score."""
        _, y_arr = self._prepare_data(X, y)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    @property
    def pseudo_r2_(self) -> float:
        """Compute deviance pseudo R^2: 1 - (D_residual / D_null)."""
        if not hasattr(self, "deviance_history_") or len(self.deviance_history_) < 2:
            return 0.0
        d_null = self.deviance_history_[0]
        d_res = getattr(self, "deviance_", d_null)
        if d_null <= 0.0:
            return 1.0 if d_res <= 0.0 else 0.0
        return float(max(0.0, min(1.0, 1.0 - (d_res / d_null))))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "family": self.family,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "l2_reg": self.l2_reg,
        }

    def set_params(self, **params: Any) -> GeneralizedLinearModel:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LogisticRegressionScratch(GeneralizedLinearModel):
    """
    Logistic Regression for Binary Classification via Bernoulli IRLS.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        l2_reg: float = 1e-5,
        family: str = "binomial",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            family="binomial",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            l2_reg=l2_reg,
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict binary class probabilities [P(y=0), P(y=1)]."""
        p1 = super().predict(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete binary class labels in {0, 1}."""
        p1 = super().predict(X)
        return (p1 >= threshold).astype(np.int64)

    def score(self, X: Any, y: Any) -> float:
        """Compute binary classification accuracy."""
        _, y_arr = self._prepare_data(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))


class PoissonRegressionScratch(GeneralizedLinearModel):
    """
    Poisson Regression for Count Data Targets via Log-Link IRLS.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        l2_reg: float = 1e-5,
        family: str = "poisson",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            family="poisson",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            l2_reg=l2_reg,
        )


class GammaRegressionScratch(GeneralizedLinearModel):
    """
    Gamma Regression for Strictly Positive, Skewed Continuous Targets.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        l2_reg: float = 1e-5,
        family: str = "gamma",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            family="gamma",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            l2_reg=l2_reg,
        )

"""
Bayesian Linear Regression Engine from Scratch.

Provides fully analytical conjugate Gaussian posterior inference, predictive
distribution decomposition into epistemic and aleatoric uncertainties,
Automatic Relevance Determination (ARD) / Empirical Bayes hyperparameter
optimization, and stochastic posterior sampling.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Conjugate Gaussian Prior.

    Prior:
        w ~ N(0, alpha^-1 * I)
    Likelihood:
        y | X, w, beta ~ N(X w + b, beta^-1 * I)
    Posterior:
        p(w | y, X) = N(m_N, S_N)
        S_N^-1 = alpha * I + beta * X^T X
        m_N = beta * S_N * X^T y

    Predictive Distribution:
        p(y_* | x_*, X, y) = N(mu_*, sigma_*^2)
        mu_* = x_*^T m_N + b
        sigma_*^2 = (1 / beta) + x_*^T S_N x_*
                    [aleatoric]   [epistemic]

    Parameters
    ----------
    alpha : float, default=1.0
        Prior precision on regression weights.
    beta : float, default=1.0
        Likelihood noise precision (inverse variance: 1 / sigma^2).
    fit_intercept : bool, default=True
        Whether to calculate an intercept for this model.
    max_iter : int, default=300
        Maximum iterations for ARD / Empirical Bayes hyperparameter updates.
    tol : float, default=1e-4
        Convergence tolerance for ARD evidence maximization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Fitted attributes
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.cov_: np.ndarray | None = None
        self.alpha_: float = self.alpha
        self.beta_: float = self.beta
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None
        self.scores_: list[float] = []

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

    def fit(self, X: Any, y: Any) -> BayesianLinearRegression:
        """
        Fit Bayesian Linear Regression via analytical conjugate Gaussian posterior.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BayesianLinearRegression
            Fitted estimator.
        """
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            self.x_mean_ = np.mean(X_arr, axis=0)
            self.y_mean_ = float(np.mean(y_arr))
            X_c = X_arr - self.x_mean_
            y_c = y_arr - self.y_mean_
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = 0.0
            X_c = X_arr
            y_c = y_arr

        # S_N^-1 = alpha * I + beta * X_c^T X_c
        precision_matrix = self.alpha * np.eye(n_features, dtype=np.float64) + self.beta * (
            X_c.T @ X_c
        )
        try:
            cov = np.linalg.inv(precision_matrix)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(precision_matrix)

        # m_N = beta * S_N * X_c^T y_c
        weights = self.beta * (cov @ (X_c.T @ y_c))

        if self.fit_intercept:
            bias = float(self.y_mean_ - float(self.x_mean_ @ weights))
        else:
            bias = 0.0

        self.weights = weights
        self.bias = bias
        self.cov_ = cov
        self.alpha_ = self.alpha
        self.beta_ = self.beta
        return self

    def fit_ard(self, X: Any, y: Any) -> BayesianLinearRegression:
        """
        Fit Bayesian Linear Regression with Automatic Relevance Determination (ARD)
        via Empirical Bayes / Evidence Approximation (Type-II Maximum Likelihood).
        """
        X_arr, y_arr = self._prepare_data(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            self.x_mean_ = np.mean(X_arr, axis=0)
            self.y_mean_ = float(np.mean(y_arr))
            X_c = X_arr - self.x_mean_
            y_c = y_arr - self.y_mean_
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = 0.0
            X_c = X_arr
            y_c = y_arr

        XtX = X_c.T @ X_c
        Xty = X_c.T @ y_c

        alpha = float(self.alpha)
        beta = float(self.beta)
        self.scores_ = []

        # Eigenvalues of unscaled XtX
        eigvals = np.linalg.eigvalsh(XtX)
        eigvals = np.maximum(eigvals, 0.0)

        for _ in range(self.max_iter):
            precision_mat = alpha * np.eye(n_features, dtype=np.float64) + beta * XtX
            try:
                cov = np.linalg.inv(precision_mat)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(precision_mat)

            weights = beta * (cov @ Xty)

            # Effective number of parameters gamma = sum(lambda_i / (alpha + lambda_i))
            effective_lambdas = beta * eigvals
            gamma = float(np.sum(effective_lambdas / (alpha + effective_lambdas + 1e-12)))

            w_norm_sq = float(weights @ weights)
            alpha_new = gamma / (w_norm_sq + 1e-12)

            residuals = y_c - X_c @ weights
            sse = float(residuals @ residuals)
            dof = max(n_samples - gamma, 1e-12)
            beta_new = dof / (sse + 1e-12)

            # Compute log marginal likelihood
            log_det = float(np.sum(np.log(alpha + effective_lambdas + 1e-12)))
            log_lik = 0.5 * (
                n_features * np.log(max(alpha, 1e-12))
                + n_samples * np.log(max(beta, 1e-12))
                - beta * sse
                - alpha * w_norm_sq
                - log_det
                - n_samples * np.log(2.0 * np.pi)
            )
            self.scores_.append(log_lik)

            if abs(alpha_new - alpha) < self.tol and abs(beta_new - beta) < self.tol:
                alpha = alpha_new
                beta = beta_new
                break

            alpha = alpha_new
            beta = beta_new

        self.alpha_ = float(alpha)
        self.beta_ = float(beta)
        self.weights = weights
        self.cov_ = cov

        if self.fit_intercept:
            self.bias = float(self.y_mean_ - float(self.x_mean_ @ weights))
        else:
            self.bias = 0.0

        return self

    def predict(
        self, X: Any, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict using the Bayesian posterior distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test inputs.
        return_std : bool, default=False
            Whether to return predictive standard deviations decomposing
            into aleatoric noise and epistemic uncertainty.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predictive posterior mean.
        y_std : ndarray of shape (n_samples,), optional
            Predictive standard deviation (only if return_std=True).
        """
        if self.weights is None or self.cov_ is None or self.x_mean_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() or fit_ard() first.")

        X_arr, _ = self._prepare_data(X)
        mu = X_arr @ self.weights + self.bias

        if not return_std:
            return mu

        # Epistemic variance: x_*^T S_N x_*
        if self.fit_intercept:
            X_c = X_arr - self.x_mean_
        else:
            X_c = X_arr

        # Vectorized row-wise quadratic form: diag(X_c S_N X_c^T)
        epistemic_var = np.sum((X_c @ self.cov_) * X_c, axis=1)
        epistemic_var = np.maximum(epistemic_var, 0.0)

        # Total variance = aleatoric (1 / beta) + epistemic (x_*^T S_N x_*)
        aleatoric_var = 1.0 / max(self.beta_, 1e-12)
        total_std = np.sqrt(aleatoric_var + epistemic_var)

        return mu, total_std

    def sample_weights(
        self,
        n_samples: int = 5,
        seed: int | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Draw stochastic parameter weight vectors from the Gaussian posterior N(m_N, S_N).
        """
        if self.weights is None or self.cov_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if random_state is not None:
            seed = random_state
        rng = np.random.default_rng(seed)
        # Ensure covariance is strictly symmetric positive semi-definite
        cov_sym = 0.5 * (self.cov_ + self.cov_.T)
        return rng.multivariate_normal(self.weights, cov_sym, size=n_samples)

    def score(self, X: Any, y: Any) -> float:
        """Compute R^2 score against target values."""
        _, y_true = self._prepare_data(X, y)
        y_pred = self.predict(X, return_std=False)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> BayesianLinearRegression:
        for key, value in params.items():
            setattr(self, key, value)
        return self

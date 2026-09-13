"""
Closed-Form Analytical Solvers for Linear Regression from Scratch.

This module provides exact, non-iterative linear algebra solvers:
1. SVD (Singular Value Decomposition / Moore-Penrose Pseudoinverse) - The gold standard
   for numerical stability; handles singular, collinear, and rank-deficient feature spaces.
2. Cholesky Decomposition (X^T X = L L^T) - The fastest analytical solver; utilizes triangular
   forward and back substitution.
3. QR Decomposition (X = Q R) - Numerically robust orthogonal factorization.
4. Normal Equation ((X^T X)^-1 X^T y) - The classical textbook method with condition number diagnostics.
"""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
import scipy.linalg


class ClosedFormLinearRegression:
    """
    Closed-form analytical Linear Regression estimator.

    Parameters
    ----------
    method : {'svd', 'cholesky', 'qr', 'normal_equation'}, default='svd'
        Numerical solver algorithm:
        - 'svd': Singular Value Decomposition. Most robust, handles rank-deficient matrices.
        - 'cholesky': Cholesky factorization of X^T X. Fastest, requires full rank.
        - 'qr': QR decomposition of X. Robust orthogonal factorization.
        - 'normal_equation': Naive direct inversion of (X^T X). For pedagogical comparisons.
    rcond : float, default=1e-15
        Cutoff for small singular values in SVD solver.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned coefficients.
    bias : float
        Learned intercept.
    condition_number_ : float
        Condition number kappa = sigma_max / sigma_min of the centered feature matrix.
    rank_ : int
        Effective numerical rank of the feature matrix.
    singular_values_ : np.ndarray
        Singular values of the centered feature matrix.
    """

    def __init__(
        self,
        method: Literal["svd", "cholesky", "qr", "normal_equation"] = "svd",
        rcond: float = 1e-15,
        fit_intercept: bool = True,
    ) -> None:
        if method == "normal":
            method = "normal_equation"
        valid_methods = ("svd", "cholesky", "qr", "normal_equation")
        if method not in valid_methods:
            raise ValueError(f"Unknown solver method '{method}'. Valid options: {valid_methods}")
        self.method = method
        self.rcond = float(rcond)
        self.fit_intercept = bool(fit_intercept)

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.condition_number_: float = 1.0
        self.rank_: int = 0
        self.singular_values_: np.ndarray | None = None
        self.n_features_in_: int = 0

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> "ClosedFormLinearRegression":
        """
        Fit closed-form linear model on training inputs.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray or pd.Series of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : ClosedFormLinearRegression
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Dimension mismatch: X {X_arr.shape} vs y {y_arr.shape}")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # Center data if fitting intercept (numerically superior to adding a column of 1s)
        if self.fit_intercept:
            mu_X = np.mean(X_arr, axis=0)
            mu_y = float(np.mean(y_arr))
            X_c = X_arr - mu_X
            y_c = y_arr - mu_y
        else:
            mu_X = np.zeros(n_features, dtype=np.float64)
            mu_y = 0.0
            X_c = X_arr
            y_c = y_arr

        # Compute SVD metrics and condition number
        singular_vals = np.linalg.svd(X_c, compute_uv=False)
        self.singular_values_ = singular_vals
        s_max = singular_vals[0]
        s_min = singular_vals[-1]
        self.condition_number_ = float(s_max / s_min) if s_min > 1e-15 else float("inf")

        cutoff = self.rcond * s_max
        self.rank_ = int(np.sum(singular_vals > cutoff))

        # Solve according to requested numerical method
        if self.method == "svd":
            # X = U Sigma V^T => w = V Sigma^+ U^T y
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
            # w = Vt.T @ (S_inv * (U.T @ y_c))
            w = Vt.T @ (S_inv * (U.T @ y_c))

        elif self.method == "cholesky":
            # X^T X w = X^T y  where X^T X = L L^T
            if self.rank_ < n_features:
                raise np.linalg.LinAlgError(
                    f"Cholesky solver requires full column rank. Matrix rank is {self.rank_} < {n_features}."
                )
            A = X_c.T @ X_c
            b_vec = X_c.T @ y_c
            try:
                L = np.linalg.cholesky(A)
                # Forward substitution: L z = b_vec
                z = scipy.linalg.solve_triangular(L, b_vec, lower=True)
                # Back substitution: L^T w = z
                w = scipy.linalg.solve_triangular(L.T, z, lower=False)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(f"Cholesky decomposition failed: {e}")

        elif self.method == "qr":
            # X = Q R => R w = Q^T y
            if self.rank_ < n_features:
                raise np.linalg.LinAlgError(
                    f"QR solver requires full column rank. Matrix rank is {self.rank_} < {n_features}."
                )
            Q, R = np.linalg.qr(X_c, mode="reduced")
            w = scipy.linalg.solve_triangular(R, Q.T @ y_c)

        elif self.method == "normal_equation":
            # Naive inversion: w = (X^T X)^-1 X^T y
            if self.rank_ < n_features or self.condition_number_ > 1e12:
                raise np.linalg.LinAlgError(
                    f"Matrix is singular or ill-conditioned (condition number={self.condition_number_:.2e}). "
                    "Normal equation inversion is mathematically unstable; use method='svd'."
                )
            A = X_c.T @ X_c
            w = np.linalg.inv(A) @ (X_c.T @ y_c)

        self.weights = w
        if self.fit_intercept:
            self.bias = float(mu_y - np.dot(mu_X, w))
        else:
            self.bias = 0.0

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict continuous target using learned weights."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
            )
        return (X_arr @ self.weights) + self.bias

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """Return R^2 coefficient of determination."""
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "method": self.method,
            "fit_intercept": self.fit_intercept,
            "rcond": self.rcond,
        }

    def set_params(self, **params: Any) -> ClosedFormLinearRegression:
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}.")
        return self

    def __repr__(self) -> str:
        status = "Fitted" if self.weights is not None else "Unfitted"
        return f"ClosedFormLinearRegression(method='{self.method}', kappa={self.condition_number_:.2f}) [{status}]"


class NewtonLinearRegression:
    """
    Second-order Newton-Raphson Linear Regression optimizer.

    Solves the quadratic MSE loss surface using exact analytical Hessian computation:
        g = (2 / N) * X^T (X w - y) + 2 * alpha * w
        H = (2 / N) * X^T X + 2 * alpha * I
        Delta w = - (H + damping * I)^(-1) * g

    Because the MSE loss surface is strictly quadratic and convex, Newton's method
    reaches the exact global optimum in precisely 1 iteration.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularization parameter (Ridge penalty).
    damping : float, default=1e-6
        Levenberg-Marquardt damping parameter added to the diagonal of H.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    max_iters : int, default=10
        Maximum number of Newton steps.
    tol : float, default=1e-8
        Convergence tolerance on weight step norm.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        damping: float = 1e-6,
        fit_intercept: bool = True,
        max_iters: int = 10,
        tol: float = 1e-8,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if damping < 0:
            raise ValueError(f"damping must be >= 0, got {damping}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")

        self.alpha = float(alpha)
        self.damping = float(damping)
        self.fit_intercept = fit_intercept
        self.max_iters = int(max_iters)
        self.tol = float(tol)

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_iter_: int = 0
        self.grad_norm_: float = 0.0
        self.hessian_condition_number_: float = 0.0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> NewtonLinearRegression:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            mu_X = np.mean(X_arr, axis=0)
            mu_y = float(np.mean(y_arr))
            X_design = X_arr - mu_X
            y_design = y_arr - mu_y
        else:
            mu_X = np.zeros(n_features)
            mu_y = 0.0
            X_design = X_arr
            y_design = y_arr

        # Exact analytical Hessian for MSE loss
        # H = (2 / N) * X^T X + 2 * alpha * I
        H = (2.0 / n_samples) * (X_design.T @ X_design)
        if self.alpha > 0.0:
            H += 2.0 * self.alpha * np.eye(n_features)

        # Condition number of the Hessian
        try:
            self.hessian_condition_number_ = float(np.linalg.cond(H))
        except Exception:
            self.hessian_condition_number_ = float("inf")

        # Damped Hessian for numerical stability
        H_damped = H.copy()
        if self.damping > 0.0:
            H_damped += self.damping * np.eye(n_features)

        w = np.zeros(n_features, dtype=np.float64)
        n_iter = 0

        for iteration in range(1, self.max_iters + 1):
            # Exact gradient: g = (2 / N) * X^T (X w - y) + 2 * alpha * w
            residuals = (X_design @ w) - y_design
            g = (2.0 / n_samples) * (X_design.T @ residuals)
            if self.alpha > 0.0:
                g += 2.0 * self.alpha * w

            self.grad_norm_ = float(np.linalg.norm(g))
            if self.grad_norm_ < self.tol:
                break

            # Solve Newton system: H * delta_w = -g
            try:
                delta_w = -scipy.linalg.solve(H_damped, g, assume_a="pos")
            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
                delta_w = -np.linalg.pinv(H_damped) @ g

            w += delta_w
            n_iter += 1

            # Check post-update gradient
            residuals_new = (X_design @ w) - y_design
            g_new = (2.0 / n_samples) * (X_design.T @ residuals_new)
            if self.alpha > 0.0:
                g_new += 2.0 * self.alpha * w
            self.grad_norm_ = float(np.linalg.norm(g_new))

            if self.grad_norm_ < self.tol or np.linalg.norm(delta_w) < self.tol:
                break

        self.weights = w
        self.bias = float(mu_y - np.dot(mu_X, w)) if self.fit_intercept else 0.0
        self.n_iter_ = n_iter
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("NewtonLinearRegression is not fitted yet.")
        X_arr = (
            X.values.astype(np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )
        return X_arr @ self.weights + self.bias

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        from src.linear_regression import compute_r2

        return compute_r2(y, self.predict(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "damping": self.damping,
            "fit_intercept": self.fit_intercept,
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> NewtonLinearRegression:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        status = "Fitted" if self.weights is not None else "Unfitted"
        return f"NewtonLinearRegression(alpha={self.alpha}, damping={self.damping}, n_iter={self.n_iter_}) [{status}]"

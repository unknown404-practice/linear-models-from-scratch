"""
Linear Regression from Scratch using Vectorized Gradient Descent & Coordinate Descent.

This module implements a production-grade, numerically stable, and memory-efficient
Linear Regression estimator trained via:
1. Gradient Descent (Full-Batch, Mini-Batch, SGD with Momentum/Nesterov, RMSprop, Adam)
2. Exact Cyclic Coordinate Descent with Soft-Thresholding for L1 Lasso and ElasticNet

Mathematical Foundations:
-------------------------
1. Linear Hypothesis:
   Given input matrix X in R^{N x D} and target vector y in R^N:
       ŷ = X @ w + b
   where:
       N: Number of samples (rows)
       D: Number of features (columns)
       w: Weight vector in R^D
       b: Scalar bias (intercept) in R

2. Objective Function - Regularized Mean Squared Error:
       J(w, b) = (1 / N) * ||ŷ - y||_2^2 + α * [(1 - ρ) * ||w||_2^2 + ρ * ||w||_1]
   where:
       α >= 0: Regularization strength
       ρ in [0, 1]: ElasticNet mixing parameter (ρ=0 is Ridge L2, ρ=1 is Lasso L1)

3. Exact Gradient Derivation (MSE + Ridge L2):
   Let residual error vector e = ŷ - y in R^N:
       ∇_w J = (2 / N) * X^T @ e + 2 * α * (1 - ρ) * w
       ∇_b J = (2 / N) * 1^T @ e = 2 * mean(e)

4. Coordinate Descent with Soft-Thresholding (L1 Lasso):
   For each feature coordinate j in 1..D:
       r^{(-j)} = (y - ŷ) + X_{:, j} * w_j
       ρ_j = (1 / N) * X_{:, j}^T @ r^{(-j)}
       z_j = (1 / N) * ||X_{:, j}||_2^2
       w_j <- S(ρ_j, α * ρ) / (z_j + α * (1 - ρ))
   where S(v, γ) = sign(v) * max(|v| - γ, 0) is the soft-thresholding operator.
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np
import pandas as pd

from src.optimizers import Optimizer, get_optimizer
from src.schedulers import LRScheduler


def soft_threshold(z: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """
    Soft-thresholding proximal operator for L1 penalty:
        S(z, γ) = sign(z) * max(|z| - γ, 0)

    Shrinks values strictly inside [-γ, γ] to exact zero,
    and shifts values outside by γ towards zero.
    """
    if gamma <= 0.0:
        return z
    if isinstance(z, np.ndarray):
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)
    if abs(z) <= gamma:
        return 0.0
    return math.copysign(abs(z) - gamma, z)


def standardize_features(
    X: np.ndarray | pd.DataFrame,
    return_scaler: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | np.ndarray:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in X:
        z = (x - μ) / σ

    Features with zero variance (constant columns) are scaled with σ = 1.0 to
    prevent zero-division and NaN generation.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Array of shape (n_samples, n_features).
    return_scaler : bool, default=True
        Whether to return the computed mean and std parameters.

    Returns
    -------
    X_scaled : np.ndarray
        Standardized array of shape (n_samples, n_features).
    scaler : dict
        Dictionary containing 'mean' and 'std' arrays.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got {X_arr.ndim}D array instead.")

    # Compute empirical mean μ and standard deviation σ along axis 0
    mean = np.mean(X_arr, axis=0)
    std = np.std(X_arr, axis=0)

    # Prevent division by zero for constant features
    std_safe = np.where(std == 0.0, 1.0, std)

    # Standardize: z = (X - μ) / σ
    X_scaled = (X_arr - mean) / std_safe

    if return_scaler:
        scaler = {"mean": mean, "std": std_safe}
        return X_scaled, scaler
    return X_scaled


def apply_scaler(
    X: np.ndarray | pd.DataFrame,
    scaler: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Apply precomputed mean and std scalers to feature matrix X.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Array of shape (n_samples, n_features).
    scaler : dict
        Dictionary containing 'mean' and 'std' numpy arrays.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled array of shape (n_samples, n_features).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    mean = scaler["mean"]
    std = scaler["std"]
    return (X_arr - mean) / std


def compute_mse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Compute Mean Squared Error (MSE):
        MSE = (1 / N) * Σ (y_i - ŷ_i)^2
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Shape mismatch: y_true {y_t.shape} vs y_pred {y_p.shape}")
    errors = y_t - y_p
    return float(np.mean(errors**2))


def compute_r2(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """
    Compute coefficient of determination R^2 score:
        R^2 = 1 - (SS_res / SS_tot)
    where:
        SS_res = Σ (y_i - ŷ_i)^2
        SS_tot = Σ (y_i - ȳ)^2
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


class GradientDescentLinearRegression:
    """
    Production-grade Linear Regression model from scratch.

    Supports:
    - Full-batch gradient descent and Mini-batch stochastic gradient descent.
    - Advanced first-order optimizers: SGD, Momentum, Nesterov, RMSprop, Adam.
    - Learning rate schedules: Constant, Step, Exponential, Cosine Annealing.
    - Regularization: Ridge (L2), Lasso (L1), and ElasticNet.
    - Exact Cyclic Coordinate Descent for exact sparse feature selection.

    Parameters
    ----------
    lr : float, default=0.01
        Initial learning rate (step size η).
    n_iters : int, default=1000
        Number of epochs or coordinate descent iterations.
    seed : int or None, default=42
        Random seed for reproducible mini-batch shuffling.
    batch_size : int or None, default=None
        Size of mini-batches. None implies full-batch gradient descent.
    standardize : bool, default=False
        Whether to standardize features internally during training.
    optimizer : str or Optimizer, default='sgd'
        Optimizer engine ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam').
    scheduler : LRScheduler or None, default=None
        Learning rate scheduler.
    penalty : str, default='none'
        Regularization penalty: 'none', 'l2' (Ridge), 'l1' (Lasso), or 'elasticnet'.
    alpha : float, default=0.0
        Regularization strength constant.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (1.0 = Lasso, 0.0 = Ridge).
    tol : float, default=1e-5
        Convergence tolerance for early stopping.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        seed: int | None = 42,
        batch_size: int | None = None,
        standardize: bool = False,
        optimizer: str | Optimizer = "sgd",
        scheduler: LRScheduler | None = None,
        penalty: Literal["none", "l2", "ridge", "l1", "lasso", "elasticnet"] = "none",
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        tol: float = 1e-5,
        learning_rate: float | None = None,
        n_epochs: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if learning_rate is not None:
            lr = learning_rate
        if n_epochs is not None:
            n_iters = n_epochs
        if random_state is not None:
            seed = random_state
        if lr <= 0:
            raise ValueError(f"Learning rate lr must be positive, got {lr}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be at least 1, got {n_iters}")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 or None, got {batch_size}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")

        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.seed = seed
        self.batch_size = batch_size
        self.standardize = standardize
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.penalty = str(penalty).strip().lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.tol = float(tol)

        # Model parameters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None
        self.history_: list[float] = []

        # Internal standardization parameters
        self.scaler_: dict[str, np.ndarray] | None = None

    def _prepare_inputs(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert and validate matrix X and vector y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")

        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValueError("X contains NaN or Inf values. Clean input data before fitting.")

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            if y_arr.ndim != 1:
                raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"Sample size mismatch: X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]} elements."
                )
            if np.isnan(y_arr).any() or np.isinf(y_arr).any():
                raise ValueError("y contains NaN or Inf values.")

        return X_arr, y_arr

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> "GradientDescentLinearRegression":
        """Fit linear model with gradient descent."""
        self._fit_internal(X, y, record_history=True)
        return self

    def fit_with_history(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_interval: int = 1,
    ) -> tuple["GradientDescentLinearRegression", list[float]]:
        """Fit linear model and record MSE history after each epoch."""
        if eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {eval_interval}")
        history = self._fit_internal(X, y, record_history=True, eval_interval=eval_interval)
        return self, history

    def _fit_internal(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        record_history: bool = False,
        eval_interval: int = 1,
    ) -> list[float]:
        """Internal core gradient descent optimization loop."""
        X_arr, y_arr = self._prepare_inputs(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # Internal Standardization if requested
        if self.standardize:
            X_train, self.scaler_ = standardize_features(X_arr, return_scaler=True)
        else:
            X_train = X_arr
            self.scaler_ = None

        # Initialize weights and bias to zeros
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        history: list[float] = []
        rng = np.random.default_rng(self.seed)

        # Initialize modular optimizers for weights and bias
        opt_w = get_optimizer(self.optimizer, lr=self.lr)
        opt_b = get_optimizer(self.optimizer, lr=self.lr)

        is_minibatch = self.batch_size is not None and self.batch_size < n_samples
        bs = self.batch_size if is_minibatch else n_samples

        for epoch in range(self.n_iters):
            # Dynamic Learning Rate Schedule
            if self.scheduler is not None:
                current_lr = self.scheduler.get_lr(epoch)
                opt_w.lr = current_lr
                opt_b.lr = current_lr

            if not is_minibatch:
                # ==========================================
                # FULL-BATCH GRADIENT DESCENT
                # ==========================================
                y_pred = X_train @ w + b
                errors = y_pred - y_arr

                grad_w = (2.0 / n_samples) * (X_train.T @ errors)
                grad_b = 2.0 * float(np.mean(errors))

                # Ridge (L2) weight decay penalty
                if self.penalty in ("l2", "ridge") and self.alpha > 0.0:
                    grad_w += 2.0 * self.alpha * w
                elif self.penalty == "elasticnet" and self.alpha > 0.0:
                    grad_w += 2.0 * self.alpha * (1.0 - self.l1_ratio) * w

                # Apply optimizer update step
                w = np.asarray(opt_w.update("w", w, grad_w), dtype=np.float64)
                b = float(opt_b.update("b", b, grad_b))

            else:
                # ==========================================
                # MINI-BATCH GRADIENT DESCENT
                # ==========================================
                shuffled_indices = rng.permutation(n_samples)

                for start_idx in range(0, n_samples, bs):
                    end_idx = min(start_idx + bs, n_samples)
                    batch_idx = shuffled_indices[start_idx:end_idx]
                    batch_n = end_idx - start_idx

                    X_batch = X_train[batch_idx]
                    y_batch = y_arr[batch_idx]

                    y_pred_batch = X_batch @ w + b
                    errors_batch = y_pred_batch - y_batch

                    grad_w = (2.0 / batch_n) * (X_batch.T @ errors_batch)
                    grad_b = 2.0 * float(np.mean(errors_batch))

                    if self.penalty in ("l2", "ridge") and self.alpha > 0.0:
                        grad_w += 2.0 * self.alpha * w
                    elif self.penalty == "elasticnet" and self.alpha > 0.0:
                        grad_w += 2.0 * self.alpha * (1.0 - self.l1_ratio) * w

                    w = np.asarray(opt_w.update("w", w, grad_w), dtype=np.float64)
                    b = float(opt_b.update("b", b, grad_b))

            # Record MSE history if requested
            if record_history and (epoch % eval_interval == 0 or epoch == self.n_iters - 1):
                current_pred = X_train @ w + b
                current_mse = float(np.mean((current_pred - y_arr) ** 2))
                history.append(current_mse)

        # Store learned parameters
        self.weights = w
        self.bias = float(b)
        self.history_ = history

        # Compute unscaled coefficients for direct interpretability
        if self.standardize and self.scaler_ is not None:
            mu = self.scaler_["mean"]
            sigma = self.scaler_["std"]
            self.coef_ = w / sigma
            self.intercept_ = float(b - np.sum((w * mu) / sigma))
        else:
            self.coef_ = w.copy()
            self.intercept_ = float(b)

        return history

    def fit_coordinate_descent(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        max_iters: int | None = None,
        tol: float | None = None,
    ) -> "GradientDescentLinearRegression":
        """
        Fit linear model using exact Cyclic Coordinate Descent with Soft-Thresholding.

        Guarantees exact parameter sparsity (w_j = 0.0) for L1 Lasso and ElasticNet penalties.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Training input matrix.
        y : np.ndarray or pd.Series of shape (n_samples,)
            Target values.
        max_iters : int or None, default=None
            Maximum coordinate descent passes over all features. Defaults to self.n_iters.
        tol : float or None, default=None
            Convergence tolerance for maximum absolute weight change. Defaults to self.tol.

        Returns
        -------
        self : GradientDescentLinearRegression
        """
        iters = max_iters if max_iters is not None else self.n_iters
        threshold = tol if tol is not None else self.tol

        X_arr, y_arr = self._prepare_inputs(X, y)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.standardize:
            X_train, self.scaler_ = standardize_features(X_arr, return_scaler=True)
        else:
            X_train = X_arr
            self.scaler_ = None

        # Precompute column squared norms: z_j = (1 / N) * ||X_{:, j}||^2
        z = np.mean(X_train**2, axis=0)
        # Avoid division by zero on constant columns
        z = np.where(z == 0.0, 1.0, z)

        # Initialize weights and bias
        w = np.zeros(n_features, dtype=np.float64)
        b = float(np.mean(y_arr))

        # Effective penalties for soft-thresholding
        if self.penalty in ("l1", "lasso"):
            l1_pen = self.alpha
            l2_pen = 0.0
        elif self.penalty == "elasticnet":
            l1_pen = self.alpha * self.l1_ratio
            l2_pen = self.alpha * (1.0 - self.l1_ratio)
        else:
            l1_pen = 0.0
            l2_pen = self.alpha

        # Residual vector: r = y - (Xw + b)
        residuals = y_arr - (X_train @ w + b)

        for iteration in range(iters):
            w_max_change = 0.0

            # Cyclic update across each coordinate
            for j in range(n_features):
                w_j_old = w[j]
                X_j = X_train[:, j]

                # Compute correlation with partial residual:
                # rho_j = (1 / N) * X_j^T @ (residuals + X_j * w_j_old)
                rho_j = float(np.mean(X_j * (residuals + X_j * w_j_old)))

                # Soft-threshold update
                if l1_pen > 0.0:
                    w_j_new = float(soft_threshold(rho_j, l1_pen) / (z[j] + l2_pen))
                else:
                    w_j_new = float(rho_j / (z[j] + l2_pen))

                diff = w_j_new - w_j_old
                if diff != 0.0:
                    w[j] = w_j_new
                    # Update residuals in O(N) time without full recomputation
                    residuals -= diff * X_j
                    w_max_change = max(w_max_change, abs(diff))

            # Update unregularized bias
            b_old = b
            b = float(np.mean(y_arr - X_train @ w))
            residuals -= b - b_old

            if w_max_change < threshold:
                break

        self.weights = w
        self.bias = b

        if self.standardize and self.scaler_ is not None:
            mu = self.scaler_["mean"]
            sigma = self.scaler_["std"]
            self.coef_ = w / sigma
            self.intercept_ = float(b - np.sum((w * mu) / sigma))
        else:
            self.coef_ = w.copy()
            self.intercept_ = float(b)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict target values using the learned linear model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted continuous values: ŷ = X @ w + b.
        """
        if self.weights is None:
            raise RuntimeError(
                "Model is not fitted yet. Call 'fit' or 'fit_coordinate_descent' first."
            )

        X_arr, _ = self._prepare_inputs(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: Model was trained on {self.n_features_in_} features, "
                f"but input has {X_arr.shape[1]} features."
            )

        if self.standardize and self.scaler_ is not None:
            X_scaled = apply_scaler(X_arr, self.scaler_)
            return (X_scaled @ self.weights) + self.bias
        else:
            return (X_arr @ self.weights) + self.bias

    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """
        Compute the coefficient of determination R^2 on test data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray or pd.Series of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X)
        return compute_r2(y, y_pred)

    @property
    def loss_history_(self) -> list[float]:
        """Alias for history_ tracking MSE loss over epochs."""
        return self.history_

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "lr": self.lr,
            "n_iters": self.n_iters,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "standardize": self.standardize,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> GradientDescentLinearRegression:
        """Set parameters for this estimator."""
        if "learning_rate" in params:
            self.lr = params.pop("learning_rate")
        if "n_epochs" in params:
            self.n_iters = params.pop("n_epochs")
        if "random_state" in params:
            self.seed = params.pop("random_state")
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}.")
        return self

    def __repr__(self) -> str:
        status = "Fitted" if self.weights is not None else "Unfitted"
        return (
            f"GradientDescentLinearRegression(lr={self.lr}, n_iters={self.n_iters}, "
            f"optimizer={self.optimizer}, penalty={self.penalty}, alpha={self.alpha}) [{status}]"
        )

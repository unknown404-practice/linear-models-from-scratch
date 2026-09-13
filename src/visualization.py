"""
Loss Surface Geometry & Optimization Trajectory Visualizer Engine.

Provides pure mathematical loss surface computations and publication-grade
2D contour, 3D surface, and gradient quiver visualizers for analyzing convex
loss landscapes and optimizer trajectories.
"""

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.optimizers import get_optimizer, Optimizer


def compute_loss_grid(
    X_sub: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    w0_range: tuple[float, float],
    w1_range: tuple[float, float],
    bias: float = 0.0,
    grid_size: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute vectorized 2D MSE loss surface over a meshgrid of weights (w0, w1).

    Parameters
    ----------
    X_sub : array-like of shape (N, 2)
        Two-feature design matrix.
    y : array-like of shape (N,)
        Target vector.
    w0_range : tuple of (min, max)
        Range for weight 0.
    w1_range : tuple of (min, max)
        Range for weight 1.
    bias : float, default=0.0
        Fixed intercept value.
    grid_size : int, default=60
        Number of grid intervals along each axis.

    Returns
    -------
    W0 : np.ndarray of shape (grid_size, grid_size)
    W1 : np.ndarray of shape (grid_size, grid_size)
    Loss : np.ndarray of shape (grid_size, grid_size)
    """
    X_arr = np.asarray(X_sub, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    w0_vals = np.linspace(w0_range[0], w0_range[1], grid_size)
    w1_vals = np.linspace(w1_range[0], w1_range[1], grid_size)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)

    # Vectorized computation across all (W0, W1) coordinates simultaneously
    W_stack = np.stack([W0.ravel(), W1.ravel()], axis=0)  # Shape: (2, grid_size^2)
    preds = (X_arr @ W_stack) + bias  # Shape: (N, grid_size^2)
    errors = preds - y_arr[:, np.newaxis]
    loss_flat = np.mean(errors**2, axis=0)
    Loss = loss_flat.reshape(grid_size, grid_size)

    return W0, W1, Loss


def plot_loss_contours(
    X_sub: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    w0_range: tuple[float, float],
    w1_range: tuple[float, float],
    bias: float = 0.0,
    trajectories: dict[str, Any] | None = None,
    levels: int = 35,
    cmap: str = "viridis",
    title: str | None = "2D MSE Loss Contours & Optimization Paths",
    xlabel: str = "Weight w0",
    ylabel: str = "Weight w1",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render 2D filled contour map of MSE loss surface with optional trajectory overlays.

    Parameters
    ----------
    X_sub : array-like of shape (N, 2)
    y : array-like of shape (N,)
    w0_range : tuple of (float, float)
    w1_range : tuple of (float, float)
    bias : float, default=0.0
    trajectories : dict, optional
        Mapping of name -> (path_array, color, marker_fmt) or name -> path_array.
    levels : int, default=35
    cmap : str, default="viridis"
    title : str, optional
    xlabel : str, default="Weight w0"
    ylabel : str, default="Weight w1"
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    W0, W1, Loss = compute_loss_grid(X_sub, y, w0_range, w1_range, bias=bias)

    cp = ax.contourf(W0, W1, Loss, levels=levels, cmap=cmap, alpha=0.9)
    fig.colorbar(cp, ax=ax, label="MSE Loss J(w0, w1)")
    ax.contour(W0, W1, Loss, levels=levels, colors="white", alpha=0.3, linewidths=0.6)

    # Compute analytical OLS global minimum for reference
    X_arr = np.asarray(X_sub, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel() - bias
    try:
        w_ols = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
        if w0_range[0] <= w_ols[0] <= w0_range[1] and w1_range[0] <= w_ols[1] <= w1_range[1]:
            ax.plot(
                w_ols[0],
                w_ols[1],
                "*",
                color="yellow",
                markersize=15,
                markeredgecolor="black",
                label=f"OLS Optimum ({w_ols[0]:.2f}, {w_ols[1]:.2f})",
                zorder=6,
            )
    except Exception:
        pass

    # Render trajectories if supplied
    if trajectories:
        for name, data in trajectories.items():
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                path = np.asarray(data[0])
                color = data[1] if isinstance(data[1], str) else None
                marker = data[2] if len(data) > 2 and isinstance(data[2], str) else ".-"
            else:
                path = np.asarray(data)
                color = None
                marker = ".-"

            kw = {"label": name, "lw": 2, "markersize": 6, "zorder": 5}
            if color:
                kw["color"] = color

            ax.plot(path[:, 0], path[:, 1], marker, **kw)
            ax.plot(
                path[0, 0],
                path[0, 1],
                "o",
                color="white",
                markeredgecolor="black",
                markersize=8,
                zorder=6,
            )

        ax.legend(loc="upper right", framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_loss_surface_3d(
    X_sub: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    w0_range: tuple[float, float],
    w1_range: tuple[float, float],
    bias: float = 0.0,
    trajectories: dict[str, Any] | None = None,
    elev: float = 28.0,
    azim: float = -55.0,
    cmap: str = "viridis",
    title: str = "3D Quadratic Convex Loss Surface J(w0, w1)",
    ax: Any | None = None,
) -> tuple[plt.Figure, Any]:
    """
    Render 3D wireframe / surface visualization of MSE loss with optional descent paths.
    """
    if ax is None:
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    W0, W1, Loss = compute_loss_grid(X_sub, y, w0_range, w1_range, bias=bias)

    surf = ax.plot_surface(W0, W1, Loss, cmap=cmap, alpha=0.82, edgecolor="none", antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="MSE Loss")

    X_arr = np.asarray(X_sub, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if trajectories:
        for name, data in trajectories.items():
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                path = np.asarray(data[0])
                color = data[1] if isinstance(data[1], str) else None
            else:
                path = np.asarray(data)
                color = None

            z_path = [float(np.mean(((X_arr @ wp) + bias - y_arr) ** 2)) for wp in path]
            kw = {"lw": 2.5, "markersize": 6, "label": name, "zorder": 10}
            if color:
                kw["color"] = color
            ax.plot(
                path[:, 0],
                path[:, 1],
                z_path,
                ".-",
                **kw,
            )

        ax.legend(loc="upper left")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Weight w0")
    ax.set_ylabel("Weight w1")
    ax.set_zlabel("MSE Loss J(w)")
    ax.view_init(elev=elev, azim=azim)

    return fig, ax


def plot_gradient_quiver(
    X_sub: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    w0_range: tuple[float, float],
    w1_range: tuple[float, float],
    bias: float = 0.0,
    stride: int = 5,
    levels: int = 30,
    cmap: str = "Blues",
    title: str = "Loss Contours with Negative Gradient Vector Field (-grad J)",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render 2D contour map overlaid with negative gradient quiver arrows (-grad J)
    illustrating steepest descent directions orthogonal to level sets.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    W0, W1, Loss = compute_loss_grid(X_sub, y, w0_range, w1_range, bias=bias)

    ax.contourf(W0, W1, Loss, levels=levels, cmap=cmap, alpha=0.85)
    ax.contour(W0, W1, Loss, levels=levels, colors="gray", alpha=0.4, linewidths=0.6)

    # Subsample grid for legible quiver arrows
    W0_sub = W0[::stride, ::stride]
    W1_sub = W1[::stride, ::stride]

    X_arr = np.asarray(X_sub, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples = len(y_arr)

    # Compute negative gradients at subsampled grid positions
    U_neg = np.zeros_like(W0_sub)
    V_neg = np.zeros_like(W1_sub)

    for i in range(W0_sub.shape[0]):
        for j in range(W0_sub.shape[1]):
            w = np.array([W0_sub[i, j], W1_sub[i, j]])
            err = (X_arr @ w) + bias - y_arr
            grad = (2.0 / n_samples) * (X_arr.T @ err)
            # Point along descent direction: -grad
            U_neg[i, j] = -grad[0]
            V_neg[i, j] = -grad[1]

    # Normalize vectors for consistent arrow length display
    norm = np.sqrt(U_neg**2 + V_neg**2)
    norm_safe = np.where(norm == 0.0, 1.0, norm)
    U_norm = U_neg / norm_safe
    V_norm = V_neg / norm_safe

    ax.quiver(
        W0_sub,
        W1_sub,
        U_norm,
        V_norm,
        color="crimson",
        alpha=0.85,
        scale=25,
        width=0.0035,
        headwidth=3.5,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Weight w0")
    ax.set_ylabel("Weight w1")

    return fig, ax


def compare_optimizer_trajectories(
    X_sub: np.ndarray | pd.DataFrame | None = None,
    y: np.ndarray | pd.Series | None = None,
    lr: float = 0.05,
    n_iters: int = 60,
    optimizers: list[str] | None = None,
    bias: float = 0.0,
    *,
    X: np.ndarray | pd.DataFrame | None = None,
    learning_rate: float | None = None,
    n_epochs: int | None = None,
    **kwargs: Any,
) -> dict[str, tuple[np.ndarray, list[float]]]:
    """
    Run multiple optimizers from the exact same initialization (0, 0)
    on a 2-feature problem, collecting step trajectories and loss values.

    Parameters
    ----------
    X_sub : array-like of shape (N, 2)
    y : array-like of shape (N,)
    lr : float, default=0.05
    n_iters : int, default=60
    optimizers : list of str, default=["sgd", "momentum", "rmsprop", "adam"]
    bias : float, default=0.0

    Returns
    -------
    dict mapping optimizer name -> (path_array of shape (n_iters + 1, 2), loss_list)
    """
    if X_sub is None:
        if X is not None:
            X_sub = X
        else:
            raise ValueError("Must provide X or X_sub")
    if learning_rate is not None:
        lr = learning_rate
    if n_epochs is not None:
        n_iters = n_epochs

    if optimizers is None:
        optimizers = ["sgd", "momentum", "rmsprop", "adam"]

    X_arr = np.asarray(X_sub, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_samples = len(y_arr)

    results: dict[str, tuple[np.ndarray, list[float]]] = {}

    opt_configs = {
        "sgd": ("SGD", {"lr": lr, "momentum": 0.0}),
        "momentum": ("Momentum", {"lr": lr, "momentum": 0.9}),
        "rmsprop": ("RMSprop", {"lr": lr, "decay_rate": 0.9}),
        "adam": ("Adam", {"lr": lr, "beta1": 0.9, "beta2": 0.999}),
    }

    for opt_key in optimizers:
        key_lower = opt_key.lower()
        label, kwargs = opt_configs.get(key_lower, (opt_key.title(), {"lr": lr}))
        opt: Optimizer = get_optimizer(key_lower, **kwargs)

        w = np.array([0.0, 0.0], dtype=np.float64)
        path = [w.copy()]
        initial_loss = float(np.mean(((X_arr @ w) + bias - y_arr) ** 2))
        losses = [initial_loss]

        for _ in range(n_iters):
            err = (X_arr @ w) + bias - y_arr
            grad_w = (2.0 / n_samples) * (X_arr.T @ err)
            w = opt.update("w", w, grad_w)
            path.append(w.copy())
            losses.append(float(np.mean(((X_arr @ w) + bias - y_arr) ** 2)))

        results[label] = (np.array(path), losses)

    return results


def visualize_conditioning_impact(
    X_unscaled: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_indices: tuple[int, int] | None = None,
    lr: float = 0.02,
    n_iters: int = 50,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Generate side-by-side comparative 2D contour figures demonstrating the
    dramatic effect of condition number κ(X) on loss landscape geometry and
    gradient descent trajectories.

    Panel A: Unstandardized X (high κ, eccentric canyon, oscillating gradient descent).
    Panel B: Standardized Z (low κ, circular isotropic bowl, smooth direct convergence).
    """
    X_df = pd.DataFrame(X_unscaled) if not isinstance(X_unscaled, pd.DataFrame) else X_unscaled
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    n_cols = X_df.shape[1]
    if feature_indices is None:
        idx0, idx1 = (0, 1) if n_cols == 2 else (0, min(2, n_cols - 1))
    else:
        idx0, idx1 = feature_indices
        if max(idx0, idx1) >= n_cols or idx0 == idx1:
            idx0, idx1 = (0, 1 if n_cols > 1 else 0)

    X_raw_sub = X_df.iloc[:, [idx0, idx1]].values.astype(np.float64)

    # Standardize features for Panel B
    mean_sub = np.mean(X_raw_sub, axis=0)
    std_sub = np.std(X_raw_sub, axis=0)
    std_safe = np.where(std_sub == 0.0, 1.0, std_sub)
    X_std_sub = (X_raw_sub - mean_sub) / std_safe

    # Center target for clean zero-bias visualization
    y_centered = y_arr - np.mean(y_arr)

    # Compute condition numbers
    kappa_raw = float(np.linalg.cond(X_raw_sub))
    kappa_std = float(np.linalg.cond(X_std_sub))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: Unstandardized Features (Ill-Conditioned Ravine) ---
    w_ols_raw = np.linalg.lstsq(X_raw_sub, y_centered, rcond=None)[0]
    span0_raw = max(0.5, abs(w_ols_raw[0]) * 1.5)
    span1_raw = max(0.5, abs(w_ols_raw[1]) * 1.5)
    w0_rng_raw = (w_ols_raw[0] - span0_raw, w_ols_raw[0] + span0_raw)
    w1_rng_raw = (w_ols_raw[1] - span1_raw, w_ols_raw[1] + span1_raw)

    # Gradient descent on unstandardized
    path_raw = [np.array([w0_rng_raw[0] * 0.8, w1_rng_raw[0] * 0.8])]
    w_curr_raw = path_raw[0].copy()
    for _ in range(n_iters):
        err = (X_raw_sub @ w_curr_raw) - y_centered
        grad = (2.0 / len(y_centered)) * (X_raw_sub.T @ err)
        w_curr_raw -= lr * grad
        path_raw.append(w_curr_raw.copy())

    traj_raw = {"GD Path (Oscillating)": (np.array(path_raw), "red", ".-")}

    plot_loss_contours(
        X_raw_sub,
        y_centered,
        w0_range=w0_rng_raw,
        w1_range=w1_rng_raw,
        bias=0.0,
        trajectories=traj_raw,
        levels=35,
        cmap="magma",
        title=f"Unstandardized (kappa = {kappa_raw:.1f}) - Eccentric Ravine",
        ax=axes[0],
    )

    # --- Panel B: Standardized Features (Well-Conditioned Isotropic Bowl) ---
    w_ols_std = np.linalg.lstsq(X_std_sub, y_centered, rcond=None)[0]
    span0_std = max(1.0, abs(w_ols_std[0]) * 1.5)
    span1_std = max(1.0, abs(w_ols_std[1]) * 1.5)
    w0_rng_std = (w_ols_std[0] - span0_std, w_ols_std[0] + span0_std)
    w1_rng_std = (w_ols_std[1] - span1_std, w_ols_std[1] + span1_std)

    # Gradient descent on standardized
    path_std = [np.array([w0_rng_std[0] * 0.8, w1_rng_std[0] * 0.8])]
    w_curr_std = path_std[0].copy()
    for _ in range(n_iters):
        err = (X_std_sub @ w_curr_std) - y_centered
        grad = (2.0 / len(y_centered)) * (X_std_sub.T @ err)
        w_curr_std -= lr * grad
        path_std.append(w_curr_std.copy())

    traj_std = {"GD Path (Smooth)": (np.array(path_std), "#00ff00", ".-")}

    plot_loss_contours(
        X_std_sub,
        y_centered,
        w0_range=w0_rng_std,
        w1_range=w1_rng_std,
        bias=0.0,
        trajectories=traj_std,
        levels=35,
        cmap="viridis",
        title=f"Standardized (kappa = {kappa_std:.1f}) - Circular Bowl",
        ax=axes[1],
    )

    plt.tight_layout()
    return fig, (axes[0], axes[1])

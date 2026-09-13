"""
Unit Tests for Loss Geometry & Trajectory Visualizer Engine (src/visualization.py).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization import (
    compute_loss_grid,
    plot_loss_contours,
    plot_loss_surface_3d,
    plot_gradient_quiver,
    compare_optimizer_trajectories,
    visualize_conditioning_impact,
)


@pytest.fixture
def synthetic_2d_data():
    """Generates synthetic 2-feature regression dataset with known minimum."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    true_w = np.array([1.5, -2.0])
    y = X @ true_w + 0.1 * np.random.randn(n_samples)
    return X, y, true_w


def test_compute_loss_grid_shape_and_minimum(synthetic_2d_data):
    X, y, true_w = synthetic_2d_data
    grid_size = 40
    w0_range = (0.0, 3.0)
    w1_range = (-3.5, -0.5)

    W0, W1, Loss = compute_loss_grid(
        X, y, w0_range=w0_range, w1_range=w1_range, bias=0.0, grid_size=grid_size
    )

    assert W0.shape == (grid_size, grid_size)
    assert W1.shape == (grid_size, grid_size)
    assert Loss.shape == (grid_size, grid_size)
    assert np.all(Loss >= 0.0)

    # Find grid coordinates of minimum loss
    min_idx = np.unravel_index(np.argmin(Loss), Loss.shape)
    best_w0 = W0[min_idx]
    best_w1 = W1[min_idx]

    # Grid minimum should be very close to true weights
    assert np.isclose(best_w0, true_w[0], atol=0.2)
    assert np.isclose(best_w1, true_w[1], atol=0.2)


def test_plot_loss_contours_returns_axes_and_renders_paths(synthetic_2d_data):
    X, y, _ = synthetic_2d_data
    dummy_traj = {"TestSGD": (np.array([[0.0, 0.0], [0.5, -0.5], [1.0, -1.5]]), "#1f77b4", "o-")}

    fig, ax = plot_loss_contours(
        X,
        y,
        w0_range=(-1.0, 3.0),
        w1_range=(-3.0, 1.0),
        bias=0.0,
        trajectories=dummy_traj,
        levels=25,
        title="Test Contour",
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Contour"
    plt.close(fig)


def test_plot_loss_surface_3d_projection(synthetic_2d_data):
    X, y, _ = synthetic_2d_data
    dummy_traj = {"TestPath": (np.array([[0.0, 0.0], [1.0, -1.0]]), "#ff7f0e", ".-")}

    fig, ax = plot_loss_surface_3d(
        X,
        y,
        w0_range=(-1.0, 3.0),
        w1_range=(-3.0, 1.0),
        bias=0.0,
        trajectories=dummy_traj,
        elev=25,
        azim=-45,
    )

    assert isinstance(fig, plt.Figure)
    assert ax.name == "3d"
    plt.close(fig)


def test_plot_gradient_quiver(synthetic_2d_data):
    X, y, _ = synthetic_2d_data

    fig, ax = plot_gradient_quiver(
        X,
        y,
        w0_range=(0.0, 3.0),
        w1_range=(-3.5, -0.5),
        bias=0.0,
        stride=5,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    # Check that quiver collection was added to axes
    collections = ax.collections
    assert len(collections) > 0
    plt.close(fig)


def test_compare_optimizer_trajectories_convergence(synthetic_2d_data):
    X, y, true_w = synthetic_2d_data
    n_iters = 30
    trajectories = compare_optimizer_trajectories(
        X, y, lr=0.05, n_iters=n_iters, optimizers=["sgd", "momentum", "rmsprop", "adam"]
    )

    assert "SGD" in trajectories
    assert "Momentum" in trajectories
    assert "RMSprop" in trajectories
    assert "Adam" in trajectories

    for name, (path, losses) in trajectories.items():
        assert path.shape == (n_iters + 1, 2)
        assert len(losses) == n_iters + 1
        # Initial point is (0, 0)
        assert np.allclose(path[0], [0.0, 0.0])
        # Loss must decrease over optimization
        assert losses[-1] < losses[0]


def test_visualize_conditioning_impact_figure(synthetic_2d_data):
    X, y, _ = synthetic_2d_data
    # Artificially create disparate feature scales
    X_unscaled = X.copy()
    X_unscaled[:, 0] *= 100.0  # feature 0 has massive scale

    fig, axes = visualize_conditioning_impact(
        X_unscaled, y, feature_indices=(0, 1), lr=0.01, n_iters=30
    )

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    assert axes[0].get_title().startswith("Unstandardized")
    assert axes[1].get_title().startswith("Standardized")
    plt.close(fig)

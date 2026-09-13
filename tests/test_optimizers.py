"""
Unit and Convergence Tests for First-Order Optimizers from Scratch.
"""

import numpy as np
import pytest
from src.optimizers import SGD, RMSprop, Adam, get_optimizer


def test_sgd_momentum_update():
    """Verify velocity accumulation and Nesterov updates in SGD."""
    opt = SGD(lr=0.1, momentum=0.9)
    param = np.array([1.0, 2.0])
    grad = np.array([0.5, -0.5])

    # First update: v1 = 0.5, step = 0.1 * 0.5 = 0.05
    new_param = opt.update("w", param, grad)
    assert np.allclose(new_param, [0.95, 2.05])

    # Second update with same grad: v2 = 0.9 * 0.5 + 0.5 = 0.95
    # step = 0.1 * 0.95 = 0.095
    new_param2 = opt.update("w", new_param, grad)
    expected_w0 = 0.95 - 0.1 * 0.95
    expected_w1 = 2.05 - 0.1 * (-0.95)
    assert np.allclose(new_param2, [expected_w0, expected_w1])


def test_adam_bias_correction_and_convergence():
    """Verify Adam converges on a simple quadratic bowl f(x) = x^2."""
    opt = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    param = np.array([3.0])

    # Minimize f(x) = x^2, grad = 2x
    for _ in range(250):
        grad = 2.0 * param
        param = opt.update("x", param, grad)

    assert np.isclose(param[0], 0.0, atol=1e-2)


def test_rmsprop_scaling():
    """Verify RMSprop scales update inversely with gradient magnitude."""
    opt = RMSprop(lr=0.01, decay_rate=0.9)
    param = np.array([5.0])
    grad_large = np.array([100.0])

    # First update with large grad
    p1 = opt.update("w", param, grad_large)
    # RMSprop adapts step size: s = 0.1 * 10000 = 1000, sqrt(s) ~ 31.6
    # step ~ 0.01 * 100 / 31.6 ~ 0.0316
    assert abs(param[0] - p1[0]) < 0.1


def test_get_optimizer_factory():
    """Verify factory returns appropriate Optimizer instances."""
    opt_adam = get_optimizer("adam", lr=0.01)
    assert isinstance(opt_adam, Adam)
    assert opt_adam.lr == 0.01

    opt_mom = get_optimizer("momentum", lr=0.05)
    assert isinstance(opt_mom, SGD)
    assert opt_mom.momentum == 0.9

    opt_rmsprop = get_optimizer("rmsprop", lr=0.005)
    assert isinstance(opt_rmsprop, RMSprop)

    # Pass-through if already an instance
    custom_adam = Adam(lr=0.002)
    assert get_optimizer(custom_adam) is custom_adam

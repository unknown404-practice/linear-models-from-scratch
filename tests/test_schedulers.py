"""
Unit Tests for Learning Rate Schedulers.
"""

import pytest
from src.schedulers import ConstantLR, StepLR, ExponentialLR, CosineAnnealingLR


def test_constant_lr():
    sched = ConstantLR(initial_lr=0.05)
    assert sched.get_lr(0) == 0.05
    assert sched.get_lr(100) == 0.05


def test_step_lr():
    sched = StepLR(initial_lr=0.1, step_size=10, gamma=0.5)
    assert sched.get_lr(0) == 0.1
    assert sched.get_lr(9) == 0.1
    assert sched.get_lr(10) == 0.05
    assert sched.get_lr(20) == 0.025


def test_exponential_lr():
    sched = ExponentialLR(initial_lr=0.1, gamma=0.9)
    assert sched.get_lr(0) == 0.1
    assert pytest.approx(sched.get_lr(1)) == 0.09
    assert pytest.approx(sched.get_lr(2)) == 0.081


def test_cosine_annealing_lr():
    sched = CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=0.0)
    assert sched.get_lr(0) == 0.1
    assert pytest.approx(sched.get_lr(50), abs=1e-4) == 0.05
    assert pytest.approx(sched.get_lr(100), abs=1e-4) == 0.0
    # Past T_max should clamp to eta_min
    assert sched.get_lr(150) == 0.0

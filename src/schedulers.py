"""
Learning Rate Schedulers from Scratch.

This module provides dynamic learning rate schedules to modulate optimization step size:
- ConstantLR: Unchanging learning rate
- StepLR: Decays learning rate by gamma every step_size epochs
- ExponentialLR: Decays learning rate exponentially each epoch by factor gamma
- CosineAnnealingLR: Smooth cosine decay from initial_lr to eta_min over T_max epochs
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    def __init__(self, initial_lr: float) -> None:
        if initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {initial_lr}")
        self.initial_lr = float(initial_lr)

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        """Compute the learning rate for the given epoch."""
        pass


class ConstantLR(LRScheduler):
    """Constant learning rate scheduler."""

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr


class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs:
        lr_t = initial_lr * gamma^(epoch // step_size)
    """

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(initial_lr=initial_lr)
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def get_lr(self, epoch: int) -> float:
        power = epoch // self.step_size
        return self.initial_lr * (self.gamma**power)


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate exponentially by gamma each epoch:
        lr_t = initial_lr * gamma^epoch
    """

    def __init__(self, initial_lr: float, gamma: float = 0.95) -> None:
        super().__init__(initial_lr=initial_lr)
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = float(gamma)

    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma**epoch)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine Annealing schedule (Loshchilov & Hutter, 2016):
        lr_t = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * min(epoch, T_max) / T_max))
    """

    def __init__(
        self,
        initial_lr: float,
        T_max: int | None = None,
        eta_min: float = 0.0,
        t_max: int | None = None,
    ) -> None:
        super().__init__(initial_lr=initial_lr)
        actual_t_max = T_max if T_max is not None else t_max
        if actual_t_max is None:
            raise TypeError("CosineAnnealingLR requires 'T_max' or 't_max' parameter")
        if actual_t_max < 1:
            raise ValueError(f"T_max must be >= 1, got {actual_t_max}")
        if eta_min < 0 or eta_min > initial_lr:
            raise ValueError(f"eta_min must satisfy 0 <= eta_min <= initial_lr, got {eta_min}")
        self.T_max = int(actual_t_max)
        self.eta_min = float(eta_min)

    def get_lr(self, epoch: int) -> float:
        if epoch >= self.T_max:
            return self.eta_min
        fraction = epoch / self.T_max
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.eta_min + (self.initial_lr - self.eta_min) * cosine_factor

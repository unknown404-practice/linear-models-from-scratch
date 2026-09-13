"""
First-Order Optimizers from Scratch using Vectorized NumPy.

This module provides clean, modular optimizer implementations:
- SGD (with classical momentum and Nesterov accelerated gradient)
- RMSprop (Root Mean Square Propagation)
- Adam (Adaptive Moment Estimation with bias-corrected first and second moments)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Optimizer(ABC):
    """Abstract base class for first-order gradient optimizers."""

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate lr must be positive, got {lr}")
        self.lr = float(lr)

    @abstractmethod
    def update(
        self, param_name: str, param: np.ndarray | float, grad: np.ndarray | float
    ) -> np.ndarray | float:
        """
        Compute updated parameter value given current parameter and gradient.

        Parameters
        ----------
        param_name : str
            Unique key identifying the parameter (e.g. 'w' or 'b') for state tracking.
        param : np.ndarray or float
            Current parameter values.
        grad : np.ndarray or float
            Current gradient evaluated at param.

        Returns
        -------
        updated_param : np.ndarray or float
            Parameter value after applying the optimization update step.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator states (e.g. velocities, moving averages)."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional Momentum and Nesterov acceleration.

    Update Rule:
        v_{t+1} = momentum * v_t + grad
        If nesterov:
            param_{t+1} = param_t - lr * (grad + momentum * v_{t+1})
        Else:
            param_{t+1} = param_t - lr * v_{t+1}

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    momentum : float, default=0.0
        Momentum factor in [0, 1).
    nesterov : bool, default=False
        Whether to enable Nesterov Accelerated Gradient (NAG).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(lr=lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.velocities: dict[str, np.ndarray | float] = {}

    def update(
        self, param_name: str, param: np.ndarray | float, grad: np.ndarray | float
    ) -> np.ndarray | float:
        if self.momentum == 0.0:
            return param - self.lr * grad

        if param_name not in self.velocities:
            if isinstance(param, np.ndarray):
                self.velocities[param_name] = np.zeros_like(param, dtype=np.float64)
            else:
                self.velocities[param_name] = 0.0

        v = self.velocities[param_name]
        v = self.momentum * v + grad
        self.velocities[param_name] = v

        if self.nesterov:
            return param - self.lr * (grad + self.momentum * v)
        return param - self.lr * v

    def reset(self) -> None:
        self.velocities.clear()


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation (RMSprop) optimizer.

    Maintains a moving average of the squared gradients to normalize the gradient
    step across varying curvatures:
        s_{t+1} = decay_rate * s_t + (1 - decay_rate) * grad^2
        param_{t+1} = param_t - (lr / (sqrt(s_{t+1}) + eps)) * grad

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    decay_rate : float, default=0.99
        Discounting factor for the history/coming gradient.
    eps : float, default=1e-8
        Small constant for numerical stability (avoids zero division).
    """

    def __init__(self, lr: float = 0.01, decay_rate: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__(lr=lr)
        if not (0.0 < decay_rate < 1.0):
            raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}")
        self.decay_rate = float(decay_rate)
        self.eps = float(eps)
        self.squared_grads: dict[str, np.ndarray | float] = {}

    def update(
        self, param_name: str, param: np.ndarray | float, grad: np.ndarray | float
    ) -> np.ndarray | float:
        if param_name not in self.squared_grads:
            if isinstance(param, np.ndarray):
                self.squared_grads[param_name] = np.zeros_like(param, dtype=np.float64)
            else:
                self.squared_grads[param_name] = 0.0

        s = self.squared_grads[param_name]
        s = self.decay_rate * s + (1.0 - self.decay_rate) * (grad**2)
        self.squared_grads[param_name] = s

        step = (self.lr / (np.sqrt(s) + self.eps)) * grad
        return param - step

    def reset(self) -> None:
        self.squared_grads.clear()


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Combines the heuristics of Momentum (first moment of gradients) and RMSprop
    (second moment of gradients) with exact analytical bias corrections:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param_t = param_{t-1} - (lr / (sqrt(v_hat) + eps)) * m_hat

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    eps : float, default=1e-8
        Term added to denominator to improve numerical stability.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self.m: dict[str, np.ndarray | float] = {}
        self.v: dict[str, np.ndarray | float] = {}
        self.t: dict[str, int] = {}

    def update(
        self, param_name: str, param: np.ndarray | float, grad: np.ndarray | float
    ) -> np.ndarray | float:
        if param_name not in self.m:
            if isinstance(param, np.ndarray):
                self.m[param_name] = np.zeros_like(param, dtype=np.float64)
                self.v[param_name] = np.zeros_like(param, dtype=np.float64)
            else:
                self.m[param_name] = 0.0
                self.v[param_name] = 0.0
            self.t[param_name] = 0

        self.t[param_name] += 1
        t = self.t[param_name]

        # Update biased 1st and 2nd moment estimates
        m = self.beta1 * self.m[param_name] + (1.0 - self.beta1) * grad
        v = self.beta2 * self.v[param_name] + (1.0 - self.beta2) * (grad**2)
        self.m[param_name] = m
        self.v[param_name] = v

        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1.0 - (self.beta1**t))
        v_hat = v / (1.0 - (self.beta2**t))

        # Update parameter
        step = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return param - step

    def reset(self) -> None:
        self.m.clear()
        self.v.clear()
        self.t.clear()


def get_optimizer(optimizer: str | Optimizer, lr: float = 0.01, **kwargs: Any) -> Optimizer:
    """
    Factory function returning an Optimizer instance.

    Parameters
    ----------
    optimizer : str or Optimizer
        Name of optimizer ('sgd', 'momentum', 'rmsprop', 'adam') or an Optimizer instance.
    lr : float, default=0.01
        Learning rate if instantiating from string name.
    **kwargs : Any
        Additional keyword arguments forwarded to optimizer constructor
        (e.g., momentum, beta, beta1, beta2, nesterov).

    Returns
    -------
    opt : Optimizer
    """
    if isinstance(optimizer, Optimizer):
        return optimizer

    name = str(optimizer).strip().lower()
    if name in ("sgd", "vanilla"):
        kwargs.setdefault("momentum", 0.0)
        return SGD(lr=lr, **kwargs)
    elif name == "momentum":
        kwargs.setdefault("momentum", 0.9)
        return SGD(lr=lr, **kwargs)
    elif name == "nesterov":
        kwargs.setdefault("momentum", 0.9)
        kwargs.setdefault("nesterov", True)
        return SGD(lr=lr, **kwargs)
    elif name == "rmsprop":
        if "beta" in kwargs:
            kwargs.setdefault("decay_rate", kwargs.pop("beta"))
        return RMSprop(lr=lr, **kwargs)
    elif name == "adam":
        return Adam(lr=lr, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Supported options: 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'."
        )

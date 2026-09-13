# Extending the Platform

This guide explains how to add new algorithms without modifying existing code.

## 1. Adding a Custom Loss Function
Inherit from `BaseEstimator` or wrap your gradient inside `GradientDescentLinearRegression`:
```python
from src.linear_regression import GradientDescentLinearRegression
import numpy as np


class LogCoshLinearRegression(GradientDescentLinearRegression):
    def _compute_gradients(self, X, y, y_pred):
        N = len(y)
        error = y_pred - y
        # Gradient of log(cosh(e)) is tanh(e)
        grad_w = (1.0 / N) * (X.T @ np.tanh(error))
        grad_b = np.mean(np.tanh(error))
        return grad_w, grad_b
```

## 2. Adding a New GLM Family
In `src/glm.py`, extend the Exponential Dispersion Family dictionary with new link and variance functions:
```python
# Define:
# - link(mu): g(mu)
# - inv_link(eta): g^-1(eta)
# - dlink(mu): g'(mu)
# - variance(mu): V(mu)
# - deviance(y, mu): unit deviance formula
```
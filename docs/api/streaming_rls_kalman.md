# API Reference: Streaming Regression (RLS & Kalman)

Located in [`src/streaming.py`](../../src/streaming.py).

## Classes

### 1. `RecursiveLeastSquares`
Rank-1 Sherman-Morrison $\mathcal{O}(D^2)$ streaming updates with exponential forgetting factor $\lambda \in (0, 1]$:
$$k_t = \frac{P_{t-1} x_t}{\lambda + x_t^T P_{t-1} x_t}, \quad w_t = w_{t-1} + k_t (y_t - x_t^T w_{t-1})$$

```python
from src.streaming import RecursiveLeastSquares

rls = RecursiveLeastSquares(forgetting_factor=0.99)
for x_t, y_t in stream_generator():
    rls.partial_fit(x_t, y_t)

y_pred = rls.predict(X_val)
```

### 2. `KalmanFilterRegression`
Dynamic latent weight tracking with process noise $Q$ and observation noise $R$.
```python
from src.streaming import KalmanFilterRegression

kalman = KalmanFilterRegression(q_cov=1e-5, r_cov=1e-2)
kalman.partial_fit(X_stream, y_stream)
```
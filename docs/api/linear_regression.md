# API Reference: `GradientDescentLinearRegression`

Located in [`src/linear_regression.py`](../../src/linear_regression.py).

## Class Overview
Vectorized Ordinary Least Squares (OLS) and regularized linear regression fitted via first-order gradient descent or cyclic coordinate descent.

### Constructor Arguments
```python
GradientDescentLinearRegression(
    learning_rate: float = 0.01,
    n_epochs: int = 1000,
    batch_size: Optional[int] = None,   # None = Full-batch, int = Mini-batch
    penalty: Optional[str] = None,      # None, 'l1', 'l2', 'elasticnet'
    alpha: float = 0.0,                 # Regularization strength
    l1_ratio: float = 0.5,              # ElasticNet mixing ratio (0 = L2, 1 = L1)
    optimizer: str = 'sgd',             # 'sgd', 'momentum', 'rmsprop', 'adam'
    scheduler: str = 'constant',        # 'constant', 'step', 'exp', 'cosine'
    tol: float = 1e-6,
    random_state: Optional[int] = None
)
```

### Minimal Usage
```python
from src.linear_regression import GradientDescentLinearRegression
import numpy as np

X = np.random.randn(100, 5)
y = X @ np.array([1.5, -2.0, 0.5, 0.0, 3.1]) + 0.5

model = GradientDescentLinearRegression(learning_rate=0.05, n_epochs=500, optimizer="adam")
model.fit(X, y)

y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"Fitted R2: {r2:.4f}, Weights: {model.weights_}, Bias: {model.bias_:.4f}")
```
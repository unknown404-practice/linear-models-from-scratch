# API Reference: Regularized Regression (Ridge, Lasso, ElasticNet)

Located in [`src/linear_regression.py`](../../src/linear_regression.py).

## Formulations

### 1. Ridge Regression (L2)
Minimizes $J(w) = \frac{1}{N} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$.
```python
model = GradientDescentLinearRegression(penalty="l2", alpha=0.1, optimizer="adam")
model.fit(X_train, y_train)
```

### 2. Lasso Regression (L1 via Cyclic Coordinate Descent)
Minimizes $J(w) = \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \|w\|_1$.
Uses the Soft-Thresholding operator to produce exact zero coefficients.
```python
model = GradientDescentLinearRegression(penalty="l1", alpha=0.05, n_epochs=300)
model.fit(X_train, y_train)
print("Sparsity (zero weights):", np.sum(model.weights_ == 0.0))
```

### 3. ElasticNet Regression
Minimizes $J(w) = \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \rho \|w\|_1 + \frac{1}{2} \alpha (1 - \rho) \|w\|_2^2$.
```python
model = GradientDescentLinearRegression(penalty="elasticnet", alpha=0.05, l1_ratio=0.7)
model.fit(X_train, y_train)
```
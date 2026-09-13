# API Reference: Conformal Prediction

Located in [`src/conformal.py`](../../src/conformal.py).

## Class Overview
`ConformalLinearRegression` generates distribution-free prediction intervals with non-asymptotic coverage guarantees:
$$P(y \in [\hat{y} - \hat{q}, \hat{y} + \hat{q}]) \ge 1 - \alpha$$

### Minimal Usage
```python
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression

base_model = ClosedFormLinearRegression(method="svd")
conformal = ConformalLinearRegression(estimator=base_model, alpha=0.10, cal_size=0.20)
conformal.fit(X_train, y_train)

# Generate lower and upper bounds
y_pred, y_lower, y_upper = conformal.predict_interval(X_test)
coverage = conformal.score_coverage(X_test, y_test)
print(f"Calibrated cutoff q: {conformal.q_hat_:.4f}, Test coverage: {coverage * 100:.2f}%")
```
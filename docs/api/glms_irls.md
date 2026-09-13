# API Reference: Generalized Linear Models (GLMs)

Located in [`src/glm.py`](../../src/glm.py).

## Classes
- `GeneralizedLinearModel`: Base engine for Exponential Dispersion Families.
- `LogisticRegressionScratch`: Binary classification via logit link.
- `PoissonRegressionScratch`: Non-negative count regression via log link.
- `GammaRegressionScratch`: Positive continuous regression via log link.

### Minimal Usage
```python
from src.glm import PoissonRegressionScratch, LogisticRegressionScratch

# Poisson count regression
poisson = PoissonRegressionScratch(alpha_l2=0.01, max_iter=30)
poisson.fit(X_counts, y_counts)
y_pred_counts = poisson.predict(X_counts)
print(f"Poisson Pseudo-R2: {poisson.pseudo_r2_:.4f}")

# Logistic binary classification
logistic = LogisticRegressionScratch(max_iter=30)
logistic.fit(X_binary, y_binary)
probs = logistic.predict_proba(X_binary)
```
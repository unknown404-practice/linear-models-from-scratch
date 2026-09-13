# API Reference: Quantile Regression

Located in [`src/quantile.py`](../../src/quantile.py).

## Class Overview
`QuantileRegressorScratch` minimizes the asymmetric pinball loss:
$$\rho_\tau(u) = u (\tau - \mathbb{I}(u < 0))$$
using smoothed Iteratively Reweighted Least Squares (IRLS).

### Minimal Usage
```python
from src.quantile import QuantileRegressorScratch

# Fit lower 10th percentile, median (50th), and upper 90th percentile
q10 = QuantileRegressorScratch(quantile=0.10).fit(X, y)
q50 = QuantileRegressorScratch(quantile=0.50).fit(X, y)
q90 = QuantileRegressorScratch(quantile=0.90).fit(X, y)

pred_low = q10.predict(X_test)
pred_med = q50.predict(X_test)
pred_high = q90.predict(X_test)
```
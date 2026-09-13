# API Reference: Robust Regression (Huber & RANSAC)

Located in [`src/robust.py`](../../src/robust.py).

## Classes

### 1. `HuberRegressorScratch`
Piecewise smooth loss function capping residual gradients at $\pm \delta$:
$$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{if } |r| > \delta \end{cases}$$

```python
from src.robust import HuberRegressorScratch

huber = HuberRegressorScratch(epsilon=1.35, learning_rate=0.05, max_iter=500)
huber.fit(X_outliers, y_outliers)
print("Robust weights:", huber.weights_)
```

### 2. `RANSACRegressorScratch`
Iterative Random Sample Consensus algorithm that fits minimal subsets, discovers inliers, and refits on the consensus set.

```python
from src.robust import RANSACRegressorScratch
from src.solvers import ClosedFormLinearRegression

ransac = RANSACRegressorScratch(
    estimator=ClosedFormLinearRegression(method="svd"),
    min_samples=10,
    residual_threshold=1.5,
    max_trials=100,
)
ransac.fit(X_outliers, y_outliers)
print(f"Inliers discovered: {np.sum(ransac.inlier_mask_)} / {len(y_outliers)}")
```
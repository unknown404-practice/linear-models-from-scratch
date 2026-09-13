# Testing Strategy & Verification Invariants

The test suite in `tests/` contains **85 unit tests** across 18 specialized test files:

### Invariant Test Categories:
1. **Convergence Invariants:** Verification that gradient descent, Adam, and IRLS converge to analytical least-squares optima to within $10^{-4}$ MSE tolerance.
2. **Exact Sparsity Tests:** Verification that L1 Lasso and ElasticNet coordinate descent set non-predictive weights to exact $0.0$ floating-point representation.
3. **Outlier Resilience Invariants:** Tests confirming Huber and RANSAC estimators maintain $R^2 > 0.45$ under catastrophic $50\times$ leverage outlier corruption.
4. **Uncertainty Monotonicity:** Verification that Bayesian epistemic uncertainty strictly increases as test points extrapolate beyond the convex hull of training data.
5. **Coverage Calibration Invariants:** Empirical verification that conformal prediction intervals achieve $\ge 1 - \alpha$ nominal coverage on held-out test splits.
6. **Streaming Equivalence:** Verification that streaming RLS asymptotically converges to full-batch OLS parameters.
7. **Serialization Invariance:** Verification that every model serialized to JSON produces zero deviation ($0.0000$ error) when reloaded and scored.
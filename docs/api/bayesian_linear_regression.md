# API Reference: Bayesian Linear Regression

Located in [`src/bayesian.py`](../../src/bayesian.py).

## Class Overview
`BayesianLinearRegression` provides exact analytical conjugate Gaussian posteriors, uncertainty decomposition, and Automatic Relevance Determination (ARD).

### Mathematical Posterior
$$S_N = (\alpha I + \beta X^T X)^{-1}, \quad m_N = \beta S_N X^T y$$

### Uncertainty Decomposition
$$\sigma_*^2(x_*) = \frac{1}{\beta} + x_*^T S_N x_*$$

### Minimal Usage
```python
from src.bayesian import BayesianLinearRegression

bayes = BayesianLinearRegression(alpha=1.0, beta=25.0)
bayes.fit(X_train, y_train)

# Mean prediction and standard deviation
y_mean, y_std = bayes.predict(X_test, return_std=True)

# Sample 10 weight vectors from posterior
w_samples = bayes.sample_weights(n_samples=10, random_state=42)
```
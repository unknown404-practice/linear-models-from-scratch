# Frequently Asked Questions (FAQ) & Troubleshooting

### 1. How is this repository different from `scikit-learn`?
`scikit-learn` wraps optimized Cython/C/Fortran libraries (`liblinear`, LAPACK). While fast, its internals are opaque for educational understanding. This repository re-implements linear models using **pure, vectorized NumPy**—preserving the exact same mathematical properties and `fit`/`predict`/`score` API contract, but with crystal-clear code where every line maps directly to a textbook equation.

### 2. Can I use these models in production?
Yes. Every estimator is:
- Fully vectorized for fast BLAS GEMV/GEMM execution.
- Covered by strict zero-data-leakage unit tests.
- Equipped with pure-JSON model persistence (`save_model_json` / `load_model_json`), eliminating Python `pickle` security vulnerabilities.

### 3. Which solver should I use: Gradient Descent vs. Closed-Form vs. Newton-Raphson?
- **Gradient Descent (`GradientDescentLinearRegression`):** Best for very large sample counts ($N > 100,000$) where memory prohibits constructing $X^T X$, or when streaming mini-batches.
- **Closed-Form (`ClosedFormLinearRegression(method="svd")`):** Best for small-to-medium feature counts ($D < 2,000$). Extremely fast and handles rank-deficient/collinear matrices safely.
- **Newton-Raphson (`NewtonLinearRegression`):** Exactly 1-step analytical convergence on quadratic MSE loss surfaces with Levenberg-Marquardt damping.

### 4. How do I handle categorical features and missing values without data leakage?
Use our pure-NumPy pipeline components in `src/pipeline.py`:
- `SimpleImputerScratch`: Imputes continuous or categorical columns.
- `OneHotEncoderScratch`: Encodes discrete categories with `handle_unknown="ignore"`.
- `ColumnTransformerScratch`: Bundles transformers across distinct column slices.
- `PipelineScratch`: Chains transformers and terminal estimators, ensuring statistics are fitted strictly on training data.

### 5. What is the difference between Epistemic and Aleatoric uncertainty?
In `BayesianLinearRegression`:
$$\sigma_*^2(x_*) = \underbrace{\frac{1}{\beta}}_{\text{Aleatoric (Noise)}} + \underbrace{x_*^T S_N x_*}_{\text{Epistemic (Ignorance)}}$$
- **Aleatoric:** Inherent random noise in measurements that cannot be reduced with more training data.
- **Epistemic:** Uncertainty stemming from lack of training data in that region of feature space. It vanishes near dense training clusters and expands quadratically in out-of-distribution (OOD) space.

### 6. When should I choose Conformal Prediction over Bayesian Regression?
- Choose **Bayesian Regression** when you want an interpretable posterior distribution over parameters $w$ and have reasonable prior information.
- Choose **Conformal Prediction** when you cannot guarantee Gaussian errors or linear assumptions, but require **guaranteed finite-sample coverage** $P(y \in \mathcal{C}(x)) \ge 1 - \alpha$ regardless of the underlying data distribution.

### 7. How do I extend the library with a custom optimizer or loss?
See [`docs/design/extensibility.md`](design/extensibility.md) for step-by-step templates.
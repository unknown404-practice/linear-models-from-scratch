# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## How to Update This File
When submitting a pull request that introduces new features, fixes, or breaking changes:
1. Add an entry under the `[Unreleased]` section below.
2. Group entries under `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`.
3. Include links to relevant PRs or issues if applicable.

---

## [Unreleased]
### Planned
- PyPI release under `linear-models-from-scratch`.
- Web-based documentation site hosted on GitHub Pages via `mkdocs-material`.
- GPU acceleration backend via CuPy fallback for billion-row matrices.

---

## [0.1.0] - 2026-09-13
### Initial Public Release
A production-grade, vectorized NumPy implementation of linear models from first principles,
bridging pedagogical clarity with production rigor, uncertainty quantification, and competition baselines.

### Added
#### Core Solvers & Optimizers
- **Vectorized OLS Gradient Descent**: Full-batch and mini-batch stochastic gradient descent with deterministic seed control.
- **First-Order Optimizers**: Heavy-ball Momentum, RMSprop with exponential moving averages, and Adam with first/second moment bias correction.
- **Dynamic Learning Rate Schedulers**: `ConstantLR`, `StepLR`, `ExponentialLR`, and `CosineAnnealingLR` (Loshchilov & Hutter, 2016).
- **Analytical Closed-Form Solvers**: Exact Moore-Penrose pseudoinverse via SVD, Cholesky decomposition ($X^T X = L L^T$), QR orthogonal factorization, and classical Normal Equations with condition number ($\kappa$) diagnostics.
- **Second-Order Newton-Raphson**: Exact single-step quadratic convergence for MSE cost.

#### Regularization & Non-Linear Features
- **Ridge (L2)**: Vectorized analytic shrinkage and weight decay penalty.
- **Lasso (L1)**: Exact cyclic coordinate descent with soft-thresholding operator $\mathcal{S}_\lambda(z) = \text{sign}(z)\max(|z|-\lambda, 0)$, guaranteeing exact parameter sparsity.
- **ElasticNet (L1 + L2)**: Cyclic coordinate descent combining $\ell_1$ sparsity and $\ell_2$ grouping.
- **Polynomial Features**: From-scratch $D$-dimensional polynomial expansion with interaction terms and degree configuration.

#### Robust Regression
- **Huber Regressor**: Piecewise quadratic/linear loss with Huber parameter $\delta$, resilient to heavy-tailed response outliers.
- **RANSAC Regressor**: Random Sample Consensus meta-estimator with configurable inlier residual thresholding and sub-sample iteration bounds.

#### Uncertainty Quantification
- **Bayesian Linear Regression**: Conjugate Gaussian-Inverse-Gamma prior formulation, analytical posterior weight distribution $\mathcal{N}(\mu_N, \Sigma_N)$, and decomposed epistemic vs. aleatoric predictive confidence bands.
- **Automatic Relevance Determination (ARD)**: Hierarchical sparse Gaussian priors with per-feature hyperparameter pruning.
- **Inductive Split Conformal Prediction**: Distribution-free prediction intervals $[ \hat{y} - \hat{q}, \hat{y} + \hat{q} ]$ with non-asymptotic finite-sample coverage guarantee $P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$.

#### Generalized Linear Models (GLMs)
- **IRLS Engine**: Iteratively Reweighted Least Squares solver supporting non-normal exponential dispersion families.
- **Logistic Regression**: Binary cross-entropy classification via logit link.
- **Poisson Regression**: Non-negative rate count regression with canonical log link.
- **Gamma Regression**: Strictly positive skewed continuous regression with inverse link.

#### Streaming & Quantile Regression
- **Recursive Least Squares (RLS)**: Online Woodbury matrix inversion updates with exponential forgetting factor $\lambda \in (0, 1]$ for non-stationary tracking.
- **Kalman Filter Regression**: Linear-Gaussian state-space tracking with process noise $Q$ and observation covariance $R$.
- **Quantile Regressor**: Asymmetric pinball loss $\rho_\tau(u) = u(\tau - \mathbb{I}(u < 0))$ for conditional median ($\tau = 0.50$) and risk corridor estimation ($\tau \in \{0.10, 0.90\}$).

#### Pipeline & Infrastructure
- **Composition Pipeline**: Scikit-learn-compatible `PipelineScratch` supporting `.fit()`, `.predict()`, `.transform()`, and `.score()`.
- **Column Transformers**: Heterogeneous schema handling with `ColumnTransformerScratch`.
- **Custom Transformers**: `StandardScalerScratch`, `SimpleImputerScratch`, `OneHotEncoderScratch`.
- **Model Selection**: `KFoldScratch`, `cross_val_score_scratch`, and exhaustive `GridSearchCVScratch`.
- **Diagnostics**: OLS standard errors, Student's t-statistics, p-values, Variance Inflation Factor (VIF), Breusch-Pagan heteroscedasticity test, and Durbin-Watson autocorrelation test.
- **Loss Geometry Visualizer**: 2D filled contour maps, 3D loss surface projections, negative gradient quiver vector fields, and multi-optimizer trajectory comparisons.
- **Dual Persistence**: Dual format model serialization supporting human-readable JSON state dictionaries and Python binary pickle format.

#### Packaging & CI/CD
- Modern `pyproject.toml` configuration conforming to PEP 517 / PEP 621.
- Dual package discovery (`src` and `linear_models_from_scratch`).
- GitHub Actions CI matrix testing Python 3.11, 3.12, and 3.13.
- Automated headless execution check for all 19 Jupyter notebooks (`scripts/check_notebooks.py`).
- Code quality tooling with Ruff (linter/formatter) and Mypy (static typing).
- Pre-commit hook definitions in `.pre-commit-config.yaml`.

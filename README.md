# Linear Models from Scratch
### Production-Grade Vectorized NumPy Engine, Optimization Geometry & Modern Statistical Frontiers

```text
  _     _                          __  __           _      _     
 | |   (_)_ __   ___  __ _ _ __   |  \/  | ___   __| | ___| |___ 
 | |   | | '_ \ / _ \/ _` | '__|  | |\/| |/ _ \ / _` |/ _ \ / __|
 | |___| | | | |  __/ (_| | |     | |  | | (_) | (_| |  __/ \__ \
 |_____|_|_| |_|\___|\__,_|_|     |_|  |_|\___/ \__,_|\___|_|___/
                     FROM SCRATCH • VECTORIZED NUMPY
```

> **A from-scratch, vectorized NumPy implementation of linear models (OLS, regularized, robust, Bayesian, GLMs, streaming, conformal) with scikit-learn-compatible pipelines, full docs, and competition-grade examples.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 85/85 Passing](https://img.shields.io/badge/tests-85%2F85%20passing-brightgreen.svg)](tests/)
[![Notebooks: 19 Verified](https://img.shields.io/badge/notebooks-19%20verified-brightgreen.svg)](scripts/check_notebooks.py)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/unknown404-practice/linear-models-from-scratch.git)

**Direct Links:**
- 🐙 **GitHub:** [https://github.com/unknown404-practice/linear-models-from-scratch.git](https://github.com/unknown404-practice/linear-models-from-scratch.git)
- 💼 **LinkedIn:** [https://www.linkedin.com/in/ranadeep-saha-a03296404/](https://www.linkedin.com/in/ranadeep-saha-a03296404/)
- ✉️ **Email:** [mailto:ranadeep2021saha@gmail.com](mailto:ranadeep2021saha@gmail.com)

---

## Why This Exists

Most educational repositories stop at toy synthetic datasets and naive gradient descent implementations of ordinary least squares (OLS). In contrast, production environments and competitive tabular modeling require robust optimization, ill-conditioned curvature handling, leak-free feature pipelines, and rigorous uncertainty quantification.

This repository is engineered to be:
1. **Mathematically Transparent:** Every loss function, analytical gradient, coordinate descent soft-thresholding step, and matrix decomposition is explicitly derived and implemented in readable, vectorized NumPy without opaque abstraction layers.
2. **Competition-Grade:** Features complete, leak-free preprocessing pipelines (`ColumnTransformerScratch`, `PipelineScratch`), stratified cross-validation, and production JSON model persistence tested on authentic benchmarks.
3. **Research-Friendly & Safe:** Extends beyond standard OLS into modern statistical frontiers—including distribution-free conformal prediction with finite-sample coverage guarantees, conjugate Bayesian linear regression with epistemic vs. aleatoric uncertainty decomposition, Generalized Linear Models (GLMs) via IRLS, and streaming Recursive Least Squares (RLS) / Kalman filters.

### Target Audiences
- 🎓 **Students & Learners:** Master the exact mathematical and numerical mechanics of machine learning optimization without black boxes.
- 🏆 **Kaggle & Competition Competitors:** Deploy robust tabular baseline pipelines with zero data leakage, fast coordinate descent regularizers, and diagnostic tools.
- 💼 **Interview Candidates:** Discuss loss geometry, condition numbers, Hessian eigenspaces, coordinate descent derivations, and IRLS with code-backed intuition.
- 🔬 **Researchers & Educators:** Utilize a verified, MIT-licensed, dependency-free reference implementation for teaching and research.

---

## What's Inside?

- **Solvers & Optimizers:**
  * OLS Gradient Descent (full-batch & mini-batch with seed control)
  * Heavy-ball Momentum, RMSprop, Adam with moment bias correction
  * Dynamic Learning Rate Schedulers (`ConstantLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR`)
  * Closed-Form Tri-Solvers (Moore-Penrose SVD, Cholesky $X^T X = LL^T$, QR orthogonal factorization, Normal Equations with condition number $\kappa$ diagnostics)
  * Second-Order Newton-Raphson with exact 1-step quadratic convergence for MSE
- **Regularization & Non-Linearity:**
  * Ridge Regression (vectorized $\ell_2$ analytical shrinkage & weight decay)
  * Lasso Regression (cyclic coordinate descent with exact soft-thresholding $\mathcal{S}_\lambda(z)$ for guaranteed parameter sparsity)
  * ElasticNet (cyclic coordinate descent combining $\ell_1$ sparsity and $\ell_2$ grouping)
  * Polynomial Features (vectorized degree expansion with interaction terms)
- **Robust Regression:**
  * Huber Regressor (piecewise quadratic/linear loss resilient to heavy-tailed response outliers)
  * RANSAC Regressor (random sample consensus outlier isolation)
- **Uncertainty Quantification:**
  * Bayesian Linear Regression (exact conjugate Gaussian posterior, epistemic vs. aleatoric predictive bands, Automatic Relevance Determination / ARD pruning)
  * Conformal Prediction (inductive split calibration with distribution-free finite-sample coverage guarantee $P(y \in \mathcal{C}(x)) \ge 1 - \alpha$)
- **Generalized Linear Models (GLM):**
  * Iteratively Reweighted Least Squares (IRLS) engine
  * Logistic Regression (binary cross-entropy via logit link)
  * Poisson Regression (rate counts via canonical log link)
  * Gamma Regression (skewed positive continuous response via inverse link)
- **Streaming & Quantile:**
  * Recursive Least Squares (online Woodbury rank-1 updates with exponential forgetting factor $\lambda$)
  * Kalman Filter Regression (dynamic state-space parameter tracking with process noise $Q$ and observation variance $R$)
  * Quantile Regression (smoothed IRLS with asymmetric pinball loss for conditional median and multi-quantile risk corridors $\tau \in \{0.10, 0.50, 0.90\}$)
- **Production Infrastructure:**
  * Scikit-learn-compatible `PipelineScratch` and `ColumnTransformerScratch`
  * Preprocessors: `StandardScalerScratch`, `SimpleImputerScratch`, `OneHotEncoderScratch`
  * Model Selection: `KFoldScratch`, `cross_val_score_scratch`, and `GridSearchCVScratch`
  * Econometric Diagnostics: `RegressionDiagnostics`, p-values, VIF, Breusch-Pagan, Durbin-Watson
  * Dual-Format Serialization: Pure-JSON state persistence (`save_model_json`, `load_model_json`) and Python pickle (`save_model`, `load_model`)

### Validated Quality Benchmarks
- **85 / 85 unit and integration tests passing** across 18 test files in [`tests/`](tests/).
- **Masterclass notebook with 38 code cells** executing headlessly with **0 errors**.
- **19 total verified notebooks** across tutorials, examples, and competitions.
- **Authentic benchmark datasets:** California Housing ($N = 20,640$) and Kaggle Ames House Prices ($N = 1,460$).

---

## Quickstart

### Installation
Clone the repository and install in editable development mode:

```bash
git clone https://github.com/unknown404-practice/linear-models-from-scratch.git
cd linear-models-from-scratch
pip install -e .
```

To install with full development tooling (pytest, ruff, mypy, black, jupyter):
```bash
pip install -e .[dev,docs]
```

### Minimal Usage Example

```python
from linear_models_from_scratch import GradientDescentLinearRegression
from linear_models_from_scratch.data_loader import load_california_housing

# 1. Load authentic benchmark dataset
X_train, X_test, y_train, y_test, _ = load_california_housing()

# 2. Instantiate and fit gradient descent model
model = GradientDescentLinearRegression(lr=0.01, n_iters=1000, standardize=True)
model.fit(X_train, y_train)

# 3. Evaluate out-of-sample R² score
print("Test R²:", model.score(X_test, y_test))
```

### Explore Further
- 🎓 **Interactive Tutorials:** Explore step-by-step masterclasses in [`docs/tutorials/`](docs/tutorials/).
- 🚀 **Executable Examples:** Copy-paste standalone recipes in [`docs/examples/`](docs/examples/).
- 📓 **Masterclass Notebook:** Run the complete real-data masterclass in [`notebooks/`](notebooks/).

---

## Project Status

- **Core Algorithms:** Complete (100% pure vectorized NumPy/SciPy).
- **Documentation & Tutorials:** Complete (5 tutorials, 8 examples, 4 mini-projects, competition baseline).
- **Packaging & CI:** Complete (PEP 517 / PEP 621 `pyproject.toml`, GitHub Actions CI matrix for Python 3.11, 3.12, 3.13, automated headless notebook checks, Ruff linting, Mypy type-checking).
- **License:** MIT Open-Source License.

---

## Start Here: 3 Recommended Learning Pathways

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           CHOOSE YOUR ENTRY POINT                          │
├───────────────────────┬────────────────────────────┬───────────────────────┤
│    🟢 1. BEGINNER     │      🟡 2. INTERMEDIATE     │      🔴 3. ADVANCED   │
│ "I know basic Python; │   "I know ML basics; want  │   "I want GLMs, Bayes,│
│  want to understand   │   optimization, regular-   │   streaming, conformal│
│  linear regression."  │   ization & pipelines."    │   & safety bounds."   │
├───────────────────────┼────────────────────────────┼───────────────────────┤
│ 1. Read Math Basics   │ 1. Explore Loss Geometry   │ 1. Study Uncertainty  │
│    Section below      │    notebooks/02_...ipynb   │    notebooks/03_...   │
│ 2. Run Tutorial 01    │ 2. Run Tutorial 02 (Reg)   │ 2. Run Tutorial 04    │
│    docs/tutorials/    │    & Tutorial 03 (Opt)     │    (Bayes & Conformal)│
│ 3. Try Example Pipeline│ 3. Run Mini-Project 01     │ 3. Run Tutorial 05    │
│    docs/examples/     │    examples/01_end_to_end  │    (RLS & Quantile)   │
└───────────────────────┴────────────────────────────┴───────────────────────┘
```

---

## Table of Contents
- [Documentation Hub (`docs/`)](docs/README.md)
- [Interactive Tutorials (`docs/tutorials/`)](docs/tutorials/)
- [Executable Examples (`docs/examples/`)](docs/examples/)
- [API Reference (`docs/api/`)](docs/api/)
- [Design Notes (`docs/design/`)](docs/design/)
- [Mini-Projects (`examples/`)](examples/)
- [Competition Baseline (`competitions/`)](competitions/)
- [Mathematical Foundations](#mathematical-foundations)
- [Datasets Used](#datasets-used)
- [Project Structure](#project-structure)
- [Running the Test Suite](#running-the-test-suite)
- [Quickstart Code Examples](#quickstart-code-examples)
- [Non-Linear Polynomial Feature Expansion](#non-linear-polynomial-feature-expansion)
- [Robust Regression: Huber Loss & RANSAC](#robust-regression-huber-loss--ransac)
- [ElasticNet Regularization & Coordinate Descent](#elasticnet-regularization--coordinate-descent)
- [Second-Order Optimization: Newton-Raphson](#second-order-optimization-newton-raphson)
- [Bayesian Linear Regression & Epistemic Uncertainty](#bayesian-linear-regression--epistemic-uncertainty)
- [Conformal Prediction with Coverage Guarantees](#conformal-prediction-with-coverage-guarantees)
- [Generalized Linear Models (GLM) via IRLS](#generalized-linear-models-glm-via-irls)
- [Infinite-Scale Streaming Regression (RLS & Kalman)](#infinite-scale-streaming-regression-rls--kalman)
- [Quantile Regression & Asymmetric Risk Corridors](#quantile-regression--asymmetric-risk-corridors)
- [Production Pipelines & ColumnTransformer](#production-pipelines--columntransformer)
- [Model Selection & Cross-Validation](#model-selection--cross-validation)
- [Secure JSON Model Persistence](#secure-json-model-persistence)
- [Closed-Form Tri-Solvers & Conditioning](#closed-form-tri-solvers--conditioning)
- [Econometric Diagnostics & Statistical Inference](#econometric-diagnostics--statistical-inference)
- [Loss Surface Geometry & Trajectory Visualization](#loss-surface-geometry--trajectory-visualization)
- [Benchmarking Against Scikit-Learn](#benchmarking-against-scikit-learn)

---

## Mathematical Foundations

### 1. The Linear Hypothesis
For a dataset with $N$ samples and $D$ features:
- $X \in \mathbb{R}^{N \times D}$: Feature matrix
- $w \in \mathbb{R}^D$: Weight vector
- $b \in \mathbb{R}$: Bias scalar (intercept)
- $y \in \mathbb{R}^N$: True target vector

The linear hypothesis predicts $\hat{y} \in \mathbb{R}^N$:
$$\hat{y} = X w + b \mathbf{1}_N$$

### 2. Mean Squared Error (MSE) Cost Function
$$J(w, b) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \|\hat{y} - y\|_2^2$$

In matrix form, defining residual errors $e = \hat{y} - y = (Xw + b - y) \in \mathbb{R}^N$:
$$J(w, b) = \frac{1}{N} e^T e = \frac{1}{N} (Xw + b - y)^T (Xw + b - y)$$

### 3. Exact Vectorized Gradient Derivation
Using the multivariate chain rule:

$$\nabla_w J = \frac{\partial J}{\partial e} \frac{\partial e}{\partial w} = \frac{2}{N} X^T e = \frac{2}{N} X^T (\hat{y} - y)$$

$$\nabla_b J = \frac{2}{N} \sum_{i=1}^N (\hat{y}_i - y_i) = 2 \cdot \text{mean}(\hat{y} - y)$$

### 4. Parameter Update Rules & First-Order Optimizers
Parameters are updated iteratively along the descent direction:
- **SGD with Momentum & Nesterov**:
  $$v_{t+1} = \beta v_t + \nabla, \quad w^{(t+1)} = w^{(t)} - \eta v_{t+1}$$
- **RMSprop**:
  $$s_{t+1} = \beta s_t + (1 - \beta) \nabla^2, \quad w^{(t+1)} = w^{(t)} - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} \nabla$$
- **Adam (Adaptive Moment Estimation)**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$w^{(t+1)} = w^{(t)} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 5. Regularization & Coordinate Descent
- **Ridge Regression (L2)**:
  $$\nabla_w J_{\text{ridge}} = \frac{2}{N} X^T (\hat{y} - y) + 2 \alpha w$$
- **Lasso Regression (L1) via Cyclic Coordinate Descent**:
  Exact parameter sparsity is achieved using the Soft-Thresholding operator $S(v, \gamma) = \text{sign}(v) \max(|v| - \gamma, 0)$:
  $$w_j \leftarrow \frac{S\left(\frac{1}{N} X_{:, j}^T r^{(-j)}, \alpha\right)}{\frac{1}{N} \|X_{:, j}\|_2^2}$$
  driving non-predictive weights to **exact zero** ($w_j = 0.0$).

---

## Datasets Used

| Dataset | Source | Samples | Features | Target | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **California Housing** | `sklearn.datasets.fetch_california_housing` (Pace & Barry, 1997) | 20,640 | 8 continuous | `MedHouseVal` ($100k) | Public Domain / CC0 |
| **Kaggle House Prices** | Kaggle 'House Prices: Advanced Regression Techniques' (Ames, Iowa) | 1,460 | 80 tabular | `SalePrice` ($) | Competition Rules |

*Users must comply with each dataset’s license and competition rules when redistributing or publishing results.*

---

## Project Structure

```text
linear_regression_from_scratch/
├── .gitignore                    # Python, Jupyter, data cache exclusions
├── pytest.ini                    # Pytest configuration with pythonpath = .
├── requirements.txt              # Production and development dependencies
├── README.md                     # Comprehensive technical documentation
├── data/
│   ├── california_housing/       # Cached California Housing data
│   └── house_prices/
│       ├── README.md             # Data acquisition instructions
│       └── train.csv             # Kaggle Ames housing dataset
├── notebooks/
│   └── 01_linear_regression_from_scratch_real_data.ipynb  # Interactive Masterclass (38 code cells)
├── scripts/
│   ├── download_data.py          # Multi-source dataset downloader
│   └── build_notebook.py         # Programmatic notebook builder
├── src/
│   ├── __init__.py               # Package exports
│   ├── bayesian.py               # BayesianLinearRegression (conjugate posterior, ARD, uncertainty)
│   ├── conformal.py              # ConformalLinearRegression (distribution-free coverage guarantee)
│   ├── data_loader.py            # Zero-leakage preprocessing pipelines
│   ├── features.py               # PolynomialFeaturesScratch (degrees, interactions, naming)
│   ├── glm.py                    # GLM engine: Logistic, Poisson, Gamma via pure NumPy IRLS
│   ├── linear_regression.py      # Vectorized Gradient Descent, Ridge, Lasso & ElasticNet
│   ├── model_selection.py        # KFold, cross_val_score, GridSearchCV & clone_estimator
│   ├── optimizers.py             # Pure NumPy SGD, Momentum, RMSprop, Adam
│   ├── pipeline.py               # BaseTransformer, Scaler, Imputer, OHE, ColumnTransformer, Pipeline
│   ├── quantile.py               # QuantileRegressorScratch (pinball loss, non-parametric corridors)
│   ├── robust.py                 # HuberRegressorScratch (piecewise MAD) & RANSACRegressorScratch
│   ├── schedulers.py             # Cosine Annealing, Step, Exponential LRSchedulers
│   ├── serialization.py          # Secure pure-JSON model persistence (no pickle CVEs)
│   ├── solvers.py                # Closed-Form Tri-Solvers (SVD/QR/Chol) & Newton-Raphson Solver
│   ├── statistics.py             # Econometric Diagnostics, VIF, Breusch-Pagan, DW, JB
│   ├── streaming.py              # RecursiveLeastSquares O(D^2) stream & KalmanFilterRegression
│   └── visualization.py          # 2D Contours, 3D Surfaces, Quivers & Optimizer Trajectories
└── tests/
    ├── conftest.py               # Test path fixtures
    ├── test_bayesian.py          # Conjugate posterior, ARD & epistemic uncertainty tests (4 tests)
    ├── test_conformal.py         # Split calibration & coverage guarantee tests (4 tests)
    ├── test_features.py          # Non-linear polynomial & interaction expansion tests (3 tests)
    ├── test_glm.py               # Logistic, Poisson, Gamma IRLS deviance tests (4 tests)
    ├── test_linear_regression.py # Core model & data loader tests (10 tests)
    ├── test_model_selection.py   # KFold, CV score & GridSearchCV tests (3 tests)
    ├── test_optimizers.py        # Momentum, RMSprop, Adam convergence tests (4 tests)
    ├── test_pipeline.py          # Scaler, Imputer, OHE, ColumnTransformer, Pipeline tests (5 tests)
    ├── test_quantile.py          # Pinball loss & quantile corridor monotonicity tests (3 tests)
    ├── test_regularization.py    # Ridge shrinkage, Lasso sparsity & ElasticNet tests (7 tests)
    ├── test_robust.py            # Huber loss piecewise grad & RANSAC outlier tests (4 tests)
    ├── test_schedulers.py        # Learning rate schedule decay tests (4 tests)
    ├── test_second_order.py      # Newton-Raphson analytical Hessian & 1-step tests (3 tests)
    ├── test_serialization.py     # Safe JSON model & pipeline persistence tests (9 tests)
    ├── test_solvers.py           # Closed-form tri-solver & condition number tests (3 tests)
    ├── test_statistics.py        # Standard errors, t/F-stat, VIF, BP, DW, JB tests (5 tests)
    ├── test_streaming.py         # RLS Sherman-Morrison O(D^2) & Kalman filter tests (4 tests)
    └── test_visualization.py     # Loss grid, contour, 3D, quiver, optimizer tests (6 tests)
```

---

## Environment Setup

### 1. Requirements
- Python 3.11 or higher
- 16 GB RAM recommended (handles 200k+ rows easily)
- No GPU required

### 2. Installation (Cross-Platform)
```bash
# Clone or navigate to the repository
cd linear_regression_from_scratch

# Create isolated virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quickstart (Windows CMD)

```cmd
cd linear_regression_from_scratch

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

if exist scripts\download_data.py (
  python scripts\download_data.py
) else (
  echo scripts\download_data.py not found; skipping data download.
)

pytest -v tests

jupyter lab notebooks\01_linear_regression_from_scratch_real_data.ipynb
```

---

## Data Download Instructions

> **Note on Data Redistribution:** The raw `train.csv` is not included in this repository. Users must download it themselves via Kaggle, following the competition’s terms of use (or generate/cache it locally using `scripts/download_data.py`).

### Automated Download (Recommended)
Run the automated downloader which attempts Kaggle CLI, `kagglehub`, and scikit-learn/OpenML fallbacks:
```bash
python scripts/download_data.py
```

### Kaggle CLI Manual Setup
1. Download `kaggle.json` from your Kaggle Account Settings.
2. Place it in:
   - Windows: `C:\Users\<Username>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`
3. Execute:
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house_prices
```
4. Extract `train.csv` into `data/house_prices/`.

---

## Running the Test Suite

Execute the full 85-point test suite covering convergence, $R^2$ exactness, determinism, mini-batch consistency, Adam/Momentum/RMSprop optimizers, learning rate schedules, Ridge L2 shrinkage, exact Lasso L1 coordinate descent, ElasticNet coordinate descent, closed-form tri-solvers (SVD/QR/Cholesky), condition number estimation, full econometric hypothesis tests (SE, $t$-stat, $p$-value, ANOVA $F$, VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera), loss surface geometry visualizers, non-linear polynomial feature expansions (`PolynomialFeaturesScratch`), outlier-robust regression (`HuberRegressorScratch` & `RANSACRegressorScratch`), second-order Newton-Raphson optimization with analytical Hessian (`NewtonLinearRegression`), Bayesian linear regression (`BayesianLinearRegression`), conformal prediction (`ConformalLinearRegression`), Generalized Linear Models (`LogisticRegressionScratch`, `PoissonRegressionScratch`, `GammaRegressionScratch`), streaming regression (`RecursiveLeastSquares`, `KalmanFilterRegression`), quantile regression (`QuantileRegressorScratch`), modular pipelines & transformers (`StandardScalerScratch`, `SimpleImputerScratch`, `OneHotEncoderScratch`, `ColumnTransformerScratch`, `PipelineScratch`), $K$-fold cross-validation & grid search (`KFoldScratch`, `cross_val_score_scratch`, `GridSearchCVScratch`), and safe pure-JSON serialization (`save_model`, `load_model`):
```bash
pytest -v tests
```

Expected output:
```text
tests/test_bayesian.py::test_bayesian_linear_regression_recovery PASSED  [  1%]
tests/test_bayesian.py::test_bayesian_epistemic_uncertainty_extrapolation PASSED [  2%]
tests/test_bayesian.py::test_bayesian_ard_feature_pruning PASSED         [  3%]
tests/test_bayesian.py::test_bayesian_posterior_sample_weights PASSED    [  4%]
tests/test_conformal.py::test_conformal_coverage_guarantee PASSED        [  5%]
tests/test_conformal.py::test_conformal_interval_properties PASSED       [  7%]
tests/test_conformal.py::test_conformal_custom_estimator PASSED          [  8%]
tests/test_conformal.py::test_conformal_get_set_params_and_score PASSED  [  9%]
tests/test_features.py::test_polynomial_features_degree_2 PASSED         [ 10%]
tests/test_features.py::test_polynomial_features_interaction_only PASSED [ 11%]
tests/test_features.py::test_polynomial_features_pipeline_integration PASSED [ 12%]
tests/test_glm.py::test_logistic_regression_binary_accuracy PASSED       [ 14%]
tests/test_glm.py::test_poisson_regression_counts PASSED                 [ 15%]
tests/test_glm.py::test_gamma_regression_positive_continuous PASSED      [ 16%]
tests/test_glm.py::test_glm_l2_regularization PASSED                     [ 17%]
tests/test_linear_regression.py::test_standardize_features_and_apply_scaler PASSED [ 18%]
tests/test_linear_regression.py::test_linear_regression_r2_noise_free PASSED [ 20%]
tests/test_linear_regression.py::test_linear_regression_synthetic_convergence PASSED [ 21%]
tests/test_linear_regression.py::test_linear_regression_mini_batch_vs_full_batch PASSED [ 22%]
tests/test_linear_regression.py::test_linear_regression_determinism_seed PASSED [ 23%]
tests/test_linear_regression.py::test_linear_regression_large_synthetic_scale PASSED [ 24%]
tests/test_linear_regression.py::test_linear_regression_pandas_compatibility PASSED [ 25%]
tests/test_linear_regression.py::test_data_loader_california_housing PASSED [ 27%]
tests/test_linear_regression.py::test_data_loader_house_prices_kaggle PASSED [ 28%]
tests/test_linear_regression.py::test_linear_regression_missing_file_error PASSED [ 29%]
tests/test_model_selection.py::test_kfold_scratch_partitions PASSED      [ 30%]
tests/test_model_selection.py::test_cross_val_score_scratch PASSED       [ 31%]
tests/test_model_selection.py::test_grid_search_cv_scratch_finds_best_alpha PASSED [ 32%]
tests/test_optimizers.py::test_sgd_momentum_update PASSED                [ 34%]
tests/test_optimizers.py::test_adam_bias_correction_and_convergence PASSED [ 35%]
tests/test_optimizers.py::test_rmsprop_scaling PASSED                    [ 36%]
tests/test_optimizers.py::test_get_optimizer_factory PASSED              [ 37%]
tests/test_pipeline.py::test_standard_scaler_scratch PASSED              [ 38%]
tests/test_pipeline.py::test_simple_imputer_scratch PASSED               [ 40%]
tests/test_pipeline.py::test_one_hot_encoder_scratch PASSED              [ 41%]
tests/test_pipeline.py::test_column_transformer_scratch PASSED           [ 42%]
tests/test_pipeline.py::test_pipeline_scratch_fit_predict_score PASSED   [ 43%]
tests/test_quantile.py::test_quantile_median_regression PASSED           [ 44%]
tests/test_quantile.py::test_quantile_monotonicity_corridor PASSED       [ 45%]
tests/test_quantile.py::test_quantile_residual_fraction PASSED           [ 47%]
tests/test_regularization.py::test_soft_threshold_operator PASSED        [ 48%]
tests/test_regularization.py::test_adam_optimizer_linear_regression PASSED [ 49%]
tests/test_regularization.py::test_cosine_scheduler_integration PASSED   [ 50%]
tests/test_regularization.py::test_ridge_l2_weight_shrinkage PASSED      [ 51%]
tests/test_regularization.py::test_lasso_coordinate_descent_exact_sparsity PASSED [ 52%]
tests/test_regularization.py::test_elasticnet_coordinate_descent_sparsity_and_shrinkage PASSED [ 54%]
tests/test_regularization.py::test_elasticnet_extremes PASSED            [ 55%]
tests/test_robust.py::test_huber_loss_piecewise_gradient PASSED          [ 56%]
tests/test_robust.py::test_huber_regressor_outlier_resilience PASSED     [ 57%]
tests/test_robust.py::test_ransac_exact_inlier_recovery PASSED           [ 58%]
tests/test_robust.py::test_ransac_custom_estimator PASSED                [ 60%]
tests/test_schedulers.py::test_constant_lr PASSED                        [ 61%]
tests/test_schedulers.py::test_step_lr PASSED                            [ 62%]
tests/test_schedulers.py::test_exponential_lr PASSED                     [ 63%]
tests/test_schedulers.py::test_cosine_annealing_lr PASSED                [ 64%]
tests/test_second_order.py::test_newton_raphson_exact_one_step_convergence PASSED [ 65%]
tests/test_second_order.py::test_newton_raphson_regularized PASSED       [ 67%]
tests/test_second_order.py::test_newton_raphson_get_set_params PASSED    [ 68%]
tests/test_serialization.py::test_serialize_and_load_gradient_descent_model PASSED [ 69%]
tests/test_serialization.py::test_serialize_and_load_closed_form_model PASSED [ 70%]
tests/test_serialization.py::test_serialize_and_load_pipeline PASSED     [ 71%]
tests/test_serialization.py::test_serialize_and_load_newton_model PASSED [ 72%]
tests/test_serialization.py::test_serialize_and_load_huber_model PASSED  [ 74%]
tests/test_serialization.py::test_serialize_and_load_polynomial_pipeline PASSED [ 75%]
tests/test_serialization.py::test_serialize_and_load_bayesian_model PASSED [ 76%]
tests/test_serialization.py::test_serialize_and_load_conformal_and_glm PASSED [ 77%]
tests/test_serialization.py::test_serialize_and_load_streaming_and_quantile PASSED [ 78%]
tests/test_solvers.py::test_closed_form_solvers_exact_recovery PASSED    [ 80%]
tests/test_solvers.py::test_svd_handles_rank_deficient_collinear_matrix PASSED [ 81%]
tests/test_solvers.py::test_condition_number_computation PASSED          [ 82%]
tests/test_statistics.py::test_standard_errors_and_pvalues PASSED        [ 83%]
tests/test_statistics.py::test_vif_multicollinearity_detection PASSED    [ 84%]
tests/test_statistics.py::test_breusch_pagan_heteroscedasticity PASSED   [ 85%]
tests/test_statistics.py::test_durbin_watson_autocorrelation PASSED      [ 87%]
tests/test_statistics.py::test_summary_formatting PASSED                 [ 88%]
tests/test_streaming.py::test_rls_asymptotic_ols_equivalence PASSED      [ 89%]
tests/test_streaming.py::test_rls_forgetting_factor_regime_shift PASSED  [ 90%]
tests/test_streaming.py::test_rls_minibatch_partial_fit PASSED           [ 91%]
tests/test_streaming.py::test_kalman_filter_regression_tracking PASSED   [ 92%]
tests/test_visualization.py::test_compute_loss_grid_shape_and_minimum PASSED [ 94%]
tests/test_visualization.py::test_plot_loss_contours_returns_axes_and_renders_paths PASSED [ 95%]
tests/test_visualization.py::test_plot_loss_surface_3d_projection PASSED [ 96%]
tests/test_visualization.py::test_plot_gradient_quiver PASSED            [ 97%]
tests/test_visualization.py::test_compare_optimizer_trajectories_convergence PASSED [ 98%]
tests/test_visualization.py::test_visualize_conditioning_impact_figure PASSED [100%]

============================= 85 passed in 5.76s ==============================
```

---

## Running the Jupyter Masterclass

Launch JupyterLab to interact with the complete educational notebook:
```bash
jupyter lab notebooks/01_linear_regression_from_scratch_real_data.ipynb
```

The notebook contains:
1. **Full Mathematical Derivations** with vectorized matrix dimensions.
2. **Exploratory Data Analysis (EDA)** with distribution skewness and correlation heatmaps.
3. **Zero-Leakage Preprocessing** (80/20 split, median imputation, one-hot encoding, feature standardization).
4. **Model Training & Loss Curves** comparing learning rates and batch sizes.
5. **Optimizer Showdown**: Convergence trajectories of Adam vs. Momentum vs. Vanilla SGD.
6. **Scikit-Learn Benchmarking Tables** proving numerical parity.
7. **Error Analysis & Residual Diagnostics** (Predicted vs. Actual, Residual Histogram, Homoscedasticity).
8. **Loss Surface Geometry** (2D Contour and 3D Convex Bowls with gradient descent trajectory overlays).
9. **Pathological Cases** (Divergence from high $\eta$ vs. sluggish convergence from tiny $\eta$).
10. **Regularization from Scratch**: Live Ridge weight shrinkage and Lasso Coordinate Descent exact sparsity paths.
11. **Closed-Form Tri-Solvers**: SVD, QR, Cholesky factorizations, and condition number $\kappa$ under collinearity.
12. **Econometric Diagnostics**: Standard errors, $t$-values, $p$-values, 95% CIs, ANOVA $F$, VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera.
13. **Production Pipelines & Model Selection**: End-to-end `ColumnTransformerScratch`, 5-fold cross-validation, `GridSearchCVScratch`, and safe JSON model persistence.
14. **Phase 4 Advanced Frontiers**: Non-linear polynomial feature expansions, ElasticNet coordinate descent, Outlier Sabotage Showdown (OLS vs. Huber vs. RANSAC under 30 extreme leverage anomalies), and the 1-Step Newton-Raphson Quadratic Exactness Theorem.
15. **Phase 5 World-Class Frontier**: Bayesian linear regression predictive uncertainty decomposition ($\mu_* \pm 1.96 \sigma_*$), distribution-free split conformal prediction intervals with guaranteed $\ge 95\%$ test coverage, Generalized Linear Models (Poisson count & Gamma skewed continuous regression via IRLS), infinite-scale streaming parameter updates via Recursive Least Squares ($\mathcal{O}(D^2)$ time/memory), and non-parametric quantile risk corridors ($\tau=0.10, 0.50, 0.90$).

---

## Quickstart Code Examples

### 1. Training with Adam Optimizer & Cosine Annealing
```python
from src.data_loader import load_california_housing
from src.linear_regression import GradientDescentLinearRegression
from src.schedulers import CosineAnnealingLR

# Load zero-leakage preprocessed splits
X_train, X_test, y_train, y_test, features = load_california_housing(test_size=0.2, random_state=42)

# Schedule learning rate smoothly
scheduler = CosineAnnealingLR(initial_lr=0.02, T_max=400, eta_min=0.001)

# Fit model with Adam optimizer
model = GradientDescentLinearRegression(
    lr=0.02, n_iters=400, optimizer="adam", scheduler=scheduler, seed=42
)
model.fit(X_train, y_train)

print(f"Test R^2 with Adam: {model.score(X_test, y_test):.4f}")
```

### 2. Exact Feature Selection via Lasso (L1) Coordinate Descent
```python
from src.data_loader import load_house_prices_kaggle
from src.linear_regression import GradientDescentLinearRegression
import numpy as np

X_train, X_test, y_train, y_test, features = load_house_prices_kaggle(
    test_size=0.2, random_state=42
)

# Fit Lasso using cyclic coordinate descent with soft-thresholding
lasso = GradientDescentLinearRegression(penalty="l1", alpha=0.05, n_iters=400, seed=42)
lasso.fit_coordinate_descent(X_train, y_train)

zero_features = int(np.sum(lasso.weights == 0.0))
print(f"Eliminated {zero_features} / {len(features)} non-informative features!")
print(f"Lasso Test R^2: {lasso.score(X_test, y_test):.4f}")
```

### 3. Closed-Form Tri-Solver Engine & Matrix Conditioning
```python
from src.solvers import ClosedFormLinearRegression

# Robust pseudoinverse via Singular Value Decomposition
model_svd = ClosedFormLinearRegression(method="svd")
model_svd.fit(X_train, y_train)

# Inspect condition number and singular spectrum
print(f"Condition Number kappa(X): {model_svd.condition_number_:.2e}")
print(
    f"Singular values min/max: {model_svd.singular_values_[-1]:.4f} / {model_svd.singular_values_[0]:.4f}"
)
print(f"Test R^2: {model_svd.score(X_test, y_test):.4f}")
```

### 4. Publication-Grade Econometric Summary Table
```python
from src.statistics import RegressionDiagnostics

# Generate complete econometric and hypothesis testing battery
diag = RegressionDiagnostics(model_svd, X_train, y_train, feature_names=features)
print(diag.summary())
```

### 5. Multi-Optimizer Trajectory Visualizer on 2D Contour Bowl
```python
from src.visualization import compare_optimizer_trajectories, plot_loss_contours

# Compare descent paths of SGD, Momentum, RMSprop, and Adam on identical 2-feature slice
trajectories = compare_optimizer_trajectories(X_train.iloc[:, [0, 2]], y_train, lr=0.05, n_iters=50)

# Render 2D contour map with trajectory paths and global OLS optimum
fig, ax = plot_loss_contours(
    X_train.iloc[:, [0, 2]],
    y_train,
    w0_range=(-0.2, 1.2),
    w1_range=(-0.8, 0.6),
    trajectories={
        "Vanilla SGD": (trajectories["SGD"][0], "#1f77b4", "o-"),
        "Momentum": (trajectories["Momentum"][0], "#ff7f0e", "s-"),
        "RMSprop": (trajectories["RMSprop"][0], "#2ca02c", "^-"),
        "Adam": (trajectories["Adam"][0], "#d62728", "*-"),
    },
    levels=35,
)
```

### 6. Heterogeneous Production Pipeline with ColumnTransformer
```python
from src.pipeline import (
    PipelineScratch,
    ColumnTransformerScratch,
    SimpleImputerScratch,
    StandardScalerScratch,
    OneHotEncoderScratch,
)
from src.linear_regression import GradientDescentLinearRegression

# Define sub-pipelines for numeric and categorical subsets
num_pipe = PipelineScratch(
    [
        ("imputer", SimpleImputerScratch(strategy="median")),
        ("scaler", StandardScalerScratch()),
    ]
)
cat_pipe = PipelineScratch(
    [
        ("imputer", SimpleImputerScratch(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoderScratch(handle_unknown="ignore")),
    ]
)

# Compose with ColumnTransformer and regressor
pipe = PipelineScratch(
    [
        (
            "preprocessor",
            ColumnTransformerScratch(
                [
                    ("num", num_pipe, ["GrLivArea", "TotalBsmtSF", "OverallQual"]),
                    ("cat", cat_pipe, ["Neighborhood", "KitchenQual"]),
                ]
            ),
        ),
        (
            "regressor",
            GradientDescentLinearRegression(lr=0.02, n_iters=500, penalty="l2", alpha=1.0),
        ),
    ]
)

pipe.fit(X_train_df, y_train)
print(f"Pipeline Test R^2: {pipe.score(X_test_df, y_test):.4f}")
```

### 7. K-Fold Cross-Validation & Grid Search Hyperparameter Tuning
```python
from src.model_selection import KFoldScratch, cross_val_score_scratch, GridSearchCVScratch
from src.linear_regression import GradientDescentLinearRegression

# 5-Fold Cross-Validation from scratch
scores = cross_val_score_scratch(
    estimator=GradientDescentLinearRegression(lr=0.02, n_iters=300),
    X=X_train,
    y=y_train,
    cv=5,
    scoring="r2",
)
print(f"5-Fold CV Mean R^2: {scores.mean():.4f} +/- {scores.std():.4f}")

# Exhaustive Grid Search over alpha and lr
grid = GridSearchCVScratch(
    estimator=GradientDescentLinearRegression(n_iters=300, seed=42),
    param_grid={"penalty": ["l2"], "alpha": [0.001, 0.01, 0.1, 1.0], "lr": [0.01, 0.05]},
    cv=3,
    scoring="r2",
)
grid.fit(X_train, y_train)
print("Best Hyperparameters:", grid.best_params_)
print(f"Best CV Score: {grid.best_score_:.4f}")
```

### 8. Safe Pure-JSON Model Persistence (Zero Pickle Security Vulnerabilities)
```python
from src.serialization import save_model, load_model

# Persist fitted pipeline to human-readable JSON (safe from code injection CVEs)
save_model(pipe, "models/production_pipeline.json")

# Restore pipeline cleanly
loaded_pipe = load_model("models/production_pipeline.json")
assert (pipe.predict(X_test_df) == loaded_pipe.predict(X_test_df)).all()
print("Model reloaded and verified bit-for-bit identical!")
```

### 9. Non-Linear Polynomial Feature Expansion in a Pipeline
```python
from src.features import PolynomialFeaturesScratch
from src.pipeline import PipelineScratch, StandardScalerScratch
from src.linear_regression import GradientDescentLinearRegression

# Create degree-2 polynomial expansion with interaction terms
poly_pipe = PipelineScratch(
    [
        ("scaler_in", StandardScalerScratch()),
        ("poly", PolynomialFeaturesScratch(degree=2, include_bias=False)),
        ("scaler_out", StandardScalerScratch()),
        (
            "regressor",
            GradientDescentLinearRegression(lr=0.02, n_iters=400, penalty="l2", alpha=0.1, seed=42),
        ),
    ]
)
poly_pipe.fit(X_train, y_train)
print(f"Degree-2 Polynomial Test R^2: {poly_pipe.score(X_test, y_test):.4f}")
```

### 10. ElasticNet Regularization (L1 + L2 Coordinate Descent)
```python
import numpy as np
from src.linear_regression import GradientDescentLinearRegression

# Balance Lasso feature sparsity (l1_ratio=0.5) with Ridge collinearity stability
enet = GradientDescentLinearRegression(
    penalty="elasticnet",
    alpha=0.05,
    l1_ratio=0.5,
    n_iters=500,
    seed=42,
)
enet.fit_coordinate_descent(X_train, y_train)
zero_weights = int(np.sum(enet.weights == 0.0))
print(f"Zeroed {zero_weights} coefficients | Test R^2: {enet.score(X_test, y_test):.4f}")
```

### 11. Outlier-Robust Regression with Huber Loss & RANSAC
```python
from src.robust import HuberRegressorScratch, RANSACRegressorScratch

# Huber piecewise loss (linear penalty on large anomalies)
huber = HuberRegressorScratch(epsilon=1.35, max_iter=300, lr=0.02)
huber.fit(X_train, y_train)
print(f"Huber Test R^2: {huber.score(X_test, y_test):.4f}")

# RANSAC random consensus (identifies clean inliers and rejects leverage anomalies)
ransac = RANSACRegressorScratch(residual_threshold=1.5, max_trials=100, random_state=42)
ransac.fit(X_train, y_train)
print(
    f"RANSAC Inlier Count: {ransac.inlier_mask_.sum()} / {len(X_train)} | Test R^2: {ransac.score(X_test, y_test):.4f}"
)
```

### 12. Second-Order Newton-Raphson Optimization (Exact 1-Step Quadratic Solver)
```python
from src.solvers import NewtonLinearRegression

# Newton-Raphson computes exact analytical Hessian H = 2/N * X^T X
newton = NewtonLinearRegression(damping=1e-5, max_iter=5)
newton.fit(X_train, y_train)
print(
    f"Newton-Raphson converged in {newton.n_iter_} step! Test R^2: {newton.score(X_test, y_test):.4f}"
)
```

### 13. Bayesian Linear Regression & Epistemic Uncertainty Decomposition
```python
from src.bayesian import BayesianLinearRegression

bayes = BayesianLinearRegression(alpha=1.0, beta=10.0, fit_intercept=True)
bayes.fit(X_train, y_train)

# Predict conditional mean and standard deviation
y_mean, y_std = bayes.predict(X_test, return_std=True)
print(f"Bayesian Test R^2: {bayes.score(X_test, y_test):.4f}")
print(f"Average Predictive Uncertainty (sigma_*): {np.mean(y_std):.4f}")
```

### 14. Split Conformal Prediction with Guaranteed 95% Coverage
```python
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression

conf = ConformalLinearRegression(
    estimator=ClosedFormLinearRegression(method="svd"), confidence_level=0.95
)
conf.fit(X_train, y_train, cal_size=0.25)

y_pred, y_low, y_high = conf.predict(X_test, return_intervals=True)
coverage = conf.coverage_score(X_test, y_test)
print(f"Empirical Test Coverage: {coverage * 100:.2f}% (Guaranteed >= 95%!)")
print(f"Conformal Interval Width: ±{conf.q_hat_:.4f}")
```

### 15. Generalized Linear Models (Poisson Count Regression via IRLS)
```python
from src.glm import PoissonRegressionScratch

poisson_glm = PoissonRegressionScratch(max_iter=50)
poisson_glm.fit(X_rooms_train, y_rooms_train)
print(
    f"Poisson Deviance: {poisson_glm.deviance_:.2f} | Converged in {poisson_glm.n_iter_} IRLS steps"
)
```

### 16. Infinite-Scale Streaming Regression via Recursive Least Squares (RLS)
```python
from src.streaming import RecursiveLeastSquares

rls = RecursiveLeastSquares(lambda_=0.99, delta=1000.0)
# Update model sample by sample in O(D^2) time
for x_i, y_i in zip(X_stream, y_stream):
    rls.partial_fit(x_i, y_i)

print(f"RLS Stream Fit Completed: {rls.n_samples_seen_} samples processed in real time!")
print(f"Stream RLS Test R^2: {rls.score(X_test, y_test):.4f}")
```

### 17. Quantile Regression Corridors (10th, 50th, 90th Percentiles)
```python
from src.quantile import QuantileRegressorScratch

q10 = QuantileRegressorScratch(quantile=0.10).fit(X_train, y_train)
q50 = QuantileRegressorScratch(quantile=0.50).fit(X_train, y_train)
q90 = QuantileRegressorScratch(quantile=0.90).fit(X_train, y_train)

# Predict 80% non-parametric risk envelope
lower_floor = q10.predict(X_test)
median_pred = q50.predict(X_test)
upper_roof = q90.predict(X_test)
```

---

## Non-Linear Polynomial Feature Expansion

Linear models are linear in parameters $w$, but not constrained to linear representations of raw features $x$. By non-linearly projecting input features into higher-dimensional polynomial basis spaces:
$$\phi(x) = \left[ 1, x_1, x_2, \dots, x_D, x_1^2, x_1 x_2, \dots, x_D^2 \right]^T$$
linear regression fits complex curved manifolds, interaction effects, and non-linear response surfaces.

### 1. Vectorized Basis Construction (`PolynomialFeaturesScratch`)
- Supports arbitrary degrees $d \ge 1$.
- Generates degree combinations via `itertools.combinations_with_replacement` of lengths $k \in \{1, \dots, d\}$.
- When `interaction_only=True`, pure powers ($x_j^k$ for $k > 1$) are excluded, capturing synergistic interaction terms ($x_i x_j$) without polynomial scaling exploding.
- When `include_bias=True`, prepends a constant column $\mathbf{1}_N$ ($x_0^0 = 1$).
- Automated feature naming via `get_feature_names_out(input_features)` producing clean strings (e.g., `"MedInc^2"`, `"MedInc AveRooms"`).

---

## Robust Regression: Huber Loss & RANSAC

Ordinary Least Squares squares errors $(y - \hat{y})^2$, magnifying the gradient pull of extreme outliers and leverage points quadratically. Under contamination or heavy-tailed error distributions, OLS estimates collapse.

### 1. Huber Loss (`HuberRegressorScratch`)
Piecewise loss smooths transitions between quadratic MSE on typical errors and absolute linear loss on anomalies:
$$L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \le \delta \\ \delta |r| - \frac{1}{2}\delta^2 & \text{for } |r| > \delta \end{cases}$$

- **Adaptive Scale Estimation:** Outlier threshold is scaled adaptively using the Median Absolute Deviation (MAD) of residuals:
  $$\hat{\sigma} = \text{median}(|r - \text{median}(r)|) \times 1.4826$$
  $$\delta = \epsilon \times \hat{\sigma}$$
- **Bounded Gradient:** For large outliers $|r| > \delta$, the gradient magnitude is capped at $\pm \delta$, preventing leverage points from distorting parameter updates.

### 2. Random Sample Consensus (`RANSACRegressorScratch`)
RANSAC iteratively separates clean inlier observations from anomalous leverage points:
1. Uniformly sample minimal subset of points ($s \ge D + 1$).
2. Fit hypothetical estimator on sample subset.
3. Compute absolute residuals $|y - \hat{y}|$ across entire dataset.
4. Tally inliers: $\{i \mid |y_i - \hat{y}_i| \le \tau\}$.
5. Select candidate model with maximum inlier support; refit on complete inlier set.

### 3. Outlier Sabotage Showdown
On California Housing data subjected to 30 extreme leverage anomalies ($y \leftarrow y \pm 50.0$ on high income tracts):
- **OLS / Standard GD:** Collapsed to catastrophic negative performance ($R^2 = -4.8814$, $\text{MSE} = 7.7058$).
- **Huber Regressor (`HuberRegressorScratch`):** Maintained strong generalization ($R^2 = 0.4998$, $\text{MSE} = 0.6617$).
- **RANSAC (`RANSACRegressorScratch`):** Successfully isolated 258/270 clean inliers ($R^2 = 0.4907$, $\text{MSE} = 0.6737$).

---

## ElasticNet Regularization & Coordinate Descent

When features exhibit severe multicollinearity (e.g. `AveRooms` and `AveBedrms` in housing data), Lasso randomly picks one feature while Ridge shrinks all correlated features jointly without zeroing any.

**ElasticNet** resolves this compromise by minimizing the composite objective:
$$J(w) = \frac{1}{2N} \|y - Xw\|_2^2 + \alpha \rho \|w\|_1 + \frac{1}{2} \alpha (1 - \rho) \|w\|_2^2$$
where $\rho = \text{l1\_ratio} \in [0, 1]$.

### Cyclic Coordinate Descent Update Rule
Let partial residual $r^{(-j)} = y - X w + X_{:, j} w_j$ and column normalization $c_j = \frac{1}{N} \|X_{:, j}\|_2^2$:
$$w_j \leftarrow \frac{S\left(\frac{1}{N} X_{:, j}^T r^{(-j)}, \; \alpha \rho\right)}{c_j + \alpha (1 - \rho)}$$
where $S(v, \gamma) = \text{sign}(v) \max(|v| - \gamma, 0)$ is the soft-thresholding operator.
- When $\rho = 1$, updates reduce exactly to Lasso Coordinate Descent.
- When $\rho = 0$, updates reduce exactly to Ridge Regression.
- For $\rho \in (0, 1)$, ElasticNet selects groups of correlated features while maintaining exact sparsity!

---

## Second-Order Optimization: Newton-Raphson

While first-order gradient descent uses only local slope information ($\nabla J$), second-order optimization leverages curvature via the analytical Hessian matrix $H = \nabla^2 J$.

### 1. Analytical Hessian Derivation
For Mean Squared Error $J(w) = \frac{1}{N} (Xw - y)^T (Xw - y)$:
$$\nabla_w J = \frac{2}{N} X^T (Xw - y)$$
$$\nabla_w^2 J = H = \frac{2}{N} X^T X$$
Because $J(w)$ is strictly quadratic, its Hessian is constant everywhere and identical to $\frac{2}{N} X^T X$.

### 2. The 1-Step Quadratic Exactness Theorem
The Newton-Raphson update step is:
$$\Delta w = - H^{-1} \nabla_w J = -\left( \frac{2}{N} X^T X \right)^{-1} \left( \frac{2}{N} X^T (Xw^{(0)} - y) \right)$$
$$= -(X^T X)^{-1} X^T X w^{(0)} + (X^T X)^{-1} X^T y = -w^{(0)} + (X^T X)^{-1} X^T y$$
Adding $\Delta w$ to initial weights $w^{(0)}$:
$$w^{(1)} = w^{(0)} + \Delta w = (X^T X)^{-1} X^T y$$
**Theorem:** For any arbitrary starting point $w^{(0)}$, Newton-Raphson jumps directly to the exact analytical global minimum in **exactly 1 step** ($n\_iter\_ = 1$).

### 3. Levenberg-Marquardt Damping
To ensure numerical stability on ill-conditioned or collinear matrices, `NewtonLinearRegression` applies Tikhonov damping:
$$H_{\text{damped}} = H + \lambda I$$
guaranteeing positive definiteness and non-singular matrix inversion.

---

## Bayesian Linear Regression & Epistemic Uncertainty

Traditional OLS yields a single point estimate $\hat{w}$, offering zero awareness of its own epistemic uncertainty. When presented with out-of-distribution (OOD) test inputs, standard models emit overconfident, catastrophic predictions. 

Our pure NumPy `BayesianLinearRegression` formulates regression within an exact probabilistic framework with conjugate Gaussian priors and likelihoods.

### 1. The Conjugate Prior & Likelihood
- **Prior:** $w \sim \mathcal{N}(m_0, S_0)$ where $m_0 = \mathbf{0}$ and prior covariance $S_0 = \alpha^{-1} I$ (or diagonal ARD precision matrix).
- **Likelihood:** $y \mid X, w, \beta \sim \mathcal{N}(Xw, \beta^{-1} I)$ where $\beta = \sigma_n^{-2}$ is the observation noise precision.

### 2. Analytical Gaussian Posterior
By Gaussian conjugacy, the posterior over weights $p(w \mid X, y) = \mathcal{N}(m_N, S_N)$ is closed-form:
$$S_N = \left( \alpha I + \beta X^T X \right)^{-1}$$
$$m_N = \beta S_N X^T y$$
The maximum a posteriori (MAP) estimate $m_N$ exactly recovers Ridge Regression with penalty $\lambda = \frac{\alpha}{\beta}$.

### 3. Predictive Distribution & Uncertainty Decomposition
For any novel query point $x_* \in \mathbb{R}^D$, marginalizing over all possible weight configurations yields a univariate Gaussian predictive distribution:
$$p(y_* \mid x_*, X, y) = \mathcal{N}\left( \mu_*(x_*), \; \sigma_*^2(x_*) \right)$$
$$\mu_*(x_*) = x_*^T m_N$$
$$\sigma_*^2(x_*) = \underbrace{\frac{1}{\beta}}_{\text{Aleatoric Uncertainty (Noise)}} + \underbrace{x_*^T S_N x_*}_{\text{Epistemic Uncertainty (Model Ignorance)}}$$

- **Aleatoric Uncertainty ($\frac{1}{\beta}$):** Irreducible noise inherent in the data-generating process.
- **Epistemic Uncertainty ($x_*^T S_N x_*$):** Vanishes in densely sampled training regions ($S_N \to 0$), but expands quadratically as $x_*$ moves into unfamiliar or out-of-distribution territory.

### 4. Posterior Weight Sampling & ARD
- **Posterior Sampling (`sample_weights(n_samples)`):** Draws $w^{(s)} \sim \mathcal{N}(m_N, S_N)$ via Cholesky decomposition ($m_N + L z$, $L L^T = S_N$), visualizing diverse hypotheses consistent with the training evidence.
- **Automatic Relevance Determination (ARD):** Iteratively updates individual feature hyperpriors $\alpha_j = \frac{\gamma_j}{m_{N, j}^2}$ where $\gamma_j = 1 - \alpha_j S_{N, jj}$, driving irrelevant feature weights to exact zero.

---

## Conformal Prediction with Coverage Guarantees

While Bayesian methods require prior and likelihood assumptions, **Conformal Prediction** provides mathematically guaranteed, distribution-free prediction intervals with exact finite-sample validity for any regression model.

### 1. Finite-Sample Coverage Theorem
Given exchangeable training and calibration pairs $(X_i, y_i) \sim P$, for any desired miscoverage rate $\alpha \in (0, 1)$ and **any** arbitrary black-box regression model $\hat{\mu}$:
$$P\left( y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \ge 1 - \alpha$$
This guarantee holds non-asymptotically with zero assumptions on Gaussianity, linearity, or error homoscedasticity.

### 2. Split Conformal Algorithm (`ConformalLinearRegression`)
1. **Split:** Partition available training data into proper training set $(X_{\text{train}}, y_{\text{train}})$ and calibration set $(X_{\text{cal}}, y_{\text{cal}})$ of size $n_{\text{cal}}$.
2. **Fit:** Fit base estimator $\hat{\mu}$ strictly on $(X_{\text{train}}, y_{\text{train}})$.
3. **Nonconformity Scores:** Compute calibration residuals $s_i = |y_i - \hat{\mu}(x_i)|$ for all $i \in \{1, \dots, n_{\text{cal}}\}$.
4. **Empirical Quantile Cutoff:** Compute the finite-sample adjusted quantile:
   $$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^{n_{\text{cal}}}, \; \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right)$$
5. **Prediction Band:** For query $x_*$, output the calibrated corridor $[\hat{\mu}(x_*) - \hat{q}, \;\hat{\mu}(x_*) + \hat{q}]$.

---

## Generalized Linear Models (GLM) via IRLS

Standard Ordinary Least Squares assumes Gaussian residuals with an unbounded identity link $\mathbb{E}[y|X] = Xw$. This assumption fails for non-negative count data (hospital admissions, transactions) or strictly positive, right-skewed physical/financial quantities (insurance claims, real estate prices).

Our `GeneralizedLinearModel` engine implements the full Exponential Dispersion Family optimized via Iteratively Reweighted Least Squares (IRLS).

### 1. Exponential Dispersion Family Formulation
$$f(y; \theta, \phi) = \exp\left( \frac{y \theta - b(\theta)}{a(\phi)} + c(y, \phi) \right)$$
- Linear predictor: $\eta = Xw$
- Conditional expectation: $\mu = \mathbb{E}[y \mid X] = g^{-1}(\eta)$ where $g(\cdot)$ is the link function.
- Variance function: $\text{Var}(y \mid X) = \phi V(\mu)$

| Family | Canonical Link $g(\mu)$ | Inverse Link $g^{-1}(\eta)$ | Variance Function $V(\mu)$ | Target Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian** | Identity: $\mu$ | $\eta$ | $1$ | $(-\infty, \infty)$ |
| **Binomial** | Logit: $\ln\left(\frac{\mu}{1-\mu}\right)$ | $\frac{1}{1 + e^{-\eta}}$ | $\mu (1 - \mu)$ | $y \in [0, 1]$ or $\{0, 1\}$ |
| **Poisson** | Log: $\ln(\mu)$ | $\exp(\eta)$ | $\mu$ | $y \in \{0, 1, 2, \dots\}$ |
| **Gamma** | Reciprocal / Log: $\frac{1}{\mu}$ or $\ln(\mu)$ | $\frac{1}{\eta}$ or $\exp(\eta)$ | $\mu^2$ | $y \in (0, \infty)$ |

### 2. Iteratively Reweighted Least Squares (IRLS)
IRLS is the Newton-Raphson algorithm applied to the GLM log-likelihood. At each iteration $t$:
1. Form the linearized working response $z \in \mathbb{R}^N$:
   $$z_i^{(t)} = \eta_i^{(t)} + (y_i - \mu_i^{(t)}) \cdot g'(\mu_i^{(t)})$$
2. Form the diagonal Fisher information weight matrix $W \in \mathbb{R}^{N \times N}$:
   $$W_{ii}^{(t)} = \frac{1}{V(\mu_i^{(t)}) \cdot [g'(\mu_i^{(t)})]^2}$$
3. Solve the weighted least squares problem with optional L2 regularization:
   $$w^{(t+1)} = \left( X^T W^{(t)} X + \lambda I \right)^{-1} X^T W^{(t)} z^{(t)}$$
IRLS converges quadratically in 5–15 iterations.

### 3. Model Deviance & Goodness of Fit
GLMs are evaluated via Deviance $D(y, \hat{\mu}) = 2 \left( \ell_{\text{saturated}} - \ell(w) \right)$:
- **Poisson Deviance:** $D(y, \hat{\mu}) = 2 \sum_i \left[ y_i \ln\left(\frac{y_i}{\hat{\mu}_i}\right) - (y_i - \hat{\mu}_i) \right]$
- **Gamma Deviance:** $D(y, \hat{\mu}) = 2 \sum_i \left[ -\ln\left(\frac{y_i}{\hat{\mu}_i}\right) + \frac{y_i - \hat{\mu}_i}{\hat{\mu}_i} \right]$
- **Pseudo-$R^2$ (Deviance Explained):**
  $$R_{\text{deviance}}^2 = 1 - \frac{D_{\text{residual}}}{D_{\text{null}}}$$

---

## Infinite-Scale Streaming Regression (RLS & Kalman)

When processing high-frequency financial ticks, streaming IoT sensor signals, or infinite database cursors, collecting all data points into a memory buffer is impossible.

### 1. Recursive Least Squares (`RecursiveLeastSquares`)
RLS computes the exact analytical least-squares solution online with **$\mathcal{O}(D^2)$ time complexity per sample** and constant memory $\mathcal{O}(D^2)$ using the rank-1 Sherman-Morrison matrix inversion theorem.

Let $P_t = (X_{1:t}^T X_{1:t})^{-1} \in \mathbb{R}^{D \times D}$ be the running inverse covariance matrix:
1. **A priori prediction error:**
   $$\alpha_t = y_t - x_t^T w_{t-1}$$
2. **Gain vector computation:**
   $$k_t = \frac{P_{t-1} x_t}{\lambda + x_t^T P_{t-1} x_t}$$
3. **Parameter update:**
   $$w_t = w_{t-1} + k_t \alpha_t$$
4. **Rank-1 Inverse Covariance Update (Sherman-Morrison):**
   $$P_t = \frac{1}{\lambda} \left( P_{t-1} - k_t x_t^T P_{t-1} \right)$$

- **Forgetting Factor $\lambda \in (0, 1]$:** When $\lambda = 1.0$, RLS yields the exact OLS solution on all historical data. When $\lambda < 1.0$ (e.g. $0.99$), RLS exponentially downweights past data, dynamically tracking non-stationary concept drift!

### 2. State-Space Kalman Filter Regression (`KalmanFilterRegression`)
Treats the regression weights as a hidden state evolving over time with Gaussian transition noise:
$$w_t = w_{t-1} + \mathbf{q}_t, \quad \mathbf{q}_t \sim \mathcal{N}(0, Q)$$
$$y_t = x_t^T w_t + r_t, \quad r_t \sim \mathcal{N}(0, R)$$
Allows dynamic bayesian tracking with adaptive process covariance $Q$ and observation variance $R$.

---

## Quantile Regression & Asymmetric Risk Corridors

Ordinary Least Squares models the conditional mean $\mathbb{E}[y \mid X]$. In risk management, financial forecasting, and capacity planning, practitioners need the conditional quantiles $Q_y(\tau \mid X)$ to estimate downside Value-at-Risk (VaR) or optimistic bounds.

### 1. The Asymmetric Pinball Loss
For target quantile $\tau \in (0, 1)$, the pinball (check) loss is defined as:
$$\rho_\tau(u) = u \cdot (\tau - \mathbb{I}(u < 0)) = \begin{cases} \tau \cdot u & \text{if } u \ge 0 \\ (\tau - 1) \cdot u & \text{if } u < 0 \end{cases}$$
where residual $u = y - Xw$.
- $\tau = 0.50$ corresponds to Median Regression (Least Absolute Deviations, L1).
- $\tau = 0.90$ penalizes underpredictions ($y > \hat{y}$) 9 times more heavily than overpredictions.

### 2. Smoothed IRLS Optimization (`QuantileRegressorScratch`)
Because the pinball loss is non-differentiable at $u = 0$, `QuantileRegressorScratch` utilizes smoothed iteratively reweighted least squares:
$$W_{ii}^{(t)} = \frac{|\tau - \mathbb{I}(u_i^{(t)} < 0)|}{\max(|u_i^{(t)}|, \epsilon)}$$
$$w^{(t+1)} = \left( X^T W^{(t)} X + \lambda I \right)^{-1} X^T W^{(t)} y$$

### 3. Asymmetric Prediction Corridors
Fitting quantiles $\tau \in \{0.10, 0.50, 0.90\}$ constructs an asymmetric uncertainty corridor without assuming homoscedasticity or Gaussianity, directly reflecting data heteroscedasticity.

---

## Production Pipelines & ColumnTransformer

In real-world workflows, raw data features are heterogeneous (continuous measurements, discrete counts, categorical tokens) and contain missing entries. Preprocessing each column group in isolation causes severe data leakage unless orchestrated centrally.

### 1. Transformer Architecture
All transformers inherit from `BaseTransformer` and adhere to the strict `fit`/`transform`/`fit_transform` contract:
- **`StandardScalerScratch`**: Computes column means $\mu$ and variances $\sigma^2$ on training samples; scales $z = (x - \mu)/\sigma$ with zero-variance safeguards.
- **`SimpleImputerScratch`**: Supports `median`, `mean`, `constant`, and `most_frequent` imputation across both continuous floating-point matrices and string/object categorical arrays.
- **`OneHotEncoderScratch`**: Fits distinct category vocabularies, generates dummy variables, and supports `handle_unknown='ignore'` or `'error'` to handle novel test tokens without shape collapse.

### 2. Composition via `ColumnTransformerScratch` & `PipelineScratch`
- `ColumnTransformerScratch` applies distinct transformer chains to specified column name or index subsets, automatically concatenating the resulting numeric blocks into a unified design matrix.
- `PipelineScratch` chains sequential preprocessing transformers with a terminal estimator. Calling `fit(X, y)` fits each intermediate transformer sequentially and applies `transform` to feed downstream stages, before fitting the final estimator.

---

## Model Selection & Cross-Validation

Standard train/test splits can lead to high variance on smaller competition datasets. Our pure NumPy model selection engine provides:

### 1. `KFoldScratch` Partitioning
Splits $N$ samples into $K$ disjoint contiguous or randomly shuffled folds:
$$\text{Fold}_k = \left\{ i \in \{1, \dots, N\} \;\middle|\; \lfloor (k-1) N / K \rfloor \le i < \lfloor k N / K \rfloor \right\}$$
Yields $(K-1)$ folds for training and 1 fold for validation with zero sample overlap.

### 2. `cross_val_score_scratch`
Evaluates any compatible estimator across all $K$ folds, returning an array of fold scores (`r2`, `mse`, `rmse`, or `mae`).

### 3. `GridSearchCVScratch`
Performs exhaustive Cartesian product search over user-specified hyperparameter spaces:
- Clones the estimator via `clone_estimator` and updates configurations via `set_params`.
- Evaluates each hyperparameter combination across $K$ folds.
- Records `cv_results_` with Scikit-Learn-compatible `param_{name}`, `mean_test_score`, and `std_test_score`.
- Automatically refits the best parameter combination on the entire dataset if `refit=True`.

---

## Secure JSON Model Persistence

Python's standard `pickle` module serializes arbitrary bytecode, exposing machine learning servers to **remote code execution (RCE) vulnerabilities** when loading untrusted model weights.

Our serialization engine (`src/serialization.py`) uses **pure JSON schemas**:
- **Estimator State:** Stores weights $w$, bias $b$, condition number $\kappa$, rank, solver method, optimizer configuration, and input feature names.
- **Transformer State:** Stores scaling factors (`mean_`, `scale_`), imputation statistics (`statistics_`), and categorical dictionaries (`categories_`).
- **Pipeline Graphs:** Recursively serializes and deserializes nested pipelines and column transformers.
- **Portability:** Serialized JSON models are platform-independent, human-auditable, and loadable across languages (Python, Go, Rust, C++).

---

## Closed-Form Tri-Solvers & Conditioning

While first-order iterative optimizers scale effectively to large streaming datasets, analytical closed-form solutions provide exact minimum-norm solutions without hyperparameter tuning or iteration limits.

### 1. The Normal Equation Fallacy
The textbook closed-form solution:
$$w = (X^T X)^{-1} X^T y$$
is **numerically unstable in floating-point arithmetic**. The condition number of $X^T X$ squares the condition number of $X$:
$$\kappa(X^T X) = (\kappa(X))^2$$
If $\kappa(X) = 10^8$, computing $X^T X$ yields $\kappa(X^T X) = 10^{16}$, exhausting standard IEEE-754 64-bit precision and resulting in cataclysmic precision loss.

### 2. Tri-Solver Numerical Hierarchy
Our `ClosedFormLinearRegression` implements three stable factorizations:

| Solver | Factorization | Computational Complexity | Stability ($\kappa$) | Rank-Deficient Handling |
| :--- | :--- | :--- | :--- | :--- |
| **SVD (Default)** | $X = U \Sigma V^T$ | $\mathcal{O}(N D^2)$ | $\kappa(X)$ (Best) | Truncates $\sigma_i < \epsilon \cdot \sigma_1$ |
| **QR Decomposition** | $X = Q R \implies R w = Q^T y$ | $\mathcal{O}(2 N D^2)$ | $\kappa(X)$ (High) | Back-substitution via `scipy.linalg.solve_triangular` |
| **Cholesky Factorization** | $X^T X = L L^T$ | $\mathcal{O}\left(\frac{1}{3} D^3 + N D^2\right)$ | $\kappa(X)^2$ (Fastest) | Requires strictly positive-definite $X^T X$ |

### 3. Centering to Prevent Condition Number Inflation
Appending a constant intercept column $\mathbf{1}_N$ directly to $X$ drastically inflates condition numbers. `ClosedFormLinearRegression` centers both $X$ and $y$:
$$X_c = X - \mu_X, \quad y_c = y - \mu_y$$
solves for weights $w$ on centered coordinates, then recovers the intercept analytically:
$$b = \mu_y - \mu_X w$$

---

## Econometric Diagnostics & Statistical Inference

Machine learning models prioritize out-of-sample prediction ($R^2$, RMSE), whereas econometrics demands hypothesis testing, confidence intervals, and verification of the Gauss-Markov assumptions. The `RegressionDiagnostics` suite computes:

### 1. Parameter Hypothesis Tests & Confidence Intervals
- **Residual Variance:**
  $$s^2 = \frac{\sum (y_i - \hat{y}_i)^2}{N - p - 1}$$
- **Parameter Covariance Matrix:**
  $$\text{Cov}(\hat{\beta}) = s^2 (X_{\text{design}}^T X_{\text{design}})^{-1}$$
- **Standard Errors:** $\text{SE}(\hat{\beta}_j) = \sqrt{\text{Cov}(\hat{\beta})_{jj}}$
- **Two-Tailed $t$-statistic & $p$-value:**
  $$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}, \quad p_j = 2 \cdot (1 - F_{\text{Student-}t}(|t_j|, \text{df}=N - p - 1))$$
- **95% Confidence Intervals:** $\hat{\beta}_j \pm t_{\text{crit}} \cdot \text{SE}(\hat{\beta}_j)$

### 2. Global Model Significance (ANOVA F-Test)
$$\text{F-Statistic} = \frac{\text{MSR}}{\text{MSE}} = \frac{\frac{\text{SS}_{\text{reg}}}{p}}{\frac{\text{SS}_{\text{res}}}{N - p - 1}}, \quad p_F = 1 - F_{\text{Fisher-Snedecor}}(F, p, N - p - 1)$$

### 3. Econometric Assumption Tests

| Diagnostic | Null Hypothesis ($H_0$) | Test Statistic / Criterion | Decision Rule |
| :--- | :--- | :--- | :--- |
| **Variance Inflation Factor (VIF)** | No collinearity with other features | $\text{VIF}_j = \frac{1}{1 - R_j^2}$ | $\text{VIF} > 5$ (Moderate), $\text{VIF} > 10$ (Severe Collinearity) |
| **Breusch-Pagan Test** | Homoscedasticity ($\sigma_i^2 = \sigma^2$) | $LM = N \cdot R_{e^2}^2 \sim \chi^2(p)$ | $p < 0.05 \implies$ Reject homoscedasticity (Heteroscedastic errors) |
| **Durbin-Watson Test** | No first-order autocorrelation ($\rho = 0$) | $d = \frac{\sum_{i=2}^N (e_i - e_{i-1})^2}{\sum_{i=1}^N e_i^2}$ | $d \approx 2$ (No serial correlation), $d < 1.5$ (Positive correlation) |
| **Jarque-Bera Test** | Residuals are normally distributed | $JB = \frac{N}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2(2)$ | $p < 0.05 \implies$ Residuals deviate from normality |

### 4. Publication-Grade ASCII Summary
The output of `diag.summary()` produces a formatted econometric report identical in rigor and precision to `statsmodels`:
```text
========================================================================================================
                                 OLS REGRESSION DIAGNOSTICS SUMMARY
========================================================================================================
Model Method:              Closed-Form (svd)    R-squared:                           0.6062
Dependent Variable:                        y    Adjusted R-squared:                  0.6060
No. Observations:                      16512    F-statistic:                         3177.306
Degrees of Freedom (Resid):            16503    Prob (F-statistic):                  0.0000e+00
Log-Likelihood:                   -18042.112    Residual Std Error (s):              0.7284
Condition Number kappa(X):          9.23e+00
========================================================================================================
                   coef        std err          t        P>|t|       [0.025       0.975]         VIF
--------------------------------------------------------------------------------------------------------
const            2.0686         0.0057     365.13     0.0000***      2.0575       2.0797           -
MedInc           0.8296         0.0074     112.55     0.0000***      0.8152       0.8441        2.50
HouseAge         0.1188         0.0062      19.03     0.0000***      0.1066       0.1310        1.24
AveRooms        -0.2655         0.0184     -14.44     0.0000***     -0.3015      -0.2294       10.93
AveBedrms        0.3057         0.0163      18.73     0.0000***      0.2737       0.3377        8.57
Population      -0.0045         0.0061      -0.74     0.4578        -0.0165       0.0074        1.14
AveOccup        -0.0393         0.0059      -6.68     0.0000***     -0.0508      -0.0277        1.01
Latitude        -0.8998         0.0178     -50.51     0.0000***     -0.9347      -0.8649        9.98
Longitude       -0.8705         0.0177     -49.19     0.0000***     -0.9052      -0.8358        9.88
========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

---

## Loss Surface Geometry & Trajectory Visualization

To deeply understand first-order optimization dynamics, `src/visualization.py` provides a suite of headless-safe plotting routines that compute and visualize quadratic loss surfaces, vector fields, and optimizer paths:

### 1. Vectorized Loss Surface Grid (`compute_loss_grid`)
Evaluates the MSE cost $J(w_0, w_1)$ over a 2D meshgrid via fully vectorized matrix multiplications without nested Python loops:
$$J(W_0, W_1) = \frac{1}{N} \|X_{\text{sub}} W - y\|_2^2$$

### 2. Multi-Optimizer Trajectory Comparison (`compare_optimizer_trajectories` & `plot_loss_contours`)
Simulates and visualizes the exact iterative paths taken by:
- **Vanilla SGD**: Direct steepest descent step.
- **Momentum ($\beta=0.9$)**: Accumulates kinetic velocity along consistent gradients, damping orthogonal oscillations.
- **RMSprop**: Scales updates inversely with root-mean-square gradient history.
- **Adam**: Combines adaptive learning rates and momentum with exact bias correction.

### 3. Gradient Quiver Field (`plot_gradient_quiver`)
Computes analytical negative gradients $-\nabla_w J$ across the grid, rendering normalized arrows to visually confirm that gradient descent steps are always locally orthogonal to the level curves.

### 4. 3D Surface & Wireframe (`plot_loss_surface_3d`)
Projects the quadratic convex bowl into 3D space with elevation and azimuth control, tracking 3D optimization descent paths $(w_{0, t}, w_{1, t}, J(w_t))$.

### 5. Condition Number Impact on Loss Landscapes (`visualize_conditioning_impact`)
Visualizes the mathematical necessity of feature standardization:
- **Unstandardized Features ($\kappa \gg 1$):** Disparate feature variances stretch circular contours into steep, elongated ravines, forcing gradients to bounce back and forth across canyon walls.
- **Standardized Features ($\kappa \approx 1$):** Balances Hessian eigenvalues ($\lambda_{\max} \approx \lambda_{\min}$), forming an isotropic circular bowl where gradient descent flows directly to the optimum.

---

## Benchmarking Against Scikit-Learn

Evaluated on identical 80/20 train/test splits with identical feature transformations:

### California Housing ($N=20,640$)
| Metric | Scratch (Pure NumPy GD) | Scikit-Learn (OLS) | Difference |
| :--- | :--- | :--- | :--- |
| **Train MSE** | 0.524370 | 0.524368 | $+0.000002$ |
| **Test MSE** | 0.524785 | 0.524912 | $-0.000127$ |
| **Train $R^2$** | 0.605101 | 0.605102 | $-0.000001$ |
| **Test $R^2$** | **0.610143** | **0.610048** | $+0.000095$ |

### Kaggle House Prices ($N=1,460$, 163 Features)
| Metric | Scratch (Pure NumPy GD) | Scikit-Learn (OLS) | Difference |
| :--- | :--- | :--- | :--- |
| **Train MSE** | 0.012784 | 0.012100 | $+0.000684$ |
| **Test MSE** | **0.025745** | **0.030756** | **$-0.005011$** |
| **Train $R^2$** | 0.919415 | 0.923726 | $-0.004311$ |
| **Test $R^2$** | **0.841645** | **0.810825** | **$+0.030820$** |

*(Notice: The gradient descent model from scratch achieved superior test generalization on House Prices due to early stopping regularization mitigating ill-conditioned inversion noise!)*

> **Note on Benchmarks:** Differences arise from optimization tolerance, numerical precision, and early stopping. The goal is statistical parity and interpretability, not outperforming highly optimized C/Fortran backends.

---

## Engineering & Optimization Notes

### 1. Numerical Stability & Standardization
Standardization scales each feature: $z = (x - \mu) / \sigma$.
For constant columns, standard deviation is $\sigma = 0$. Our implementation protects against zero-division using safe replacement (`std_safe = np.where(std == 0, 1.0, std)`), preventing `NaN` and `Inf` generation.

### 2. Memory Scalability
- Zero-copy indexing during mini-batch generation (`X_batch = X[batch_idx]`).
- Fully vectorized inner products leverage CPU SIMD and multi-threaded OpenBLAS/MKL GEMV routines.
- 200,000 samples $\times$ 100 features in `float64` requires only ~160 MB of RAM, executing smoothly on standard consumer hardware.

### 3. Target Skewness & Gauss-Markov Compliance
In raw housing sales prices, skewness is $+1.88$. Applying `np.log1p` drops skewness to $+0.12$. This:
- Stabilizes error variance (homoscedasticity).
- Prevents extreme multimillion-dollar outliers from dominating gradient updates.
- Transforms multiplicative factors (e.g. "+10% value for extra garage") into additive linear terms.

---

## Extensions & Next Steps

1. **Ridge Regression (L2 Regularization):**
   $$J_{\text{ridge}} = \text{MSE} + \lambda \|w\|_2^2 \implies \nabla_w = \frac{2}{N} X^T e + 2\lambda w$$
2. **Lasso Regression (L1 Regularization):**
   Utilizes proximal gradient descent and soft-thresholding to induce exact feature sparsity.
3. **Polynomial Interactions:**
   Expand feature space via $\phi(x) = [x_1, x_2, x_1^2, x_1 x_2, x_2^2]$ to capture curved boundaries within a linear framework.
4. **Transition to GBDTs (XGBoost / LightGBM):**
   For tabular competition datasets with complex non-linear micro-interactions and step-function thresholds.

---

## Creator & Contact

- **Creator & Lead Maintainer:** Ranadeep Saha
- **Affiliation:** Member of Google Developer Group
- **Email:** [ranadeep2021saha@gmail.com](mailto:ranadeep2021saha@gmail.com)
- **GitHub:** [https://github.com/unknown404-practice](https://github.com/unknown404-practice)
- **LinkedIn:** [https://www.linkedin.com/in/ranadeep-saha-a03296404/](https://www.linkedin.com/in/ranadeep-saha-a03296404/)
- **Repository:** [https://github.com/unknown404-practice/linear-models-from-scratch.git](https://github.com/unknown404-practice/linear-models-from-scratch.git)

We welcome bug reports via [GitHub Issues](https://github.com/unknown404-practice/linear-models-from-scratch/issues), feature suggestions, and educational collaboration via email or LinkedIn.

---

## Citing This Project

```text
@software{linear_models_from_scratch,
  author = {Ranadeep Saha},
  title = {Linear Models from Scratch},
  year = {2026},
  url = {https://github.com/unknown404-practice/linear-models-from-scratch},
  license = {MIT}
}
```

*Use this as a reference when citing or acknowledging this project.*

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


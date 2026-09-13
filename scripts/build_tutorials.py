"""
Build 5 Step-by-Step Educational Tutorial Notebooks in docs/tutorials/
"""

import nbformat as nbf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = ROOT_DIR / "docs" / "tutorials"
TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)

PATH_SETUP_CODE = """import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Robust import setup: traverse up until 'src' directory is found
root_dir = Path.cwd().resolve()
while not (root_dir / "src").exists() and root_dir != root_dir.parent:
    root_dir = root_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
"""


def make_nb():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    }
    return nb


def add_md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def add_code(nb, code):
    nb.cells.append(nbf.v4.new_code_cell(code.strip()))


# =============================================================================
# TUTORIAL 01: LINEAR REGRESSION BASICS
# =============================================================================
def build_tutorial_01():
    nb = make_nb()
    add_md(
        nb,
        """# Tutorial 01: Linear Regression & Gradient Descent From Scratch

### Learning Objectives
1. Understand the Linear Hypothesis $\\hat{y} = Xw + b$ and the Mean Squared Error (MSE) objective.
2. Mathematically derive the exact vectorized gradients $\\nabla_w J$ and $\\nabla_b J$.
3. Train `GradientDescentLinearRegression` on synthetic 1D data and observe convergence.
4. Compare first-order Gradient Descent against the closed-form Normal Equations.
5. Solve 3 practical exercises exploring learning rates, normalization, and mini-batches.
""",
    )

    add_md(
        nb,
        """---
## 1. Mathematical Foundations

### The Linear Hypothesis
For a dataset with $N$ observations and $D$ features:
$$\\hat{y} = Xw + b \\mathbf{1}_N$$
where $X \\in \\mathbb{R}^{N \\times D}$, $w \\in \\mathbb{R}^D$, $b \\in \\mathbb{R}$, and $\\hat{y} \\in \\mathbb{R}^N$.

### Mean Squared Error (MSE)
$$J(w, b) = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2 = \\frac{1}{N} \\|\\hat{y} - y\\|_2^2$$

### Vectorized Gradients
Let residual error vector $e = \\hat{y} - y \\in \\mathbb{R}^N$:
$$\\nabla_w J = \\frac{2}{N} X^T e = \\frac{2}{N} X^T (\\hat{y} - y)$$
$$\\nabla_b J = \\frac{2}{N} \\sum_{i=1}^N (\\hat{y}_i - y_i) = 2 \\cdot \\text{mean}(\\hat{y} - y)$$

### Parameter Updates
$$w^{(t+1)} = w^{(t)} - \\eta \\nabla_w J, \\quad b^{(t+1)} = b^{(t)} - \\eta \\nabla_b J$$
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.linear_regression import GradientDescentLinearRegression, compute_mse, compute_r2
from src.solvers import ClosedFormLinearRegression
""",
    )

    add_md(
        nb,
        """## 2. Generating Synthetic 1D Data
We generate $N=100$ samples with ground truth slope $w = 3.5$, intercept $b = 1.2$, and Gaussian noise $\\epsilon \\sim \\mathcal{N}(0, 0.5^2)$.
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 100
X_1d = np.random.uniform(-3, 3, size=(N, 1))
true_w = 3.5
true_b = 1.2
noise = np.random.normal(0, 0.5, size=N)
y_1d = (X_1d[:, 0] * true_w) + true_b + noise

print(f"Dataset generated: N={N}, X shape={X_1d.shape}, y shape={y_1d.shape}")
print(f"Ground Truth: w={true_w}, b={true_b}")
""",
    )

    add_md(
        nb,
        """## 3. Training Gradient Descent & Tracking Loss
We instantiate `GradientDescentLinearRegression` with learning rate $\\eta = 0.05$ and train for $200$ epochs.
""",
    )

    add_code(
        nb,
        """gd_model = GradientDescentLinearRegression(lr=0.05, n_iters=200, optimizer="sgd")
gd_model.fit(X_1d, y_1d)

y_pred_gd = gd_model.predict(X_1d)
r2_gd = gd_model.score(X_1d, y_1d)
mse_gd = compute_mse(y_1d, y_pred_gd)

print(f"Fitted Parameters: w={gd_model.weights[0]:.4f}, b={gd_model.bias:.4f}")
print(f"Metrics: R2={r2_gd:.4f}, MSE={mse_gd:.4f}")
print(f"Epochs trained: {len(gd_model.loss_history_)}")
""",
    )

    add_md(
        nb,
        """## 4. Comparing GD vs. Analytical Closed-Form
We fit `ClosedFormLinearRegression(method="normal")` on the exact same data to verify analytical equivalence.
""",
    )

    add_code(
        nb,
        """cf_model = ClosedFormLinearRegression(method="normal_equation")
cf_model.fit(X_1d, y_1d)

print(f"Closed-Form Normal Equations: w={cf_model.weights[0]:.4f}, b={cf_model.bias:.4f}")
print(f"Weight difference (GD vs CF): {abs(gd_model.weights[0] - cf_model.weights[0]):.6f}")
print(f"Bias difference (GD vs CF):   {abs(gd_model.bias - cf_model.bias):.6f}")
""",
    )

    add_md(
        nb,
        """## 5. Visualizing the Fitted Line & Loss Convergence
""",
    )

    add_code(
        nb,
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Plot 1: Fitted Line
x_line = np.linspace(-3.2, 3.2, 100).reshape(-1, 1)
y_line_gd = gd_model.predict(x_line)

axes[0].scatter(X_1d, y_1d, alpha=0.7, color="tab:blue", label="Data Points")
axes[0].plot(x_line, y_line_gd, color="tab:red", lw=2.5, label=f"GD Fit: y = {gd_model.weights[0]:.2f}x + {gd_model.bias:.2f}")
axes[0].plot(x_line, x_line[:, 0] * true_w + true_b, color="black", linestyle="--", lw=1.5, label="Ground Truth")
axes[0].set_title("1D Linear Regression Fit", fontweight="bold")
axes[0].set_xlabel("Feature x")
axes[0].set_ylabel("Target y")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss Convergence Curve
axes[1].plot(gd_model.loss_history_, color="tab:purple", lw=2)
axes[1].set_title("MSE Cost vs Epoch", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
    )

    add_md(
        nb,
        """---
## Exercises

### Exercise 1: Exploring Learning Rate Sensitivity
Try learning rates $\\eta \\in [0.001, 0.05, 1.2]$. What happens to the loss curve when $\\eta$ is too small vs too large?

<details>
<summary><b>Click to View Solution</b></summary>

```python
for test_lr in [0.001, 0.05, 0.9]:
    m = GradientDescentLinearRegression(lr=test_lr, n_iters=100)
    m.fit(X_1d, y_1d)
    print(f"lr={test_lr:<6} -> Final Loss: {m.loss_history_[-1]:.4f} | w: {m.weights[0]:.3f}")
# lr=0.001 converges very slowly; lr=0.05 converges smoothly; lr >= 1.0 oscillates or diverges!
```
</details>

### Exercise 2: Adding Multiple Input Features
Generate 2D synthetic data $y = 1.5 x_1 - 2.0 x_2 + 0.5 + \\epsilon$ and fit `GradientDescentLinearRegression`. Verify that recovered weights match.

<details>
<summary><b>Click to View Solution</b></summary>

```python
X_2d = np.random.randn(200, 2)
y_2d = 1.5 * X_2d[:, 0] - 2.0 * X_2d[:, 1] + 0.5 + np.random.normal(0, 0.1, 200)
m_2d = GradientDescentLinearRegression(lr=0.1, n_iters=200).fit(X_2d, y_2d)
print("Recovered weights:", np.round(m_2d.weights, 3), "Bias:", round(m_2d.bias, 3))
```
</details>
""",
    )
    with open(
        TUTORIALS_DIR / "tutorial_01_linear_regression_basics.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)
    print("Created tutorial_01_linear_regression_basics.ipynb")


# =============================================================================
# TUTORIAL 02: REGULARIZATION AND BIAS-VARIANCE
# =============================================================================
def build_tutorial_02():
    nb = make_nb()
    add_md(
        nb,
        """# Tutorial 02: Regularization (Ridge, Lasso, ElasticNet) & Bias-Variance Tradeoff

### Learning Objectives
1. Understand why ill-conditioned or collinear features cause exploding weights.
2. Master L2 (Ridge) shrinkage, L1 (Lasso) soft-thresholding, and ElasticNet compromise.
3. Compare coordinate descent vs. gradient descent for sparse feature selection.
4. Plot the Bias-Variance tradeoff curve across regularization strength $\\alpha$.
""",
    )

    add_md(
        nb,
        """---
## 1. Mathematical Formulations

### Ridge Regression (L2 Penalty)
$$J(w) = \\frac{1}{N} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_2^2$$
Shrinks all coefficients continuously toward zero without zeroing any out.

### Lasso Regression (L1 Penalty via Coordinate Descent)
$$J(w) = \\frac{1}{2N} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_1$$
Cyclic updates via the Soft-Thresholding operator $S(v, \\gamma) = \\text{sign}(v) \\max(|v| - \\gamma, 0)$:
$$w_j \\leftarrow \\frac{S\\left(\\frac{1}{N} X_{:, j}^T r^{(-j)}, \\; \\alpha\\right)}{\\frac{1}{N} \\|X_{:, j}\\|_2^2}$$
Produces exact mathematical zeros ($w_j = 0.0$), performing automatic feature selection!

### ElasticNet Regression
$$J(w) = \\frac{1}{2N} \\|y - Xw\\|_2^2 + \\alpha \\rho \\|w\\|_1 + \\frac{1}{2} \\alpha (1 - \\rho) \\|w\\|_2^2$$
Combines Lasso sparsity with Ridge grouping for correlated feature blocks.
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.linear_regression import GradientDescentLinearRegression, compute_mse, compute_r2
""",
    )

    add_md(
        nb,
        """## 2. Generating Multicollinear High-Dimensional Data
We generate $N=150$ samples with $D=15$ features, where only 4 features are truly predictive, and 2 pairs are highly correlated ($r \\approx 0.98$).
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N, D = 150, 15
X_raw = np.random.randn(N, D)
# Inject severe collinearity: column 1 ~ column 0, column 3 ~ column 2
X_raw[:, 1] = X_raw[:, 0] * 0.98 + np.random.normal(0, 0.05, N)
X_raw[:, 3] = X_raw[:, 2] * 0.99 + np.random.normal(0, 0.02, N)

# True weights: only 4 non-zero
w_true = np.zeros(D)
w_true[0] = 3.0
w_true[2] = -2.5
w_true[4] = 1.8
w_true[6] = 4.0

y = X_raw @ w_true + 1.0 + np.random.normal(0, 0.5, N)

# Train/test split
X_train, X_test = X_raw[:100], X_raw[100:]
y_train, y_test = y[:100], y[100:]
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"True non-zero features: indices [0, 2, 4, 6]")
""",
    )

    add_md(
        nb,
        """## 3. Fitting OLS vs. Ridge vs. Lasso vs. ElasticNet
""",
    )

    add_code(
        nb,
        """models = {
    "OLS (No Reg)": GradientDescentLinearRegression(penalty="none", lr=0.05, n_iters=400, optimizer="adam"),
    "Ridge (L2)":   GradientDescentLinearRegression(penalty="l2", alpha=1.0, lr=0.05, n_iters=400, optimizer="adam"),
    "Lasso (L1)":   GradientDescentLinearRegression(penalty="l1", alpha=0.05, lr=0.05, n_iters=400, optimizer="adam"),
    "ElasticNet":   GradientDescentLinearRegression(penalty="elasticnet", alpha=0.05, l1_ratio=0.5, lr=0.05, n_iters=400, optimizer="adam")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    zeros = np.sum(np.abs(model.weights) < 1e-4)
    print(f"{name:<15} -> Test R2: {test_r2:.4f} | Zero weights: {zeros:>2}/{D} | Weight L2 norm: {np.linalg.norm(model.weights):.2f}")
""",
    )

    add_md(
        nb,
        """## 4. Visualizing Sparsity & Weight Shrinkage
""",
    )

    add_code(
        nb,
        """fig, ax = plt.subplots(figsize=(11, 4.5))
x_idx = np.arange(D)
width = 0.2

ax.bar(x_idx - 1.5*width, models["OLS (No Reg)"].weights, width=width, label="OLS (No Reg)", alpha=0.8)
ax.bar(x_idx - 0.5*width, models["Ridge (L2)"].weights, width=width, label="Ridge (L2)", alpha=0.8)
ax.bar(x_idx + 0.5*width, models["Lasso (L1)"].weights, width=width, label="Lasso (L1)", alpha=0.8)
ax.bar(x_idx + 1.5*width, w_true, width=width, label="Ground Truth", color="black", alpha=0.5)

ax.set_xticks(x_idx)
ax.set_xticklabels([f"x{i}" for i in range(D)])
ax.set_title("Coefficient Comparison Across Regularizers", fontweight="bold")
ax.set_ylabel("Weight Value")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )

    add_md(
        nb,
        """---
## Exercises

### Exercise 1: Tuning Regularization Strength $\\alpha$
Iterate $\\alpha \\in [10^{-3}, 10^2]$ for Lasso and plot the number of non-zero features vs $\\alpha$.

<details>
<summary><b>Click to View Solution</b></summary>

```python
alphas = np.logspace(-3, 1, 15)
non_zeros = []
for a in alphas:
    m = GradientDescentLinearRegression(penalty="l1", alpha=a, lr=0.05, n_iters=300, optimizer="adam").fit(X_train, y_train)
    non_zeros.append(np.sum(np.abs(m.weights) > 1e-3))

plt.figure(figsize=(6, 3))
plt.semilogx(alphas, non_zeros, marker='o')
plt.title("Lasso Sparsity vs Alpha")
plt.xlabel("Alpha")
plt.ylabel("Number of Non-Zero Features")
plt.grid(True, alpha=0.3)
plt.show()
```
</details>
""",
    )
    with open(
        TUTORIALS_DIR / "tutorial_02_regularization_and_bias_variance.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)
    print("Created tutorial_02_regularization_and_bias_variance.ipynb")


# =============================================================================
# TUTORIAL 03: OPTIMIZERS AND SCHEDULERS
# =============================================================================
def build_tutorial_03():
    nb = make_nb()
    add_md(
        nb,
        """# Tutorial 03: First-Order Optimizers (Momentum, RMSprop, Adam) & Schedulers

### Learning Objectives
1. Understand why vanilla SGD oscillates along ill-conditioned loss ravines.
2. Master Exponential Moving Averages of gradients (Momentum) and squared gradients (RMSprop).
3. Implement Adam's bias correction mechanism.
4. Compare Cosine Annealing vs. Exponential learning rate schedules.
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.linear_regression import GradientDescentLinearRegression
from src.schedulers import ConstantLR, StepLR, ExponentialLR, CosineAnnealingLR
""",
    )

    add_md(
        nb,
        """## 1. Comparing Optimizers on Ill-Conditioned Surfaces
We generate synthetic data with disparate feature variances (condition number $\\kappa > 100$).
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 500
# Feature 0 has scale 10.0, Feature 1 has scale 0.1 (eccentric quadratic bowl)
X_ill = np.column_stack([
    np.random.normal(0, 10.0, N),
    np.random.normal(0, 0.1, N),
    np.random.normal(0, 1.0, N)
])
y_ill = 2.0 * X_ill[:, 0] + 5.0 * X_ill[:, 1] - 3.0 * X_ill[:, 2] + np.random.normal(0, 0.5, N)

opts = ["sgd", "momentum", "rmsprop", "adam"]
trained_models = {}

for opt in opts:
    m = GradientDescentLinearRegression(lr=0.01, n_iters=250, optimizer=opt, seed=42)
    m.fit(X_ill, y_ill)
    trained_models[opt] = m
    print(f"Optimizer {opt.upper():<10} -> Final MSE: {m.loss_history_[-1]:.4f}")
""",
    )

    add_md(
        nb,
        """## 2. Convergence Trajectories Plot
""",
    )

    add_code(
        nb,
        """plt.figure(figsize=(9, 4.5))
for opt, m in trained_models.items():
    plt.plot(m.loss_history_, lw=2, label=opt.upper())
plt.yscale("log")
plt.title("Loss Convergence Comparison on Ill-Conditioned Problem", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )

    add_md(
        nb,
        """## 3. Learning Rate Schedulers
We compare Constant learning rate against Cosine Annealing:
$$\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\frac{\\pi t}{T_{\\max}}\\right)\\right)$$
""",
    )

    add_code(
        nb,
        """sched_const = ConstantLR(initial_lr=0.05)
sched_cosine = CosineAnnealingLR(initial_lr=0.05, T_max=250, eta_min=1e-4)

m_const = GradientDescentLinearRegression(lr=0.05, n_iters=250, scheduler=sched_const, optimizer="adam")
m_cosine = GradientDescentLinearRegression(lr=0.05, n_iters=250, scheduler=sched_cosine, optimizer="adam")

m_const.fit(X_ill, y_ill)
m_cosine.fit(X_ill, y_ill)

print(f"Constant LR Final Loss:        {m_const.loss_history_[-1]:.5f}")
print(f"Cosine Annealing Final Loss:  {m_cosine.loss_history_[-1]:.5f}")
""",
    )
    with open(
        TUTORIALS_DIR / "tutorial_03_optimizers_and_schedulers.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)
    print("Created tutorial_03_optimizers_and_schedulers.ipynb")


# =============================================================================
# TUTORIAL 04: UNCERTAINTY AND CONFORMAL PREDICTION
# =============================================================================
def build_tutorial_04():
    nb = make_nb()
    add_md(
        nb,
        """# Tutorial 04: Probabilistic Bayesian Regression & Conformal Safety

### Learning Objectives
1. Understand the difference between Aleatoric noise and Epistemic model ignorance.
2. Derive the closed-form Gaussian posterior $p(w \\mid X, y) = \\mathcal{N}(m_N, S_N)$.
3. Sample credible regression parameter hypotheses via Cholesky decomposition.
4. Guarantee distribution-free finite-sample coverage via Split Conformal Prediction:
   $$P(y \\in [\\hat{y} - \\hat{q}, \\hat{y} + \\hat{q}]) \\ge 1 - \\alpha$$
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.bayesian import BayesianLinearRegression
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression
""",
    )

    add_md(
        nb,
        """## 1. Fitting Bayesian Linear Regression & Epistemic Decomposition
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 100
X = np.random.uniform(-2, 2, size=(N, 1))
y = 2.0 * X[:, 0] + 0.5 + np.random.normal(0, 0.2, size=N)

bayes = BayesianLinearRegression(alpha=1.0, beta=25.0)
bayes.fit(X, y)

# Query range including out-of-distribution extrapolation (x in [-4, 4])
X_query = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
y_mean, y_std = bayes.predict(X_query, return_std=True)

print(f"Posterior Mean: w={bayes.weights[0]:.4f}, b={bayes.bias:.4f}")
print(f"In-distribution std (at x=0):  {bayes.predict(np.array([[0.0]]), return_std=True)[1][0]:.4f}")
print(f"Out-of-distribution std (x=4): {bayes.predict(np.array([[4.0]]), return_std=True)[1][0]:.4f}")
""",
    )

    add_md(
        nb,
        """## 2. Visualizing Bayesian Credible Bands & Posterior Weight Sampling
""",
    )

    add_code(
        nb,
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Plot 1: Predictive Mean + 2-Sigma Uncertainty Envelope
axes[0].scatter(X, y, color="tab:blue", alpha=0.7, label="Training Data ([-2, 2])")
axes[0].plot(X_query, y_mean, color="tab:red", lw=2, label="Posterior Mean")
axes[0].fill_between(
    X_query[:, 0],
    y_mean - 2 * y_std,
    y_mean + 2 * y_std,
    color="tab:red",
    alpha=0.2,
    label="±2σ Predictive Band"
)
axes[0].set_title("Bayesian Uncertainty Expansion (OOD Awareness)", fontweight="bold")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Sampled Hypothesis Lines from Posterior
w_samples = bayes.sample_weights(n_samples=20, random_state=42)
for i in range(20):
    axes[1].plot(X_query, X_query @ w_samples[i] + bayes.bias, color="tab:purple", alpha=0.35)

axes[1].scatter(X, y, color="tab:blue", alpha=0.7, zorder=5)
axes[1].set_title("Posterior Weight Samples (Hypothesis Bundle)", fontweight="bold")
axes[1].set_xlabel("x")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
    )

    add_md(
        nb,
        """## 3. Conformal Prediction with Guaranteed Finite-Sample Coverage
Conformal prediction calibration produces exact coverage bounds without assuming Gaussianity.
""",
    )

    add_code(
        nb,
        """base_est = ClosedFormLinearRegression(method="svd")
conformal = ConformalLinearRegression(estimator=base_est, alpha=0.10, cal_size=0.25, random_state=42)
conformal.fit(X, y)

# Generate test split
X_test = np.random.uniform(-2, 2, size=(200, 1))
y_test = 2.0 * X_test[:, 0] + 0.5 + np.random.normal(0, 0.2, size=200)

cov_score = conformal.score_coverage(X_test, y_test)
print(f"Nominal Confidence Level: 90.00%")
print(f"Calibrated Cutoff q_hat:    {conformal.q_hat_:.4f}")
print(f"Empirical Test Coverage:   {cov_score * 100:.2f}%")
""",
    )
    with open(
        TUTORIALS_DIR / "tutorial_04_uncertainty_and_conformal_prediction.ipynb",
        "w",
        encoding="utf-8",
    ) as f:
        nbf.write(nb, f)
    print("Created tutorial_04_uncertainty_and_conformal_prediction.ipynb")


# =============================================================================
# TUTORIAL 05: STREAMING AND QUANTILE REGRESSION
# =============================================================================
def build_tutorial_05():
    nb = make_nb()
    add_md(
        nb,
        """# Tutorial 05: Streaming Recursive Least Squares & Quantile Regression

### Learning Objectives
1. Master online updating via Recursive Least Squares (RLS) with $\\mathcal{O}(D^2)$ time/memory.
2. Track non-stationary concept drift using the exponential forgetting factor $\\lambda$.
3. Formulate the asymmetric pinball loss for Quantile Regression:
   $$\\rho_\\tau(u) = u (\\tau - \\mathbb{I}(u < 0))$$
4. Build asymmetric risk ribbons (10th percentile floor, median, 90th percentile ceiling).
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.streaming import RecursiveLeastSquares
from src.quantile import QuantileRegressorScratch
""",
    )

    add_md(
        nb,
        """## 1. Online Learning with RLS under Concept Drift
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
y = np.zeros(N)

# Dynamic regime shift at t = 500
for t in range(N):
    if t < 500:
        y[t] = 2.0 * X[t, 0] - 1.5 * X[t, 1] + 0.5 + np.random.normal(0, 0.1)
    else:
        y[t] = -1.0 * X[t, 0] + 3.0 * X[t, 1] - 0.5 + np.random.normal(0, 0.1)

rls = RecursiveLeastSquares(lambda_=0.98)
w0_history = []

for t in range(N):
    rls.partial_fit(X[t], y[t])
    w0_history.append(float(rls.weights[0]))

print(f"RLS adapted weights after drift: w0={rls.weights[0]:.4f} (ground truth: -1.0), w1={rls.weights[1]:.4f} (ground truth: 3.0)")
""",
    )

    add_md(
        nb,
        """## 2. Quantile Regression on Heteroscedastic Data
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 300
x_fan = np.random.uniform(0.5, 5.0, N)
# Error variance expands with x (heteroscedastic fan)
noise_fan = np.random.normal(0, 0.3 * x_fan, N)
y_fan = 2.0 * x_fan + noise_fan
X_fan = x_fan.reshape(-1, 1)

# Fit 10th percentile, Median (50th), and 90th percentile
q10 = QuantileRegressorScratch(quantile=0.10).fit(X_fan, y_fan)
q50 = QuantileRegressorScratch(quantile=0.50).fit(X_fan, y_fan)
q90 = QuantileRegressorScratch(quantile=0.90).fit(X_fan, y_fan)

x_grid = np.linspace(0.5, 5.0, 100).reshape(-1, 1)
pred_10 = q10.predict(x_grid)
pred_50 = q50.predict(x_grid)
pred_90 = q90.predict(x_grid)

# Plotting the asymmetric risk corridor
plt.figure(figsize=(9, 4.5))
plt.scatter(x_fan, y_fan, color="tab:blue", alpha=0.6, label="Heteroscedastic Observations")
plt.plot(x_grid, pred_50, color="black", lw=2, label="Median Fit (τ = 0.50)")
plt.fill_between(x_grid[:, 0], pred_10, pred_90, color="tab:orange", alpha=0.3, label="80% Quantile Corridor [τ=0.10, τ=0.90]")
plt.title("Quantile Regression: Asymmetric Risk Corridor on Heteroscedastic Data", fontweight="bold")
plt.xlabel("Feature x")
plt.ylabel("Target y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    with open(
        TUTORIALS_DIR / "tutorial_05_streaming_and_quantile_regression.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)
    print("Created tutorial_05_streaming_and_quantile_regression.ipynb")


def main():
    print("Building 5 step-by-step tutorials...")
    build_tutorial_01()
    build_tutorial_02()
    build_tutorial_03()
    build_tutorial_04()
    build_tutorial_05()
    print("All 5 tutorials built successfully!")


if __name__ == "__main__":
    main()

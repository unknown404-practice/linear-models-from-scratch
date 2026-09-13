"""
Build Companion Notebooks:
1. notebooks/02_optimizers_and_geometry.ipynb
2. notebooks/03_uncertainty_and_risk.ipynb
"""

import nbformat as nbf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

PATH_SETUP_CODE = """import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
# NOTEBOOK 02: OPTIMIZERS AND GEOMETRY
# =============================================================================
def build_notebook_02():
    nb = make_nb()
    add_md(
        nb,
        """# Companion Masterclass: Loss Surface Geometry & Optimization Dynamics

### An In-Depth Study of Hessian Curvature, Condition Numbers, and First- vs. Second-Order Solvers

**Audience:** ML engineers, researchers, and competitive data scientists  
**Topics:**
- Quadratic loss geometry & analytical Hessian $H = \\frac{2}{N} X^T X$
- Condition number $\\kappa = \\lambda_{\\max} / \\lambda_{\\min}$ and canyon stretching
- Comparative trajectories: SGD, Momentum, RMSprop, Adam, and Newton-Raphson
- Theoretical convergence bounds: $\\eta < \\frac{2}{\\lambda_{\\max}}$
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.visualization import (
    compute_loss_grid,
    plot_loss_contours,
    plot_loss_surface_3d,
    plot_gradient_quiver,
    compare_optimizer_trajectories,
    visualize_conditioning_impact
)
from src.solvers import ClosedFormLinearRegression, NewtonLinearRegression
from src.linear_regression import GradientDescentLinearRegression
""",
    )

    add_md(
        nb,
        """---
## 1. The Condition Number & Loss Canyon Curvature

For linear regression with MSE cost, the loss function is strictly quadratic:
$$J(w) = \\frac{1}{N} (Xw - y)^T (Xw - y)$$
The Hessian matrix is constant everywhere:
$$H = \\nabla^2 J(w) = \\frac{2}{N} X^T X$$

The **Condition Number** $\\kappa(H)$ measures the eccentricity of the quadratic bowl:
$$\\kappa(H) = \\frac{\\lambda_{\\max}(H)}{\\lambda_{\\min}(H)}$$
- $\\kappa \\approx 1$: Perfectly circular, isotropic loss contours. Gradient vectors point directly toward the global minimum.
- $\\kappa \\gg 1$: Highly stretched elliptical canyons. Gradient vectors point almost perpendicular to the optimal descent path, causing vanilla SGD to oscillate violently!
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 200
X_raw = np.column_stack([np.random.normal(0, 15.0, N), np.random.normal(0, 0.2, N)])
y_raw = 2.0 * X_raw[:, 0] + 5.0 * X_raw[:, 1] + np.random.normal(0, 0.5, N)
fig, axes = visualize_conditioning_impact(X_raw, y_raw)
plt.show()
""",
    )

    add_md(
        nb,
        """## 2. Multi-Optimizer Trajectory Showdown

We project the descent paths of **SGD, Momentum, RMSprop, Adam, and Newton-Raphson** on the exact same 2D loss surface.
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 100
# Generate moderately ill-conditioned 2D data (kappa ~ 15)
X_demo = np.column_stack([
    np.random.normal(0, 3.0, N),
    np.random.normal(0, 0.8, N)
])
y_demo = 1.5 * X_demo[:, 0] - 2.0 * X_demo[:, 1] + np.random.normal(0, 0.2, N)

trajectories = compare_optimizer_trajectories(
    X=X_demo,
    y=y_demo,
    optimizers=["sgd", "momentum", "rmsprop", "adam"],
    learning_rate=0.08,
    n_epochs=60,
)
fig, ax = plot_loss_contours(
    X_sub=X_demo,
    y=y_demo,
    w0_range=(-0.5, 2.5),
    w1_range=(-3.0, 1.0),
    trajectories=trajectories,
    title="Multi-Optimizer Convergence on Ill-Conditioned Loss Surface"
)
plt.show()
""",
    )

    add_md(
        nb,
        """## 3. Second-Order Newton-Raphson: 1-Step Quadratic Exactness

Because MSE is quadratic, a single Newton step jumps directly to the global optimum:
$$\\Delta w = - H^{-1} \\nabla_w J = - \\left( \\frac{2}{N} X^T X \\right)^{-1} \\left( \\frac{2}{N} X^T (Xw^{(0)} - y) \\right) = (X^T X)^{-1} X^T y - w^{(0)}$$
$$w^{(1)} = w^{(0)} + \\Delta w = (X^T X)^{-1} X^T y$$
""",
    )

    add_code(
        nb,
        """newton = NewtonLinearRegression()
newton.fit(X_demo, y_demo)

print(f"Newton-Raphson converged in: {newton.n_iter_} step(s)!")
print(f"Final gradient norm: {newton.grad_norm_:.8f}")
print(f"Recovered Weights: {np.round(newton.weights, 4)}, Bias: {newton.bias:.4f}")
""",
    )
    with open(NOTEBOOKS_DIR / "02_optimizers_and_geometry.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/02_optimizers_and_geometry.ipynb")


# =============================================================================
# NOTEBOOK 03: UNCERTAINTY AND RISK
# =============================================================================
def build_notebook_03():
    nb = make_nb()
    add_md(
        nb,
        """# Companion Masterclass: Uncertainty Quantification, Conformal Safety & Risk Modeling

### From Point Predictions to Calibrated Decision Bounds

**Topics:**
1. Limits of Ordinary Least Squares point estimates.
2. Bayesian Linear Regression: exact conjugate Gaussian posterior ($m_N, S_N$).
3. Epistemic vs. Aleatoric uncertainty decomposition.
4. Out-of-Distribution (OOD) detection.
5. Distribution-free Conformal Prediction with non-asymptotic coverage guarantees.
6. Quantile Regression via smoothed IRLS for asymmetric downside risk corridors.
""",
    )

    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.bayesian import BayesianLinearRegression
from src.conformal import ConformalLinearRegression
from src.quantile import QuantileRegressorScratch
from src.solvers import ClosedFormLinearRegression
""",
    )

    add_md(
        nb,
        """---
## 1. Epistemic vs. Aleatoric Uncertainty in Bayesian Regression

For query point $x_*$:
$$p(y_* \\mid x_*, X, y) = \\mathcal{N}(\\mu_*(x_*), \\sigma_*^2(x_*))$$
$$\\sigma_*^2(x_*) = \\underbrace{\\frac{1}{\\beta}}_{\\text{Aleatoric Noise}} + \\underbrace{x_*^T S_N x_*}_{\\text{Epistemic Uncertainty}}$$

- **Aleatoric Noise ($1/\\beta$):** Irreducible variance in observations.
- **Epistemic Uncertainty ($x_*^T S_N x_*$):** Model ignorance. Vanishes near training data clusters, but explodes in out-of-distribution space!
""",
    )

    add_code(
        nb,
        """np.random.seed(42)
N = 100
X_train = np.random.uniform(-1.5, 1.5, (N, 1))
y_train = 2.5 * X_train[:, 0] + 0.8 + np.random.normal(0, 0.2, N)

bayes = BayesianLinearRegression(alpha=1.0, beta=25.0)
bayes.fit(X_train, y_train)

# Query grid spanning familiar inliers ([-1.5, 1.5]) and severe OOD extrapolation ([-4, 4])
X_eval = np.linspace(-4, 4, 300).reshape(-1, 1)
y_mean, y_std = bayes.predict(X_eval, return_std=True)

plt.figure(figsize=(10, 4.5))
plt.scatter(X_train, y_train, color="tab:blue", alpha=0.7, label="Training Observations (x in [-1.5, 1.5])")
plt.plot(X_eval, y_mean, color="tab:red", lw=2, label="Posterior Mean")
plt.fill_between(X_eval[:, 0], y_mean - 2*y_std, y_mean + 2*y_std, color="tab:red", alpha=0.2, label="±2σ Predictive Band")
plt.title("Bayesian Epistemic Uncertainty Expansion in OOD Territory", fontweight="bold")
plt.xlabel("Feature x")
plt.ylabel("Target y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )

    add_md(
        nb,
        """## 2. Conformal Prediction: Distribution-Free Coverage Guarantees

Unlike Bayesian methods, **Split Conformal Prediction** guarantees coverage:
$$P(y_{\\text{test}} \\in [\\hat{\\mu}(x) - \\hat{q}, \\hat{\\mu}(x) + \\hat{q}]) \\ge 1 - \\alpha$$
under ANY arbitrary noise distribution without normality assumptions.
""",
    )

    add_code(
        nb,
        """# Heavy-tailed Laplace noise
y_laplace = 2.5 * X_train[:, 0] + 0.8 + np.random.laplace(0, 0.3, N)

conformal = ConformalLinearRegression(
    estimator=ClosedFormLinearRegression(method="svd"),
    alpha=0.05,  # 95% nominal confidence
    cal_size=0.25,
    random_state=42
)
conformal.fit(X_train, y_laplace)

# Test evaluation
X_test = np.random.uniform(-1.5, 1.5, (200, 1))
y_test = 2.5 * X_test[:, 0] + 0.8 + np.random.laplace(0, 0.3, 200)

cov = conformal.score_coverage(X_test, y_test)
print(f"Nominal Confidence Level: 95.00%")
print(f"Empirical Coverage Score: {cov * 100:.2f}% (>= 95.0% guaranteed)")
print(f"Calibrated nonconformity cutoff q_hat: {conformal.q_hat_:.4f}")
""",
    )

    add_md(
        nb,
        """## 3. Quantile Regression: Modeling Conditional Percentiles & Asymmetric Risk

Ordinary least squares models only the mean $\\mathbb{E}[y \\mid X]$. For financial risk and Value-at-Risk (VaR), we model conditional quantiles:
$$\\rho_\\tau(u) = u (\\tau - \\mathbb{I}(u < 0))$$
""",
    )

    add_code(
        nb,
        """# Heteroscedastic fan data
x_fan = np.random.uniform(1.0, 6.0, 300)
y_fan = 3.0 * x_fan + np.random.normal(0, 0.5 * x_fan, 300)
X_fan = x_fan.reshape(-1, 1)

q_floor = QuantileRegressorScratch(quantile=0.05).fit(X_fan, y_fan)
q_median = QuantileRegressorScratch(quantile=0.50).fit(X_fan, y_fan)
q_ceil = QuantileRegressorScratch(quantile=0.95).fit(X_fan, y_fan)

x_plot = np.linspace(1.0, 6.0, 100).reshape(-1, 1)
pred_floor = q_floor.predict(x_plot)
pred_median = q_median.predict(x_plot)
pred_ceil = q_ceil.predict(x_plot)

plt.figure(figsize=(10, 4.5))
plt.scatter(x_fan, y_fan, color="tab:blue", alpha=0.5, label="Heteroscedastic Data")
plt.plot(x_plot, pred_median, color="black", lw=2, label="Median LAD (τ=0.50)")
plt.fill_between(x_plot[:, 0], pred_floor, pred_ceil, color="tab:green", alpha=0.25, label="90% Quantile Corridor [τ=0.05, τ=0.95]")
plt.title("Asymmetric Quantile Risk Corridor on Expanding Variance", fontweight="bold")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    with open(NOTEBOOKS_DIR / "03_uncertainty_and_risk.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/03_uncertainty_and_risk.ipynb")


def main():
    print("Building companion notebooks...")
    build_notebook_02()
    build_notebook_03()
    print("Companion notebooks built successfully!")


if __name__ == "__main__":
    main()

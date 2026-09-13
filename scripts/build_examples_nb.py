"""
Build 8 Visual Example Notebooks in docs/examples/,
2 Mini-Project Notebooks in examples/, and 1 Competition Baseline in competitions/.
All notebooks feature pure-NumPy model workflows with clear inline Matplotlib visualizations.
"""

import nbformat as nbf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_EX_DIR = ROOT_DIR / "docs" / "examples"
EXAMPLES_DIR = ROOT_DIR / "examples"
COMPETITIONS_DIR = ROOT_DIR / "competitions"

for d in [DOCS_EX_DIR, EXAMPLES_DIR, COMPETITIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
# 1. docs/examples/example_house_prices_baseline.ipynb
# =============================================================================
def build_ex_house_prices():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Kaggle Ames House Prices Tabular Baseline

This example demonstrates how to build a leak-free tabular regression baseline for Kaggle House Prices using `ColumnTransformerScratch` and `ClosedFormLinearRegression`, accompanied by a diagnostic prediction fit plot.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.pipeline import PipelineScratch, ColumnTransformerScratch, StandardScalerScratch, SimpleImputerScratch, OneHotEncoderScratch
from src.solvers import ClosedFormLinearRegression
from src.linear_regression import compute_r2, compute_mse

# Load real dataset
df = pd.read_csv(root_dir / "data" / "house_prices" / "train.csv")
num_cols = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]
cat_cols = ["Neighborhood"]

X = df[num_cols + cat_cols]
y = np.log1p(df["SalePrice"].values)

# Pipeline
preprocessor = ColumnTransformerScratch([
    ("num", PipelineScratch([("imp", SimpleImputerScratch(strategy="median")), ("scaler", StandardScalerScratch())]), num_cols),
    ("cat", PipelineScratch([("imp", SimpleImputerScratch(strategy="most_frequent")), ("ohe", OneHotEncoderScratch(handle_unknown="ignore"))]), cat_cols)
])

pipe = PipelineScratch([("prep", preprocessor), ("reg", ClosedFormLinearRegression(method="svd"))])
pipe.fit(X, y)
y_pred = pipe.predict(X)

r2 = compute_r2(y, y_pred)
rmse = np.sqrt(compute_mse(y, y_pred))
print(f"Fitted Baseline R2: {r2:.4f}")
print(f"Fitted RMSE (log scale): {rmse:.4f}")

# Diagnostic scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(y, y_pred, alpha=0.5, edgecolors='none', color='#2563eb', label='Trained Houses')
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y = x)')
plt.xlabel("Actual Log SalePrice")
plt.ylabel("Predicted Log SalePrice")
plt.title(f"Ames Housing Baseline Pipeline (R² = {r2:.3f}, RMSE = {rmse:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- How to handle mixed numeric and categorical data using pure-NumPy `ColumnTransformerScratch`.
- Log-transforming skewed targets for stabilized variance and linear relationships.
- Inspecting prediction accuracy visually against the identity reference line.
""",
    )
    with open(DOCS_EX_DIR / "example_house_prices_baseline.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 2. docs/examples/example_california_housing_pipeline.ipynb
# =============================================================================
def build_ex_california_housing():
    nb = make_nb()
    add_md(
        nb,
        """# Example: California Housing End-to-End Pipeline

Demonstrates scaling, degree-2 polynomial expansion, and Ridge regression on California Housing with a visual prediction alignment plot.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.pipeline import PipelineScratch, StandardScalerScratch
from src.features import PolynomialFeaturesScratch
from src.linear_regression import GradientDescentLinearRegression, compute_r2

df = pd.read_csv(root_dir / "data" / "california_housing" / "california_housing.csv")
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
X = df[features].values[:2000]
y = df["MedHouseVal"].values[:2000]

pipe = PipelineScratch([
    ("scaler", StandardScalerScratch()),
    ("poly", PolynomialFeaturesScratch(degree=2, interaction_only=True)),
    ("ridge", GradientDescentLinearRegression(penalty="l2", alpha=0.5, lr=0.02, n_iters=300, optimizer="adam"))
])

pipe.fit(X, y)
r2_val = pipe.score(X, y)
print(f"Pipeline R2 Score on 2,000 California Tracts: {r2_val:.4f}")

y_pred = pipe.predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(y, y_pred, alpha=0.4, color='#059669', edgecolors='none', label='Census Tracts')
plt.plot([0, 5], [0, 5], 'r--', lw=2, label='Identity Line')
plt.xlabel("Actual Median House Value ($100k)")
plt.ylabel("Predicted Value ($100k)")
plt.title(f"California Housing Polynomial Ridge Pipeline (R² = {r2_val:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- Chaining feature scaling, polynomial interaction terms, and regularized GD in a single unified pipeline.
- Evaluating non-linear feature expansion performance on spatial housing data.
""",
    )
    with open(
        DOCS_EX_DIR / "example_california_housing_pipeline.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)


# =============================================================================
# 3. docs/examples/example_robust_regression_with_ransac.ipynb
# =============================================================================
def build_ex_robust():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Robust Regression with Huber & RANSAC

Demonstrates isolating extreme leverage outliers using `HuberRegressorScratch` and `RANSACRegressorScratch` with a visual comparison of fitted slopes.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.robust import HuberRegressorScratch, RANSACRegressorScratch
from src.solvers import ClosedFormLinearRegression

np.random.seed(42)
N = 120
x = np.random.uniform(-3, 3, N)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.3, N)

# Inject 15 extreme outliers
x[:15] = np.random.uniform(-3, -1, 15)
y[:15] += 25.0
X = x.reshape(-1, 1)

ols = ClosedFormLinearRegression(method="svd").fit(X, y)
huber = HuberRegressorScratch(epsilon=1.35, lr=0.05, max_iter=300).fit(X, y)
ransac = RANSACRegressorScratch(residual_threshold=1.0, max_trials=100).fit(X, y)

print(f"OLS Slope:    {ols.weights[0]:.4f} (severely corrupted by outliers)")
print(f"Huber Slope:  {huber.weights[0]:.4f} (attenuated outlier gradients)")
print(f"RANSAC Slope: {ransac.estimator_.weights[0]:.4f} (ground truth ~ 2.0)")
print(f"RANSAC Inliers detected: {np.sum(ransac.inlier_mask_)} / {N}")

# Visualize Outlier Rejection
plt.figure(figsize=(8, 5))
inliers = ransac.inlier_mask_
outliers = ~inliers
plt.scatter(x[inliers], y[inliers], color='#2563eb', alpha=0.8, label=f'Inliers ({np.sum(inliers)})')
plt.scatter(x[outliers], y[outliers], color='#dc2626', marker='x', s=60, label=f'Outliers ({np.sum(outliers)})')

grid_x = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.plot(grid_x, ols.predict(grid_x), 'r--', lw=2, label=f'OLS (Slope: {ols.weights[0]:.2f})')
plt.plot(grid_x, huber.predict(grid_x), 'g-.', lw=2, label=f'Huber (Slope: {huber.weights[0]:.2f})')
plt.plot(grid_x, ransac.predict(grid_x), 'b-', lw=2.5, label=f'RANSAC (Slope: {ransac.estimator_.weights[0]:.2f})')

plt.title("Robust Regression: OLS vs Huber vs RANSAC on Contaminated Data")
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
        """### What You Learned
- Why OLS breaks under leverage outliers.
- How Huber loss clips gradients while RANSAC discards contaminated observations completely.
""",
    )
    with open(
        DOCS_EX_DIR / "example_robust_regression_with_ransac.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)


# =============================================================================
# 4. docs/examples/example_glm_logistic_poisson.ipynb
# =============================================================================
def build_ex_glm():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Generalized Linear Models (GLM) via IRLS

Demonstrates count modeling via `PoissonRegressionScratch` and binary classification via `LogisticRegressionScratch` with dual visual diagnostic curves.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.glm import PoissonRegressionScratch, LogisticRegressionScratch

np.random.seed(42)
N = 300
X = np.random.randn(N, 2)

# 1. Poisson Counts: y ~ Poisson(exp(Xw + b))
eta = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.2
mu = np.exp(eta)
y_counts = np.random.poisson(mu)

poisson = PoissonRegressionScratch(max_iter=25).fit(X, y_counts)
print(f"Poisson Pseudo-R2: {poisson.pseudo_r2_:.4f}")
print(f"Recovered Weights: {np.round(poisson.weights, 3)} (truth: [0.8, -0.5])")

# 2. Logistic Binary: y ~ Bernoulli(sigmoid(Xw + b))
p = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 0] - 1.5 * X[:, 1])))
y_bin = (np.random.rand(N) < p).astype(int)

logistic = LogisticRegressionScratch(max_iter=25).fit(X, y_bin)
print(f"Logistic Accuracy: {logistic.score(X, y_bin)*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Poisson counts vs linear predictor
sorted_idx = np.argsort(eta)
axes[0].scatter(eta, y_counts, alpha=0.4, color='#7c3aed', label='Observed Counts')
axes[0].plot(eta[sorted_idx], poisson.predict(X)[sorted_idx], 'r-', lw=2, label='Poisson Mean Rate λ')
axes[0].set_title("Poisson Count Regression (Log Link)")
axes[0].set_xlabel("Linear Predictor η = Xw + b")
axes[0].set_ylabel("Count y")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Logistic classification sigmoid curve
p_pred = logistic.predict_proba(X)
log_eta = X @ logistic.weights + logistic.bias
s_idx = np.argsort(log_eta)
axes[1].scatter(log_eta, y_bin, alpha=0.4, c=y_bin, cmap='coolwarm', edgecolors='none', label='Class Labels')
axes[1].plot(log_eta[s_idx], p_pred[s_idx], 'k-', lw=2, label='Fitted Sigmoid σ(η)')
axes[1].axhline(0.5, color='gray', linestyle=':', label='Decision Threshold')
axes[1].set_title("Logistic Classification (Logit Link)")
axes[1].set_xlabel("Log-odds η = Xw + b")
axes[1].set_ylabel("Probability P(y=1)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- Extending linear models to discrete event counts and classification using IRLS.
- Viewing fitted non-linear link functions and probability curves.
""",
    )
    with open(DOCS_EX_DIR / "example_glm_logistic_poisson.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 5. docs/examples/example_bayesian_linear_regression.ipynb
# =============================================================================
def build_ex_bayesian():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Bayesian Linear Regression & Posterior Sampling

Demonstrates exact conjugate Gaussian posterior inference, parameter weight sampling, and epistemic credible intervals.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.bayesian import BayesianLinearRegression

np.random.seed(42)
N = 80
X = np.random.uniform(-2, 2, size=(N, 1))
y = 1.8 * X[:, 0] + 0.4 + np.random.normal(0, 0.25, size=N)

bayes = BayesianLinearRegression(alpha=1.0, beta=16.0).fit(X, y)
w_samples = bayes.sample_weights(n_samples=5, random_state=42)

print(f"Posterior MAP Weights: {bayes.weights[0]:.4f}, Bias: {bayes.bias:.4f}")
print("Sampled weight parameters:\\n", np.round(w_samples, 4))

# Visualize Posterior Mean & Sampled Hypotheses
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y, color='#1e293b', alpha=0.7, label='Observations')
grid_X = np.linspace(-2.2, 2.2, 100).reshape(-1, 1)

# Sampled hypotheses from posterior
for i in range(len(w_samples)):
    plt.plot(grid_X[:, 0], grid_X @ w_samples[i] + bayes.bias, color='#f59e0b', alpha=0.6,
             linestyle='--', label='Posterior Sample' if i == 0 else "")

# MAP line
y_mean, y_std = bayes.predict(grid_X, return_std=True)
plt.plot(grid_X[:, 0], y_mean, color='#2563eb', lw=2.5, label='MAP Posterior Mean')
plt.fill_between(grid_X[:, 0], y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
                 color='#93c5fd', alpha=0.3, label='95% Epistemic Credible Interval')

plt.title("Bayesian Linear Regression: Posterior Hypotheses & Epistemic Uncertainty")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- Calculating analytical posterior distributions over weights.
- Sampling plausible functions from parameter covariance $S_N$.
- Distinguishing epistemic variance from observation noise.
""",
    )
    with open(DOCS_EX_DIR / "example_bayesian_linear_regression.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 6. docs/examples/example_conformal_prediction_coverage.ipynb
# =============================================================================
def build_ex_conformal():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Conformal Prediction Finite-Sample Coverage

Demonstrates distribution-free prediction intervals and calibrating coverage across test splits with visual interval ribbons.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression

np.random.seed(42)
N = 400
X = np.random.uniform(-3, 3, size=(N, 2))
# Heavy-tailed Student-t noise (violates Gaussianity!)
y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.standard_t(df=3, size=N)

base = ClosedFormLinearRegression(method="svd")
conformal = ConformalLinearRegression(estimator=base, alpha=0.10, cal_size=0.25, random_state=42)
conformal.fit(X, y)

# Out-of-sample evaluation
X_test = np.random.uniform(-3, 3, size=(200, 2))
y_test = 2.0 * X_test[:, 0] - 1.5 * X_test[:, 1] + np.random.standard_t(df=3, size=200)

coverage = conformal.score_coverage(X_test, y_test)
print("Nominal Confidence Level: 90.00%")
print(f"Calibrated Cutoff q_hat:   {conformal.q_hat_:.4f}")
print(f"Empirical Coverage:        {coverage * 100:.2f}% (guaranteed >= 90%)")

# Visualize Conformal Prediction Interval Coverage
y_pred_test, lower_test, upper_test = conformal.predict_interval(X_test)
covered = (y_test >= lower_test) & (y_test <= upper_test)

sort_idx = np.argsort(y_pred_test)
plt.figure(figsize=(9, 5))
plt.fill_between(np.arange(len(y_test)), lower_test[sort_idx], upper_test[sort_idx],
                 color='#dbeafe', alpha=0.7, label=f'90% Conformal Band (q̂ = {conformal.q_hat_:.2f})')
plt.plot(np.arange(len(y_test)), y_pred_test[sort_idx], 'b-', lw=1.5, label='Point Prediction')
plt.scatter(np.arange(len(y_test))[covered[sort_idx]], y_test[sort_idx][covered[sort_idx]],
            color='#16a34a', s=25, alpha=0.7, label=f'Covered Points ({np.mean(covered)*100:.1f}%)')
plt.scatter(np.arange(len(y_test))[~covered[sort_idx]], y_test[sort_idx][~covered[sort_idx]],
            color='#dc2626', s=35, marker='x', label='Outside Interval')

plt.title(f"Conformal Prediction Under Heavy-Tailed Noise (Empirical Coverage: {coverage*100:.1f}%)")
plt.xlabel("Test Sample Index (Sorted by Prediction)")
plt.ylabel("Target y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- Conformal prediction guarantees finite-sample coverage even under heavy-tailed non-Gaussian errors.
- Visualizing distribution-free prediction uncertainty bands.
""",
    )
    with open(
        DOCS_EX_DIR / "example_conformal_prediction_coverage.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)


# =============================================================================
# 7. docs/examples/example_streaming_rls_kalman.ipynb
# =============================================================================
def build_ex_streaming():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Streaming Regression with RLS & Kalman Filter

Demonstrates infinite-scale streaming updates with constant $\\mathcal{O}(D^2)$ memory and real-time parameter tracking.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.streaming import RecursiveLeastSquares, KalmanFilterRegression

np.random.seed(42)
N = 1000
X = np.random.randn(N, 2)
y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0 + np.random.normal(0, 0.1, N)

rls = RecursiveLeastSquares(lambda_=1.0)
w0_history, w1_history = [], []
for i in range(N):
    rls.partial_fit(X[i], y[i])
    w0_history.append(rls.weights[0])
    w1_history.append(rls.weights[1])

print(f"RLS Recovered Parameters: w={np.round(rls.weights, 4)}, b={rls.bias:.4f}")
print(f"Samples Processed: {rls.n_samples_seen_}")

plt.figure(figsize=(8, 4.5))
plt.plot(w0_history, label='Estimated w₀ (Truth: 3.0)', color='#2563eb', lw=1.8)
plt.axhline(3.0, color='#2563eb', linestyle=':', alpha=0.7)
plt.plot(w1_history, label='Estimated w₁ (Truth: -2.0)', color='#dc2626', lw=1.8)
plt.axhline(-2.0, color='#dc2626', linestyle=':', alpha=0.7)
plt.title(f"Recursive Least Squares Parameter Convergence Over {N} Live Stream Events")
plt.xlabel("Sample Count Seen (t)")
plt.ylabel("Estimated Weight Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    add_md(
        nb,
        """### What You Learned
- Processing live data point-by-point without storing past observations.
- Observing exponential parameter convergence with Sherman-Morrison rank-1 updates.
""",
    )
    with open(DOCS_EX_DIR / "example_streaming_rls_kalman.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 8. docs/examples/example_quantile_regression_risk.ipynb
# =============================================================================
def build_ex_quantile():
    nb = make_nb()
    add_md(
        nb,
        """# Example: Quantile Regression & Asymmetric Loss

Demonstrates asymmetric pinball loss minimization for financial risk corridors and conditional percentiles.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.quantile import QuantileRegressorScratch

np.random.seed(42)
x = np.random.uniform(1, 10, 300)
# Heteroscedastic noise
y = 1.5 * x + np.random.normal(0, 0.4 * x, 300)
X = x.reshape(-1, 1)

q10 = QuantileRegressorScratch(quantile=0.10).fit(X, y)
q50 = QuantileRegressorScratch(quantile=0.50).fit(X, y)
q90 = QuantileRegressorScratch(quantile=0.90).fit(X, y)

print(f"10th Percentile Slope: {q10.weights[0]:.4f} (downside floor)")
print(f"50th Percentile Slope: {q50.weights[0]:.4f} (median LAD fit)")
print(f"90th Percentile Slope: {q90.weights[0]:.4f} (upside ceiling)")

# Visualize Quantile Regression Pinball Envelope
grid_x = np.linspace(1, 10, 100).reshape(-1, 1)
pred_10 = q10.predict(grid_x)
pred_50 = q50.predict(grid_x)
pred_90 = q90.predict(grid_x)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.4, color='#64748b', s=25, label='Heteroscedastic Observations')
plt.fill_between(grid_x[:, 0], pred_10, pred_90, color='#fed7aa', alpha=0.5, label='80% Risk Corridor (τ=0.10 to 0.90)')
plt.plot(grid_x[:, 0], pred_10, 'r--', lw=2, label=f'10th Percentile Floor (Slope: {q10.weights[0]:.2f})')
plt.plot(grid_x[:, 0], pred_50, 'k-', lw=2.5, label=f'50th Percentile Median (Slope: {q50.weights[0]:.2f})')
plt.plot(grid_x[:, 0], pred_90, 'g--', lw=2, label=f'90th Percentile Ceiling (Slope: {q90.weights[0]:.2f})')

plt.title("Quantile Regression: Asymmetric Loss Pinball Risk Envelopes")
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
        """### What You Learned
- Modeling conditional percentiles to capture expanding variance without parametric assumptions.
- Constructing risk corridors for asymmetric downside/upside forecasting.
""",
    )
    with open(DOCS_EX_DIR / "example_quantile_regression_risk.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 9. examples/02_model_comparison_dashboard.ipynb
# =============================================================================
def build_dashboard():
    nb = make_nb()
    add_md(
        nb,
        """# Mini-Project 02: Model Comparison Dashboard

A comprehensive benchmark comparing OLS, Ridge, Lasso, ElasticNet, Huber, and RANSAC on California Housing with a visual performance benchmark chart.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.solvers import ClosedFormLinearRegression, NewtonLinearRegression
from src.linear_regression import GradientDescentLinearRegression, compute_mse, compute_r2
from src.robust import HuberRegressorScratch, RANSACRegressorScratch
from src.pipeline import StandardScalerScratch

df = pd.read_csv(root_dir / "data" / "california_housing" / "california_housing.csv")
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
X = df[features].values
y = df["MedHouseVal"].values

# Standardize
scaler = StandardScalerScratch()
X_scaled = scaler.fit_transform(X)

# 80/20 split
split = int(0.8 * len(X))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

suite = {
    "OLS (SVD)": ClosedFormLinearRegression(method="svd"),
    "Newton-Raphson": NewtonLinearRegression(),
    "Ridge (L2)": GradientDescentLinearRegression(penalty="l2", alpha=1.0, lr=0.02, n_iters=300, optimizer="adam"),
    "Lasso (L1)": GradientDescentLinearRegression(penalty="l1", alpha=0.01, lr=0.02, n_iters=300, optimizer="adam"),
    "ElasticNet": GradientDescentLinearRegression(penalty="elasticnet", alpha=0.01, l1_ratio=0.5, lr=0.02, n_iters=300, optimizer="adam"),
    "Huber Regressor": HuberRegressorScratch(epsilon=1.35, lr=0.02, max_iter=300)
}

dashboard = []
for name, model in suite.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dashboard.append({
        "Model": name,
        "Test R2": compute_r2(y_test, y_pred),
        "Test RMSE": np.sqrt(compute_mse(y_test, y_pred))
    })

res_df = pd.DataFrame(dashboard)
print(res_df.to_string(index=False))

# Plot performance benchmark
plt.figure(figsize=(9, 4.5))
x_pos = np.arange(len(res_df))
plt.bar(x_pos - 0.2, res_df["Test R2"], width=0.4, label="Test R²", color="#3b82f6")
plt.bar(x_pos + 0.2, res_df["Test RMSE"], width=0.4, label="Test RMSE", color="#ef4444")
plt.xticks(x_pos, res_df["Model"], rotation=20, ha="right")
plt.title("Model Comparison Benchmark on California Housing")
plt.ylabel("Score / Metric")
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
""",
    )
    with open(EXAMPLES_DIR / "02_model_comparison_dashboard.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 10. examples/03_uncertainty_aware_predictions.ipynb
# =============================================================================
def build_uncertainty_project():
    nb = make_nb()
    add_md(
        nb,
        """# Mini-Project 03: Uncertainty-Aware Predictions Pipeline

Combines Bayesian Linear Regression (epistemic uncertainty) and Conformal Prediction (guaranteed coverage) with visual error bar comparisons.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.bayesian import BayesianLinearRegression
from src.conformal import ConformalLinearRegression
from src.solvers import ClosedFormLinearRegression

df = pd.read_csv(root_dir / "data" / "california_housing" / "california_housing.csv")
X = df[["MedInc", "HouseAge"]].values[:1000]
y = df["MedHouseVal"].values[:1000]

# 1. Bayesian Epistemic Uncertainty
bayes = BayesianLinearRegression(alpha=1.0, beta=5.0).fit(X, y)
y_mean, y_std = bayes.predict(X[:10], return_std=True)

# 2. Conformal Interval with 90% Guarantee
conformal = ConformalLinearRegression(estimator=ClosedFormLinearRegression(method="svd"), alpha=0.10, cal_size=0.20, random_state=42)
conformal.fit(X, y)
preds, lower, upper = conformal.predict_interval(X[:10])

comparison_table = pd.DataFrame({
    "True y": y[:10],
    "Bayes Mean": y_mean,
    "Epistemic std": y_std,
    "Conformal Low (90%)": lower,
    "Conformal High (90%)": upper
})
print(comparison_table.round(3).to_string(index=False))

# Plot comparison of Epistemic vs Conformal intervals
plt.figure(figsize=(10, 5))
sample_idx = np.arange(10)
plt.errorbar(sample_idx - 0.15, y_mean, yerr=1.96 * y_std, fmt='o', color='#2563eb',
             capsize=4, label='Bayesian 95% Credible Interval (Epistemic)')
y_err_lower = np.maximum(0, preds - lower)
y_err_upper = np.maximum(0, upper - preds)
plt.errorbar(sample_idx + 0.15, preds, yerr=[y_err_lower, y_err_upper], fmt='s', color='#16a34a',
             capsize=4, label='Conformal 90% Coverage Band (Finite-Sample)')
plt.scatter(sample_idx, y[:10], color='#dc2626', marker='*', s=120, zorder=5, label='Actual Value')
plt.xticks(sample_idx, [f"Tract {i+1}" for i in sample_idx])
plt.ylabel("Median House Value ($100k)")
plt.title("Epistemic Uncertainty vs. Conformal Prediction Bands")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""",
    )
    with open(EXAMPLES_DIR / "03_uncertainty_aware_predictions.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


# =============================================================================
# 11. competitions/house_prices_competition_baseline.ipynb
# =============================================================================
def build_competition_baseline():
    nb = make_nb()
    add_md(
        nb,
        """# Competition Baseline: Kaggle Ames House Prices

A complete, Kaggle-ready baseline workflow featuring:
1. Feature selection & preprocessing via `ColumnTransformerScratch`.
2. K-Fold Cross Validation via `cross_val_score_scratch`.
3. ElasticNet Coordinate Descent fitting.
4. Visual cross-validation stability and prediction fit plots.
""",
    )
    add_code(
        nb,
        PATH_SETUP_CODE
        + """
from src.pipeline import PipelineScratch, ColumnTransformerScratch, StandardScalerScratch, SimpleImputerScratch, OneHotEncoderScratch
from src.linear_regression import GradientDescentLinearRegression
from src.model_selection import KFoldScratch, cross_val_score_scratch

df = pd.read_csv(root_dir / "data" / "house_prices" / "train.csv")
num_cols = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]
cat_cols = ["Neighborhood"]

X = df[num_cols + cat_cols]
y = np.log1p(df["SalePrice"].values)

pipe = PipelineScratch([
    ("prep", ColumnTransformerScratch([
        ("num", PipelineScratch([("imp", SimpleImputerScratch(strategy="median")), ("scaler", StandardScalerScratch())]), num_cols),
        ("cat", PipelineScratch([("imp", SimpleImputerScratch(strategy="most_frequent")), ("ohe", OneHotEncoderScratch(handle_unknown="ignore"))]), cat_cols)
    ])),
    ("reg", GradientDescentLinearRegression(penalty="elasticnet", alpha=0.02, l1_ratio=0.5, lr=0.05, n_iters=300, optimizer="adam"))
])

# 5-Fold Cross Validation
cv = KFoldScratch(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score_scratch(pipe, X, y, cv=cv, scoring="r2")

print(f"5-Fold Cross-Validation R2: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print("Cross-validation fold scores:", np.round(cv_scores, 4))

# Plot CV Folds and Mock Predictions
pipe.fit(X, y)
y_pred = pipe.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar([f"Fold {i+1}" for i in range(5)], cv_scores, color='#6366f1', alpha=0.85)
axes[0].axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean R²: {np.mean(cv_scores):.3f}')
axes[0].set_title("5-Fold Cross-Validation R² Scores")
axes[0].set_ylabel("R² Score")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].scatter(y, y_pred, alpha=0.4, color='#0284c7', s=20)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
axes[1].set_title("In-Sample Predictions vs Actual Log SalePrice")
axes[1].set_xlabel("Actual Log SalePrice")
axes[1].set_ylabel("Predicted Log SalePrice")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
    )
    with open(
        COMPETITIONS_DIR / "house_prices_competition_baseline.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)


def main():
    print("Building example notebooks and project templates with rich visualizations...")
    build_ex_house_prices()
    build_ex_california_housing()
    build_ex_robust()
    build_ex_glm()
    build_ex_bayesian()
    build_ex_conformal()
    build_ex_streaming()
    build_ex_quantile()
    build_dashboard()
    build_uncertainty_project()
    build_competition_baseline()
    print("All example notebooks and templates built successfully!")


if __name__ == "__main__":
    main()

"""
Build 8 Compact Example Notebooks in docs/examples/,
2 Mini-Project Notebooks in examples/, and 1 Competition Baseline in competitions/.
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

This example demonstrates how to build a leak-free tabular regression baseline for Kaggle House Prices using `ColumnTransformerScratch` and `ClosedFormLinearRegression`.
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

print(f"Fitted Baseline R2: {compute_r2(y, y_pred):.4f}")
print(f"Fitted RMSE (log scale): {np.sqrt(compute_mse(y, y_pred)):.4f}")
""",
    )
    add_md(
        nb,
        """### What You Learned
- How to handle mixed numeric and categorical data using pure-NumPy `ColumnTransformerScratch`.
- Log-transforming skewed targets for improved linear fit.
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

Demonstrates scaling, degree-2 polynomial expansion, and Ridge regression on California Housing.
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
print(f"Pipeline R2 Score on 2,000 California Tracts: {pipe.score(X, y):.4f}")
""",
    )
    add_md(
        nb,
        """### What You Learned
- Chaining feature scaling, polynomial interaction terms, and regularized GD in a single unified pipeline.
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

Demonstrates isolating extreme leverage outliers using `HuberRegressorScratch` and `RANSACRegressorScratch`.
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
""",
    )
    add_md(
        nb,
        """### What You Learned
- Why OLS breaks under leverage outliers.
- How Huber loss clips gradients and RANSAC discards contaminated observations.
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

Demonstrates count modeling via `PoissonRegressionScratch` and binary classification via `LogisticRegressionScratch`.
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
""",
    )
    add_md(
        nb,
        """### What You Learned
- Extending linear models to discrete event counts and classification using IRLS.
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

Demonstrates exact conjugate Gaussian posterior inference and parameter weight sampling.
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
""",
    )
    add_md(
        nb,
        """### What You Learned
- Calculating analytical posterior distributions over weights.
- Sampling hypotheses from parameter covariance $S_N$.
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

Demonstrates distribution-free prediction intervals and calibrating coverage across test splits.
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
print(f"Nominal Confidence Level: 90.00%")
print(f"Calibrated Cutoff q_hat:   {conformal.q_hat_:.4f}")
print(f"Empirical Coverage:        {coverage * 100:.2f}% (guaranteed >= 90%)")
""",
    )
    add_md(
        nb,
        """### What You Learned
- Conformal prediction guarantees finite-sample coverage even under heavy-tailed non-Gaussian errors.
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

Demonstrates infinite-scale streaming updates with constant $\\mathcal{O}(D^2)$ memory.
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
for i in range(N):
    rls.partial_fit(X[i], y[i])

print(f"RLS Recovered Parameters: w={np.round(rls.weights, 4)}, b={rls.bias:.4f}")
print(f"Samples Processed: {rls.n_samples_seen_}")
""",
    )
    add_md(
        nb,
        """### What You Learned
- Processing live data point-by-point without storing past observations.
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

Demonstrates asymmetric pinball loss minimization for financial risk corridors.
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
""",
    )
    add_md(
        nb,
        """### What You Learned
- Modeling conditional percentiles to capture expanding variance without parametric assumptions.
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

A comprehensive benchmark comparing OLS, Ridge, Lasso, ElasticNet, Huber, and RANSAC on California Housing.
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

Combines Bayesian Linear Regression (epistemic uncertainty) and Conformal Prediction (guaranteed coverage).
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
4. Mock submission file generation.
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
""",
    )
    with open(
        COMPETITIONS_DIR / "house_prices_competition_baseline.ipynb", "w", encoding="utf-8"
    ) as f:
        nbf.write(nb, f)


def main():
    print("Building example notebooks and project templates...")
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

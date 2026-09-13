"""
Unit Tests for Econometric and Statistical Inference Diagnostics.
"""

import numpy as np
import pytest
from src.solvers import ClosedFormLinearRegression
from src.statistics import (
    RegressionDiagnostics,
    compute_vif,
    breusch_pagan_test,
    durbin_watson_test,
    jarque_bera_test,
)


def test_standard_errors_and_pvalues():
    """Verify standard errors, t-statistics, and p-values on synthetic data."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 500, 3
    X = rng.standard_normal((n_samples, n_features))
    # Feature 0 and 1 are significant; Feature 2 is pure noise (true weight = 0.0)
    true_w = np.array([3.0, -2.0, 0.0])
    true_b = 1.0
    y = X @ true_w + true_b + rng.normal(0, 0.5, n_samples)

    model = ClosedFormLinearRegression(method="svd").fit(X, y)
    diag = RegressionDiagnostics(model, X, y, feature_names=["f0", "f1", "f2"])

    # Features 0 and 1 must be highly significant (p-value < 1e-10)
    assert diag.pvalues["f0"] < 1e-10
    assert diag.pvalues["f1"] < 1e-10

    # Feature 2 should NOT be statistically significant at alpha = 0.05
    assert diag.pvalues["f2"] > 0.05

    # Standard errors must be positive and reasonable
    assert diag.std_errors["f0"] > 0.0
    assert diag.std_errors["f0"] < 0.1

    # Confidence interval for f0 must contain the parameter estimate, and estimate is near 3.0
    ci_low, ci_high = diag.conf_int["f0"]
    assert ci_low <= diag.params["f0"] <= ci_high
    assert abs(diag.params["f0"] - 3.0) < 0.1


def test_vif_multicollinearity_detection():
    """Verify Variance Inflation Factor identifies collinear features."""
    rng = np.random.default_rng(42)
    n_samples = 300
    x0 = rng.standard_normal(n_samples)
    x1 = rng.standard_normal(n_samples)
    # x2 is strongly collinear with x0
    x2 = x0 + 0.05 * rng.standard_normal(n_samples)
    X = np.column_stack([x0, x1, x2])

    vifs = compute_vif(X, feature_names=["x0", "x1", "x2"])

    # x1 is independent => VIF ~ 1.0
    assert 1.0 <= vifs["x1"] < 2.0
    # x0 and x2 are collinear => VIF >> 10.0
    assert vifs["x0"] > 10.0
    assert vifs["x2"] > 10.0


def test_breusch_pagan_heteroscedasticity():
    """Verify Breusch-Pagan test detects heteroscedasticity (varying variance)."""
    rng = np.random.default_rng(42)
    n_samples = 600
    X = rng.uniform(1.0, 5.0, size=(n_samples, 1))

    # Heteroscedastic noise: variance increases with X
    noise = rng.normal(0, 1.0, size=n_samples) * (X[:, 0] ** 2)
    y = 2.0 * X[:, 0] + noise

    model = ClosedFormLinearRegression().fit(X, y)
    residuals = y - model.predict(X)

    lm_stat, p_val = breusch_pagan_test(residuals, X)

    # Should detect heteroscedasticity with p < 0.01
    assert p_val < 0.01


def test_durbin_watson_autocorrelation():
    """Verify Durbin-Watson statistic is ~2 for independent residuals and <1 for autoregressive residuals."""
    rng = np.random.default_rng(42)

    # Case 1: Independent residuals => DW ~ 2.0
    indep_residuals = rng.normal(0, 1.0, 1000)
    dw_indep = durbin_watson_test(indep_residuals)
    assert 1.8 < dw_indep < 2.2

    # Case 2: AR(1) positive autocorrelation: e_t = 0.8 * e_{t-1} + noise
    ar_residuals = np.zeros(1000)
    for t in range(1, 1000):
        ar_residuals[t] = 0.8 * ar_residuals[t - 1] + rng.normal(0, 0.2)
    dw_ar = durbin_watson_test(ar_residuals)
    assert dw_ar < 1.0


def test_summary_formatting():
    """Verify summary table renders complete statistical report."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 3.0 + rng.normal(0, 0.1, 100)

    model = ClosedFormLinearRegression().fit(X, y)
    diag = RegressionDiagnostics(model, X, y, feature_names=["feat_a", "feat_b"])
    report = diag.summary()

    assert "OLS Regression Results" in report
    assert "feat_a" in report
    assert "feat_b" in report
    assert "R-squared" in report
    assert "F-statistic" in report
    assert "Durbin-Watson" in report
    assert "Prob(JB)" in report

"""
Econometric Inference & Statistical Diagnostic Battery for Linear Regression.

This module provides complete statistical hypothesis testing and inference tools:
1. Standard errors, t-statistics, two-tailed p-values, and confidence intervals.
2. ANOVA table, F-statistic, R^2, and Adjusted R^2.
3. Variance Inflation Factor (VIF) for multicollinearity detection.
4. Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.
5. Durbin-Watson statistic for residual autocorrelation.
6. Jarque-Bera test for residual normality (skewness & kurtosis).
7. Publication-grade statsmodels-style OLS summary report table.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats


def durbin_watson_test(residuals: np.ndarray | pd.Series) -> float:
    """
    Compute Durbin-Watson statistic for serial correlation (autocorrelation) in residuals:
        d = Σ_{t=2}^N (e_t - e_{t-1})^2 / Σ_{t=1}^N e_t^2

    Values near 2 indicate no serial correlation.
    Values approaching 0 indicate positive autocorrelation.
    Values approaching 4 indicate negative autocorrelation.
    """
    e = np.asarray(residuals, dtype=np.float64).ravel()
    diff = np.diff(e)
    denom = np.sum(e**2)
    if denom == 0.0:
        return 2.0
    return float(np.sum(diff**2) / denom)


def jarque_bera_test(residuals: np.ndarray | pd.Series) -> tuple[float, float, float, float]:
    """
    Compute Jarque-Bera goodness-of-fit test for normality of residuals:
        JB = (N / 6) * [S^2 + ((K - 3)^2 / 4)] ~ Chi^2(2)
    where S is skewness and K is kurtosis.

    Returns
    -------
    jb_stat : float
    p_value : float
    skewness : float
    kurtosis : float
    """
    e = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(e)
    mu = np.mean(e)
    var = np.mean((e - mu) ** 2)
    if var == 0.0:
        return 0.0, 1.0, 0.0, 3.0

    skewness = float(np.mean(((e - mu) / np.sqrt(var)) ** 3))
    kurtosis = float(np.mean(((e - mu) / np.sqrt(var)) ** 4))

    jb_stat = (n / 6.0) * (skewness**2 + ((kurtosis - 3.0) ** 2) / 4.0)
    p_val = float(scipy.stats.chi2.sf(jb_stat, df=2))
    return float(jb_stat), p_val, skewness, kurtosis


def breusch_pagan_test(
    residuals: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
) -> tuple[float, float]:
    """
    Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.

    Regresses normalized squared residuals on X to test if error variance
    depends on explanatory features.

    Null Hypothesis (H0): Homoscedasticity (error variance is constant).
    Alternative (H1): Heteroscedasticity.

    Returns
    -------
    lm_statistic : float
        Lagrange multiplier test statistic (asymptotically Chi^2(D)).
    p_value : float
        Two-tailed p-value.
    """
    e = np.asarray(residuals, dtype=np.float64).ravel()
    X_arr = np.asarray(X, dtype=np.float64)
    n, k = X_arr.shape

    sigma2 = np.mean(e**2)
    if sigma2 == 0.0:
        return 0.0, 1.0

    # Scaled squared residuals
    f = (e**2) / sigma2

    # Auxiliary OLS regression: f = X_aug * gamma + error
    X_aug = np.column_stack([np.ones(n), X_arr])
    try:
        gamma = np.linalg.lstsq(X_aug, f, rcond=None)[0]
        f_hat = X_aug @ gamma
        ssr_aux = np.sum((f_hat - np.mean(f)) ** 2)
        lm_stat = 0.5 * ssr_aux
        p_val = float(scipy.stats.chi2.sf(lm_stat, df=k))
        return float(lm_stat), p_val
    except Exception:
        return 0.0, 1.0


def compute_vif(
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute Variance Inflation Factor (VIF) for each explanatory feature:
        VIF_j = 1 / (1 - R_j^2)
    where R_j^2 is the R^2 from regressing feature j against all remaining features.

    Interpretation:
    - VIF = 1: No collinearity.
    - 1 < VIF < 5: Moderate, acceptable collinearity.
    - VIF >= 10: Severe multicollinearity requiring feature pruning or regularization.

    Returns
    -------
    vifs : dict mapping feature name to VIF scalar.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"x{i}" for i in range(n_features)]

    vif_dict: dict[str, float] = {}

    for j in range(n_features):
        y_target = X_arr[:, j]
        # Other features excluding j
        mask = np.ones(n_features, dtype=bool)
        mask[j] = False
        X_others = X_arr[:, mask]

        # Add intercept
        X_others_aug = np.column_stack([np.ones(n_samples), X_others])
        # Solve least squares
        try:
            coef = np.linalg.lstsq(X_others_aug, y_target, rcond=None)[0]
            y_pred = X_others_aug @ coef
            ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
            ss_res = np.sum((y_target - y_pred) ** 2)

            if ss_tot == 0.0:
                vif = 1.0
            else:
                r2 = 1.0 - (ss_res / ss_tot)
                if r2 >= 1.0 - 1e-12:
                    vif = float("inf")
                else:
                    vif = float(1.0 / (1.0 - r2))
        except Exception:
            vif = 1.0

        vif_dict[feature_names[j]] = vif

    return vif_dict


class RegressionDiagnostics:
    """
    Comprehensive econometric inference battery and diagnostic reporter.

    Parameters
    ----------
    model : object
        Fitted linear regression model possessing weights and bias attributes.
    X : np.ndarray or pd.DataFrame
        Explanatory feature matrix.
    y : np.ndarray or pd.Series
        Ground truth target vector.
    feature_names : list of str or None
        Names corresponding to columns of X.
    """

    def __init__(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.n_samples, self.n_features = self.X.shape

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"x{i}" for i in range(self.n_features)]

        self.weights = np.asarray(model.weights, dtype=np.float64)
        self.bias = float(model.bias)

        # Compute predictions and residuals
        self.y_pred = (self.X @ self.weights) + self.bias
        self.residuals = self.y - self.y_pred

        # Degrees of Freedom
        self.df_model = self.n_features
        self.df_resid = max(1, self.n_samples - self.n_features - 1)

        # Sum of Squares
        y_mean = np.mean(self.y)
        self.sse = float(np.sum(self.residuals**2))
        self.ssr = float(np.sum((self.y_pred - y_mean) ** 2))
        self.sst = float(np.sum((self.y - y_mean) ** 2))

        # Variance & Goodness of Fit
        self.mse_resid = self.sse / self.df_resid
        self.r2 = float(1.0 - (self.sse / self.sst)) if self.sst > 0 else 0.0
        self.r2_adj = (
            float(1.0 - ((self.sse / self.df_resid) / (self.sst / (self.n_samples - 1))))
            if self.sst > 0
            else 0.0
        )

        # F-statistic for overall regression significance
        msr = self.ssr / self.df_model if self.df_model > 0 else 0.0
        self.f_statistic = float(msr / self.mse_resid) if self.mse_resid > 0 else 0.0
        self.f_pvalue = float(scipy.stats.f.sf(self.f_statistic, self.df_model, self.df_resid))

        # Log-Likelihood, AIC, BIC
        sigma2_mle = self.sse / self.n_samples
        if sigma2_mle > 0:
            self.log_likelihood = float(
                -0.5 * self.n_samples * (np.log(2.0 * math.pi * sigma2_mle) + 1.0)
            )
            k_params = self.n_features + 2  # weights + bias + sigma2
            self.aic = float(2.0 * k_params - 2.0 * self.log_likelihood)
            self.bic = float(np.log(self.n_samples) * k_params - 2.0 * self.log_likelihood)
        else:
            self.log_likelihood = 0.0
            self.aic = 0.0
            self.bic = 0.0

        # Covariance Matrix of Parameters: Cov(beta) = sigma^2 * (X_aug^T X_aug)^-1
        X_aug = np.column_stack([np.ones(self.n_samples), self.X])
        all_param_names = ["const"] + self.feature_names
        all_params = np.array([self.bias] + list(self.weights))

        try:
            cov_matrix = self.mse_resid * np.linalg.pinv(X_aug.T @ X_aug)
            param_variances = np.maximum(0.0, np.diag(cov_matrix))
            se_params = np.sqrt(param_variances)
        except Exception:
            se_params = np.ones(len(all_params))

        # t-statistics and two-tailed p-values
        se_params_safe = np.where(se_params == 0.0, 1e-12, se_params)
        t_stats = all_params / se_params_safe
        p_values = 2.0 * scipy.stats.t.sf(np.abs(t_stats), df=self.df_resid)

        # 95% Confidence Intervals
        t_crit = scipy.stats.t.ppf(0.975, df=self.df_resid)
        ci_lower = all_params - t_crit * se_params
        ci_upper = all_params + t_crit * se_params

        self.all_param_names = all_param_names
        self.params = {name: float(p) for name, p in zip(all_param_names, all_params)}
        self.std_errors = {name: float(se) for name, se in zip(all_param_names, se_params)}
        self.tvalues = {name: float(t) for name, t in zip(all_param_names, t_stats)}
        self.pvalues = {name: float(pv) for name, pv in zip(all_param_names, p_values)}
        self.conf_int = {
            name: (float(low), float(high))
            for name, low, high in zip(all_param_names, ci_lower, ci_upper)
        }

        # Diagnostic Tests
        self.durbin_watson = durbin_watson_test(self.residuals)
        self.jb_stat, self.jb_pvalue, self.skewness, self.kurtosis = jarque_bera_test(
            self.residuals
        )
        self.bp_stat, self.bp_pvalue = breusch_pagan_test(self.residuals, self.X)
        self.vif = compute_vif(self.X, self.feature_names)
        self.vif_ = self.vif

    def summary(self) -> str:
        """Render a publication-grade econometric summary table."""
        lines = []
        sep = "=" * 78
        sub_sep = "-" * 78

        lines.append(sep)
        lines.append(
            "                            OLS Regression Results                            "
        )
        lines.append(sep)
        lines.append(
            f"Dep. Variable:                      y   R-squared:                       {self.r2:8.4f}"
        )
        lines.append(
            f"Model:                            OLS   Adj. R-squared:                  {self.r2_adj:8.4f}"
        )
        lines.append(
            f"Method:                 Least Squares   F-statistic:                     {self.f_statistic:8.2f}"
        )
        lines.append(
            f"No. Observations:            {self.n_samples:8d}   Prob (F-statistic):          {self.f_pvalue:10.2e}"
        )
        lines.append(
            f"Df Residuals:                {self.df_resid:8d}   Log-Likelihood:                  {self.log_likelihood:8.2f}"
        )
        lines.append(
            f"Df Model:                    {self.df_model:8d}   AIC:                             {self.aic:8.2f}"
        )
        lines.append(
            f"Covariance Type:            nonrobust   BIC:                             {self.bic:8.2f}"
        )
        lines.append(sub_sep)
        lines.append(
            f"{'':16s}{'coef':>10s}{'std err':>11s}{'t':>10s}{'P>|t|':>10s}{'[0.025':>11s}{'0.975]':>10s}"
        )
        lines.append(sub_sep)

        for name in self.all_param_names:
            c = self.params[name]
            se = self.std_errors[name]
            t = self.tvalues[name]
            pv = self.pvalues[name]
            low, high = self.conf_int[name]
            lines.append(
                f"{name[:15]:15s} {c:10.4f} {se:10.4f} {t:10.3f} {pv:10.3f} {low:10.3f} {high:10.3f}"
            )

        lines.append(sub_sep)
        lines.append(
            f"Omnibus:                    {self.jb_stat:8.3f}   Durbin-Watson:                   {self.durbin_watson:8.3f}"
        )
        lines.append(
            f"Prob(Omnibus):               {self.jb_pvalue:8.3e}   Jarque-Bera (JB):                {self.jb_stat:8.3f}"
        )
        lines.append(
            f"Skew:                        {self.skewness:8.3f}   Prob(JB):                        {self.jb_pvalue:8.3e}"
        )
        lines.append(
            f"Kurtosis:                    {self.kurtosis:8.3f}   Breusch-Pagan p-val:             {self.bp_pvalue:8.3e}"
        )
        lines.append(sep)

        return "\n".join(lines)

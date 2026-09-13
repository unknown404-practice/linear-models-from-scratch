"""
Linear Models from Scratch
==========================

A production-grade, pedagogical, and vectorized NumPy implementation of linear models,
optimization routines, regularization methods, robust estimators, Bayesian uncertainty,
conformal prediction intervals, generalized linear models (GLMs), streaming filters,
quantile regression, and scikit-learn-compatible composition pipelines.

Modules
-------
- `linear_regression`: Gradient descent, closed-form solvers, and regularized regression.
- `solvers`: Analytical solvers (QR, SVD, Cholesky, Normal Equation, 1-step Newton-Raphson).
- `optimizers`: First-order optimizers (SGD with momentum, RMSprop, Adam).
- `schedulers`: Dynamic learning rate schedules (StepLR, ExponentialLR, CosineAnnealingLR).
- `features`: Polynomial feature mapping with interaction terms.
- `robust`: Huber loss regression and RANSAC outlier-resilient fitting.
- `bayesian`: Analytical Bayesian linear regression with posterior predictive intervals and ARD.
- `conformal`: Inductive split conformal prediction with distribution-free finite-sample guarantees.
- `glm`: Generalized linear models via Iteratively Reweighted Least Squares (IRLS).
- `streaming`: Online Recursive Least Squares (RLS) and state-space Kalman filters.
- `quantile`: Pinball-loss quantile regression for risk corridors and conditional medians.
- `pipeline`: Scikit-learn compatible Pipeline, ColumnTransformer, and custom Transformers.
- `model_selection`: Stratified K-Fold CV, cross_val_score, and GridSearchCV.
- `statistics`: OLS inferential statistics (p-values, VIF, Breusch-Pagan, Durbin-Watson).
- `visualization`: 2D/3D loss surface geometry and optimizer trajectory plotting.
- `serialization`: Dual format persistence (JSON state and Python pickle).
- `data_loader`: Real-world regression dataset loaders and preprocessing pipelines.
"""

from __future__ import annotations

from pathlib import Path

# Version single source of truth
__version__ = "0.1.0"
__author__ = "Ranadeep Saha"

from src.linear_regression import (
    GradientDescentLinearRegression,
    standardize_features,
    apply_scaler,
    compute_mse,
    compute_r2,
    soft_threshold,
)
from src.optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    get_optimizer,
)
from src.schedulers import (
    LRScheduler,
    ConstantLR,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
)
from src.solvers import (
    ClosedFormLinearRegression,
    NewtonLinearRegression,
)
from src.statistics import (
    RegressionDiagnostics,
    compute_vif,
    breusch_pagan_test,
    durbin_watson_test,
    jarque_bera_test,
)
from src.visualization import (
    compute_loss_grid,
    plot_loss_contours,
    plot_loss_surface_3d,
    plot_gradient_quiver,
    compare_optimizer_trajectories,
    visualize_conditioning_impact,
)
from src.features import (
    PolynomialFeaturesScratch,
)
from src.robust import (
    HuberRegressorScratch,
    RANSACRegressorScratch,
)
from src.pipeline import (
    BaseTransformer,
    StandardScalerScratch,
    SimpleImputerScratch,
    OneHotEncoderScratch,
    ColumnTransformerScratch,
    PipelineScratch,
)
from src.model_selection import (
    clone_estimator,
    KFoldScratch,
    cross_val_score_scratch,
    GridSearchCVScratch,
)
from src.serialization import (
    save_model,
    load_model,
    save_model_json,
    load_model_json,
)
from src.bayesian import (
    BayesianLinearRegression,
)
from src.conformal import (
    ConformalLinearRegression,
)
from src.glm import (
    GeneralizedLinearModel,
    LogisticRegressionScratch,
    PoissonRegressionScratch,
    GammaRegressionScratch,
)
from src.streaming import (
    RecursiveLeastSquares,
    KalmanFilterRegression,
)
from src.quantile import (
    QuantileRegressorScratch,
)
from src.data_loader import (
    load_california_housing,
    load_house_prices_kaggle,
    clean_and_split_regression_data,
)


# =============================================================================
# CONVENIENCE SUBCLASSES AND ALIASES FOR SCIKIT-LEARN COMPATIBILITY
# =============================================================================


class Ridge(GradientDescentLinearRegression):
    """Ridge (L2-regularized) Linear Regression from Scratch."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(penalty="l2", alpha=alpha, **kwargs)


class Lasso(GradientDescentLinearRegression):
    """Lasso (L1-regularized) Linear Regression from Scratch."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(penalty="l1", alpha=alpha, **kwargs)


class ElasticNet(GradientDescentLinearRegression):
    """ElasticNet (L1 + L2 regularized) Linear Regression from Scratch."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        super().__init__(penalty="elasticnet", alpha=alpha, l1_ratio=l1_ratio, **kwargs)


# Estimator aliases
HuberRegressor = HuberRegressorScratch
RANSACRegressor = RANSACRegressorScratch
ConformalPredictor = ConformalLinearRegression
QuantileRegressor = QuantileRegressorScratch
PolynomialFeatures = PolynomialFeaturesScratch
LogisticRegression = LogisticRegressionScratch
PoissonRegression = PoissonRegressionScratch
GammaRegression = GammaRegressionScratch

# Pipeline and transformation aliases
StandardScaler = StandardScalerScratch
SimpleImputer = SimpleImputerScratch
OneHotEncoder = OneHotEncoderScratch
ColumnTransformer = ColumnTransformerScratch
Pipeline = PipelineScratch

# Model selection aliases
KFold = KFoldScratch
cross_val_score = cross_val_score_scratch
GridSearchCV = GridSearchCVScratch


__all__ = [
    "__version__",
    "__author__",
    # Core Regression Models
    "GradientDescentLinearRegression",
    "ClosedFormLinearRegression",
    "NewtonLinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    # Robust & Non-linear
    "HuberRegressorScratch",
    "HuberRegressor",
    "RANSACRegressorScratch",
    "RANSACRegressor",
    "PolynomialFeaturesScratch",
    "PolynomialFeatures",
    # Uncertainty Quantification
    "BayesianLinearRegression",
    "ConformalLinearRegression",
    "ConformalPredictor",
    # Generalized Linear Models
    "GeneralizedLinearModel",
    "LogisticRegressionScratch",
    "LogisticRegression",
    "PoissonRegressionScratch",
    "PoissonRegression",
    "GammaRegressionScratch",
    "GammaRegression",
    # Streaming & Quantile
    "RecursiveLeastSquares",
    "KalmanFilterRegression",
    "QuantileRegressorScratch",
    "QuantileRegressor",
    # Pipeline & Transformers
    "BaseTransformer",
    "StandardScalerScratch",
    "StandardScaler",
    "SimpleImputerScratch",
    "SimpleImputer",
    "OneHotEncoderScratch",
    "OneHotEncoder",
    "ColumnTransformerScratch",
    "ColumnTransformer",
    "PipelineScratch",
    "Pipeline",
    # Model Selection & CV
    "clone_estimator",
    "KFoldScratch",
    "KFold",
    "cross_val_score_scratch",
    "cross_val_score",
    "GridSearchCVScratch",
    "GridSearchCV",
    # Optimizers & Schedulers
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    "get_optimizer",
    "LRScheduler",
    "ConstantLR",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    # Diagnostics & Stats
    "RegressionDiagnostics",
    "compute_vif",
    "breusch_pagan_test",
    "durbin_watson_test",
    "jarque_bera_test",
    # Visualization
    "compute_loss_grid",
    "plot_loss_contours",
    "plot_loss_surface_3d",
    "plot_gradient_quiver",
    "compare_optimizer_trajectories",
    "visualize_conditioning_impact",
    # Serialization
    "save_model",
    "load_model",
    "save_model_json",
    "load_model_json",
    # Data loaders & utilities
    "load_california_housing",
    "load_house_prices_kaggle",
    "clean_and_split_regression_data",
    "standardize_features",
    "apply_scaler",
    "compute_mse",
    "compute_r2",
    "soft_threshold",
]

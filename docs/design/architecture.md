# Architectural Blueprint

## 1. Modular Hierarchy
The repository is strictly factored into single-responsibility modules under `src/`:

```
src/
├── linear_regression.py   # GD engine, regularized GD, Coordinate Descent
├── solvers.py             # Closed-form tri-solvers (Normal, QR, SVD), Newton-Raphson
├── optimizers.py          # Momentum, RMSprop, Adam parameter optimizers
├── schedulers.py          # Constant, Step, Exponential, Cosine learning rate decay
├── robust.py              # Huber loss, MAD adaptive scale, RANSAC consensus
├── features.py            # Polynomial combinations and interaction terms
├── bayesian.py            # Conjugate Gaussian posterior, uncertainty decomposition, ARD
├── conformal.py           # Split conformal calibration, finite-sample coverage score
├── glm.py                 # Exponential dispersion family via IRLS (Logistic, Poisson, Gamma)
├── streaming.py           # Recursive Least Squares (Sherman-Morrison), Kalman Filter
├── quantile.py            # Asymmetric pinball loss, smoothed IRLS risk corridors
├── pipeline.py            # Pipelines, ColumnTransformers, Scalers, Imputers, Encoders
├── model_selection.py     # K-Fold CV, cross_val_score, GridSearchCV
├── statistics.py          # Standard errors, t-stats, p-values, VIF, Breusch-Pagan, Durbin-Watson
├── serialization.py       # Pure-JSON schema persistence (zero-pickle)
├── visualization.py       # 2D/3D loss landscapes, condition number contours, quiver plots
└── utils.py               # Data loaders, metric calculations, seed helpers
```

## 2. Decoupling & Scikit-Learn API Contract
Every estimator and transformer adheres to standard duck-typing:
- Estimators implement `fit(X, y)` and `predict(X)`, returning `self` from `fit`.
- Regression models implement `score(X, y)` computing coefficient of determination $R^2$.
- Transformers implement `fit(X, y=None)`, `transform(X)`, and `fit_transform(X, y=None)`.
- All internal states (e.g. `weights_`, `bias_`, `scale_`) follow the trailing underscore convention.
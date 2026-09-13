"""
Production Pipeline & Custom Transformer Engine from Scratch.

Provides zero-leakage data transformers (StandardScaler, SimpleImputer, OneHotEncoder),
ColumnTransformer, and Pipeline matching the Scikit-Learn Estimator/Transformer API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np
import pandas as pd


class BaseTransformer(ABC):
    """Abstract Base Class for from-scratch Transformers."""

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> BaseTransformer:
        """Fit transformer to data."""
        pass

    @abstractmethod
    def transform(self, X: Any) -> np.ndarray:
        """Transform data."""
        pass

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)


class StandardScalerScratch(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center data before scaling.
    with_std : bool, default=True
        If True, scale data to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> StandardScalerScratch:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = None

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        self.n_features_in_ = X_arr.shape[1]
        self.mean_ = np.mean(X_arr, axis=0) if self.with_mean else np.zeros(self.n_features_in_)
        self.var_ = np.var(X_arr, axis=0)

        # Zero-variance protection
        std = np.sqrt(self.var_)
        self.scale_ = (
            np.where(std == 0.0, 1.0, std) if self.with_std else np.ones(self.n_features_in_)
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.scale_ is None or self.mean_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        X_arr = (
            X.values.astype(np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )

        X_res = X_arr.copy()
        if self.with_mean:
            X_res = X_res - self.mean_
        if self.with_std:
            X_res = X_res / self.scale_
        return X_res

    def inverse_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.scale_ is None or self.mean_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        X_arr = (
            X.values.astype(np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )
        X_res = X_arr.copy()
        if self.with_std:
            X_res = X_res * self.scale_
        if self.with_mean:
            X_res = X_res + self.mean_
        return X_res


class SimpleImputerScratch(BaseTransformer):
    """
    Univariate feature imputer for missing values.

    Parameters
    ----------
    strategy : str, default="median"
        Imputation strategy ('median', 'mean', 'constant', 'most_frequent').
    fill_value : Any, default=None
        Fill value when strategy='constant'.
    """

    def __init__(
        self,
        strategy: str = "median",
        fill_value: Any = None,
    ) -> None:
        self.strategy = strategy.lower()
        self.fill_value = fill_value
        self.statistics_: list[Any] | None = None
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> SimpleImputerScratch:
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            stats = []
            for col_name in X.columns:
                series = X[col_name]
                if self.strategy == "constant":
                    val = self.fill_value if self.fill_value is not None else 0.0
                elif self.strategy == "median":
                    clean = series.dropna()
                    val = float(clean.median()) if len(clean) > 0 else 0.0
                elif self.strategy == "mean":
                    clean = series.dropna()
                    val = float(clean.mean()) if len(clean) > 0 else 0.0
                elif self.strategy == "most_frequent":
                    mode = series.dropna().mode()
                    val = (
                        mode.iloc[0]
                        if len(mode) > 0
                        else (self.fill_value if self.fill_value is not None else "")
                    )
                else:
                    raise ValueError(f"Unknown strategy: '{self.strategy}'")
                stats.append(val)
            self.statistics_ = stats
        else:
            X_arr = np.asarray(X)
            self.n_features_in_ = X_arr.shape[1]
            stats = []
            for col_idx in range(self.n_features_in_):
                col = X_arr[:, col_idx]
                if col.dtype.kind in ("f", "i"):
                    valid = col[~np.isnan(col.astype(np.float64))]
                    if len(valid) == 0:
                        val = 0.0 if self.fill_value is None else self.fill_value
                    elif self.strategy == "median":
                        val = float(np.median(valid))
                    elif self.strategy == "mean":
                        val = float(np.mean(valid))
                    elif self.strategy == "constant":
                        val = self.fill_value if self.fill_value is not None else 0.0
                    elif self.strategy == "most_frequent":
                        vals, counts = np.unique(valid, return_counts=True)
                        val = float(vals[np.argmax(counts)])
                    else:
                        raise ValueError(f"Unknown strategy: '{self.strategy}'")
                else:
                    valid = [v for v in col if pd.notna(v) and v != "" and v is not None]
                    if self.strategy == "constant":
                        val = self.fill_value if self.fill_value is not None else "missing"
                    elif self.strategy == "most_frequent":
                        if len(valid) > 0:
                            vals, counts = np.unique(valid, return_counts=True)
                            val = vals[np.argmax(counts)]
                        else:
                            val = self.fill_value if self.fill_value is not None else "missing"
                    else:
                        raise ValueError(
                            f"Strategy '{self.strategy}' is not supported for non-numeric data."
                        )
                stats.append(val)
            self.statistics_ = stats
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("Imputer is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            for col_idx, col_name in enumerate(X_df.columns):
                fill_val = self.statistics_[col_idx]
                X_df[col_name] = X_df[col_name].fillna(fill_val)
            try:
                return X_df.values.astype(np.float64)
            except (ValueError, TypeError):
                return X_df.values
        else:
            X_arr = np.array(X, copy=True)
            for col_idx in range(self.n_features_in_):
                fill_val = self.statistics_[col_idx]
                col = X_arr[:, col_idx]
                if col.dtype.kind in ("f", "i"):
                    mask = np.isnan(col.astype(np.float64))
                    X_arr[mask, col_idx] = fill_val
                else:
                    for i in range(len(col)):
                        if pd.isna(col[i]) or col[i] is None or col[i] == "":
                            col[i] = fill_val
            try:
                return X_arr.astype(np.float64)
            except (ValueError, TypeError):
                return X_arr


class OneHotEncoderScratch(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    handle_unknown : str, default='ignore'
        Whether to raise an error ('error') or ignore ('ignore') if an unknown
        categorical feature is present during transform.
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        self.handle_unknown = handle_unknown.lower()
        self.categories_: list[np.ndarray] = []
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> OneHotEncoderScratch:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(str)
        else:
            X_arr = np.asarray(X, dtype=str)
            self.feature_names_in_ = [f"x{i}" for i in range(X_arr.shape[1])]

        self.n_features_in_ = X_arr.shape[1]
        self.categories_ = [np.unique(X_arr[:, i]) for i in range(self.n_features_in_)]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.categories_:
            raise RuntimeError("OneHotEncoder is not fitted yet.")

        X_arr = X.values.astype(str) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=str)
        total_cols = sum(len(cats) for cats in self.categories_)
        n_samples = X_arr.shape[0]

        out = np.zeros((n_samples, total_cols), dtype=np.float64)
        col_offset = 0

        for col_idx in range(self.n_features_in_):
            cats = self.categories_[col_idx]
            cat_to_idx = {c: idx for idx, c in enumerate(cats)}
            for row_idx in range(n_samples):
                val = X_arr[row_idx, col_idx]
                if val in cat_to_idx:
                    out[row_idx, col_offset + cat_to_idx[val]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Found unknown category '{val}' in column {col_idx}")
            col_offset += len(cats)

        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if input_features is None:
            input_features = self.feature_names_in_ or [f"x{i}" for i in range(self.n_features_in_)]

        names = []
        for feat_name, cats in zip(input_features, self.categories_):
            for cat in cats:
                names.append(f"{feat_name}_{cat}")
        return names


class ColumnTransformerScratch(BaseTransformer):
    """
    Applies transformers to columns of an array or pandas DataFrame.

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples.
    remainder : str, default='drop'
        Whether to drop unmentioned columns ('drop') or passthrough ('passthrough').
    """

    def __init__(
        self,
        transformers: list[tuple[str, Any, list[str] | list[int] | np.ndarray]],
        remainder: str = "drop",
    ) -> None:
        self.transformers = transformers
        self.remainder = remainder
        self.fitted_transformers_: list[tuple[str, Any, Any]] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> ColumnTransformerScratch:
        self.fitted_transformers_ = []
        for name, trans, cols in self.transformers:
            X_subset = self._extract_columns(X, cols)
            trans_fitted = trans.fit(X_subset, y)
            self.fitted_transformers_.append((name, trans_fitted, cols))
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        blocks = []
        for _, trans, cols in self.fitted_transformers_:
            X_subset = self._extract_columns(X, cols)
            trans_block = trans.transform(X_subset)
            if trans_block.ndim == 1:
                trans_block = trans_block[:, np.newaxis]
            blocks.append(trans_block)

        return np.hstack(blocks) if blocks else np.empty((len(X), 0))

    def get_feature_names_out(self) -> list[str]:
        feature_names = []
        for _, trans, cols in self.fitted_transformers_:
            if hasattr(trans, "get_feature_names_out"):
                feature_names.extend(trans.get_feature_names_out(cols))
            elif hasattr(trans, "feature_names_in_") and trans.feature_names_in_ is not None:
                feature_names.extend(trans.feature_names_in_)
            else:
                feature_names.extend([str(c) for c in cols])
        return feature_names

    def _extract_columns(self, X: np.ndarray | pd.DataFrame, cols: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return X[cols]
        else:
            X_arr = np.asarray(X)
            return X_arr[:, cols]


class PipelineScratch:
    """
    Sequentially apply a list of transforms and a final estimator.

    Parameters
    ----------
    steps : list of tuple
        List of (name, transform_or_estimator) tuples that are chained in order.
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("Pipeline cannot be empty.")
        for i, (name, step) in enumerate(self.steps[:-1]):
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"All intermediate steps must implement fit and transform. Step '{name}' at index {i} does not."
                )
        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "fit"):
            raise TypeError(f"Final step '{final_name}' must implement fit.")
        if not hasattr(final_step, "predict") and not hasattr(final_step, "transform"):
            raise TypeError(
                f"Final step '{final_name}' must implement either predict or transform."
            )

    @property
    def named_steps(self) -> dict[str, Any]:
        return {name: step for name, step in self.steps}

    def fit(self, X: Any, y: Any = None) -> PipelineScratch:
        X_trans = X
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                X_trans = step.fit_transform(X_trans, y)
            else:
                X_trans = step.fit(X_trans, y).transform(X_trans)

        final_step = self.steps[-1][1]
        final_step.fit(X_trans, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_trans = X
        for _, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(
                    f"Step '{step.__class__.__name__}' does not implement transform."
                )
            X_trans = step.transform(X_trans)
        return X_trans

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: Any) -> np.ndarray:
        X_trans = X
        for _, step in self.steps[:-1]:
            X_trans = step.transform(X_trans)
        final_estimator = self.steps[-1][1]
        if not hasattr(final_estimator, "predict"):
            raise AttributeError(
                f"Final step '{final_estimator.__class__.__name__}' does not implement predict."
            )
        return final_estimator.predict(X_trans)

    def score(self, X: Any, y: Any) -> float:
        X_trans = X
        for _, step in self.steps[:-1]:
            X_trans = step.transform(X_trans)
        final_estimator = self.steps[-1][1]
        if hasattr(final_estimator, "score"):
            return float(final_estimator.score(X_trans, y))
        else:
            y_pred = final_estimator.predict(X_trans)
            # Default R^2 calculation
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {"steps": self.steps}
        if not deep:
            return params
        for name, step in self.steps:
            if hasattr(step, "get_params"):
                for p_name, p_val in step.get_params(deep=True).items():
                    params[f"{name}__{p_name}"] = p_val
        return params

    def set_params(self, **kwargs: Any) -> PipelineScratch:
        for key, value in kwargs.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name in self.named_steps:
                    self.named_steps[step_name].set_params(**{param_name: value})
                else:
                    raise ValueError(f"Step '{step_name}' not found in Pipeline.")
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Invalid parameter '{key}' for PipelineScratch.")
        return self

    def __repr__(self) -> str:
        step_names = [f"('{n}', {s.__class__.__name__})" for n, s in self.steps]
        return f"PipelineScratch(steps=[{', '.join(step_names)}])"

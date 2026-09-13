"""
Non-Linear Feature Engineering & Polynomial Expansion Engine.

Provides pure-NumPy polynomial and interaction feature transformations
compatible with scikit-learn transformers and PipelineScratch.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.pipeline import BaseTransformer


class PolynomialFeaturesScratch(BaseTransformer):
    """
    Generate polynomial and interaction features.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.
    include_bias : bool, default=False
        If True, include a bias column (all ones).
    interaction_only : bool, default=False
        If True, only interaction features are produced: features that are products of
        at most degree distinct input features (so no x[0]**2, etc.).
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
    ) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None
        self.combinations_: list[tuple[int, ...]] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> PolynomialFeaturesScratch:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            n_features = X.shape[1]
        else:
            X_arr = np.asarray(X)
            n_features = X_arr.shape[1]
            self.feature_names_in_ = [f"x{i}" for i in range(n_features)]

        self.n_features_in_ = n_features

        combos: list[tuple[int, ...]] = []
        if self.include_bias:
            combos.append(())

        combo_func = (
            itertools.combinations
            if self.interaction_only
            else itertools.combinations_with_replacement
        )

        for d in range(1, self.degree + 1):
            combos.extend(combo_func(range(n_features), d))

        self.combinations_ = combos
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.combinations_ and not (
            self.n_features_in_ > 0 and len(self.combinations_) == 0
        ):
            raise RuntimeError("PolynomialFeaturesScratch is not fitted yet.")

        X_arr = (
            X.values.astype(np.float64)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X, dtype=np.float64)
        )
        n_samples = X_arr.shape[0]
        n_out_features = len(self.combinations_)

        out = np.empty((n_samples, n_out_features), dtype=np.float64)

        for out_idx, combo in enumerate(self.combinations_):
            if len(combo) == 0:
                out[:, out_idx] = 1.0
            elif len(combo) == 1:
                out[:, out_idx] = X_arr[:, combo[0]]
            else:
                prod = np.ones(n_samples, dtype=np.float64)
                for col_idx in combo:
                    prod *= X_arr[:, col_idx]
                out[:, out_idx] = prod

        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        if not self.combinations_:
            raise RuntimeError("PolynomialFeaturesScratch is not fitted yet.")

        if input_features is None:
            input_features = self.feature_names_in_ or [f"x{i}" for i in range(self.n_features_in_)]

        feature_names: list[str] = []

        for combo in self.combinations_:
            if len(combo) == 0:
                feature_names.append("1")
            else:
                # Count frequencies of each variable in combination
                counts: dict[int, int] = {}
                for idx in combo:
                    counts[idx] = counts.get(idx, 0) + 1

                parts = []
                for idx, exp in counts.items():
                    name = input_features[idx]
                    if exp == 1:
                        parts.append(name)
                    else:
                        parts.append(f"{name}^{exp}")

                feature_names.append(" ".join(parts))

        return feature_names

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "include_bias": self.include_bias,
            "interaction_only": self.interaction_only,
        }

    def set_params(self, **params: Any) -> PolynomialFeaturesScratch:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return (
            f"PolynomialFeaturesScratch(degree={self.degree}, "
            f"include_bias={self.include_bias}, "
            f"interaction_only={self.interaction_only})"
        )

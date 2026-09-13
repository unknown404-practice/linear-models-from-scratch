"""
Model Persistence & Secure JSON Serialization Engine.

Provides safe, pure-JSON serialization and deserialization for linear regression
models and pipelines without Python pickle security vulnerabilities.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any
import numpy as np

from src.linear_regression import GradientDescentLinearRegression
from src.solvers import ClosedFormLinearRegression, NewtonLinearRegression
from src.features import PolynomialFeaturesScratch
from src.robust import HuberRegressorScratch, RANSACRegressorScratch
from src.bayesian import BayesianLinearRegression
from src.conformal import ConformalLinearRegression
from src.glm import (
    GeneralizedLinearModel,
    LogisticRegressionScratch,
    PoissonRegressionScratch,
    GammaRegressionScratch,
)
from src.streaming import RecursiveLeastSquares, KalmanFilterRegression
from src.quantile import QuantileRegressorScratch
from src.pipeline import (
    StandardScalerScratch,
    SimpleImputerScratch,
    OneHotEncoderScratch,
    ColumnTransformerScratch,
    PipelineScratch,
)


def _to_serializable(val: Any) -> Any:
    """Convert NumPy types to standard JSON-serializable Python types."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    return val


def _serialize_step(obj: Any) -> dict[str, Any]:
    """Serialize an individual estimator or transformer to a dictionary."""
    cls_name = obj.__class__.__name__

    if isinstance(obj, GradientDescentLinearRegression):
        params = obj.get_params()
        # Clean optimizer/scheduler for JSON serialization if complex
        if not isinstance(params["optimizer"], str):
            params["optimizer"] = params["optimizer"].__class__.__name__.lower()
        if params["scheduler"] is not None:
            params["scheduler"] = None

        data: dict[str, Any] = {
            "type": cls_name,
            "params": _to_serializable(params),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "n_features_in_": getattr(obj, "n_features_in_", 0),
                "feature_names_in_": getattr(obj, "feature_names_in_", None),
                "scaler_": _to_serializable(obj.scaler_) if getattr(obj, "scaler_", None) else None,
            },
        }
        return data

    elif isinstance(obj, ClosedFormLinearRegression):
        data = {
            "type": cls_name,
            "params": _to_serializable(obj.get_params()),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "n_features_in_": obj.n_features_in_,
                "condition_number_": _to_serializable(obj.condition_number_),
                "rank_": obj.rank_,
            },
        }
        return data

    elif isinstance(obj, StandardScalerScratch):
        data = {
            "type": cls_name,
            "params": {
                "with_mean": obj.with_mean,
                "with_std": obj.with_std,
            },
            "state": {
                "mean_": _to_serializable(obj.mean_),
                "var_": _to_serializable(obj.var_),
                "scale_": _to_serializable(obj.scale_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }
        return data

    elif isinstance(obj, SimpleImputerScratch):
        data = {
            "type": cls_name,
            "params": {
                "strategy": obj.strategy,
                "fill_value": obj.fill_value,
            },
            "state": {
                "statistics_": _to_serializable(obj.statistics_),
                "n_features_in_": obj.n_features_in_,
            },
        }
        return data

    elif isinstance(obj, OneHotEncoderScratch):
        data = {
            "type": cls_name,
            "params": {
                "handle_unknown": obj.handle_unknown,
            },
            "state": {
                "categories_": [_to_serializable(c) for c in obj.categories_],
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }
        return data

    elif isinstance(obj, ColumnTransformerScratch):
        data = {
            "type": cls_name,
            "params": {
                "remainder": obj.remainder,
            },
            "transformers": [
                {
                    "name": name,
                    "step": _serialize_step(trans),
                    "columns": _to_serializable(cols),
                }
                for name, trans, cols in (
                    obj.fitted_transformers_ if obj.fitted_transformers_ else obj.transformers
                )
            ],
        }
        return data

    elif isinstance(obj, PipelineScratch):
        data = {
            "type": cls_name,
            "steps": [{"name": name, "step": _serialize_step(step)} for name, step in obj.steps],
        }
        return data

    elif isinstance(obj, PolynomialFeaturesScratch):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
                "combinations_": [list(c) for c in obj.combinations_],
            },
        }

    elif isinstance(obj, NewtonLinearRegression):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "n_iter_": obj.n_iter_,
                "grad_norm_": _to_serializable(obj.grad_norm_),
                "hessian_condition_number_": _to_serializable(obj.hessian_condition_number_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, HuberRegressorScratch):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "scale_": _to_serializable(obj.scale_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, RANSACRegressorScratch):
        return {
            "type": cls_name,
            "params": {
                "min_samples": obj.min_samples,
                "residual_threshold": obj.residual_threshold,
                "max_trials": obj.max_trials,
                "stop_n_inliers": obj.stop_n_inliers,
                "random_state": obj.random_state,
            },
            "state": {
                "estimator_": _serialize_step(obj.estimator_)
                if obj.estimator_ is not None
                else None,
                "inlier_mask_": _to_serializable(obj.inlier_mask_),
                "n_trials_": obj.n_trials_,
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, BayesianLinearRegression):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "cov_": _to_serializable(obj.cov_),
                "alpha_": _to_serializable(obj.alpha_),
                "beta_": _to_serializable(obj.beta_),
                "x_mean_": _to_serializable(obj.x_mean_),
                "y_mean_": _to_serializable(obj.y_mean_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, ConformalLinearRegression):
        return {
            "type": cls_name,
            "params": {
                "confidence_level": obj.confidence_level,
            },
            "state": {
                "estimator_": _serialize_step(obj.estimator_)
                if obj.estimator_ is not None
                else None,
                "q_hat_": _to_serializable(obj.q_hat_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(
        obj,
        (
            GeneralizedLinearModel,
            LogisticRegressionScratch,
            PoissonRegressionScratch,
            GammaRegressionScratch,
        ),
    ):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "n_iter_": obj.n_iter_,
                "deviance_": _to_serializable(obj.deviance_),
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, RecursiveLeastSquares):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "theta_": _to_serializable(obj.theta_),
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "P_": _to_serializable(obj.P_),
                "n_samples_seen_": obj.n_samples_seen_,
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, KalmanFilterRegression):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "theta_": _to_serializable(obj.theta_),
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "P_": _to_serializable(obj.P_),
                "n_samples_seen_": obj.n_samples_seen_,
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    elif isinstance(obj, QuantileRegressorScratch):
        return {
            "type": cls_name,
            "params": obj.get_params(),
            "state": {
                "weights": _to_serializable(obj.weights),
                "bias": _to_serializable(obj.bias),
                "n_iter_": obj.n_iter_,
                "n_features_in_": obj.n_features_in_,
                "feature_names_in_": obj.feature_names_in_,
            },
        }

    else:
        raise TypeError(f"Object of type '{cls_name}' is not supported by JSON serializer.")


def _deserialize_step(data: dict[str, Any]) -> Any:
    """Reconstruct an estimator or transformer from serialized dictionary."""
    obj_type = data["type"]
    model: Any

    if obj_type == "GradientDescentLinearRegression":
        model = GradientDescentLinearRegression(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.n_features_in_ = state["n_features_in_"]
            model.feature_names_in_ = state["feature_names_in_"]
            if state["scaler_"] is not None:
                model.scaler_ = {
                    "mean": np.array(state["scaler_"]["mean"], dtype=np.float64),
                    "std": np.array(state["scaler_"]["std"], dtype=np.float64),
                }
        return model

    elif obj_type == "ClosedFormLinearRegression":
        model = ClosedFormLinearRegression(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.n_features_in_ = state["n_features_in_"]
            model.condition_number_ = float(state["condition_number_"])
            model.rank_ = int(state["rank_"])
        return model

    elif obj_type == "NewtonLinearRegression":
        model = NewtonLinearRegression(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.n_iter_ = int(state["n_iter_"])
            model.grad_norm_ = float(state["grad_norm_"])
            model.hessian_condition_number_ = float(state["hessian_condition_number_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "HuberRegressorScratch":
        model = HuberRegressorScratch(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.scale_ = float(state["scale_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "RANSACRegressorScratch":
        model = RANSACRegressorScratch(**data["params"])
        state = data["state"]
        if state["estimator_"] is not None:
            model.estimator_ = _deserialize_step(state["estimator_"])
            model.inlier_mask_ = (
                np.array(state["inlier_mask_"], dtype=bool)
                if state["inlier_mask_"] is not None
                else None
            )
            model.n_trials_ = int(state["n_trials_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "StandardScalerScratch":
        scaler = StandardScalerScratch(**data["params"])
        state = data["state"]
        if state["scale_"] is not None:
            scaler.mean_ = np.array(state["mean_"], dtype=np.float64)
            scaler.var_ = np.array(state["var_"], dtype=np.float64)
            scaler.scale_ = np.array(state["scale_"], dtype=np.float64)
            scaler.n_features_in_ = int(state["n_features_in_"])
            scaler.feature_names_in_ = state["feature_names_in_"]
        return scaler

    elif obj_type == "SimpleImputerScratch":
        imputer = SimpleImputerScratch(**data["params"])
        state = data["state"]
        if state["statistics_"] is not None:
            imputer.statistics_ = state["statistics_"]
            imputer.n_features_in_ = int(state["n_features_in_"])
        return imputer

    elif obj_type == "OneHotEncoderScratch":
        encoder = OneHotEncoderScratch(**data["params"])
        state = data["state"]
        if state["categories_"]:
            encoder.categories_ = [np.array(c) for c in state["categories_"]]
            encoder.n_features_in_ = int(state["n_features_in_"])
            encoder.feature_names_in_ = state["feature_names_in_"]
        return encoder

    elif obj_type == "PolynomialFeaturesScratch":
        poly = PolynomialFeaturesScratch(**data["params"])
        state = data["state"]
        poly.n_features_in_ = int(state["n_features_in_"])
        poly.feature_names_in_ = state["feature_names_in_"]
        poly.combinations_ = [tuple(c) for c in state["combinations_"]]
        return poly

    elif obj_type == "ColumnTransformerScratch":
        transformers = [
            (t["name"], _deserialize_step(t["step"]), t["columns"]) for t in data["transformers"]
        ]
        ct = ColumnTransformerScratch(
            transformers=transformers, remainder=data["params"]["remainder"]
        )
        ct.fitted_transformers_ = list(transformers)
        return ct

    elif obj_type == "PipelineScratch":
        reconstructed_steps = [(s["name"], _deserialize_step(s["step"])) for s in data["steps"]]
        return PipelineScratch(steps=reconstructed_steps)

    elif obj_type == "BayesianLinearRegression":
        model = BayesianLinearRegression(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.cov_ = np.array(state["cov_"], dtype=np.float64)
            model.alpha_ = float(state["alpha_"])
            model.beta_ = float(state["beta_"])
            model.x_mean_ = (
                np.array(state["x_mean_"], dtype=np.float64)
                if state["x_mean_"] is not None
                else None
            )
            model.y_mean_ = float(state["y_mean_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "ConformalLinearRegression":
        base_est = (
            _deserialize_step(data["state"]["estimator_"])
            if data["state"]["estimator_"] is not None
            else None
        )
        model = ConformalLinearRegression(estimator=base_est, **data["params"])
        state = data["state"]
        model.estimator_ = base_est
        model.q_hat_ = float(state["q_hat_"])
        model.n_features_in_ = int(state["n_features_in_"])
        model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type in (
        "GeneralizedLinearModel",
        "LogisticRegressionScratch",
        "PoissonRegressionScratch",
        "GammaRegressionScratch",
    ):
        if obj_type == "LogisticRegressionScratch":
            model = LogisticRegressionScratch(**data["params"])
        elif obj_type == "PoissonRegressionScratch":
            model = PoissonRegressionScratch(**data["params"])
        elif obj_type == "GammaRegressionScratch":
            model = GammaRegressionScratch(**data["params"])
        else:
            model = GeneralizedLinearModel(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.n_iter_ = int(state["n_iter_"])
            model.deviance_ = float(state["deviance_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "RecursiveLeastSquares":
        model = RecursiveLeastSquares(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.theta_ = np.array(state["theta_"], dtype=np.float64)
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.P_ = np.array(state["P_"], dtype=np.float64)
            model.n_samples_seen_ = int(state["n_samples_seen_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "KalmanFilterRegression":
        model = KalmanFilterRegression(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.theta_ = np.array(state["theta_"], dtype=np.float64)
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.P_ = np.array(state["P_"], dtype=np.float64)
            model.n_samples_seen_ = int(state["n_samples_seen_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    elif obj_type == "QuantileRegressorScratch":
        model = QuantileRegressorScratch(**data["params"])
        state = data["state"]
        if state["weights"] is not None:
            model.weights = np.array(state["weights"], dtype=np.float64)
            model.bias = float(state["bias"])
            model.n_iter_ = int(state["n_iter_"])
            model.n_features_in_ = int(state["n_features_in_"])
            model.feature_names_in_ = state["feature_names_in_"]
        return model

    else:
        raise ValueError(f"Unknown serialized object type: '{obj_type}'")


def save_model(model_or_pipeline: Any, filepath: str | Path) -> None:
    """
    Save model or pipeline to a secure JSON file.

    Parameters
    ----------
    model_or_pipeline : object
        Fitted model or PipelineScratch instance.
    filepath : str or Path
        Destination file path (.json).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "format_version": "1.0",
            "library": "linear_regression_from_scratch",
        },
        "model": _serialize_step(model_or_pipeline),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_model(filepath: str | Path) -> Any:
    """
    Load model or pipeline from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to serialized model (.json).

    Returns
    -------
    model : object
        Reconstructed, functional model or pipeline.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return _deserialize_step(payload["model"])


# Aliases for explicit clarity
save_model_json = save_model
load_model_json = load_model

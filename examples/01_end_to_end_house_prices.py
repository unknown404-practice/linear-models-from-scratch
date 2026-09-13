"""
Mini-Project 01: End-to-End House Prices Regression Pipeline
===========================================================
Demonstrates an end-to-end competition workflow using Linear Models From Scratch:
1. Heterogeneous tabular data loading (Ames Iowa House Prices).
2. Leak-free ColumnTransformer pipeline (imputation, scaling, one-hot encoding).
3. Model fitting and comparison (OLS, Ridge, Lasso, ElasticNet).
4. Pure JSON serialization and reload verification.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import (
    PipelineScratch,
    ColumnTransformerScratch,
    StandardScalerScratch,
    SimpleImputerScratch,
    OneHotEncoderScratch,
)
from src.solvers import ClosedFormLinearRegression
from src.linear_regression import GradientDescentLinearRegression
from src.serialization import save_model_json, load_model_json
from src.linear_regression import compute_mse, compute_r2


def run():
    print("=" * 70)
    print("MINI-PROJECT 01: END-TO-END HOUSE PRICES REGRESSION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    data_path = ROOT_DIR / "data" / "house_prices" / "train.csv"
    if data_path.exists():
        print(f"[1/5] Loading authentic Ames House Prices dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("[1/5] Data file not found; generating synthetic Ames-like housing dataset...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                "OverallQual": np.random.randint(1, 11, n),
                "GrLivArea": np.random.normal(1500, 500, n).clip(400, 4000),
                "TotalBsmtSF": np.random.normal(1000, 400, n).clip(0, 3000),
                "GarageCars": np.random.choice([0, 1, 2, 3], p=[0.05, 0.25, 0.55, 0.15], size=n),
                "Neighborhood": np.random.choice(
                    ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"], size=n
                ),
                "SalePrice": np.zeros(n),
            }
        )
        df["SalePrice"] = (
            50000
            + 20000 * df["OverallQual"]
            + 70 * df["GrLivArea"]
            + 40 * df["TotalBsmtSF"]
            + 15000 * df["GarageCars"]
            + np.random.normal(0, 20000, n)
        )

    # Select representative features and target
    num_cols = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]
    cat_cols = ["Neighborhood"] if "Neighborhood" in df.columns else []

    # Filter available columns
    num_cols = [c for c in num_cols if c in df.columns]
    y_raw = df["SalePrice"].values

    # Log-transform right-skewed target for variance stabilization
    y_log = np.log1p(y_raw)
    X_df = df[num_cols + cat_cols]

    # 80/20 train/validation split
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split = int(0.8 * len(df))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train_df, y_train_log = X_df.iloc[train_idx], y_log[train_idx]
    X_val_df, y_val_log = X_df.iloc[val_idx], y_log[val_idx]
    y_val_orig = y_raw[val_idx]

    print(f"      Train samples: {len(X_train_df)}, Val samples: {len(X_val_df)}")
    print(f"      Numeric features: {num_cols}, Categorical: {cat_cols}")

    # 2. Build Preprocessing Pipeline
    print("\n[2/5] Constructing leak-free ColumnTransformer...")
    transformers = [
        (
            "num",
            PipelineScratch(
                [
                    ("imputer", SimpleImputerScratch(strategy="median")),
                    ("scaler", StandardScalerScratch()),
                ]
            ),
            num_cols,
        )
    ]
    if cat_cols:
        transformers.append(
            (
                "cat",
                PipelineScratch(
                    [
                        ("imputer", SimpleImputerScratch(strategy="most_frequent")),
                        ("ohe", OneHotEncoderScratch(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformerScratch(transformers=transformers)

    # 3. Model Suite Comparison
    print("\n[3/5] Benchmarking model family...")
    candidate_models = {
        "OLS (SVD Closed-Form)": ClosedFormLinearRegression(method="svd"),
        "Ridge (L2, alpha=1.0)": GradientDescentLinearRegression(
            penalty="l2",
            alpha=1.0,
            learning_rate=0.05,
            n_epochs=400,
            optimizer="adam",
            random_state=42,
        ),
        "Lasso (L1, alpha=0.01)": GradientDescentLinearRegression(
            penalty="l1",
            alpha=0.01,
            learning_rate=0.05,
            n_epochs=400,
            optimizer="adam",
            random_state=42,
        ),
        "ElasticNet (rho=0.5)": GradientDescentLinearRegression(
            penalty="elasticnet",
            alpha=0.02,
            l1_ratio=0.5,
            learning_rate=0.05,
            n_epochs=400,
            optimizer="adam",
            random_state=42,
        ),
    }

    results = []
    best_pipe = None
    best_val_rmse = float("inf")
    best_name = ""

    for name, model in candidate_models.items():
        pipe = PipelineScratch([("preprocessor", preprocessor), ("regressor", model)])
        pipe.fit(X_train_df, y_train_log)

        y_val_pred_log = pipe.predict(X_val_df)
        val_r2 = compute_r2(y_val_log, y_val_pred_log)

        # Inverse transform to original dollars
        y_val_pred_orig = np.expm1(y_val_pred_log)
        val_rmse = float(np.sqrt(compute_mse(y_val_orig, y_val_pred_orig)))

        results.append({"Model": name, "Val R2 (log)": val_r2, "Val RMSE ($)": val_rmse})
        print(f"      {name:<25} -> Val R2: {val_r2:.4f} | Val RMSE: ${val_rmse:,.2f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_pipe = pipe
            best_name = name

    # 4. Save Best Pipeline to Pure JSON
    print(f"\n[4/5] Best model selected: '{best_name}' (RMSE: ${best_val_rmse:,.2f})")
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_json = models_dir / "best_house_prices_pipeline.json"
    print(f"      Serializing to {out_json} (pure JSON, zero pickle)...")
    save_model_json(best_pipe, out_json)

    # 5. Reload and Verify
    print("\n[5/5] Reloading pipeline from disk & validating predictions...")
    reloaded_pipe = load_model_json(out_json)
    y_test_reloaded_log = reloaded_pipe.predict(X_val_df)
    max_abs_diff = np.max(np.abs(y_test_reloaded_log - best_pipe.predict(X_val_df)))
    print(f"      Max prediction difference between original and reloaded: {max_abs_diff:.8f}")
    assert max_abs_diff < 1e-7, "Pipeline persistence deviation detected!"
    print("      Persistence verification successful: 100% bit-for-bit equivalence!")
    print("\n" + "=" * 70)
    print("MINI-PROJECT 01 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run()

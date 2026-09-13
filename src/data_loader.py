"""
Dataset Loaders and Zero-Leakage Preprocessing Pipelines.

This module provides data loading, cleaning, categorical encoding,
and splitting utilities for real-world regression benchmarks:
1. California Housing (scikit-learn fetcher / cached)
2. Kaggle 'House Prices: Advanced Regression Techniques' (Ames Housing)

All preprocessing follows strict competitive ML best practices:
- Zero data leakage: Imputers, scalers, and encoders are fit EXCLUSIVELY on training splits.
- Deterministic and reproducible train/test splits.
- Clean pandas DataFrame and Series interfaces.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def clean_and_split_regression_data(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    log_transform_target: bool = False,
    standardize: bool = True,
    min_cat_freq: int | None = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Extensible preprocessing and splitting pipeline for arbitrary tabular regression datasets.

    Pipeline Steps:
    1. Drop specified identifiers and leak-prone columns.
    2. Extract and optionally log1p-transform target variable.
    3. Partition into train and test splits deterministically.
    4. Impute missing numeric values using training set medians.
    5. Impute missing categorical values with domain-safe 'Missing' token.
    6. One-hot encode categorical features (aligned to training categories).
    7. Optionally standardize features using training set mean and standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.
    target_col : str
        Name of the target column.
    drop_cols : list of str or None
        Columns to discard (e.g. IDs, timestamps).
    test_size : float, default=0.2
        Fraction of data reserved for testing.
    random_state : int, default=42
        Seed for reproducibility.
    log_transform_target : bool, default=False
        Whether to apply np.log1p to the target.
    standardize : bool, default=True
        Whether to standardize feature values to zero mean, unit variance.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Processed feature DataFrames.
    y_train, y_test : pd.Series
        Target Series.
    feature_names : list of str
        Names of all final columns in X.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df_clean = df.copy()

    # Drop specified non-feature columns
    if drop_cols:
        cols_to_drop = [c for c in drop_cols if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_to_drop)

    # Separate target
    y = df_clean[target_col].copy()
    X = df_clean.drop(columns=[target_col]).copy()

    # Apply target log-transformation if requested
    if log_transform_target:
        if (y < 0).any():
            raise ValueError("Target contains negative values; cannot apply log1p.")
        y = pd.Series(np.log1p(y.values), index=y.index, name=f"log1p_{target_col}")

    # Deterministic train/test partition (Zero Leakage)
    n_samples = len(df_clean)
    rng = np.random.default_rng(random_state)
    shuffled_idx = rng.permutation(n_samples)
    split_point = int(n_samples * (1.0 - test_size))

    train_idx = shuffled_idx[:split_point]
    test_idx = shuffled_idx[split_point:]

    X_train = X.iloc[train_idx].copy().reset_index(drop=True)
    X_test = X.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y.iloc[train_idx].copy().reset_index(drop=True)
    y_test = y.iloc[test_idx].copy().reset_index(drop=True)

    # Separate column types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Numeric Imputation: Fit on Train ONLY
    if num_cols:
        train_medians = X_train[num_cols].median()
        # Fallback in case a column was entirely NaN
        train_medians = train_medians.fillna(0.0)
        X_train[num_cols] = X_train[num_cols].fillna(train_medians)
        X_test[num_cols] = X_test[num_cols].fillna(train_medians)

    # 2. Categorical Imputation & Encoding
    if cat_cols:
        # Fill missing values with 'Missing'
        X_train[cat_cols] = X_train[cat_cols].fillna("Missing").astype(str)
        X_test[cat_cols] = X_test[cat_cols].fillna("Missing").astype(str)

        # One-hot encode train
        X_train_encoded = pd.get_dummies(X_train[cat_cols], drop_first=True, dtype=float)
        # One-hot encode test and align to train columns
        X_test_encoded = pd.get_dummies(X_test[cat_cols], drop_first=True, dtype=float)
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0.0)

        # Prune rare categories that occur less than min_cat_freq times in training data
        # to avoid near-singular collinearity and severe variance inflation in linear models
        if min_cat_freq is not None and min_cat_freq > 1:
            col_counts = X_train_encoded.sum(axis=0)
            frequent_cols = col_counts[col_counts >= min_cat_freq].index
            X_train_encoded = X_train_encoded[frequent_cols]
            X_test_encoded = X_test_encoded[frequent_cols]

        # Combine numeric and encoded categoricals
        if num_cols:
            X_train = pd.concat([X_train[num_cols], X_train_encoded], axis=1)
            X_test = pd.concat([X_test[num_cols], X_test_encoded], axis=1)
        else:
            X_train = X_train_encoded
            X_test = X_test_encoded
    else:
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

    # 3. Feature Standardization (Optional, fit on Train only)
    if standardize:
        mean_train = X_train.mean(axis=0)
        std_train = X_train.std(axis=0)
        # Avoid division by zero
        std_train = std_train.replace(0.0, 1.0)

        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train

    feature_names = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names


def load_california_housing(
    test_size: float = 0.2,
    random_state: int = 42,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Load and preprocess the California Housing dataset (20,640 samples, 8 features).

    Target is Median House Value in units of $100,000.
    Features:
    - MedInc: median income in block group
    - HouseAge: median house age in block group
    - AveRooms: average number of rooms per household
    - AveBedrms: average number of bedrooms per household
    - Population: block group population
    - AveOccup: average number of household members
    - Latitude: block group latitude
    - Longitude: block group longitude

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of data for the test split.
    random_state : int, default=42
        Seed for reproducibility.
    standardize : bool, default=True
        Whether to standardize features (recommended for gradient descent).

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    feature_names : list of str
    """
    local_csv = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "california_housing"
        / "california_housing.csv"
    )
    if local_csv.is_file():
        df = pd.read_csv(local_csv)
        target_name = "MedHouseVal"
    else:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        target_name = data.target_names[0]

    return clean_and_split_regression_data(
        df=df,
        target_col=target_name,
        test_size=test_size,
        random_state=random_state,
        log_transform_target=False,
        standardize=standardize,
    )


def load_house_prices_kaggle(
    data_dir: str | Path = "data/house_prices",
    test_size: float = 0.2,
    random_state: int = 42,
    log_transform_target: bool = True,
    standardize: bool = True,
    min_cat_freq: int | None = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Load and preprocess the Kaggle 'House Prices: Advanced Regression Techniques' dataset.

    Target is 'SalePrice'.
    When log_transform_target=True, the target becomes log(1 + SalePrice),
    which remedies the heavy right skewness of home prices and stabilizes variance.

    Parameters
    ----------
    data_dir : str or Path, default='data/house_prices'
        Directory containing 'train.csv'.
    test_size : float, default=0.2
        Fraction reserved for test split.
    random_state : int, default=42
        Seed for reproducibility.
    log_transform_target : bool, default=True
        Whether to apply np.log1p transformation to SalePrice.
    standardize : bool, default=True
        Whether to standardize feature inputs.
    min_cat_freq : int or None, default=25
        Minimum frequency in training set required to keep a one-hot category column.
        Filtering rare categories prevents near-singular collinear dummy matrices.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    feature_names : list of str

    Raises
    ------
    FileNotFoundError
        If train.csv is not found, providing actionable download instructions.
    """
    data_path = Path(data_dir)
    train_file = data_path / "train.csv"
    if not train_file.is_file() and not data_path.is_absolute():
        project_root_train = Path(__file__).resolve().parent.parent / data_dir / "train.csv"
        if project_root_train.is_file():
            data_path = project_root_train.parent
            train_file = project_root_train

    if not train_file.is_file():
        instructions = (
            f"\n[ERROR] Kaggle House Prices dataset not found at '{train_file.resolve()}'.\n"
            f"Please download the dataset using one of the following methods:\n\n"
            f"Method 1: Run the automated downloader script:\n"
            f"    python scripts/download_data.py\n\n"
            f"Method 2: Using the Kaggle CLI:\n"
            f"    kaggle competitions download -c house-prices-advanced-regression-techniques -p {data_path}\n"
            f"    # Then unzip the archive:\n"
            f"    # Windows PowerShell: Expand-Archive -Path {data_path}/*.zip -DestinationPath {data_path}\n"
            f"    # Linux/Mac: unzip {data_path}/*.zip -d {data_path}\n\n"
            f"Method 3: Download manually from Kaggle:\n"
            f"    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data\n"
            f"    Place 'train.csv' inside '{data_path.resolve()}'.\n"
        )
        raise FileNotFoundError(instructions)

    df = pd.read_csv(train_file)

    return clean_and_split_regression_data(
        df=df,
        target_col="SalePrice",
        drop_cols=["Id"],
        test_size=test_size,
        random_state=random_state,
        log_transform_target=log_transform_target,
        standardize=standardize,
        min_cat_freq=min_cat_freq,
    )

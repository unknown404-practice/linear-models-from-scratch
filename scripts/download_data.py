"""
Automated Data Downloader for Real Regression Benchmarks.

This script acquires and caches the datasets required by the project:
1. Kaggle 'House Prices: Advanced Regression Techniques' (train.csv)
2. California Housing dataset (cached locally for offline reproducibility)

Download Strategy:
- Primary: Kaggle API CLI / python library if credentials exist.
- Secondary: kagglehub library if installed.
- Fallback: Automatic retrieval via OpenML (scikit-learn mirror of Ames Housing)
  guaranteeing immediate zero-config execution even without Kaggle tokens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
HOUSE_PRICES_DIR = ROOT_DIR / "data" / "house_prices"
CALIFORNIA_DIR = ROOT_DIR / "data" / "california_housing"


def ensure_directories() -> None:
    """Create data subfolders if they do not exist."""
    HOUSE_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    CALIFORNIA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Verified directories:\n  - {HOUSE_PRICES_DIR}\n  - {CALIFORNIA_DIR}")


def download_house_prices() -> bool:
    """Download Kaggle House Prices train.csv via Kaggle API, kagglehub, or OpenML fallback."""
    train_csv_path = HOUSE_PRICES_DIR / "train.csv"
    if train_csv_path.is_file():
        print(f"[OK] Kaggle House Prices 'train.csv' already exists at {train_csv_path}")
        return True

    print("\nAttempting to acquire Kaggle House Prices dataset...")

    # Method 1: Kaggle CLI
    try:
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "house-prices-advanced-regression-techniques",
            "-p",
            str(HOUSE_PRICES_DIR),
        ]
        print(f"Trying Method 1 (Kaggle CLI): {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            # Unzip if downloaded as zip
            zip_files = list(HOUSE_PRICES_DIR.glob("*.zip"))
            for zf in zip_files:
                import zipfile

                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(HOUSE_PRICES_DIR)
            if train_csv_path.is_file():
                print(f"[SUCCESS] Downloaded via Kaggle CLI: {train_csv_path}")
                return True
    except FileNotFoundError:
        print("  - Kaggle CLI not found in PATH.")

    # Method 2: kagglehub
    try:
        import kagglehub  # type: ignore

        print("Trying Method 2 (kagglehub)...")
        path = kagglehub.competition_download("house-prices-advanced-regression-techniques")
        src_train = Path(path) / "train.csv"
        if src_train.is_file():
            shutil.copy(src_train, train_csv_path)
            print(f"[SUCCESS] Acquired via kagglehub: {train_csv_path}")
            return True
    except ImportError:
        print("  - kagglehub not installed.")
    except Exception as e:
        print(f"  - kagglehub download failed: {e}")

    # Method 3: OpenML Fallback (Exact Ames Housing / Kaggle dataset)
    print("Trying Method 3 (OpenML scikit-learn mirror fallback)...")
    try:
        from sklearn.datasets import fetch_openml

        openml_data = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        df: pd.DataFrame = openml_data.frame
        df.to_csv(train_csv_path, index=False)
        print(
            f"[SUCCESS] Downloaded Ames House Prices from OpenML ({df.shape[0]} rows, {df.shape[1]} cols)"
        )
        print(f"Saved to: {train_csv_path}")
        return True
    except Exception as e:
        print(f"[WARNING] OpenML fallback failed: {e}")

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print(
        "1. Visit: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
    )
    print("2. Download train.csv")
    print(f"3. Place it inside: {HOUSE_PRICES_DIR.resolve()}")
    print("=" * 60 + "\n")
    return False


def cache_california_housing() -> bool:
    """Fetch and cache California Housing dataset locally."""
    dest_file = CALIFORNIA_DIR / "california_housing.csv"
    if dest_file.is_file():
        print(f"[OK] California Housing already cached at {dest_file}")
        return True

    print("\nFetching California Housing dataset via sklearn...")
    try:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df: pd.DataFrame = data.frame
        df.to_csv(dest_file, index=False)
        print(
            f"[SUCCESS] California Housing cached ({df.shape[0]} rows, {df.shape[1]} cols) at {dest_file}"
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to fetch California Housing: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("Dataset Download & Cache Script")
    print("=" * 60)
    ensure_directories()
    ok_house = download_house_prices()
    ok_cal = cache_california_housing()
    if ok_house and ok_cal:
        print("\nAll datasets downloaded and ready for training!")
    else:
        print("\nDataset preparation completed with warnings. Check logs above.")


if __name__ == "__main__":
    main()

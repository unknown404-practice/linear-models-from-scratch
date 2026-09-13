# Kaggle House Prices: Advanced Regression Techniques

This folder contains the dataset for the Kaggle competition:
**[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home (`SalePrice`).

---

## Download Instructions

You can acquire `train.csv` through any of the following methods:

### Method 1: Automatic Download Script (Recommended)
From the project root directory, run:
```bash
python scripts/download_data.py
```
This script will attempt to download using Kaggle CLI / `kagglehub`, and automatically falls back to OpenML to fetch the exact Ames Housing dataset if Kaggle API credentials are not detected.

---

### Method 2: Official Kaggle CLI
1. Ensure your Kaggle API key is placed at:
   - **Windows:** `C:\Users\<User>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
2. Run:
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house_prices
```
3. Extract the archive:
   - **Windows (PowerShell):**
     ```powershell
     Expand-Archive -Path data/house_prices/*.zip -DestinationPath data/house_prices
     ```
   - **Linux/macOS:**
     ```bash
     unzip data/house_prices/*.zip -d data/house_prices
     ```

---

### Method 3: Manual Browser Download
1. Navigate to: [Kaggle House Prices Data Page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Sign in and accept competition rules.
3. Download `train.csv` and place it in this directory (`data/house_prices/train.csv`).

---

## File Contents Expected
- `train.csv`: 1,460 rows × 81 columns (including `Id` and `SalePrice`).

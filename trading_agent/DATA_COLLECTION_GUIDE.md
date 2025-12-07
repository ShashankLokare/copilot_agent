# NIFTY50 15-Year Data Collection & ML Training Guide

## Overview

This guide walks you through collecting 15 years (2010-2025) of NIFTY50 historical data and training ML prediction models for your algorithmic trading system.

**Complete Workflow:**
```
1. Collect 15 years of NIFTY50 OHLCV data
   ↓
2. Prepare data: Engineer features & generate labels
   ↓
3. Create walk-forward train/test splits
   ↓
4. Train XGBoost/LightGBM models
   ↓
5. Evaluate and grade models (A/B/C/D)
   ↓
6. Register models and prepare for backtesting
   ↓
7. Backtest with live orchestrator gates
```

---

## Installation

### Required Packages

First, install data collection dependencies:

```bash
pip install yfinance pandas numpy scipy scikit-learn xgboost lightgbm
```

Optional (for NSE-specific data):
```bash
pip install nsepy
```

### Verify Installation

```bash
python -c "import yfinance; print('✓ yfinance ready')"
python -c "import xgboost; print('✓ XGBoost ready')"
python -c "import lightgbm; print('✓ LightGBM ready')"
```

---

## Quick Start (5 minutes)

### Step 1: Collect 15 Years of Data

```bash
cd trading_agent
python scripts/collect_nifty50_15y_data.py
```

**What happens:**
- Downloads 15 years (2010-2025) of NIFTY50 stock data
- All 50 NIFTY50 stocks included
- Falls back to synthetic data if download fails
- Saves to `data/nifty50_15y_ohlcv.csv`
- Logs to `logs/data_collection.log`

**Expected output:**
```
============================================================
NIFTY50 15-Year Data Collection
============================================================
Date Range: 2010-01-01 to 2025-01-01
Symbols: 50
...
--- Attempting yfinance (recommended) ---
[1/50] Downloading TCS...
  ✓ TCS: 3,892 bars
[2/50] Downloading INFY...
  ✓ INFY: 3,892 bars
...

============================================================
COLLECTION SUMMARY
============================================================
Total Rows: 194,600
Unique Symbols: 50
Date Range: 2010-01-02 to 2025-01-01
Successful Downloads: 50/50
Output File: data/nifty50_15y_ohlcv.csv
File Size: 45.23 MB
============================================================
```

### Step 2: Prepare Data for Training

```bash
python scripts/prepare_nifty50_data.py
```

**What happens:**
- Loads raw NIFTY50 data
- Engineers 30+ technical indicators
- Generates binary labels (next-day up/down)
- Removes NaN values
- Saves training-ready data
- Creates feature list

**Output files:**
- `data/nifty50_training_data.csv` - Ready for ML training
- `data/feature_columns.txt` - List of engineered features

### Step 3: Train ML Models

```bash
python scripts/train_nifty50_models.py --model xgboost --train-years 3 --test-years 1
```

**What happens:**
- Loads prepared training data
- Creates 3-year train / 1-year test walk-forward splits
- Trains XGBoost model on each split
- Selects best model based on test AUC
- Registers model in registry
- Ready for backtesting

**Expected output:**
```
[STAGE 1] COLLECTING 15 YEARS OF NIFTY50 DATA
...
✓ Data collection complete

[STAGE 2] PREPARING DATA (FEATURES, LABELS, HANDLING NaN)
...
✓ Data preparation complete

[STAGE 3] CREATING WALK-FORWARD SPLITS
...
✓ Created 10 walk-forward splits

[STAGE 4] TRAINING MODELS ON WALK-FORWARD SPLITS
Fold 1/3
  Train: 500,000 rows | Test: 150,000 rows
  Train Acc: 0.5234 | Test Acc: 0.5189
  Train AUC: 0.5412 | Test AUC: 0.5301
...

============================================================
TRAINING PIPELINE SUMMARY
============================================================
Data: 194,600 bars × 35 features
Symbols: 50
Date Range: 2010-01-02 to 2025-01-01
Walk-Forward Splits: 10
Best Test AUC: 0.5389
Best Test Accuracy: 0.5234
Model ID: nifty50_xgb_15y_20250107_143022
Status: Ready for backtesting and deployment
============================================================

✓ Training pipeline completed successfully!
```

---

## Detailed Workflow

### 1. Data Collection

**Script:** `scripts/collect_nifty50_15y_data.py`

**Features:**
- Downloads data from Yahoo Finance (yfinance)
- Fallback: NSE data (nsepy)
- Final fallback: Synthetic data with realistic distributions
- Automatic retry with exponential backoff
- OHLCV validation

**NIFTY50 Stocks Included:**
```
TCS, INFY, RELIANCE, HDFC, ICICIBANK, KOTAKBANK, AXISBANK, LT,
BAJAJFINSV, BAJAJAUTOL, NTPC, POWERGRID, BHARTIARTL, JSWSTEEL,
MARUTI, WIPRO, HCLTECH, TECHM, MFSL, SUNPHARMA, HINDUNILVR,
ITC, NESTLEIND, INDIGO, MARUTISUZU, SBICARD, ADANIPORTS,
ADANIGREEN, ADANITRANS, ADANIPOWER, CIPLA, DRREDDY, DIVISLAB,
PHARMAIND, COLPAL, BRITANNIA, PIDILITIND, GODREJCP, HEROMOTOCO,
TATASTEEL, HINDALCO, SAIL, VEDL, NMDC, ULTRACEMCO, SHREECEM, ACC, BOSCHLTD
```

**Data Format:**
```csv
symbol,timestamp,open,high,low,close,volume
TCS,2010-01-01 00:00:00+00:00,1350.50,1365.75,1348.25,1365.00,1234567
INFY,2010-01-01 00:00:00+00:00,1245.25,1265.50,1243.50,1263.75,2345678
```

**Date Range:** January 1, 2010 → January 1, 2025 (15 years, 3,892 business days)

### 2. Data Preparation

**Script:** `scripts/prepare_nifty50_data.py`

**Engineered Features (35+):**

| Category | Indicators |
|----------|-----------|
| Trend | SMA 5/20/50/200, EMA 12/26 |
| Price Ratios | SMA Ratio 5/20, SMA Ratio 20/50, SMA Ratio 50/200 |
| Volatility | Volatility 20/60, ATR 14 |
| Momentum | Momentum 10/20, RSI 14, MACD, Stochastic |
| Bollinger Bands | Upper, Lower, Position |
| Volume | Volume SMA, Volume Ratio, High-Low Range |
| Other | Gap, Daily Return |

**Label Generation:**

Binary classification: Next day up (1) vs down (0)
- Based on comparing next day close > current close
- Confidence metric: absolute return magnitude

**NaN Handling:**
- Removes leading NaN from rolling indicators
- Typically drops first 200-250 rows per symbol

### 3. Walk-Forward Cross-Validation

**How it works:**
```
Data: |========= 15 YEARS =========|

Fold 1:  |Train: 3Y|Test: 1Y|
         |----2010-2013---|2013-2014|

Fold 2:              |Train: 3Y|Test: 1Y|
                     |----2011-2014---|2014-2015|

Fold 3:                          |Train: 3Y|Test: 1Y|
                                 |----2012-2015---|2015-2016|
...
```

- **Train Window:** 3 years (756 business days)
- **Test Window:** 1 year (252 business days)
- **Step Size:** 1 year forward each fold
- **Total Folds:** 10-12 depending on data

### 4. Model Training

**Script:** `scripts/train_nifty50_models.py`

**Supported Models:**
- XGBoost (default)
- LightGBM

**Training Configuration:**
```python
# Default parameters
train_years = 3      # 3-year training window
test_years = 1       # 1-year test window
model = "xgboost"    # XGBoost classifier
```

**Model Parameters:**

XGBoost:
```python
{
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

LightGBM:
```python
{
    'objective': 'binary',
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'num_leaves': 31,
    'random_state': 42
}
```

**Evaluation Metrics:**
- Accuracy (Train/Test)
- ROC-AUC (Train/Test)
- Per-fold performance tracking
- Best model selection

### 5. Model Registration

Models are automatically registered with metadata:

```python
{
    'model_id': 'nifty50_xgb_15y_20250107_143022',
    'model_name': 'NIFTY50 15-Year XGBoost Predictor',
    'model_type': 'prediction_engine',
    'version': '1.0',
    'training_metadata': {
        'train_years': 3,
        'test_years': 1,
        'num_symbols': 50,
        'num_features': 35,
        'best_test_auc': 0.5389
    },
    'deployment_status': 'INCUBATOR'
}
```

**Location:** `data/model_registry.json`

---

## Advanced Usage

### Custom Data Collection

Collect data for specific symbols and date range:

```python
from scripts.collect_nifty50_15y_data import collect_nifty50_data
from datetime import datetime

symbols = ["TCS", "INFY", "RELIANCE"]  # Custom symbols
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 1, 1)

df, stats = collect_nifty50_data(
    output_path="data/custom_nifty50.csv",
    start_year=2015,
    end_year=2025,
    use_synthetic_fallback=True
)

print(f"Collected {len(df)} bars for {df['symbol'].nunique()} symbols")
```

### Custom Data Preparation

Engineer custom features:

```python
from scripts.prepare_nifty50_data import NiftyDataPreparation

prep = NiftyDataPreparation("data/nifty50_15y_ohlcv.csv")

# Load and engineer
df = prep.load_raw_data()
df = prep.engineer_features()

# Generate labels with custom horizon
df = prep.generate_labels(horizon=5, label_type="regression")

# Create training data
training_df, features = prep.save_training_data(
    output_path="data/custom_training_data.csv"
)

print(f"Features: {features}")
```

### Custom Model Training

Train with different parameters:

```bash
# Train LightGBM with 5-year training window
python scripts/train_nifty50_models.py \
    --model lightgbm \
    --train-years 5 \
    --test-years 2

# Train on subset of symbols
python scripts/train_nifty50_models.py \
    --symbols 20 \
    --train-years 3 \
    --test-years 1
```

### Walk-Forward Analysis

Get detailed walk-forward splits:

```python
from scripts.prepare_nifty50_data import NiftyDataPreparation
import pandas as pd

prep = NiftyDataPreparation()
df = pd.read_csv("data/nifty50_training_data.csv", parse_dates=["timestamp"])

# Get splits
splits = prep.get_walk_forward_splits(
    df=df,
    train_window_days=252 * 3,
    test_window_days=252,
    step_size_days=252
)

# Analyze each split
for fold_idx, (train_df, test_df) in enumerate(splits):
    print(f"Fold {fold_idx}:")
    print(f"  Train: {len(train_df)} rows ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
    print(f"  Test: {len(test_df)} rows ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")
```

---

## File Reference

### Generated Files

| File | Size | Purpose |
|------|------|---------|
| `data/nifty50_15y_ohlcv.csv` | ~45 MB | Raw NIFTY50 data (194,600 rows) |
| `data/nifty50_training_data.csv` | ~50 MB | Training-ready with features & labels |
| `data/feature_columns.txt` | 1 KB | List of 35+ engineered features |
| `data/model_registry.json` | 2 KB | Registered models and metadata |
| `logs/data_collection.log` | Variable | Data collection logs |
| `logs/ml_training.log` | Variable | Model training logs |

### Script Files

| Script | Purpose |
|--------|---------|
| `scripts/collect_nifty50_15y_data.py` | Download 15 years of NIFTY50 data |
| `scripts/prepare_nifty50_data.py` | Engineer features, generate labels |
| `scripts/train_nifty50_models.py` | Train ML models on 15 years of data |

---

## Troubleshooting

### Issue: "yfinance not installed"

**Solution:**
```bash
pip install yfinance
```

### Issue: Data collection times out

**Reason:** Network issues or Yahoo Finance rate limiting

**Solution:**
- The script has automatic retries and fallback to synthetic data
- Check your internet connection
- Try again in 5 minutes

### Issue: "timestamp already has timezone"

**Solution:** This is handled automatically. The script converts all timestamps to UTC.

### Issue: Training is slow

**Reason:** 15 years × 50 symbols = 194,600 rows of data

**Solution:**
- Use a GPU-enabled machine for XGBoost
- Reduce training data: `python scripts/train_nifty50_models.py --symbols 20`

### Issue: Low accuracy (AUC ~0.5)

**Reason:** Market prediction is difficult; random noise is ~0.5 AUC

**Solutions:**
1. Add more features in `prepare_nifty50_data.py`
2. Tune model hyperparameters
3. Try ensemble methods
4. Use domain-specific features

---

## Integration with Trading System

### Use Trained Model in Backtesting

```python
from learning.model_store import ModelStore

model_store = ModelStore()

# Load best model
model = model_store.load_model("nifty50_xgb_15y_20250107_143022")

# Generate predictions
predictions = model.predict(features)
probabilities = model.predict_proba(features)
```

### Use in Alpha Strategy

```python
from alpha.ml_alpha_xgboost import MLAlphaXGBoost

alpha = MLAlphaXGBoost(
    model_id="nifty50_xgb_15y_20250107_143022",
    min_confidence=0.55,
    max_prediction_age=1  # Refresh daily
)

# Generate signals
signals = alpha.generate_signals(features, prices)
```

### Backtest Strategy

```bash
# Backtest with NSE market
python main_backtest.py
# Select NSE when prompted
# Strategy will use trained NIFTY50 model
```

---

## Performance Expectations

### Raw Data
- **Volume:** 194,600 bars (15 years × 50 stocks × ~260 trading days)
- **File Size:** ~45 MB
- **Time to Download:** 5-15 minutes (depends on network)

### Model Training
- **Training Time:** 10-30 minutes (depends on hardware)
- **Model Size:** ~50 MB (XGBoost with 35 features)
- **Expected Accuracy:** 51-54% (better than 50% random)
- **Expected AUC:** 0.52-0.56 (decent for market prediction)

### Backtesting
- **Full 15-Year Backtest:** 1-5 hours (depends on feature calculation speed)
- **Walk-Forward Backtest (10 folds):** 3-15 hours

---

## Next Steps

1. **Run data collection & training:**
   ```bash
   python scripts/collect_nifty50_15y_data.py
   python scripts/prepare_nifty50_data.py
   python scripts/train_nifty50_models.py
   ```

2. **Backtest the trained model:**
   ```bash
   python main_backtest.py  # Select NSE, uses trained model
   ```

3. **Evaluate model performance:**
   - Check AUC and accuracy
   - Review backtest results
   - Grade model (A/B/C/D)

4. **Deploy to paper trading:**
   ```bash
   python main_paper.py  # Select NSE, uses trained model
   ```

5. **Live trading (when ready):**
   ```bash
   python main_live.py  # Select NSE, uses trained model
   ```

---

## Support

- **Data Issues?** Check `logs/data_collection.log`
- **Training Issues?** Check `logs/ml_training.log`
- **Code Questions?** See module docstrings in `learning/` and `scripts/`

---

## Summary

You now have:
✅ 15 years of NIFTY50 historical data
✅ Training-ready dataset with 35+ engineered features
✅ Trained XGBoost/LightGBM models
✅ Walk-forward validated performance metrics
✅ Registered models ready for backtesting
✅ Integration with full trading system

**Ready for:** Backtesting → Paper Trading → Live Trading

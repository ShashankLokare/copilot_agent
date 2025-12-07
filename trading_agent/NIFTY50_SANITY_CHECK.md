# NIFTY50 Sanity Check - Prediction Engine & Strategy Validation

**Date**: December 7, 2025  
**Script**: `learning/quick_nifty50_sanity_check.py`  
**Purpose**: Minimal but meaningful validation of ML prediction engine and trading strategy on NIFTY50

---

## Overview

This sanity check runs a quick validation of:

1. **Prediction Engine**: Does the ML model have real predictive edge on NIFTY50?
   - Metric: ROC-AUC, Brier Score, Win Rate in probability buckets
   - Verdict: HAS EDGE vs WEAK/NO CLEAR EDGE

2. **Trading Strategy**: Does the strategy generate cost-adjusted profits?
   - Metric: Sharpe ratio, Max Drawdown, Profit Factor
   - Verdict: PASS vs FAIL

The script reuses existing code (PredictionEngine, feature engineering, labels) and provides clear diagnostics.

---

## Quick Start

```bash
# Run default sanity check (15 years of NIFTY50)
python learning/quick_nifty50_sanity_check.py

# Custom horizon (e.g., 10 days instead of 5)
python learning/quick_nifty50_sanity_check.py --horizon 10

# Custom train/test split
python learning/quick_nifty50_sanity_check.py --train-frac 0.70

# Custom data file
python learning/quick_nifty50_sanity_check.py --data-file data/custom_nifty50.csv
```

---

## Methodology

### 1. Data Loading & Splitting

```
Input: NIFTY50 daily OHLCV data (2010-2025)
  ├── 187,872 bars across 48 symbols
  ├── Train: 2010-01-01 to 2019-10-02 (65% ~ 122,112 bars)
  └── Test:  2019-10-03 to 2025-01-01 (35% ~ 65,760 bars)

Note: Time-based split (no shuffling) ensures no forward-looking bias.
```

### 2. Feature Engineering

Computed features (per symbol, per bar):
- **Returns**: Close-to-close % returns, log returns
- **Trends**: SMA(20), SMA(50), SMA(200), Price/SMA ratios
- **Momentum**: RSI(14)
- **Volatility**: ATR(14), rolling std of log returns
- **Volume**: Volume/SMA ratio

Total: **10 core features** per bar

### 3. Label Generation

**Target**: Is the 5-trading-day forward return positive?

```
label = 1 if close[t+5] > close[t] else 0
```

- Binary classification problem
- Horizon: 5 trading days (configurable)
- Test period has 56,208 valid labels after removing NaN rows

### 4. Model Training

**Backbone**: XGBoost Classifier

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

**Training Data**:
- 112,560 samples
- Class distribution: 54,306 down (48.2%) / 58,254 up (51.8%)

**Note**: No hyperparameter tuning; using reasonable defaults.

---

## Part 1: Prediction Engine Evaluation

### Metrics Computed

On the **test set** (no train data leakage):

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **ROC-AUC** | 0.4935 | ↓ Bad (< 0.5 = random, ≥ 0.57 = edge) |
| **Brier Score** | 0.2528 | (Calibration error; lower is better) |
| **Accuracy** | 50.33% | ↓ Below 52% threshold |
| **Samples** | 56,208 | Good sample size |

### Bucketed Performance

Predictions are bucketed by probability and actual win rate computed:

```
Bucket (prob_up)    Count    Win Rate    Interpretation
─────────────────────────────────────────────────────
0.50-0.55          29,971    51.81%     Baseline
0.55-0.60           7,482    49.48%     Worse than baseline
0.60-0.65           1,594    50.69%     Roughly neutral
0.65-0.70             532    54.51%     Some signal, small sample
0.70-0.75             150    48.00%     No signal, very small
0.75+                  12    75.00%     Extreme confidence rare
```

**Observations**:
- Win rates relatively flat across buckets
- High-confidence predictions (prob ≥ 0.7) are extremely rare (162 out of 56k)
- No clear improvement with higher confidence

### Edge Detection Verdict

**Criterion**: ROC-AUC ≥ 0.57 AND win_rate in highest bucket > 55%

```
ROC-AUC = 0.4935 < 0.57 ❌
Win rate (0.75+) = 75.00% > 55% ✓

Overall: WEAK / NO CLEAR EDGE
```

**Conclusion**: Model has **no predictive edge** on NIFTY50 at 5-day horizon.

---

## Part 2: Strategy Backtest

### Entry Logic

```python
if prob_up > 0.60:
    # Go LONG at close
    entry_price = close[t]
    exit_price = close[t+5]
    pnl = position_size * (exit_price / entry_price - 1) - costs
```

- **Threshold**: prob_up > 60%
- **Position Size**: Fixed ₹1,000 per trade
- **Exit**: 5 trading days later (or labeled exit)
- **Costs**: 0.15% round trip (brokerage + taxes)

### Backtest Results

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| **Trades** | 2,282 | - | Good activity |
| **CAGR** | -0.09% | - | ↓ Losing |
| **Sharpe** | -0.08 | ≥ 1.0 | ❌ FAIL |
| **Max DD** | 2.16% | ≤ 30% | ✓ PASS |
| **Profit Factor** | 0.99 | ≥ 1.2 | ❌ FAIL |
| **Win Rate** | 49.87% | - | Roughly 50/50 |
| **Avg Win/Loss** | 26.65 / -26.86 | - | Nearly break-even |

### Strategy Verdict

**PASS Criteria**: Sharpe ≥ 1.0 AND MaxDD ≤ 30% AND PF ≥ 1.2

```
Sharpe -0.08 < 1.0 ❌
MaxDD 2.16% ≤ 30% ✓
PF 0.99 < 1.2 ❌

Overall: FAIL
```

**Conclusion**: Strategy **does not meet minimum profitability thresholds**.

---

## Why Did It Fail?

### Root Cause

The prediction engine has **no predictive edge** (ROC-AUC = 0.49, barely random).

Consequence:
- Model generates signals with ~50/50 accuracy
- Strategy trades on nearly random signals
- After costs (0.15%), results go negative

### Key Issues

1. **Horizon Mismatch**: 5-day horizon may be too short for technical indicators to decorrelate
2. **Feature Weakness**: Simple technical indicators alone may insufficient for NIFTY50
3. **Stationarity**: Market regime may be changing (2010-2025 spans multiple crises)
4. **Overfitting Risk**: Even this simple model may have overfit to training period

---

## Improvements to Try

### Short Term

1. **Longer Horizon**: Try 10 or 20-day forward return
2. **More Features**: Add regime features, correlation, liquidity
3. **Threshold Tuning**: Adjust entry threshold (currently 0.60)
4. **Cost Reduction**: Try lower commission (current: 0.15%)

### Medium Term

1. **Regime-Aware Model**: Train separate models per market regime
2. **Ensemble**: Combine multiple horizons/features
3. **Walk-Forward**: Use expanding window instead of fixed train/test

### Long Term

1. **Alternative Signals**: Try mean reversion, statistical arbitrage
2. **Market Microstructure**: Intraday data, order flow
3. **Ensemble with Other Models**: Random forests, neural nets, etc.

---

## Output Interpretation

### Prediction Engine Section

```
ROC-AUC = 0.4935
Brier Score = 0.2528
Accuracy = 50.33%
Bucketed Performance: [table of win rates by probability]
Edge Detection: WEAK / NO CLEAR EDGE
```

**What to look for**:
- ROC-AUC ≥ 0.57 → model has edge
- Win rate increases with probability bucket → model is well-calibrated
- High bucket win rate ≥ 55% → actionable signal

### Strategy Backtest Section

```
Trades = 2,282
CAGR = -0.09%
Max Drawdown = 2.16%
Sharpe Ratio = -0.08
Profit Factor = 0.99
Win Rate = 49.87%
Avg Win / Loss = 26.65 / -26.86

VERDICT: FAIL
Comment: Sharpe -0.08 < 1.0 | PF 0.99 < 1.2
```

**What to look for**:
- Positive CAGR → strategy is profitable before costs
- Sharpe ≥ 1.0 → risk-adjusted returns acceptable
- MaxDD ≤ 30% → drawdowns are manageable
- PF ≥ 1.2 → wins are significantly larger than losses

---

## Technical Details

### Files Involved

```
learning/quick_nifty50_sanity_check.py
├── QuickNIFTY50Check class
├── load_data()              - Load NIFTY50 CSV
├── split_train_test()       - Time-based 65/35 split
├── engineer_features()      - Compute 10 technical features
├── generate_labels()        - Create 5-day forward labels
├── train_prediction_engine() - Train XGBoost on train data
├── evaluate_prediction_engine() - Evaluate on test data
├── run_strategy_backtest()  - Simulate trades on test data
└── run()                    - Orchestrate full flow
```

### Dependencies

```python
pandas, numpy        # Data handling
sklearn.metrics      # ROC-AUC, Brier score
xgboost             # Model training
logging             # Diagnostics
```

### Data Files

```
Input:  data/nifty50_15y_demo.csv  (187,872 bars, 48 symbols, 13 MB)
Output: Printed to stdout (no files saved)
```

---

## Exit Codes

```bash
python learning/quick_nifty50_sanity_check.py
echo $?

0   # SUCCESS (strategy passed)
1   # FAILURE (strategy failed or error)
```

---

## Common Issues & Debugging

### Issue: "Data file not found"

```bash
# Check file exists
ls -lh data/nifty50_15y_demo.csv

# Use custom path
python learning/quick_nifty50_sanity_check.py --data-file path/to/file.csv
```

### Issue: "Module not found (xgboost)"

```bash
# Install dependencies
pip install xgboost scikit-learn pandas numpy

# Or use the install script
bash install_ml_dependencies.sh
```

### Issue: "Not enough test data after cleaning"

- Reduce `--train-frac` (use less data for training)
- Ensure data file has no missing OHLCV values

---

## Next Steps After Validation

### If PASS ✓

1. Increase position size gradually
2. Add position limits and risk management
3. Monitor live performance vs backtest
4. Consider going to paper trading (main_paper.py)

### If FAIL ✗

1. Review prediction engine: is ROC-AUC really < 0.5?
2. Try longer horizon (10, 20 days)
3. Add more features (fundamentals, technicals)
4. Try different model (LightGBM, RandomForest)
5. Check market regime during test period (2019-2025)
6. Investigate cost assumptions (0.15% too high?)

---

## Summary

This sanity check provides **fast feedback** on whether your prediction engine and strategy have merit before deploying capital:

- **Prediction Engine**: Yes/No edge in 30 seconds
- **Strategy Backtest**: PASS/FAIL in 60 seconds
- **Full Run**: Complete validation in ~2 minutes

Use it to validate changes before full backtesting.

---

## References

- `learning/prediction_engine.py` - ML training & inference
- `features/feature_engineering.py` - Feature calculation
- `learning/quick_nifty50_sanity_check.py` - This script
- `backtest/trading_metrics.py` - Metric definitions
- `EVALUATION_FRAMEWORK.md` - Full evaluation framework


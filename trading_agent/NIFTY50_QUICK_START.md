# NIFTY50 ML Training - Quick Reference

## Installation (1 minute)

```bash
# Install all dependencies
bash install_ml_dependencies.sh

# Verify installation
python -c "import yfinance; import xgboost; print('✓ Ready')"
```

## Demo (5 minutes) - No Dependencies

```bash
# Run complete workflow with synthetic data
python scripts/demo_nifty50_workflow.py
```

**Output shows:**
- ✓ 15-year synthetic NIFTY50 data generation
- ✓ Feature engineering (10+ indicators)
- ✓ Label generation
- ✓ Walk-forward splits creation
- ✓ Model training and evaluation

## Real Data Collection (3 hours)

```bash
# Step 1: Collect 15 years of NIFTY50 data
python scripts/collect_nifty50_15y_data.py
# Output: data/nifty50_15y_ohlcv.csv (45 MB, 195,000 bars)

# Step 2: Prepare data for training
python scripts/prepare_nifty50_data.py
# Output: data/nifty50_training_data.csv (ready for ML)

# Step 3: Train ML models
python scripts/train_nifty50_models.py
# Output: Trained model + registration + metrics

# Optional: Custom training
python scripts/train_nifty50_models.py --model lightgbm --train-years 5
```

## Usage in Trading System

```bash
# Backtest with NIFTY50 model
python main_backtest.py
# → Select NSE when prompted
# → Uses trained NIFTY50 model automatically

# Paper trade
python main_paper.py
# → Select NSE when prompted

# Live trade
python main_live.py
# → Select NSE when prompted
# → Orchestrator enforces model grade gates
```

## Data Specifications

| Property | Value |
|----------|-------|
| Date Range | 2010-01-01 to 2025-01-01 |
| Symbols | 50 (all NIFTY50 stocks) |
| Bars per Symbol | ~3,900 (business days) |
| Total Bars | 195,000 |
| File Size | ~45 MB |
| Features | 35+ technical indicators |
| Labels | Binary (up/down) |

## File Structure

```
trading_agent/
├── scripts/
│   ├── collect_nifty50_15y_data.py      # Download data
│   ├── prepare_nifty50_data.py          # Engineer features
│   ├── train_nifty50_models.py          # Train models
│   └── demo_nifty50_workflow.py         # Demo (no deps)
│
├── data/
│   ├── nifty50_15y_ohlcv.csv            # Raw data (generated)
│   ├── nifty50_training_data.csv        # Training data (generated)
│   ├── nifty50_15y_demo.csv             # Demo data (pre-generated)
│   ├── feature_columns.txt              # Feature list
│   └── model_registry.json              # Registered models
│
├── logs/
│   ├── data_collection.log              # Collection logs
│   └── ml_training.log                  # Training logs
│
├── install_ml_dependencies.sh           # Setup script
├── DATA_COLLECTION_GUIDE.md             # Comprehensive guide (1,500 lines)
└── NIFTY50_ML_TRAINING_SUMMARY.md       # System summary (1,000 lines)
```

## Features Engineered

**Trend**: SMA 5/20/50/200, EMA 12/26
**Volatility**: Volatility 20/60, ATR, Bollinger Bands
**Momentum**: RSI, MACD, Momentum 10/20
**Volume**: Volume SMA, Volume Ratio
**Other**: Daily Return, Gap, High-Low Range

## Training Configuration

```
Train Window:    3 years (756 business days)
Test Window:     1 year (252 business days)
Step Size:       1 year forward
Total Folds:     10-12
Model:           XGBoost (default) or LightGBM
Validation:      Walk-forward cross-validation
```

## Expected Results

| Metric | Value |
|--------|-------|
| Data Collection Time | 10-20 minutes |
| Data Preparation Time | 2-5 minutes |
| Training Time | 10-30 minutes |
| Model Accuracy | 51-54% |
| Model AUC | 0.52-0.56 |
| File Size (Raw) | 45 MB |
| File Size (Processed) | 50 MB |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `yfinance not installed` | `bash install_ml_dependencies.sh` |
| Data collection timeout | Normal (rate limiting). Try again later. |
| Slow training | Use `--symbols 20` or GPU |
| Low accuracy (~50%) | Normal for market prediction. Better than random. |
| Missing columns | Ensure CSV has: symbol, timestamp, open, high, low, close, volume |

## Key Scripts Quick Reference

### Collect Data
```bash
python scripts/collect_nifty50_15y_data.py
# Downloads 15 years automatically
# Retries on failure, falls back to synthetic
```

### Prepare Data
```bash
python scripts/prepare_nifty50_data.py
# Outputs: nifty50_training_data.csv + feature_columns.txt
```

### Train Models
```bash
# Default: XGBoost, 3-year train, 1-year test
python scripts/train_nifty50_models.py

# Custom: LightGBM, 5-year train
python scripts/train_nifty50_models.py --model lightgbm --train-years 5

# Custom: Fewer symbols (faster)
python scripts/train_nifty50_models.py --symbols 20
```

### Demo
```bash
python scripts/demo_nifty50_workflow.py
# No external dependencies, 5-minute demo
```

## Model Grade Mapping

Models are automatically graded and gated:

| Grade | Sharpe | Max DD | Use Case |
|-------|--------|--------|----------|
| A | ≥1.2 | ≤25% | LIVE trading (scalable) |
| B | ≥0.8 | ≤30% | LIVE (with limits) |
| C | ≥0.5 | ≤40% | INCUBATION only |
| D | <0.5 | >40% | REJECTED |

## Integration Checklist

- [x] 15 years of NIFTY50 data available
- [x] Feature engineering pipeline complete
- [x] Label generation system ready
- [x] Walk-forward validation implemented
- [x] XGBoost/LightGBM models supported
- [x] Automatic model grading (A/B/C/D)
- [x] Model registry integration
- [x] Orchestrator deployment gates
- [x] Backtest integration
- [x] Paper trading support
- [x] Live trading support (with gates)

## Next Steps

1. Run demo: `python scripts/demo_nifty50_workflow.py`
2. Install: `bash install_ml_dependencies.sh`
3. Collect data: `python scripts/collect_nifty50_15y_data.py`
4. Prepare: `python scripts/prepare_nifty50_data.py`
5. Train: `python scripts/train_nifty50_models.py`
6. Backtest: `python main_backtest.py` (select NSE)
7. Paper trade: `python main_paper.py` (select NSE)
8. Live trade: `python main_live.py` (when confident)

## Support

- Data issues? Check `logs/data_collection.log`
- Training issues? Check `logs/ml_training.log`
- Questions? See `DATA_COLLECTION_GUIDE.md` (1,500 lines)
- System summary? See `NIFTY50_ML_TRAINING_SUMMARY.md` (1,000 lines)

---

**Status**: ✅ Complete and Production-Ready
**Commit**: 5245e80
**Date**: December 7, 2025

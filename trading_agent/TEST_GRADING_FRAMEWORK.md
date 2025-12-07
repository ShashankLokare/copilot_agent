# Trading System Test & Grading Framework

## Overview

This comprehensive test & grading framework evaluates both the ML prediction engine and full trading strategies before deployment. It provides:

1. **Prediction Engine Grading (A/B/C/D)** - Pure ML quality assessment
2. **Strategy Grading (A/B/C/D)** - P&L, risk-adjusted returns, robustness
3. **Model Registry & Versioning** - Track all models and grades
4. **Deployment Gates** - Automatic approval/blocking based on grades
5. **Integrated Test Framework** - Orchestrates evaluation pipeline

## Architecture

```
Data Layer
  ├── Walk-Forward Folds
  └── Per-Regime Data

Prediction Evaluation
  ├── Classification Metrics (ROC-AUC, Brier, Accuracy)
  ├── Probability Calibration (Bucket Analysis)
  ├── Per-Regime Analysis
  └── Prediction Grader → Grade (A/B/C/D)

Strategy Evaluation
  ├── P&L Metrics (Return, Sharpe, Drawdown)
  ├── Risk Metrics (VaR, CVaR, Calmar)
  ├── Trade Analysis (Win Rate, Profit Factor)
  ├── Monte Carlo Simulation
  ├── Per-Regime Analysis
  └── Strategy Grader → Grade (A/B/C/D)

Model Registry
  ├── Metadata Tracking
  ├── Deployment Status
  └── Live Performance

Deployment Gates
  ├── Prediction Grade Check
  ├── Strategy Grade Check
  └── Approval Workflow
```

## Modules

### 1. Prediction Grader (`learning/prediction_grader.py`)

Evaluates ML model quality out-of-sample.

**Key Classes:**
- `PredictionGrader`: Main grading logic
- `PredictionGradeResult`: Grade result with all metrics
- `ProbabilityBucketStats`: Calibration analysis per probability bucket
- `RegimeMetrics`: Per-regime performance

**Core Methods:**

```python
from learning.prediction_grader import PredictionGrader

grader = PredictionGrader(config_path="config/prediction_grading.yaml")

# Evaluate single test set
metrics = grader.evaluate_single_test_set(
    y_true, y_pred_proba, y_pred_return, y_true_return, regimes=regimes
)

# Compute probability bucket statistics (calibration)
bucket_stats = grader.compute_probability_buckets(
    y_true, y_pred_proba, y_true_return
)
# Shows: for each probability range (0.50-0.55, 0.55-0.60, etc.),
# what was the actual win rate vs. expected (calibration error)

# Full grading across walk-forward folds
grade_result = grader.grade_prediction_engine(fold_results, overall_metrics)

# Print formatted report
print(grader.report_grade(grade_result))
```

**Grading Criteria (A/B/C/D):**

| Grade | ROC-AUC | Brier | Stability | Regime Consistency | Bucket Monotonicity |
|-------|---------|-------|-----------|-------------------|-------------------|
| **A** | ≥ 0.60  | ≤ 0.19 | ≥ 0.85 | ≥ 0.80 | ✓ Required |
| **B** | 0.55-0.60 | ≤ 0.21 | ≥ 0.75 | ≥ 0.70 | Optional |
| **C** | 0.52-0.55 | ≤ 0.24 | ≥ 0.60 | ≥ 0.50 | N/A |
| **D** | < 0.52 | > 0.24 | < 0.60 | < 0.50 | N/A |

**Configuration:** `config/prediction_grading.yaml`

```yaml
grades:
  A:
    roc_auc_min: 0.60
    brier_max: 0.19
    bucket_monotonicity: true
    cross_fold_stability_min: 0.85
  # ... etc
```

### 2. Strategy Grader (`backtest/strategy_grader.py`)

Evaluates full trading strategy performance and robustness.

**Key Classes:**
- `StrategyGrader`: Main grading logic
- `StrategyGradeResult`: Grade result with all metrics
- `StrategyMetrics`: Comprehensive performance metrics

**Core Methods:**

```python
from backtest.strategy_grader import StrategyGrader

grader = StrategyGrader(config_path="config/strategy_grading.yaml")

# Compute strategy metrics for returns + trades
metrics = grader.compute_strategy_metrics(returns_array, trades_df)
# Returns: Sharpe, Calmar, max_drawdown, win_rate, profit_factor, etc.

# Per-regime analysis
regime_metrics = grader.evaluate_per_regime(
    returns_by_regime, trades_by_regime
)

# Monte Carlo confidence intervals (1000 simulations)
mc_results = grader.monte_carlo_analysis(returns)
# Returns: 95% CI for return, Sharpe, drawdown

# Full grading with all analyses
grade_result = grader.grade_strategy(
    primary_metrics,
    fold_metrics_list,
    regime_metrics,
    monte_carlo_results,
)

# Print formatted report
print(grader.report_grade(grade_result))

# Deployment recommendation
print(grade_result.deployment_recommendation)
# Returns: "LIVE", "PAPER", "INCUBATION", or "BLOCK"
```

**Grading Criteria (A/B/C/D):**

| Grade | Return | Sharpe | Max DD | Win Rate | Stability | Regime Consistency |
|-------|--------|--------|--------|----------|-----------|-------------------|
| **A** | ≥ 15% | ≥ 1.5 | ≥ -25% | ≥ 52% | ≥ 0.80 | ≥ 0.75 |
| **B** | ≥ 10% | ≥ 1.0 | ≥ -35% | ≥ 51% | ≥ 0.70 | ≥ 0.60 |
| **C** | ≥ 5% | ≥ 0.5 | ≥ -50% | ≥ 50% | ≥ 0.50 | ≥ 0.40 |
| **D** | < 5% | < 0.5 | < -50% | < 50% | < 0.50 | < 0.40 |

**Configuration:** `config/strategy_grading.yaml`

### 3. Model Metadata & Registry (`learning/model_metadata.py`)

Centralized model tracking, versioning, and deployment status.

**Key Classes:**
- `ModelMetadata`: Complete model metadata
- `ModelRegistry`: Central registry for all models

**Core Methods:**

```python
from learning.model_metadata import ModelRegistry, ModelMetadata

registry = ModelRegistry(registry_path="models/registry.json")

# Register a new model
metadata = ModelMetadata(
    model_name="xgboost_model",
    model_version="1.0.0",
    model_type="xgboost",
    training_date="2024-01-15",
    training_features=["sma_5", "volatility"],
    training_horizons=[5, 20],
    training_data_period={"start": "2020-01-01", "end": "2023-12-31"},
    training_n_samples=5000,
    hyperparameters={"max_depth": 6, "learning_rate": 0.1},
)

registry.register_model(metadata)

# Update with grading results
registry.update_grading(
    "xgboost_model",
    "1.0.0",
    grade="A",
    metrics={"roc_auc": 0.62, "brier_score": 0.18},
)

# Update deployment status
registry.update_deployment(
    "xgboost_model",
    "1.0.0",
    status="LIVE",
    notes="Approved by quant team on 2024-01-16",
)

# Update live performance metrics
registry.update_live_performance(
    "xgboost_model",
    "1.0.0",
    performance_metrics={
        "daily_pnl": 500.0,
        "win_rate": 0.55,
        "sharpe": 1.2,
    }
)

# Query registry
models = registry.list_models()
candidates = registry.get_deployment_candidates(min_grade="B")
live_models = registry.get_live_models()
registry.print_registry_summary()
```

### 4. Integrated Test Framework (`backtest/test_framework.py`)

Orchestrates prediction and strategy evaluation.

**Key Classes:**
- `TestFramework`: Main orchestrator
- `StrategyBacktestEvaluator`: Walk-forward backtesting helper

**Core Workflow:**

```python
from backtest.test_framework import TestFramework

framework = TestFramework()

# STEP 1: Evaluate prediction engine
pred_grade = framework.evaluate_prediction_engine_walk_forward(
    model_name="xgboost_model",
    model_version="1.0.0",
    fold_results=[
        {
            'y_true': y_true_fold,
            'y_pred_proba': proba_fold,
            'y_pred_return': pred_ret_fold,
            'y_true_return': actual_ret_fold,
            'metrics': {},
            'regimes': regime_fold,
        }
        for each fold in walk_forward
    ]
)

# STEP 2: Evaluate strategy
strat_grade = framework.evaluate_strategy_walk_forward(
    strategy_name="ml_alpha_strategy",
    strategy_version="1.0.0",
    fold_results=[
        (returns_fold, trades_df_fold)
        for each fold in walk_forward
    ]
)

# STEP 3: Generate reports
framework.generate_test_report(
    pred_grade,
    strat_grade,
    output_dir="reports/grading",
)

# STEP 4: Check deployment gates
is_approvable, reason = framework.check_deployment_gates(
    pred_grade.grade,
    strat_grade.grade,
)

# STEP 5: Register and deploy
if is_approvable:
    framework.registry.update_deployment(
        "xgboost_model", "1.0.0",
        status="PAPER" if strat_grade.grade == "B" else "LIVE",
    )
```

## Usage Examples

### Example 1: Grading a Prediction Engine

```python
from learning.prediction_grader import PredictionGrader
import numpy as np

# Initialize grader
grader = PredictionGrader()

# Simulate test data from walk-forward fold
fold_results = []

for fold_idx in range(3):  # 3 folds
    y_true = np.random.binomial(1, 0.5, 500)
    y_pred_proba = y_true * 0.65 + (1 - y_true) * 0.35 + np.random.normal(0, 0.05, 500)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    y_true_return = y_true * 0.02 - (1 - y_true) * 0.01 + np.random.normal(0, 0.01, 500)
    y_pred_return = y_pred_proba * 0.02 - (1 - y_pred_proba) * 0.01
    
    fold_results.append({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba,
        'y_pred_return': y_pred_return,
        'y_true_return': y_true_return,
        'metrics': grader.evaluate_single_test_set(y_true, y_pred_proba, y_pred_return, y_true_return),
        'regimes': None,
    })

# Overall metrics
overall_metrics = grader.evaluate_single_test_set(
    np.concatenate([f['y_true'] for f in fold_results]),
    np.concatenate([f['y_pred_proba'] for f in fold_results]),
    np.concatenate([f['y_pred_return'] for f in fold_results]),
    np.concatenate([f['y_true_return'] for f in fold_results]),
)

# Grade
grade_result = grader.grade_prediction_engine(fold_results, overall_metrics)

# Print report
print(grader.report_grade(grade_result))
```

### Example 2: Grading a Strategy

```python
from backtest.strategy_grader import StrategyGrader
import pandas as pd

grader = StrategyGrader()

# Simulate strategy results
fold_results = []

for fold_idx in range(3):
    returns = np.random.normal(0.0005, 0.01, 252)  # Daily returns
    trades_df = pd.DataFrame({
        'entry_price': np.random.uniform(100, 110, 50),
        'exit_price': np.random.uniform(100, 110, 50),
        'direction': np.random.choice([1, -1], 50),
        'entry_date': pd.date_range('2023-01-01', periods=50),
        'exit_date': pd.date_range('2023-01-10', periods=50),
    })
    
    fold_results.append((returns, trades_df))

# Aggregate metrics
all_returns = np.concatenate([r for r, _ in fold_results])
all_trades = pd.concat([t for _, t in fold_results], ignore_index=True)

primary_metrics = grader.compute_strategy_metrics(all_returns, all_trades)

# Fold metrics
fold_metrics = [grader.compute_strategy_metrics(r, t) for r, t in fold_results]

# Monte Carlo
mc_results = grader.monte_carlo_analysis(all_returns)

# Grade
grade_result = grader.grade_strategy(
    primary_metrics,
    fold_metrics,
    regime_metrics={},
    monte_carlo_results=mc_results,
)

# Print report
print(grader.report_grade(grade_result))
```

## Configuration Files

### `config/prediction_grading.yaml`

Controls prediction engine grading thresholds, bucket analysis, walk-forward setup.

Key sections:
- `grades`: Thresholds for A/B/C/D grades
- `probability_buckets`: Define bucket edges for calibration analysis
- `walk_forward`: Training/test window configuration
- `regime_analysis`: Per-regime analysis settings

### `config/strategy_grading.yaml`

Controls strategy grading thresholds, deployment gates, cost assumptions.

Key sections:
- `grades`: Thresholds for A/B/C/D grades (return, Sharpe, drawdown, etc.)
- `costs`: Slippage, commission, bid-ask assumptions
- `monte_carlo`: Simulation settings
- `deployment_gates`: Gates for deployment approval

## Deployment Workflow

```
1. Train Model
   ↓
2. Evaluate Prediction Engine
   ├─ Walk-forward validation
   ├─ Probability calibration
   ├─ Per-regime analysis
   └─ Grade: A/B/C/D
   ↓
3. Backtest Strategy
   ├─ Walk-forward backtest
   ├─ P&L metrics
   ├─ Per-regime analysis
   ├─ Monte Carlo
   └─ Grade: A/B/C/D
   ↓
4. Check Deployment Gates
   ├─ Prediction grade ≥ threshold?
   ├─ Strategy grade ≥ threshold?
   └─ Manual approval required?
   ↓
5. Register Model
   ├─ Save metadata & grades
   └─ Update deployment status
   ↓
6. Deploy (PAPER → LIVE)
   ├─ Monitor live performance
   └─ Update registry with live metrics
```

## Output Artifacts

### Reports

- `reports/grading/prediction_grade_report.txt` - Detailed prediction metrics
- `reports/grading/strategy_grade_report.txt` - Detailed strategy metrics
- `reports/grading/grading_summary.json` - Summary for dashboard

### Model Registry

- `models/registry.json` - Central registry with all model metadata and grades

Sample registry entry:
```json
{
  "xgboost_model": [
    {
      "model_name": "xgboost_model",
      "model_version": "1.0.0",
      "prediction_grade": "A",
      "strategy_grade": "A",
      "deployment_status": "LIVE",
      "deployment_date": "2024-01-16T10:30:00",
      "live_performance": {
        "daily_pnl": 500.0,
        "sharpe": 1.2
      }
    }
  ]
}
```

## Integration with ALPHATrader

The framework integrates with the existing trading system:

```python
from trading.trader import ALPHATrader
from alpha.ml_alpha_xgboost import MLAlpha
from learning.model_store import load_model
from backtest.test_framework import TestFramework

# 1. Load and test model
framework = TestFramework()
pred_grade = framework.evaluate_prediction_engine_walk_forward(...)
strat_grade = framework.evaluate_strategy_walk_forward(...)

# 2. Check deployment gates
is_ok, reason = framework.check_deployment_gates(pred_grade.grade, strat_grade.grade)

# 3. Deploy if approved
if is_ok:
    model = load_model("xgboost_model")
    alpha = MLAlpha(config=MLAlphaConfig(), model=model)
    trader = ALPHATrader(alphas=[alpha, ...])
    trader.live()  # or trader.paper()
```

## Best Practices

1. **Always use walk-forward validation** for time-series data (prevents look-ahead bias)
2. **Monitor per-regime metrics** to ensure strategies work across market conditions
3. **Use Monte Carlo analysis** to assess confidence in results
4. **Check probability calibration** (Brier score) for prediction engines
5. **Set appropriate gates** (don't deploy Grade C or D models)
6. **Monitor live performance** against backtest metrics
7. **Retrain regularly** (monthly or quarterly) with new data
8. **Keep detailed records** in model registry for audit trail

## Troubleshooting

**Problem**: Prediction grade is C or D
- Check ROC-AUC and Brier score
- Ensure walk-forward testing (no leakage)
- Try different features or model architecture
- Check probability calibration

**Problem**: Strategy grade is low despite good prediction metrics
- Check for overtrading (too many transactions)
- Verify transaction costs are realistic
- Ensure walk-forward backtest (no overfitting)
- Check regime consistency

**Problem**: Large IS/OOS degradation
- Model may be overfit
- Reduce model complexity
- Increase regularization
- Use more conservative thresholds

---

**Framework Version**: 1.0  
**Last Updated**: January 2024

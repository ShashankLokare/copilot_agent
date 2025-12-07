# Strategy Testing & Grading Framework

## Overview

This comprehensive framework evaluates trading strategies through multiple dimensions:

1. **Full-Pipeline Backtesting** - Real trading with transaction costs, slippage, liquidity
2. **Trading Metrics** - Complete PnL analytics (Sharpe, Calmar, Win Rate, Profit Factor, etc.)
3. **Regime Analysis** - Performance breakdown by market regime
4. **Monte Carlo Testing** - Trade reshuffling and bootstrap robustness tests
5. **Parameter Sensitivity** - Overfitting detection through parameter perturbations
6. **Automated Grading** - A/B/C/D grades based on configurable thresholds
7. **Model Registry** - Central tracking of models with their grades and deployment status

## Architecture

```
Strategy Input
    ↓
Full-Pipeline Backtest
  • Data → Features → Regime → Alpha → Signals → Risk → Portfolio → Execution
  • Transaction Costs (commission, fees, slippage)
  • Liquidity Constraints
    ↓
Equity Curve + Trade Log
    ↓
┌─────────────────────────────────────────────────────────┐
│ Performance Analysis                                    │
├─────────────────────────────────────────────────────────┤
│ • Trading Metrics (CAGR, Sharpe, Calmar, Win Rate, PF) │
│ • Regime Breakdown (per-regime performance)             │
│ • Cost Analysis (% of edge)                             │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Robustness Testing                                      │
├─────────────────────────────────────────────────────────┤
│ • Monte Carlo: Trade shuffling (preserve PnL)           │
│ • Bootstrap: Daily returns resampling                   │
│ • Parameter Sensitivity: Small perturbations            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Automated Grading (A/B/C/D)                             │
├─────────────────────────────────────────────────────────┤
│ Grade A: Sharpe≥1.2, MaxDD≤25%, PF≥1.4, MC≥80%        │
│ Grade B: Sharpe≥0.8, MaxDD≤30%, PF≥1.2, MC≥65%        │
│ Grade C: Sharpe≥0.5, MaxDD≤40%, PF≥1.1, MC≥50%        │
│ Grade D: Below Grade C thresholds                       │
└─────────────────────────────────────────────────────────┘
    ↓
Model Registry + Orchestrator
  • Gating: No LIVE for grade < B
  • Limits: Grade B has position size caps
  • Disable: Grade D strategies not loaded
```

## Core Modules

### 1. `backtest/trading_metrics.py`

Implements metric calculation for equity curves and trade logs.

**Key Classes:**
- `TradingMetrics` - Static methods for all metric calculations
- `EquityCurve` - Holds equity, returns, trades, and regime data
- `StrategyPerformance` - Dataclass with all metrics results
- `StrategyEvaluator` - Main evaluator class

**Key Metrics Computed:**

```python
# Risk-Return Metrics
sharpe_ratio = (mean_return - rf) / std_return
sortino_ratio = (mean_return - rf) / downside_std
cagr = (final_equity / initial_equity) ^ (1/years) - 1
calmar_ratio = cagr / max_drawdown

# Risk Metrics
max_drawdown = (equity - peak) / peak
var_95 = 95th percentile loss

# Trade Statistics
win_rate = num_winning_trades / total_trades
profit_factor = gross_profit / gross_loss
payoff_ratio = avg_win / abs(avg_loss)

# Efficiency
turnover = total_notional_traded / avg_equity
costs = commission + slippage + spread
```

**Example Usage:**

```python
from backtest.trading_metrics import StrategyEvaluator, EquityCurve

# Build equity curve
eq_curve = EquityCurve(
    timestamps=pd.date_range(...),
    equity=equity_array,
    daily_returns=returns_array,
    daily_pnl=pnl_array,
    cumulative_pnl=cumulative_pnl,
    trades=trade_list,
    regime_series=regime_series,
)

# Evaluate
evaluator = StrategyEvaluator(risk_free_rate=0.04)
perf = evaluator.evaluate(eq_curve)

print(f"Sharpe: {perf.sharpe_ratio:.2f}")
print(f"Max DD: {perf.max_drawdown:.2%}")
print(f"Win Rate: {perf.win_rate:.2%}")
```

### 2. `backtest/monte_carlo.py`

Robustness testing through randomization and bootstrapping.

**Trade Shuffling MC:**
- Preserves individual trade PnLs
- Randomly reorders trades (removes timing luck)
- Tests if returns come from consistent strategy logic

**Daily Returns Bootstrap:**
- Resamples daily returns with replacement
- Tests robustness to different return sequences
- Can highlight clustering/regime dependence

**Parameter Sensitivity:**
- Perturbs key parameters (risk, thresholds, sizing)
- Measures profitability across variations
- Detects overfitting to specific parameters

**Example Usage:**

```python
from backtest.monte_carlo import MonteCarloTester

# Trade shuffling
mc_trades = MonteCarloTester.trade_shuffling_mc(
    trades=trade_list,
    num_iterations=1000,
    metrics_to_compute=['sharpe_ratio', 'max_drawdown', 'final_pnl']
)

print(f"Prob Profitable: {mc_trades.prob_profitable:.2%}")
print(f"Sharpe 5th %ile: {mc_trades.percentiles['sharpe_ratio'][5]:.2f}")

# Daily returns bootstrap
mc_daily = MonteCarloTester.daily_returns_bootstrap(
    daily_returns=returns_array,
    num_iterations=1000,
    sample_size=252,
)
```

### 3. `backtest/strategy_grader.py`

Automated grading system based on performance and robustness.

**Grade Definitions:**

| Grade | Sharpe | Max DD | Profit Factor | CAGR | MC Prob Profitable | Use Case |
|-------|--------|--------|---------------|------|-------------------|----------|
| A | ≥1.2 | ≤25% | ≥1.4 | ≥15% | ≥80% | Live Scalable |
| B | ≥0.8 | ≤30% | ≥1.2 | ≥10% | ≥65% | Live Sized |
| C | ≥0.5 | ≤40% | ≥1.1 | ≥5% | ≥50% | Incubation |
| D | <0.5 | >40% | <1.1 | <5% | <50% | Rejected |

**Example Usage:**

```python
from backtest.strategy_grader import StrategyGrader

grader = StrategyGrader(config_path='config/strategy_grading.yaml')

grade_result = grader.grade(
    performance=perf,
    monte_carlo_result=mc_result,
    parameter_robustness=param_result,
)

print(f"Grade: {grade_result.grade}")
print(f"Allowed Modes: {grade_result.allowed_modes}")
print(grade_result.comments)
```

### 4. `learning/model_metadata.py`

Central registry for models with grades and deployment status.

**Key Classes:**
- `ModelRecord` - Complete model metadata
- `PredictionGradeResult` - ML model quality grades
- `StrategyGradeResult` - Trading strategy quality grades
- `ModelRegistry` - Central tracking system

**Fields Tracked:**

```python
model = ModelRecord(
    model_id="unique_id",
    model_name="display_name",
    model_type="strategy",  # or "prediction_engine"
    version="1.0",
    
    # Training period
    training_metadata=TrainingMetadata(...),
    
    # Grades
    prediction_grade=PredictionGradeResult(...),
    strategy_grade=StrategyGradeResult(...),
    
    # Deployment
    deployment_status="LIVE" | "PAPER" | "INCUBATOR" | "REJECTED",
    is_active=True/False,
    
    # Lineage
    parent_model_id="model_v0",
    related_model_ids=["sister_strategy"],
)
```

**Example Usage:**

```python
from learning.model_metadata import ModelRegistry, ModelRecord

registry = ModelRegistry(registry_path=Path('data/model_registry.json'))

# Register model
registry.register_model(model)

# Query
best_live = registry.get_best_model(model_type="strategy", deployment_status="LIVE")
print(f"Best live strategy: {best_live.model_name} ({best_live.overall_grade})")

# List all Grade A models
grade_a_models = [
    m for m in registry.list_models(is_active=True)
    if m.overall_grade == 'A'
]
```

## Configuration Files

### `config/strategy_grading.yaml`

Master configuration for grading criteria.

```yaml
grades:
  A:
    sharpe_min: 1.2
    max_drawdown_max: 0.25
    profit_factor_min: 1.4
    cagr_min: 0.15
    monte_carlo_profitable_pct_min: 0.80
  
  B:
    sharpe_min: 0.8
    max_drawdown_max: 0.30
    profit_factor_min: 1.2
    cagr_min: 0.10
    monte_carlo_profitable_pct_min: 0.65

# ... C and D thresholds

live_mode_enforcement:
  grade_a_required_for_live: true
  grade_b_allowed_for_live: true
  grade_c_allowed_for_live: false
  grade_d_allowed_for_live: false

# Position size limits for Grade B
grade_b_live_limits:
  max_position_size: 0.25  # 25% of normal
  max_daily_trades: 5
```

## Workflow Example

### 1. Backtest Strategy

```python
# Run full-pipeline backtest
equity_curve = backtest_engine.run(
    strategy=my_strategy,
    data=historical_data,
    commission=0.001,
    slippage=0.0005,
)
```

### 2. Compute Metrics

```python
evaluator = StrategyEvaluator()
performance = evaluator.evaluate(equity_curve)
```

### 3. Run Monte Carlo

```python
mc_result = MonteCarloTester.trade_shuffling_mc(
    trades=equity_curve.trades,
    num_iterations=1000,
)
```

### 4. Grade Strategy

```python
grader = StrategyGrader()
grade = grader.grade(performance, mc_result)

# grade.grade → "A" | "B" | "C" | "D"
# grade.allowed_modes → ["LIVE", "PAPER", "INCUBATOR"]
```

### 5. Register Model

```python
registry = ModelRegistry()
model = ModelRecord(
    model_id="strategy_v1",
    model_name="My Strategy",
    model_type="strategy",
    version="1.0",
    strategy_grade=grade,
)
registry.register_model(model)
```

### 6. Gate in Orchestrator

```python
# In orchestrator/orchestrator.py
model = registry.get_model("strategy_v1")

if model.strategy_grade.grade == 'D':
    logger.error(f"Rejecting {model.model_id}: Grade D")
    skip_model(model)
elif model.strategy_grade.grade == 'C':
    logger.warning(f"Loading {model.model_id} in INCUBATION mode")
    load_with_limits(model, mode='INCUBATOR')
elif model.strategy_grade.grade in ['A', 'B']:
    logger.info(f"Loading {model.model_id} in {model.deployment_status} mode")
    load_model(model)
```

## Interpretation Guide

### Sharpe Ratio
- **> 1.5**: Exceptional, expect for only best strategies
- **1.0-1.5**: Very good, exceeds most managers
- **0.5-1.0**: Good, above average
- **< 0.5**: Poor, close to random walk

*Formula*: (mean_return - risk_free_rate) / volatility

### Calmar Ratio
- **> 1.0**: Excellent recovery from drawdowns
- **0.5-1.0**: Good risk-adjusted returns
- **< 0.5**: Drawdowns are a major issue

*Formula*: CAGR / Max Drawdown

### Profit Factor
- **> 2.0**: Excellent, gross profit is 2x gross loss
- **1.5-2.0**: Very good
- **1.2-1.5**: Good
- **1.0-1.2**: Marginal
- **< 1.0**: Strategy loses money

*Formula*: Gross Profit / Gross Loss

### Monte Carlo Probability Profitable
- **> 75%**: Very robust
- **50-75%**: Reasonably robust
- **< 50%**: High sensitivity to luck or overfitting

*What it means*: In 1000 random trade reshuffles, what % ended profitable?

### Regime Concentration
- **< 50%**: Well diversified across regimes
- **50-70%**: Somewhat concentrated
- **> 70%**: High concentration, may fail in different market conditions

*What it means*: Is the strategy dependent on a single market regime (trend/range)?

## Integration with Orchestrator

The orchestrator checks grades before deploying:

```python
# orchestrator/orchestrator.py (pseudocode)

def load_model(model_id: str):
    model = registry.get_model(model_id)
    
    # Rule 1: Reject Grade D outright
    if model.overall_grade == 'D':
        raise ValueError(f"{model_id} is Grade D, not approved")
    
    # Rule 2: Grade C only in incubation
    if model.overall_grade == 'C':
        if requested_mode == 'LIVE':
            logger.error(f"Grade C {model_id} cannot go LIVE")
            return None
        mode = 'INCUBATOR'
    
    # Rule 3: Grade B can go live with position limits
    elif model.overall_grade == 'B':
        if requested_mode == 'LIVE':
            # Apply position size caps
            apply_limits(model, max_size=0.25)
    
    # Rule 4: Grade A has no restrictions
    # (can go LIVE at full size)
    
    return load_strategy(model_id, mode)
```

## Troubleshooting

**Q: Strategy shows high Sharpe but fails Monte Carlo**
- A: Likely overfitted. Test on different time periods or markets.

**Q: Strategy is Grade C, what should I do?**
- A: Run in INCUBATION mode. Monitor per-regime performance. Optimize parameters.

**Q: My strategy lost money in one regime**
- A: Check `regime_stats`. May need separate models per regime or regime filter.

**Q: Transaction costs are killing returns**
- A: Check `cost_pct_of_pnl`. If > 30%, reduce trade frequency or improve entries.

**Q: Profit Factor is only 1.1**
- A: Average loss is almost as large as average win. Consider tighter stops or better entry timing.

## Files Reference

| File | Purpose |
|------|---------|
| `backtest/trading_metrics.py` | Core metric calculations |
| `backtest/monte_carlo.py` | Robustness testing |
| `backtest/strategy_grader.py` | Grade assignment |
| `learning/model_metadata.py` | Model registry |
| `config/strategy_grading.yaml` | Grade thresholds |
| `examples/evaluation_framework_demo.ipynb` | Complete working example |

## Next Steps

1. **Implement full backtester** - Connect all trading system components
2. **Add walk-forward testing** - Test across multiple time windows
3. **Implement parameter sweeps** - Automated sensitivity analysis
4. **Deploy orchestrator gates** - Enforce grading in production
5. **Monitor live models** - Track actual vs backtest performance

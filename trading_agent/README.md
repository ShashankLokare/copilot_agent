# Trading Agent - Production-Grade Algorithmic Trading System

## Overview

This is a **modular, production-grade, multi-agent algorithmic trading system** designed for:
- **Capital preservation** - Strict risk management above all else
- **Robustness** - Clean, testable code with dependency injection
- **Extensibility** - Easy to add new data sources, alpha models, and brokers
- **Clarity** - 10 distinct layers with clear responsibilities

## Architecture

The system is organized into 10 layers:

### 1. **Orchestrator & Policy Layer** (`orchestrator/`)
Main control loop that coordinates all components. Supports three operating modes:
- **LIVE** - Real trading (requires broker integration)
- **PAPER** - Simulated trading with real logic
- **BACKTEST** - Historical replay

### 2. **Data Ingestion & Market Interface** (`data/`)
Pluggable data adapters:
- `CSVAdapter` - Load historical data from CSV
- `RESTAPIAdapter` - Template for live broker APIs
- `MarketDataProvider` - Unified interface

### 3. **Data Quality & Feature Engineering** (`features/`)
Computes technical indicators:
- Moving averages (SMA)
- Momentum indicators (RSI, MACD)
- Volatility (ATR, Bollinger Bands)
- Custom indicators

### 4. **Regime Detection** (`regime/`)
Identifies market conditions:
- **Trend/Range** detection
- **High/Low Volatility** classification
- Rule-based and ML-based detectors
- Used to adapt alpha behavior

### 5. **Alpha / Strategy Engine** (`alpha/`)
Multiple trading algorithms:
- `MomentumAlpha` - Trade momentum
- `MeanReversionAlpha` - Trade mean reversion
- `BreakoutAlpha` - Trade breakouts
- `MLAlphaXGBoost` - ML-based (extensible)

### 6. **Signal Validation & Scoring** (`signals/`)
Filters and scores raw signals:
- Validates by signal strength
- Requires confirmation from multiple alphas (optional)
- Scores by confidence and edge
- Filters by thresholds

### 7. **Risk Engine & Position Sizing** (`risk/`)
Enforces capital preservation:
- Max equity at risk per trade (default 1%)
- Daily/weekly drawdown limits
- Max concurrent positions
- ATR-based stop losses
- Kill-switch on severe drawdown

### 8. **Portfolio Construction & Optimization** (`portfolio/`)
Builds optimal portfolio:
- Equal-weight, volatility-target, risk-parity methods
- Sector and position limits
- Rebalancing logic

### 9. **Execution Engine** (`execution/`)
Handles order execution:
- Simulated execution for backtest/paper
- Slippage and spread modeling
- Retry logic
- Extensible for live brokers

### 10. **Monitoring, Analytics & Learning** (`monitoring/`, `learning/`)
Performance tracking:
- PnL, drawdown, Sharpe, Sortino
- Per-alpha and per-symbol metrics
- Trade logging
- Model retraining framework

## Quick Start

### 1. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates `data/sample_ohlcv.csv` with realistic OHLCV data.

### 2. Run Backtest

```bash
python main_backtest.py
```

Output example:
```
=== Backtest Results ===
Initial Capital:    $100,000.00
Final Value:        $125,432.10

Performance Metrics:
  Total Return:        25.43%
  Annual Return:       25.43%
  Max Drawdown:        -8.50%
  Sharpe Ratio:         1.24
  Sortino Ratio:        1.67
  Win Rate:           62.50%
  Profit Factor:        1.85
```

### 3. Run Paper Trading

```bash
python main_paper.py
```

Runs continuous simulation with the configured symbols and frequency.

### 4. Run Live Trading

```bash
python main_live.py
```

⚠️ **WARNING**: Only use after extensive testing! Strict risk limits are enforced.

## Configuration

Configuration files are in YAML format:

### `config/backtest_config.yaml`
```yaml
orchestrator:
  operation_mode: "BACKTEST"

data:
  symbols: ["AAPL", "GOOGL", "MSFT"]

risk:
  max_position_risk_pct: 1.0
  max_daily_drawdown_pct: 5.0
```

### `config/paper_config.yaml`
Simulated trading with real logic.

### `config/live_config.yaml`
Live trading - **extremely restrictive defaults**.

## Core Types

All data is passed through strongly-typed dataclasses:

```python
# Market data
Bar(symbol, timestamp, open, high, low, close, volume)
MarketState(symbol, timestamp, current_price, bid, ask, ...)

# Signals
Signal(symbol, direction, strength, timestamp, alpha_name)
ScoredSignal(symbol, direction, confidence, edge, timestamp, alpha_names)

# Risk/Position
Position(symbol, quantity, entry_price, current_price, pnl)
Order(symbol, quantity, order_type, direction, ...)
Trade(symbol, entry_price, entry_timestamp, exit_price, pnl)

# Portfolio
PortfolioState(timestamp, cash, equity, total_value, positions)
```

## Key Design Principles

1. **Dependency Injection** - Pass objects into constructors, not globals
2. **Type Hints** - Full type hints for mypy compatibility
3. **No Hard-Coded Constants** - All tunable values in config
4. **Modular Components** - Each layer is independent and testable
5. **Capital Preservation First** - Risk limits take precedence over returns
6. **Extensibility** - Easy to add:
   - New data sources (implement `DataAdapter`)
   - New alpha models (extend `AlphaModel`)
   - New brokers (implement `ExecutionAdapter`)

## Extending the System

### Adding a New Alpha Model

```python
from alpha.alpha_models import AlphaModel
from utils.types import Signal, SignalDirection

class MyAlpha(AlphaModel):
    def __init__(self):
        super().__init__(name="my_alpha")
    
    def generate_signals(self, market_state, features, regime):
        signals = []
        # Your logic here
        if your_condition:
            signals.append(Signal(
                symbol=market_state.symbol,
                direction=SignalDirection.LONG,
                strength=0.8,
                timestamp=market_state.timestamp,
                alpha_name=self.name,
                reasoning="My custom reason"
            ))
        return signals

# Add to orchestrator
orchestrator.alpha_engine.add_alpha(MyAlpha())
```

### Adding a New Broker

```python
from execution.execution_engine import ExecutionAdapter

class MyBrokerAdapter(ExecutionAdapter):
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def submit_order(self, order):
        # Call your broker's API
        response = requests.post(
            f"{self.api_endpoint}/orders",
            json={...},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response['order_id']
    
    # ... implement other methods

# Use in orchestrator
executor = MyBrokerAdapter(api_key="...", api_secret="...")
orchestrator.execution_engine = ExecutionEngine(executor)
```

### Adding a New Indicator

```python
from features.feature_engineering import FeatureEngineering

# Subclass and add custom indicators
class CustomFeatureEngineering(FeatureEngineering):
    def compute_features(self, bars, enabled_indicators):
        features = super().compute_features(bars, enabled_indicators)
        
        if "my_custom_indicator" in enabled_indicators:
            features.custom['my_custom_indicator'] = self._compute_my_indicator(bars)
        
        return features
    
    def _compute_my_indicator(self, bars):
        # Your calculation
        pass
```

## Risk Management Features

The system enforces capital preservation through multiple mechanisms:

1. **Per-Trade Risk Limit** - Max 1% equity at risk per position
2. **Daily Drawdown Limit** - Stop trading if daily loss > 5%
3. **Weekly Drawdown Limit** - Stop trading if weekly loss > 10%
4. **Position Limit** - Max 10 concurrent positions
5. **Kill-Switch** - Automatic shutdown if drawdown > 20%
6. **ATR-Based Stops** - Position-specific stops at entry ± 2*ATR

All limits are configurable but have safe defaults.

## Performance Metrics

The system tracks:
- **Total Return %** - Cumulative return
- **Annual Return %** - Annualized return
- **Max Drawdown %** - Largest peak-to-trough decline
- **Sharpe Ratio** - Risk-adjusted return
- **Sortino Ratio** - Downside risk-adjusted return
- **Win Rate** - % of profitable trades
- **Profit Factor** - Avg win / Avg loss
- **Expectancy** - Expected return per trade

## Testing & Validation

### Unit Tests

```bash
python -m pytest tests/
```

(Test suite can be added to `tests/` directory)

### Walk-Forward Testing

```python
from learning.retraining import WalkForwardValidator

validator = WalkForwardValidator(
    training_window_days=252,
    test_window_days=30,
    step_days=10
)

for train_start, train_end, test_start, test_end in validator.generate_windows(...):
    # Train model on train period
    # Test on test period
    # Calculate out-of-sample metrics
```

## Monitoring & Logging

The system logs all important events:

```
[2024-01-15T09:30:00] [INFO] === Iteration 2024-01-15 09:30:00 ===
[2024-01-15T09:30:01] [INFO] Step 1: Fetching market data...
[2024-01-15T09:30:02] [INFO] Step 2: Computing features and detecting regime...
[2024-01-15T09:30:02] [INFO] Detected regime: TREND_HIGH_VOL
[2024-01-15T09:30:03] [INFO] Step 3: Generating alpha signals...
[2024-01-15T09:30:03] [INFO] Generated 5 signals for AAPL
...
```

## Safety & Disclaimers

⚠️ **This system is for research and paper trading only by default.**

Before enabling live trading:
1. ✅ Backtest extensively
2. ✅ Paper trade for weeks
3. ✅ Start with micro positions
4. ✅ Monitor continuously
5. ✅ Keep emergency kill-switch nearby
6. ✅ Never trade with money you can't afford to lose

The authors assume **no responsibility** for trading losses.

## Dependencies

Core packages (minimal):
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pytz` - Timezone handling
- `pyyaml` - Configuration

Optional (for extended features):
- `xgboost` - ML alpha models
- `scikit-learn` - ML utilities
- `alpaca-trade-api` - Alpaca broker integration
- `nsepy` - NSE data
- `ccxt` - Crypto exchanges

## File Structure

```
trading_agent/
├── config/                 # Configuration files (YAML)
│   ├── paper_config.yaml
│   ├── live_config.yaml
│   └── backtest_config.yaml
├── data/                   # Data adapters
│   └── adapters.py
├── features/               # Feature engineering
│   └── feature_engineering.py
├── regime/                 # Regime detection
│   └── regime_detector.py
├── alpha/                  # Alpha models
│   └── alpha_models.py
├── signals/                # Signal processing
│   └── signal_processor.py
├── risk/                   # Risk management
│   └── risk_engine.py
├── portfolio/              # Portfolio construction
│   └── portfolio_engine.py
├── execution/              # Order execution
│   └── execution_engine.py
├── monitoring/             # Metrics & logging
│   └── metrics.py
├── backtest/               # Backtesting
│   └── backtester.py
├── learning/               # Model retraining
│   └── retraining.py
├── orchestrator/           # Main orchestrator
│   └── orchestrator.py
├── utils/                  # Utilities
│   └── types.py
├── tests/                  # Unit tests
├── scripts/                # Helper scripts
│   └── generate_sample_data.py
├── main_live.py            # Live trading entry point
├── main_paper.py           # Paper trading entry point
├── main_backtest.py        # Backtest entry point
└── README.md               # This file
```

## Next Steps

1. ✅ Generate sample data: `python scripts/generate_sample_data.py`
2. ✅ Run backtest: `python main_backtest.py`
3. ✅ Paper trade: `python main_paper.py`
4. ✅ Add your own alphas
5. ✅ Integrate real broker
6. ✅ Deploy live (after extensive testing!)

## Support & Contributions

This is a reference implementation. Feel free to:
- Fork and customize
- Add new alpha models
- Implement broker integrations
- Improve risk management
- Submit improvements

## License

Use freely for research and education. Not recommended for real money without thorough testing.

---

**Remember:** The goal is capital preservation first, returns second.

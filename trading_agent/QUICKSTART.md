# Quick Start Guide

This guide will get you up and running in 5 minutes.

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

1. **Navigate to project directory:**
   ```bash
   cd trading_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or with conda:
   ```bash
   conda create -n trading python=3.10
   conda activate trading
   pip install -r requirements.txt
   ```

## Quick Demo: Run a Backtest

1. **Generate sample data:**
   ```bash
   python scripts/generate_sample_data.py
   ```
   
   This creates `data/sample_ohlcv.csv` with 1 year of realistic OHLCV data for 5 stocks.

2. **Run backtest:**
   ```bash
   python main_backtest.py
   ```
   
   Expected output:
   ```
   === Algorithmic Trading System - BACKTEST MODE ===
   
   Running backtest...
   Symbols: AAPL, GOOGL, MSFT, AMZN, NVDA
   Initial Capital: $100,000.00
   
   === Backtest Results ===
   Initial Capital:    $100,000.00
   Final Value:        $125,432.10
   
   Performance Metrics:
     Total Return:        25.43%
     Annual Return:       25.43%
     Max Drawdown:         -8.50%
     Sharpe Ratio:          1.24
     Sortino Ratio:         1.67
     Win Rate:            62.50%
     Profit Factor:         1.85
   ```

3. **Customize the backtest:**
   
   Edit `config/backtest_config.yaml`:
   ```yaml
   data:
     symbols:
       - "AAPL"
       - "GOOGL"
       - "MSFT"
   
   risk:
     max_position_risk_pct: 1.0  # Risk 1% per trade
     max_daily_drawdown_pct: 5.0  # Stop at 5% loss
   ```

## Paper Trading (Simulated)

1. **Start paper trading:**
   ```bash
   python main_paper.py
   ```
   
   This runs a continuous simulation with your configured symbols.

2. **Configuration (`config/paper_config.yaml`):**
   ```yaml
   orchestrator:
     operation_mode: "PAPER"
     run_frequency: "daily"
   
   data:
     symbols: ["AAPL", "GOOGL", "MSFT"]
   ```

3. **Monitor with logs:**
   ```
   [2024-01-15T09:30:00] [INFO] === Iteration 2024-01-15 09:30:00 ===
   [2024-01-15T09:30:02] [INFO] Detected regime: TREND_HIGH_VOL
   [2024-01-15T09:30:03] [INFO] Generated 5 signals for AAPL
   [2024-01-15T09:30:04] [INFO] Approved 3 signals
   [2024-01-15T09:30:05] [INFO] Executed 3 orders
   ```

## Understanding the System

### 3-Minute Overview

The system has 10 layers:

```
1. ORCHESTRATOR     ← Main control loop
2. DATA             ← Fetch market data
3. FEATURES         ← Compute indicators (SMA, RSI, ATR, etc.)
4. REGIME           ← Detect market condition (trend, range)
5. ALPHA            ← Generate trading signals
6. SIGNALS          ← Score and filter signals
7. RISK             ← Size positions, enforce limits
8. PORTFOLIO        ← Build optimal weights
9. EXECUTION        ← Place orders
10. MONITORING      ← Track metrics
```

### Key Files to Understand

1. **`config/paper_config.yaml`** - Configuration (start here!)
2. **`orchestrator/orchestrator.py`** - Main loop
3. **`alpha/alpha_models.py`** - Trading strategies
4. **`risk/risk_engine.py`** - Risk management (MOST IMPORTANT)
5. **`utils/types.py`** - Data structures

### The Trading Pipeline

```python
# Simplified flow in orchestrator.run_iteration():

# 1. Get market data
market_state = data_provider.get_market_state(symbol, timestamp)

# 2. Compute features
features = feature_engine.compute_features(market_state.bars)

# 3. Detect regime
regime = regime_manager.get_regime(features)

# 4. Generate signals
signals = alpha_engine.generate_all_signals(market_state, features, regime)

# 5. Score signals
scored_signals = signal_processor.process(signals)

# 6. Apply risk management
for signal in scored_signals:
    assessment = risk_engine.assess_trade(signal, ...)
    if assessment.action == RiskAction.ACCEPT:
        # Create and execute order
        order = create_order(signal, assessment.approved_quantity)
        execution_engine.execute_order(order)

# 7. Record metrics
tracker.record_equity(timestamp, portfolio_value)
```

## Adding Your Own Alpha

Create a new file `alpha/my_alpha.py`:

```python
from alpha.alpha_models import AlphaModel
from utils.types import Signal, SignalDirection

class MyCustomAlpha(AlphaModel):
    def __init__(self):
        super().__init__(name="my_custom_alpha")
    
    def generate_signals(self, market_state, features, regime):
        signals = []
        
        # Your custom logic here
        if features.price > features.sma_50 and features.rsi > 50:
            signals.append(Signal(
                symbol=market_state.symbol,
                direction=SignalDirection.LONG,
                strength=0.75,
                timestamp=market_state.timestamp,
                alpha_name=self.name,
                reasoning="Price above SMA50 and RSI bullish"
            ))
        
        return signals
```

Then in `orchestrator/orchestrator.py`, add to `_initialize_alphas()`:

```python
from alpha.my_alpha import MyCustomAlpha

def _initialize_alphas(self, enabled_models):
    # ... existing code ...
    self.alpha_engine.add_alpha(MyCustomAlpha())
```

And enable in config:

```yaml
alpha:
  enabled_models:
    - "momentum"
    - "my_custom_alpha"  # Add your alpha
```

## Common Customizations

### Change Risk Limits

Edit `config/paper_config.yaml`:

```yaml
risk:
  max_position_risk_pct: 0.5      # Smaller = more conservative
  max_daily_drawdown_pct: 2.0     # Stop trading if loss > 2%
  kill_switch_drawdown_pct: 10.0  # Emergency stop at 10%
```

### Use Different Symbols

```yaml
data:
  symbols:
    - "SPY"
    - "QQQ"
    - "IWM"
```

### Change Running Frequency

```yaml
orchestrator:
  run_frequency: "hourly"  # "minute", "hourly", or "daily"
```

### Adjust Position Sizes

```yaml
portfolio:
  diversification_method: "equal_weight"      # or "volatility_target"
  max_single_position_pct: 5.0               # Max 5% per position
```

### Disable Kill-Switch (not recommended!)

```yaml
risk:
  kill_switch_enabled: false
```

## Troubleshooting

### "CSV file not found"
```bash
python scripts/generate_sample_data.py
```
This generates the required sample data.

### "Import pandas could not be resolved"
```bash
pip install pandas numpy pytz pyyaml
```

### Slow backtest
- Reduce lookback period in `features.lookback_periods`
- Disable unused indicators in `features.enabled_indicators`
- Use smaller date range

### Want different data
Modify `scripts/generate_sample_data.py`:
```python
generate_sample_data(
    symbols=["AAPL", "TSLA", "AMZN"],
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    output_path="data/my_data.csv"
)
```

Then update config:
```yaml
data:
  csv_path: "data/my_data.csv"
```

## Next Steps

1. **Run the demo backtest** (5 minutes)
   ```bash
   python scripts/generate_sample_data.py && python main_backtest.py
   ```

2. **Read the architecture document** (15 minutes)
   ```
   See ARCHITECTURE.md for detailed explanations
   ```

3. **Customize an alpha model** (30 minutes)
   ```
   Follow the "Adding Your Own Alpha" section above
   ```

4. **Paper trade with your changes** (ongoing)
   ```bash
   python main_paper.py
   ```

5. **Write unit tests** (if deploying live)
   ```bash
   pytest tests/
   ```

## Live Trading

⚠️ **WARNING**: Only proceed after extensive testing!

1. Set `operation_mode: "LIVE"` in config
2. Implement broker adapter (e.g., Alpaca, NSE)
3. Set `execution_mode: "live"`
4. Start with minimum position sizes
5. Monitor continuously

See `config/live_config.yaml` for example live configuration.

## Support

- **Questions?** Check `ARCHITECTURE.md` for deep dives
- **Issues?** Check troubleshooting above
- **Extensions?** See `README.md` for extensibility examples

---

**Happy trading! Remember: Capital preservation > Aggressive returns**

# Architecture Deep Dive

This document provides an in-depth explanation of the 10-layer trading system architecture.

## 1. Orchestrator & Policy Layer

**Location:** `orchestrator/orchestrator.py`

**Responsibility:** Central control loop that coordinates all components.

### Key Classes

- **Orchestrator**: Main class that:
  - Initializes all components
  - Implements the trading pipeline
  - Manages operating modes (LIVE, PAPER, BACKTEST)
  - Handles continuous execution

### Data Flow

```
Orchestrator.run_iteration()
    ├─> Data Ingestion (Market Data Provider)
    ├─> Feature Engineering
    ├─> Regime Detection
    ├─> Alpha Generation (All models)
    ├─> Signal Scoring
    ├─> Risk Assessment
    ├─> Portfolio Construction
    ├─> Execution
    └─> Metrics Recording
```

### Configuration

Controlled via `Config` dataclass with these sections:
- `orchestrator.operation_mode`: LIVE, PAPER, or BACKTEST
- `orchestrator.run_frequency`: minute, hourly, daily
- `orchestrator.enabled_markets`: List of markets to trade

---

## 2. Data Ingestion & Market Interface

**Location:** `data/adapters.py`

**Responsibility:** Fetches and normalizes market data from various sources.

### Key Classes

- **DataAdapter** (ABC): Abstract base class
  - `fetch_bars()`: Get historical OHLCV data
  - `fetch_latest_price()`: Get current price
  - `get_symbols()`: List available symbols

- **CSVAdapter**: Reads from CSV files
  - Expects columns: symbol, timestamp, open, high, low, close, volume
  - Automatically converts timestamps to UTC
  - Supports caching

- **RESTAPIAdapter**: Placeholder for API-based data
  - Extend this for real brokers (Alpaca, etc.)

- **MarketDataProvider**: Wrapper providing convenience methods
  - `get_bars()`: With optional caching
  - `get_market_state()`: Current price + recent bars + volatility
  - `get_available_symbols()`

### Data Format

All timestamps are **timezone-aware UTC** using `pytz`:

```python
from datetime import datetime
import pytz

timestamp = datetime.now(pytz.UTC)
```

---

## 3. Data Quality & Feature Engineering

**Location:** `features/feature_engineering.py`

**Responsibility:** Computes technical indicators and features.

### Key Classes

- **Features** (dataclass): Container with:
  - Price metrics (returns, log_returns)
  - Moving averages (SMA_20, SMA_50, SMA_200)
  - Momentum (RSI, MACD, MACD_signal, MACD_histogram)
  - Volatility (ATR, Bollinger Bands)
  - Volume metrics
  - Custom fields dict

- **FeatureEngineering**: Computes all indicators
  - `compute_features()`: Returns Features object
  - Supports selective indicator computation
  - Configurable lookback periods

### Supported Indicators

| Indicator | Method | Params |
|-----------|--------|--------|
| SMA | `_compute_sma()` | period |
| RSI | `_compute_rsi()` | period (default 14) |
| MACD | `_compute_macd()` | fast, slow, signal periods |
| ATR | `_compute_atr()` | period (default 14) |
| Bollinger Bands | `_compute_bollinger()` | period, std_dev |

### Adding Custom Indicators

```python
class CustomFeatures(FeatureEngineering):
    def compute_features(self, bars, enabled_indicators=None):
        features = super().compute_features(bars, enabled_indicators)
        
        if "my_indicator" in enabled_indicators:
            features.custom['my_indicator'] = self._compute_my_indicator(bars)
        
        return features
```

---

## 4. Regime Detection

**Location:** `regime/regime_detector.py`

**Responsibility:** Identifies market conditions to adapt strategy behavior.

### Key Classes

- **Regime** (Enum): Labels
  - TREND_HIGH_VOL
  - TREND_LOW_VOL
  - RANGE_HIGH_VOL
  - RANGE_LOW_VOL
  - UNKNOWN

- **RegimeState** (dataclass): Current regime with:
  - Regime label
  - Confidence (0-1)
  - Trend strength (-1 to 1)
  - Volatility percentile

- **RegimeDetector** (ABC): Base class
  - `detect()`: Returns RegimeState

- **SimpleRulesRegimeDetector**: Uses SMA positioning and ATR
  - Trend: Price > SMA midpoint
  - Volatility: ATR > threshold

- **MLRegimeDetector**: Uses pre-trained classifier
  - Loads from pickle file
  - Feature vector: [SMA_20, SMA_50, RSI, ATR, vol, returns]

### How Alphas Use Regime

```python
def generate_signals(self, market_state, features, regime):
    # Adjust behavior based on regime
    if regime.regime == Regime.TREND_HIGH_VOL:
        # Be more conservative in trending, volatile markets
        confidence_boost = 0.8
    elif regime.regime == Regime.RANGE_LOW_VOL:
        # Favor mean reversion in quiet, ranging markets
        confidence_boost = 0.7
    
    # ... generate signals with regime-adjusted confidence
```

---

## 5. Alpha / Strategy Engine

**Location:** `alpha/alpha_models.py`

**Responsibility:** Generate trading signals from market conditions.

### Key Classes

- **AlphaModel** (ABC): Base class
  - `generate_signals()`: Returns List[Signal]

- **MomentumAlpha**: Trades in direction of recent momentum
  - Uses RSI > 70 (overbought) → LONG
  - Uses RSI < 30 (oversold) → SHORT
  - Confirms with MACD direction

- **MeanReversionAlpha**: Trades reversions to mean
  - Price near upper Bollinger Band → SHORT
  - Price near lower Bollinger Band → LONG

- **BreakoutAlpha**: Trades breakouts
  - Price > SMA50 + threshold → LONG
  - Price < SMA50 - threshold → SHORT

- **MLAlphaXGBoost**: ML-based (extensible)
  - Loads XGBoost model
  - Uses feature vector for prediction
  - Confidence from model probability

- **AlphaEngine**: Manages multiple alphas
  - `add_alpha()`: Register new alpha
  - `generate_all_signals()`: Aggregates from all alphas

### Signal Format

```python
Signal(
    symbol="AAPL",
    direction=SignalDirection.LONG,
    strength=0.8,  # 0-1, how confident the alpha is
    timestamp=datetime.now(pytz.UTC),
    alpha_name="momentum",
    reasoning="RSI overbought + MACD positive"
)
```

### Creating Custom Alphas

```python
from alpha.alpha_models import AlphaModel
from utils.types import Signal, SignalDirection

class MyAlpha(AlphaModel):
    def __init__(self, param1=0.5, param2=0.3):
        super().__init__(name="my_alpha")
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, market_state, features, regime):
        signals = []
        
        if features.rsi > 50 and features.macd > 0:
            signals.append(Signal(
                symbol=market_state.symbol,
                direction=SignalDirection.LONG,
                strength=0.7,
                timestamp=market_state.timestamp,
                alpha_name=self.name,
                reasoning="Custom logic"
            ))
        
        return signals

# Register
orchestrator.alpha_engine.add_alpha(MyAlpha(param1=0.6))
```

---

## 6. Signal Validation & Scoring

**Location:** `signals/signal_processor.py`

**Responsibility:** Filter and score raw signals before execution.

### Key Classes

- **SignalValidator**: Filters raw signals
  - Min strength threshold
  - Optional confirmation (require 2+ alphas agreeing)

- **SignalScorer**: Scores by quality
  - Base confidence + signal strength
  - Boost for multiple alphas agreeing
  - Historical alpha performance weighting (optional)
  - Compute edge (expected return)

- **SignalFilter**: Final filtering by thresholds
  - Min confidence (default 0.5)
  - Min edge (default 0.01 = 1% expected return)

- **SignalProcessor**: Orchestrates pipeline
  - Validate → Score → Filter
  - Returns list of high-quality signals

### Signal Scoring Example

```
Raw Signal (alpha=momentum, strength=0.8)
  ↓
Validated (strength >= 0.3)
  ↓
Scored:
  - Base confidence: 0.5
  - Signal strength boost: 0.8 * 0.3 = +0.24
  - Agreement boost: 0 (single alpha)
  - Historical performance: +0.05 (if available)
  - Final confidence: 0.79
  ↓
Filtered (confidence >= 0.5 AND edge >= 0.01)
  ↓
Approved for Risk Engine
```

---

## 7. Risk Engine & Position Sizing

**Location:** `risk/risk_engine.py`

**Responsibility:** Capital preservation - the most important layer!

### Key Classes

- **PositionSizer**: Calculates position size
  - Input: entry_price, stop_loss_price, portfolio_equity
  - Output: number of shares = (equity * risk% / stop_distance)
  - ATR-based stop calculation

- **RiskEngine**: Enforces all risk constraints
  - Per-trade limit (default 1% equity at risk)
  - Daily drawdown limit (default 5%)
  - Weekly drawdown limit (default 10%)
  - Max concurrent positions (default 10)
  - Kill-switch on severe drawdown (default 20%)

- **RiskAssessment**: Decision per trade
  - Action: ACCEPT, REDUCE_SIZE, REJECT, KILL_SWITCH
  - Approved quantity
  - Reasoning

### Risk Engine Flow

```
Scored Signal
  ↓
RiskEngine.assess_trade()
  ├─> Check kill-switch? (If yes → REJECT)
  ├─> Check daily drawdown? (If yes → REJECT)
  ├─> Check weekly drawdown? (If yes → REJECT)
  ├─> Check position count? (If yes → REJECT)
  └─> Calculate position size
      └─> Return RiskAssessment
```

### Position Sizing Example

```
Signal: AAPL, LONG, confidence 0.75
Entry price: $150
ATR: $2.50
Stop loss: $150 - (2 * $2.50) = $145

Portfolio equity: $100,000
Risk per trade: 1% = $1,000

Position size = $1,000 / ($150 - $145) = $1,000 / $5 = 200 shares
Max loss: 200 * $5 = $1,000 (exactly 1% of equity)
```

---

## 8. Portfolio Construction & Optimization

**Location:** `portfolio/portfolio_engine.py`

**Responsibility:** Build optimal portfolio from approved signals.

### Key Classes

- **PortfolioWeights** (dataclass): Target allocation
  - weights: Dict[symbol → weight]
  - cash_weight: Cash allocation

- **PortfolioBuilder**: Constructs weights
  - Methods: equal_weight, volatility_target, risk_parity
  - Enforces position and sector limits

- **PortfolioRebalancer**: Generates rebalancing orders
  - Compares current vs target weights
  - Generates orders for positions exceeding drift threshold

### Weight Construction Methods

1. **Equal Weight**: Each position gets equal weight
   ```
   Weight per position = 1.0 / num_positions
   Max position: 10% (configurable)
   ```

2. **Volatility Target**: Inverse volatility weighting
   ```
   Weight ∝ 1 / (1 - confidence)
   Higher confidence → lower weight
   ```

3. **Risk Parity**: Each position contributes equally to risk
   ```
   Weight ∝ 1 / num_positions
   Simple equal weight variant
   ```

---

## 9. Execution Engine

**Location:** `execution/execution_engine.py`

**Responsibility:** Place orders with brokers (simulated or live).

### Key Classes

- **ExecutionAdapter** (ABC): Base class
  - `submit_order()`: Place order, return ID
  - `cancel_order()`: Cancel pending order
  - `get_order_status()`: Check status
  - `get_filled_orders()`: Retrieve fills

- **SimulatedExecutor**: For backtest/paper trading
  - Models slippage (default 2 bps)
  - Models spread (default 1 bps)
  - Models fill probability (default 95%)

- **LiveBrokerAdapter**: Template for real brokers
  - Extend and implement actual API calls
  - Examples: Alpaca, NSE, Binance

- **ExecutionEngine**: Orchestrator for execution
  - Retry logic (configurable)
  - Tracks submitted and filled orders
  - Matches entry/exit fills into trades

### Order Execution Flow

```
Order (symbol, quantity, direction, type)
  ↓
ExecutionEngine.execute_order()
  ├─> Try submit to adapter (up to max_retries)
  ├─> On success, record order ID
  ├─> Adapter processes fill
  └─> Track filled order
      ↓
      Convert fills into Trades
```

### Implementing a Live Broker

```python
class AlpacaAdapter(ExecutionAdapter):
    def __init__(self, api_key, secret_key):
        self.api = tradeapi.REST(api_key, secret_key, ...)
    
    def submit_order(self, order):
        resp = self.api.submit_order(
            symbol=order.symbol,
            qty=order.quantity,
            side="buy" if order.direction == SignalDirection.LONG else "sell",
            type="market",
            time_in_force="gtc"
        )
        return resp.id
    
    def get_order_status(self, order_id):
        order = self.api.get_order(order_id)
        # Map to OrderStatus enum
        ...
```

---

## 10. Monitoring, Analytics & Learning

**Location:** `monitoring/metrics.py`, `learning/retraining.py`

**Responsibility:** Track performance and enable continuous improvement.

### Monitoring

- **PerformanceMetrics** (dataclass): Comprehensive metrics
  - Win rate, profit factor, expectancy
  - Sharpe ratio, Sortino ratio
  - Max drawdown, duration
  - Per-symbol breakdowns

- **MetricsCalculator**: Computes metrics from trades
  - `calculate_metrics()`: Full performance summary
  - Sharpe calculation: `(avg_return - rf_rate) / std_dev`
  - Sortino: Like Sharpe but penalizes only downside

- **PerformanceTracker**: Records metrics over time
  - `record_equity()`: Track portfolio value
  - `record_trade()`: Log completed trade
  - `calculate_current_metrics()`: Point-in-time snapshot

- **Logger**: Simple logging utility
  - File and console output
  - Log levels: INFO, WARNING, ERROR

### Learning

- **ModelTrainer** (ABC): Base for model training
  - `train()`: Train on data
  - `save_model()`: Persist to disk
  - `load_model()`: Load from disk

- **WalkForwardValidator**: Out-of-sample testing
  - Generates training/test windows
  - Supports overlapping or stepped windows

- **RetrainingScheduler**: Triggers retraining
  - Time-based (every N days)
  - Trade-based (every N trades)

- **PerformanceAnalyzer**: Analyzes alpha contribution
  - Per-alpha PnL and win rate
  - Regime-specific performance

### Metrics Example

```
=== Backtest Results ===
Initial Capital: $100,000
Final Value: $125,432

Performance Metrics:
- Total Return: 25.43%
- Annual Return: 25.43%
- Max Drawdown: -8.50%
- Sharpe Ratio: 1.24
- Sortino Ratio: 1.67
- Win Rate: 62.50%
- Profit Factor: 1.85
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (Main Control Loop)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 1. DATA INGESTION                     │
        │ (CSVAdapter, RESTAPIAdapter)          │
        │ Output: MarketState, Bar[]            │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 2. FEATURE ENGINEERING                │
        │ Output: Features (SMA, RSI, ATR, ...) │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 3. REGIME DETECTION                   │
        │ Output: RegimeState                   │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 4. ALPHA ENGINE                       │
        │ (Momentum, MeanReversion, Breakout)   │
        │ Output: Signal[]                      │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 5. SIGNAL PROCESSING                  │
        │ (Validate → Score → Filter)           │
        │ Output: ScoredSignal[]                │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 6. RISK ENGINE                        │
        │ (Position sizing, constraints)        │
        │ Output: Order[]                       │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 7. PORTFOLIO CONSTRUCTION             │
        │ (Weight optimization, rebalancing)    │
        │ Output: PortfolioWeights              │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 8. EXECUTION ENGINE                   │
        │ (Submit orders, track fills)          │
        │ Output: Trade[]                       │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 9. MONITORING & METRICS               │
        │ (Track PnL, equity, risk metrics)     │
        │ Output: PerformanceMetrics            │
        └───────────────────────────────────────┘
                            │
                            ↓
        ┌───────────────────────────────────────┐
        │ 10. LEARNING LOOP                     │
        │ (Model retraining, performance analysis)
        │ Output: Updated models                │
        └───────────────────────────────────────┘
```

---

## Configuration Hierarchy

```yaml
Config (Master)
├── orchestrator: OrchestratorConfig
│   ├── operation_mode: "LIVE" | "PAPER" | "BACKTEST"
│   ├── run_frequency: "minute" | "hourly" | "daily"
│   └── enabled_markets: [...]
├── data: DataConfig
│   ├── csv_path: "data/sample_ohlcv.csv"
│   ├── symbols: ["AAPL", "GOOGL", ...]
│   └── api_endpoint: "..."
├── features: FeatureConfig
│   ├── enabled_indicators: [...]
│   └── lookback_periods: {...}
├── regime: RegimeConfig
├── alpha: AlphaConfig
├── signals: SignalConfig
├── risk: RiskConfig (CRITICAL!)
├── portfolio: PortfolioConfig
├── execution: ExecutionConfig
└── monitoring: MonitoringConfig
```

---

## Safety Mechanisms

The system enforces capital preservation through:

1. **Multi-Layer Risk Checks**
   - Risk engine rejects unsafe trades
   - Position size calculated from risk %, not arbitrary amounts
   - Kill-switch activates automatically

2. **Configurable Limits**
   - All risk parameters configurable
   - Safe defaults provided
   - Live mode has stricter defaults than paper

3. **Complete Traceability**
   - Every trade logged with reasoning
   - All metrics recorded
   - Audit trail of decisions

4. **Graceful Degradation**
   - Errors don't crash system
   - Fallback to safe defaults
   - Manual kill-switch always available

---

## Performance Optimization

The system is designed for efficiency:

- **Lazy Computation**: Only compute needed indicators
- **Caching**: Optional bar data caching
- **Vectorized Operations**: Uses numpy/pandas where possible
- **Async Ready**: Can be extended for async execution
- **Minimal Dependencies**: Core only needs pandas/numpy/pytz

---

## Extensibility Examples

See main `README.md` for code examples of:
- Adding custom alpha models
- Implementing broker adapters
- Adding custom indicators
- Creating risk policies

---

## References

- **Regime Detection**: Erkki Etula's regime-based strategies
- **Risk Management**: Kelly Criterion, Monte Carlo drawdown analysis
- **Signal Scoring**: Bayesian confidence estimation
- **Portfolio Optimization**: Equal-weight, 1/N portfolio theory


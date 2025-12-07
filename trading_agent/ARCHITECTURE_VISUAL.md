# Visual Architecture Guide

## System Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (Main Loop)                             │
│                  Controls all components and timing                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │ Data Ingestion (2)   │        │ Configuration (0)    │
        │ - CSV Adapter        │        │ - YAML files         │
        │ - REST API Template  │        │ - Typed settings     │
        └──────────────────────┘        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Market Data          │
        │ - Bar (OHLCV)        │
        │ - MarketState        │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Feature Engineering  │
        │ (3) - Indicators     │
        │ - SMA, RSI, MACD, ATR│
        │ - Bollinger Bands    │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ Market Regime (4)    │
        │ - Trend/Range        │
        │ - High/Low Vol       │
        │ - Confidence score   │
        └──────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   ┌─────────┐         ┌──────────────┐
   │ Alpha 1 │         │ Alpha 2/3/N  │
   │Momentum │ ────┐   │ MeanRevrsion │───┐
   └─────────┘     │   │ Breakout     │   │
                   │   └──────────────┘   │
                   │                      │
                   └──────────┬───────────┘
                              │
                    Raw Signals: Signal[]
                              │
                              ▼
        ┌──────────────────────────────────┐
        │ Signal Processing (6)            │
        │ ├─ Validate (strength filter)   │
        │ ├─ Score (confidence + edge)    │
        │ └─ Filter (thresholds)          │
        └──────────────────────────────────┘
                              │
                   Scored Signals: ScoredSignal[]
                              │
                              ▼
        ┌──────────────────────────────────┐
        │ Risk Engine (7)                  │
        │ ├─ Check kill-switch            │
        │ ├─ Check daily drawdown         │
        │ ├─ Check position count         │
        │ └─ Size positions (Kelly)       │
        └──────────────────────────────────┘
                              │
                   Approved Orders: Order[]
                              │
                              ▼
        ┌──────────────────────────────────┐
        │ Portfolio Construction (8)       │
        │ ├─ Equal weight                 │
        │ ├─ Volatility target            │
        │ └─ Rebalancing                  │
        └──────────────────────────────────┘
                              │
                   Portfolio Weights: PortfolioWeights
                              │
                              ▼
        ┌──────────────────────────────────┐
        │ Execution Engine (9)            │
        │ ├─ Submit orders                │
        │ ├─ Track fills                  │
        │ └─ Retry logic                  │
        └──────────────────────────────────┘
                              │
                   Filled Orders: Trade[]
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
            ▼                                    ▼
    ┌──────────────────┐         ┌──────────────────────────┐
    │ Monitoring (10)  │         │ Learning (10)            │
    │ - Metrics        │         │ - Walk-forward testing   │
    │ - Logging        │         │ - Model retraining       │
    │ - Performance    │         │ - Alpha analysis         │
    └──────────────────┘         └──────────────────────────┘
            │                                    │
            └─────────────────┬──────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Portfolio State      │
                    │ - Equity curve       │
                    │ - Trade history      │
                    │ - Metrics snapshot   │
                    └──────────────────────┘
                              │
                              ▼
                    [Loop back to Orchestrator]
```

## Component Interaction Matrix

```
                    Orchestrator → ... → Signal Processor
                         │                      │
                         ▼                      ▼
┌─────────┬──────────┬─────────┬────────┬──────────┬────────┬─────────────┐
│ Data    │ Features │ Regime  │ Alphas │ Signals  │ Risk   │ Portfolio   │
│ Layer   │ Engine   │ Detector│ Engine │ Proc.    │ Engine │ Builder     │
├─────────┼──────────┼─────────┼────────┼──────────┼────────┼─────────────┤
│ Input:  │ Input:   │ Input:  │ Input: │ Input:   │ Input: │ Input:      │
│ Symbols │ Bar[]    │ Features│ Market │ Signal[] │ Signal │ ScoredSignal│
│ Dates   │ Config   │ Config  │ State  │ Config   │ Config │ Config      │
│         │          │ Detector│ Regime │          │        │             │
├─────────┼──────────┼─────────┼────────┼──────────┼────────┼─────────────┤
│ Output: │ Output:  │ Output: │ Output:│ Output:  │ Output:│ Output:     │
│ Bar[]   │ Features │ Regime  │ Signal │ Scored   │ Risk   │ Weights     │
│ Price   │          │ State   │ List   │ Signal   │ Assess │ Rebal       │
│         │          │         │        │          │ Orders │ Orders      │
└─────────┴──────────┴─────────┴────────┴──────────┴────────┴─────────────┘
```

## Risk Engine Workflow

```
ScoredSignal
    │
    ├─────────────────────────┐
    │                         │
    ▼                         ▼
Kill Switch Check      Daily Drawdown Check
    │                         │
    ├─────────────────────────┤
    │
    ▼
Position Count Check
    │
    ├─────────────────────────┐
    │                         │
    ▼                         ▼
Pass ✓                  Fail ✗
    │                         │
    ▼                         ▼
Calculate            Reject Trade
Position Size           │
    │              RiskAssessment
    ▼              (action=REJECT)
Entry Price
Stop Loss (ATR)
    │
    ▼
Position Size
    │
    ▼
RiskAssessment
(action=ACCEPT)
```

## Multi-Alpha Integration

```
                        Orchestrator
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
           Momentum       MeanReversion  Breakout
           Alpha          Alpha          Alpha
             │              │              │
             └──────────┬───┴─────┬───────┘
                        │         │
                   Signal[], strength
                        │
                        ▼
            Signal Validator
                (min strength)
                        │
                   Validated[]
                        │
                        ▼
            Signal Scorer
         (confidence, edge)
                        │
                   ScoredSignal[]
                        │
                        ▼
            Signal Filter
        (min confidence, edge)
                        │
              High-Quality Signals
                        │
                        ▼
            Risk Engine (approval)
                        │
              Approved Orders
```

## Configuration Hierarchy

```
Config (Master)
    │
    ├─ orchestrator
    │   ├─ operation_mode: LIVE | PAPER | BACKTEST
    │   ├─ run_frequency: minute | hourly | daily
    │   └─ enabled_markets: [...]
    │
    ├─ data
    │   ├─ csv_path: "..."
    │   ├─ symbols: [AAPL, GOOGL, ...]
    │   └─ api_endpoint: "..."
    │
    ├─ features
    │   ├─ enabled_indicators: [SMA_20, RSI, MACD, ...]
    │   └─ lookback_periods: {...}
    │
    ├─ regime
    │   ├─ detector_type: simple_rules | ml
    │   ├─ trend_threshold: 0.5
    │   └─ volatility_threshold: 1.0
    │
    ├─ alpha
    │   └─ enabled_models: [momentum, mean_reversion, breakout]
    │
    ├─ signals
    │   ├─ min_confidence: 0.5
    │   └─ min_edge: 0.01
    │
    ├─ risk ⭐ MOST IMPORTANT
    │   ├─ max_position_risk_pct: 1.0
    │   ├─ max_daily_drawdown_pct: 5.0
    │   ├─ max_weekly_drawdown_pct: 10.0
    │   ├─ max_concurrent_positions: 10
    │   ├─ kill_switch_enabled: true
    │   └─ kill_switch_drawdown_pct: 20.0
    │
    ├─ portfolio
    │   ├─ diversification_method: equal_weight
    │   ├─ max_sector_exposure_pct: 30.0
    │   └─ max_single_position_pct: 10.0
    │
    ├─ execution
    │   ├─ execution_mode: simulated | live
    │   ├─ slippage_bps: 2.0
    │   ├─ spread_bps: 1.0
    │   └─ max_retries: 3
    │
    └─ monitoring
        ├─ log_level: INFO | DEBUG
        ├─ log_path: "logs/"
        └─ store_trades: true
```

## Data Type Relationships

```
Bar ────────────┐
                │
MarketState ────┼──→ Features ──┐
                │               │
                │               ├──→ Regime Detection
                │               │
                │               └──→ Alpha Models
                │                      │
                └──────────────┐       │
                               │       │
                    ┌──────────┴───────┘
                    │
                    ▼
                Signal (raw)
                    │
                    ▼
            ┌───────────────┐
            │ Validation    │
            │ Scoring       │
            │ Filtering     │
            └───────────────┘
                    │
                    ▼
            ScoredSignal
                    │
                    ▼
            RiskEngine
                    │
                    ▼
                Order
                    │
                    ▼
            ExecutionEngine
                    │
                    ▼
                Trade
                    │
            ┌───────┴────────┐
            │                │
            ▼                ▼
        Position         PerformanceMetrics
        PortfolioState   Metrics Tracker
```

## Operating Modes

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
└─────────────────────────────────────────────────────────────┘
    │
    ├─ BACKTEST Mode
    │   │
    │   ├─ Load historical data
    │   ├─ Replay timestamps
    │   ├─ SimulatedExecutor (perfect fills)
    │   └─ Output: Equity curve, metrics
    │
    ├─ PAPER Mode
    │   │
    │   ├─ Real-time data (simulated)
    │   ├─ Run continuously
    │   ├─ SimulatedExecutor (with slippage)
    │   └─ Output: Live metrics
    │
    └─ LIVE Mode (⚠️ Danger Zone)
        │
        ├─ Real-time data
        ├─ Run continuously
        ├─ LiveBrokerAdapter
        ├─ Real capital at risk!
        └─ Output: Real P&L
```

## Extension Points

```
┌──────────────────────────────────────────────────────┐
│ Easy to Extend (Clear Interfaces)                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│ DataAdapter (ABC)                                   │
│ ├─ CSVAdapter ✓                                     │
│ ├─ RESTAPIAdapter (template)                        │
│ └─ YourBrokerAdapter (extend)                       │
│                                                      │
│ AlphaModel (ABC)                                    │
│ ├─ MomentumAlpha ✓                                  │
│ ├─ MeanReversionAlpha ✓                             │
│ ├─ BreakoutAlpha ✓                                  │
│ └─ YourCustomAlpha (extend)                         │
│                                                      │
│ ExecutionAdapter (ABC)                              │
│ ├─ SimulatedExecutor ✓                              │
│ └─ YourBrokerAdapter (extend)                       │
│                                                      │
│ RegimeDetector (ABC)                                │
│ ├─ SimpleRulesRegimeDetector ✓                      │
│ ├─ MLRegimeDetector (template)                      │
│ └─ YourCustomDetector (extend)                      │
│                                                      │
│ FeatureEngineering                                  │
│ ├─ Built-in indicators ✓                            │
│ └─ Custom indicators (add methods)                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## File Organization

```
trading_agent/
│
├── 📄 Entry Points (3)
│   ├─ main_live.py
│   ├─ main_paper.py
│   └─ main_backtest.py
│
├── ⚙️ Config (4)
│   ├─ config/config.py (system)
│   ├─ config/paper_config.yaml
│   ├─ config/backtest_config.yaml
│   └─ config/live_config.yaml
│
├── 🔧 Core Modules (13)
│   ├─ orchestrator/ → main control
│   ├─ data/ → market data
│   ├─ features/ → indicators
│   ├─ regime/ → market conditions
│   ├─ alpha/ → strategies (5 variants)
│   ├─ signals/ → processing pipeline
│   ├─ risk/ → position sizing & limits
│   ├─ portfolio/ → weight optimization
│   ├─ execution/ → order placement
│   ├─ monitoring/ → metrics
│   ├─ backtest/ → simulator
│   ├─ learning/ → retraining
│   └─ utils/ → data types
│
├── 📚 Scripts & Tests (4)
│   ├─ scripts/generate_sample_data.py
│   ├─ tests/test_core.py
│   ├─ requirements.txt
│   └─ setup.py
│
└── 📖 Documentation (6)
    ├─ QUICKSTART.md (5 min)
    ├─ README.md (complete ref)
    ├─ ARCHITECTURE.md (technical)
    ├─ INDEX.md (navigation)
    ├─ IMPLEMENTATION_SUMMARY.md
    └─ COMPLETION_REPORT.md
```

---

**This architecture enables:**
✓ Modular development
✓ Easy testing
✓ Clear responsibilities
✓ Simple extension
✓ Maintainability
✓ Scalability


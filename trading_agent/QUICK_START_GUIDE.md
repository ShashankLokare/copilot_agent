# Quick Start Guide - Multi-Market Trading System

## 🚀 Getting Started in 2 Minutes

### Run Backtest with Market Selection
```bash
python main_backtest.py
```
Then select your market (1 for USA, 2 for NSE) when prompted.

### Run Paper Trading
```bash
python main_paper.py
```
Same market selection as backtest.

### Run Live Trading
```bash
python main_live.py
```
⚠️ **Warning**: Only run this with proper API credentials and risk limits configured.

---

## 📊 Available Markets

### USA Market
- **Stocks**: AAPL, GOOGL, MSFT, AMZN, NVDA
- **Execution Cost**: 1 bps slippage, 1 bps spread
- **Risk Limit**: 5% daily drawdown
- **Best For**: Tech stocks, high liquidity trading

### NSE (India) Market
- **Stocks**: TCS, INFY, RELIANCE, HDFCBANK, BAJAJ-AUTO
- **Execution Cost**: 3 bps slippage, 2 bps spread
- **Risk Limit**: 3% daily drawdown
- **Best For**: Emerging market exposure, NIFTY50 trading

---

## 🎯 Market Selection Example

```
==============================================================
ALGORITHMIC TRADING SYSTEM - MARKET SELECTION
==============================================================

Please select your trading market:

  1) USA Stock Market (NASDAQ/NYSE)
     - Symbols: AAPL, GOOGL, MSFT, AMZN, NVDA
     - Trading Hours: 9:30 AM - 4:00 PM EST
     - Slippage: ~1 bps

  2) Indian Stock Market (NSE NIFTY50)
     - Symbols: TCS, INFY, RELIANCE, HDFCBANK, BAJAJ-AUTO
     - Trading Hours: 9:15 AM - 3:30 PM IST
     - Slippage: ~3 bps

Enter your choice (1 for USA, 2 for NSE): [Type 1 or 2]
```

---

## 🧪 Testing & Validation

### Validate Both Markets
```bash
python test_markets.py
```
Checks configuration, data files, and data integrity.

### Run Backtests for Both Markets
```bash
python run_backtest_markets.py
```
Executes backtests for USA and NSE automatically.

### Explore Market Details
```bash
python demo_markets.py
```
Interactive menu to view and compare market configurations.

---

## 📁 Key Files

### Market Selection
- `utils/market_selector.py` - Interactive market selection

### Configurations
- `config/usa_config.yaml` - USA market parameters
- `config/nse_config.yaml` - NSE market parameters

### Sample Data
- `data/usa_ohlcv.csv` - USA market data (260 days × 5 stocks)
- `data/nse_ohlcv.csv` - NSE market data (260 days × 5 stocks)

### Entry Points
- `main_backtest.py` - Backtest mode with market selection
- `main_paper.py` - Paper trading with market selection
- `main_live.py` - Live trading with market selection

### Tests
- `test_markets.py` - Market compatibility tests
- `run_backtest_markets.py` - Automated backtests
- `demo_markets.py` - Interactive market explorer

---

## ⚙️ Market-Specific Parameters

### Execution
| Parameter | USA | NSE |
|-----------|-----|-----|
| Slippage | 1 bps | 3 bps |
| Spread | 1 bps | 2 bps |
| Order Type | MARKET | MARKET |

### Risk Management
| Parameter | USA | NSE |
|-----------|-----|-----|
| Max Daily DD | 5% | 3% |
| Max Weekly DD | 10% | 5% |
| Max Positions | 10 | 5 |
| Kill-Switch | 20% | 10% |

### Portfolio
| Parameter | USA | NSE |
|-----------|-----|-----|
| Method | equal_weight | equal_weight |
| Rebalance | weekly | weekly |
| Max Position | 10% | 10% |

---

## 🔄 Workflow Examples

### Example 1: Backtest USA Market
```bash
$ python main_backtest.py
[Market selection prompt appears]
Enter your choice (1 for USA, 2 for NSE): 1
[Backtester runs with USA config]
[Results displayed]
```

### Example 2: Backtest NSE Market
```bash
$ python main_backtest.py
[Market selection prompt appears]
Enter your choice (1 for USA, 2 for NSE): 2
[Backtester runs with NSE config]
[Results displayed]
```

### Example 3: Compare Both Markets
```bash
$ python demo_markets.py
[Interactive menu]
Option 3: Compare USA vs NSE Markets
[Side-by-side comparison shown]
```

---

## 📈 Understanding Results

### Backtest Output
```
=== Algorithmic Trading System - BACKTEST MODE (USA) ===

Initial Capital:    $100,000.00
Final Value:        $96,139.91
Total Return:           -3.86%
Annual Return:          -4.90%
Max Drawdown:          26.96%
Sharpe Ratio:            0.00
```

### Key Metrics
- **Total Return**: Overall profit/loss percentage
- **Max Drawdown**: Worst peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Win Rate**: Percentage of profitable trades

---

## ⚠️ Important Notes

### Risk Management
- NSE has stricter daily drawdown limits (3% vs 5%)
- Kill-switch activates earlier for NSE (10% vs 20%)
- Start with paper trading before live trading
- Monitor risk metrics during trading

### Data
- Sample data is historical 2023 data
- Use real-time data adapters for live trading
- Data formats must match CSV specification
- Verify data quality before trading

### Market Hours
- USA: 9:30 AM - 4:00 PM EST (Eastern Time)
- NSE: 9:15 AM - 3:30 PM IST (Indian Standard Time)
- Consider market hours when scheduling trades

---

## 🛠️ Customization

### Change Market Parameters
Edit the appropriate config file:
- USA: `config/usa_config.yaml`
- NSE: `config/nse_config.yaml`

### Add New Market
1. Create `config/<market>_config.yaml`
2. Update `utils/market_selector.py`
3. Add market enum and symbols
4. Run tests to validate

### Use Custom Data
1. Prepare CSV with columns: symbol, timestamp, open, high, low, close, volume
2. Update config file `data.csv_path`
3. Run tests to verify data loads

---

## 📚 Documentation

- **Detailed Guide**: `NSE_ADAPTATION_SUMMARY.md`
- **Completion Report**: `PHASE_3_COMPLETION_REPORT.md`
- **Architecture**: `ARCHITECTURE.md`
- **README**: `README.md`

---

## 🆘 Troubleshooting

### "Config file not found"
- Make sure config YAML file exists
- Check file path matches market selection
- Verify file is readable

### "CSV file not found"
- Generate sample data: `python scripts/generate_sample_data.py`
- Or: `python scripts/generate_nse_sample_data.py`
- Check data path in config file

### "Backtest fails to run"
- Run `python test_markets.py` to validate setup
- Check for Python errors in terminal output
- Ensure all dependencies installed

### "Market selection not working"
- Check utils/market_selector.py exists
- Make sure you're using updated entry point files
- Try restarting Python interpreter

---

## 💡 Tips

1. **Always test first**: Use backtest mode before paper/live trading
2. **Start small**: Use fewer symbols until comfortable
3. **Monitor metrics**: Check Sharpe ratio and drawdown regularly
4. **Document runs**: Keep notes on market-specific results
5. **Compare markets**: Use demo_markets.py to understand differences

---

## 📞 Support

For issues or questions:
1. Check documentation files (*.md)
2. Review configuration examples
3. Run validation tests
4. Check inline code comments
5. Review backtest results for insights

---

**Last Updated**: December 2025
**System Version**: Production-Ready with USA + NSE Markets
**Status**: ✅ Fully Operational

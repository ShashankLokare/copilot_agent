#!/usr/bin/env python3
"""
Main entry point for backtesting (default: NSE, non-interactive).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from backtest.backtester import Backtester
from utils.market_selector import get_market_config_path, Market
from datetime import datetime
from alpha.alpha_models import MLAlphaXGBoost
from signals.signal_processor import SignalProcessor, SignalFilter, SignalValidator


def main():
    """Run backtest for NSE without interactive selection."""
    market = Market.NSE
    config_path = get_market_config_path(market)
    
    print(f"\n=== Algorithmic Trading System - BACKTEST MODE ({market.value.upper()}) ===\n")
    
    # Load configuration
    try:
        config = Config.load_from_file(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = Config()
    
    # Load data
    try:
        data_adapter = CSVAdapter(config.data.csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {config.data.csv_path}")
        print("Please configure data.csv_path in config file")
        sys.exit(1)
    
    # High-conviction ML alpha (gated)
    ml_alpha = MLAlphaXGBoost(
        model_id="nifty50_xgboost_adv_20251207_155931",
        prob_long=0.53,
        prob_short=0.20,
        min_expected_edge=0.0005,
        allow_regime_buckets=None,  # allow all regimes for now; refine once logged
        allow_shorts=False,
    )
    signal_processor = SignalProcessor(
        validator=SignalValidator(min_signal_strength=0.5, require_confirmation=False),
        filter=SignalFilter(min_confidence=0.5, min_edge=0.005),
    )

    # Initialize backtester
    backtester = Backtester(
        data_adapter=data_adapter,
        initial_capital=100000.0,
        slippage_bps=config.execution.slippage_bps,
        spread_bps=config.execution.spread_bps,
        signal_processor=signal_processor,
        alphas=[ml_alpha],
    )
    
    # Run backtest
    print("Running backtest...")
    print(f"Symbols: {', '.join(config.data.symbols)}")
    print(f"Initial Capital: ${100000.0:,.2f}")
    print()
    
    results = backtester.run(
        symbols=config.data.symbols,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        timeframe="1d",
    )
    
    # Print results
    print("\n=== Backtest Results ===")
    if "error" in results:
        print(f"ERROR: {results['error']}")
    else:
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Value:        ${results['final_value']:,.2f}")
        print()
        
        metrics = results['metrics']
        print("Performance Metrics:")
        print(f"  Total Return:      {metrics['total_return_pct']:>8.2f}%")
        print(f"  Annual Return:     {metrics['annual_return_pct']:>8.2f}%")
        print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:>8.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}")
        print(f"  Win Rate:          {metrics['win_rate']*100:>8.2f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:>8.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main entry point for backtesting with market selection.
Supports both USA and Indian NSE markets.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from backtest.backtester import Backtester
from utils.market_selector import select_market, get_market_config_path
from datetime import datetime


def main():
    """Run backtest with market selection."""
    # Select market
    market = select_market()
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
    
    # Initialize backtester
    backtester = Backtester(
        data_adapter=data_adapter,
        initial_capital=100000.0,
        slippage_bps=config.execution.slippage_bps,
        spread_bps=config.execution.spread_bps,
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

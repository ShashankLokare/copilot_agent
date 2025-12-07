#!/usr/bin/env python3
"""
Interactive demo script to showcase both USA and NSE market trading modes.
Demonstrates market selection and configuration for each market.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from utils.market_selector import (
    select_market,
    get_market_config_path,
    get_market_name,
    get_market_symbols
)


def display_market_info(market_enum):
    """Display detailed information about the selected market."""
    config_path = get_market_config_path(market_enum)
    config = Config.load_from_file(config_path)
    market_name = get_market_name(market_enum)
    
    print(f"\n{'='*70}")
    print(f"MARKET INFORMATION: {market_name}")
    print('='*70)
    
    # Basic info
    print(f"\nMarket Overview:")
    print(f"  Market: {market_name}")
    print(f"  Configuration File: {config_path}")
    print(f"  Data Source: {config.data.csv_path}")
    
    # Symbols
    print(f"\nTrading Symbols ({len(config.data.symbols)} stocks):")
    for i, symbol in enumerate(config.data.symbols, 1):
        print(f"  {i}. {symbol}")
    
    # Execution Parameters
    print(f"\nExecution Parameters:")
    print(f"  Order Type: {config.execution.default_order_type}")
    print(f"  Slippage: {config.execution.slippage_bps} basis points")
    print(f"  Bid-Ask Spread: {config.execution.spread_bps} basis points")
    print(f"  Execution Mode: {config.execution.execution_mode}")
    
    # Risk Management
    print(f"\nRisk Management:")
    print(f"  Max Position Risk: {config.risk.max_position_risk_pct}%")
    print(f"  Max Daily Drawdown: {config.risk.max_daily_drawdown_pct}%")
    print(f"  Max Weekly Drawdown: {config.risk.max_weekly_drawdown_pct}%")
    print(f"  Kill-Switch Level: {config.risk.kill_switch_drawdown_pct}%")
    print(f"  Max Concurrent Positions: {config.risk.max_concurrent_positions}")
    
    # Portfolio Construction
    print(f"\nPortfolio Construction:")
    print(f"  Method: {config.portfolio.method}")
    print(f"  Rebalance Frequency: {config.portfolio.rebalance_frequency}")
    print(f"  Max Position Size: {config.portfolio.max_position_size_pct}%")
    print(f"  Max Sector Exposure: {config.portfolio.max_sector_exposure_pct}%")
    
    # Data Validation
    print(f"\nData Validation:")
    try:
        adapter = CSVAdapter(config.data.csv_path)
        available_symbols = adapter.get_symbols()
        print(f"  ✓ Data file found: {config.data.csv_path}")
        print(f"  ✓ Available symbols in data: {len(available_symbols)}")
        print(f"  ✓ All configured symbols present: {set(config.data.symbols) == set(available_symbols)}")
    except Exception as e:
        print(f"  ✗ Data validation error: {e}")
    
    print()


def compare_markets():
    """Display side-by-side comparison of USA and NSE markets."""
    from utils.market_selector import Market
    
    print(f"\n{'='*70}")
    print("MARKET COMPARISON: USA vs NSE")
    print('='*70)
    
    # Load both configs
    usa_config = Config.load_from_file(get_market_config_path(Market.USA))
    nse_config = Config.load_from_file(get_market_config_path(Market.NSE))
    
    print(f"\n{'Parameter':<30} {'USA':<20} {'NSE':<20}")
    print('-'*70)
    
    print(f"{'Slippage (bps)':<30} {usa_config.execution.slippage_bps:<20} {nse_config.execution.slippage_bps:<20}")
    print(f"{'Spread (bps)':<30} {usa_config.execution.spread_bps:<20} {nse_config.execution.spread_bps:<20}")
    print(f"{'Max Daily Drawdown':<30} {usa_config.risk.max_daily_drawdown_pct}% {nse_config.risk.max_daily_drawdown_pct}%")
    print(f"{'Kill-Switch Level':<30} {usa_config.risk.kill_switch_drawdown_pct}% {nse_config.risk.kill_switch_drawdown_pct}%")
    print(f"{'Max Positions':<30} {usa_config.risk.max_concurrent_positions:<20} {nse_config.risk.max_concurrent_positions:<20}")
    print(f"{'Rebalance Frequency':<30} {usa_config.portfolio.rebalance_frequency:<20} {nse_config.portfolio.rebalance_frequency:<20}")
    
    print(f"\n{'Symbols':<30} {'USA':<40} {'NSE':<40}")
    print('-'*70)
    usa_symbols = ', '.join(usa_config.data.symbols)
    nse_symbols = ', '.join(nse_config.data.symbols)
    print(f"{'':<30} {usa_symbols:<40} {nse_symbols:<40}")
    
    print()


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("ALGORITHMIC TRADING SYSTEM - MULTI-MARKET DEMO")
    print("="*70)
    
    while True:
        print(f"\n{'='*70}")
        print("MAIN MENU")
        print('='*70)
        print("\n1. View USA Market Configuration")
        print("2. View NSE (Indian) Market Configuration")
        print("3. Compare USA vs NSE Markets")
        print("4. Select Market & View Details")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            from utils.market_selector import Market
            display_market_info(Market.USA)
        elif choice == "2":
            from utils.market_selector import Market
            display_market_info(Market.NSE)
        elif choice == "3":
            compare_markets()
        elif choice == "4":
            market = select_market()
            display_market_info(market)
        elif choice == "5":
            print("\nExiting demo. Thank you!")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
        sys.exit(0)

#!/usr/bin/env python3
"""
Test script to validate both USA and NSE market modes work correctly.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from utils.market_selector import (
    get_market_config_path, 
    get_market_symbols,
    get_market_name,
    Market
)


def test_market_config(market: Market) -> bool:
    """Test that market configuration loads correctly."""
    print(f"\n{'='*60}")
    print(f"Testing {market.value.upper()} Market Configuration")
    print('='*60)
    
    config_path = get_market_config_path(market)
    print(f"Config Path: {config_path}")
    
    try:
        config = Config.load_from_file(config_path)
        print(f"✓ Config loaded successfully")
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        return False
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False
    
    # Verify symbols
    expected_symbols = get_market_symbols(market)
    actual_symbols = config.data.symbols
    print(f"Expected Symbols: {expected_symbols}")
    print(f"Actual Symbols:   {actual_symbols}")
    
    if set(expected_symbols) != set(actual_symbols):
        print(f"✗ Symbol mismatch!")
        return False
    print(f"✓ Symbols match")
    
    # Verify data file exists
    print(f"Data File: {config.data.csv_path}")
    if not Path(config.data.csv_path).exists():
        print(f"✗ Data file not found: {config.data.csv_path}")
        return False
    print(f"✓ Data file exists")
    
    # Load and verify data
    try:
        adapter = CSVAdapter(config.data.csv_path)
        print(f"✓ Data adapter initialized")
    except Exception as e:
        print(f"✗ Error initializing data adapter: {e}")
        return False
    
    # Show market-specific parameters
    print(f"\nMarket-Specific Parameters:")
    print(f"  Slippage: {config.execution.slippage_bps} bps")
    print(f"  Spread: {config.execution.spread_bps} bps")
    print(f"  Max Daily Drawdown: {config.risk.max_daily_drawdown_pct}%")
    print(f"  Kill-Switch Level: {config.risk.kill_switch_drawdown_pct}%")
    print(f"  Max Concurrent Positions: {config.risk.max_concurrent_positions}")
    
    print(f"\n✓ {market.value.upper()} market configuration valid")
    return True


def test_market_data(market: Market) -> bool:
    """Test that market data loads and has expected symbols."""
    from datetime import datetime, timedelta
    
    print(f"\n{'='*60}")
    print(f"Testing {market.value.upper()} Market Data")
    print('='*60)
    
    config_path = get_market_config_path(market)
    config = Config.load_from_file(config_path)
    
    try:
        adapter = CSVAdapter(config.data.csv_path)
        
        # Get available symbols
        available_symbols = adapter.get_symbols()
        print(f"✓ Available symbols: {sorted(available_symbols)}")
        
        # Fetch bars for first symbol with broad date range
        first_symbol = config.data.symbols[0]
        bars = adapter.fetch_bars(
            symbol=first_symbol,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2025, 12, 31),
            timeframe=config.data.timeframe
        )
        print(f"✓ Fetched {len(bars)} bars for {first_symbol}")
        
        if bars:
            first_bar = bars[0]
            last_bar = bars[-1]
            print(f"  Date range: {first_bar.timestamp} to {last_bar.timestamp}")
            print(f"  First bar: Open={first_bar.open:.2f}, Close={first_bar.close:.2f}, Volume={first_bar.volume:,.0f}")
        
        # Verify all expected symbols are available
        missing_symbols = set(config.data.symbols) - set(available_symbols)
        if missing_symbols:
            print(f"✗ Missing symbols in data: {missing_symbols}")
            return False
        
        print(f"✓ All {len(config.data.symbols)} expected symbols present")
        return True
    except Exception as e:
        print(f"✗ Error loading market data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all market tests."""
    print("\n" + "="*60)
    print("MARKET COMPATIBILITY TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Test each market
    for market in [Market.USA, Market.NSE]:
        market_name = get_market_name(market)
        print(f"\n>>> Testing {market_name}")
        
        # Test configuration
        if not test_market_config(market):
            all_passed = False
            continue
        
        # Test data
        if not test_market_data(market):
            all_passed = False
            continue
    
    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("System is ready for both USA and NSE market trading")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please fix the errors above before trading")
    print('='*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

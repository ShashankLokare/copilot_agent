#!/usr/bin/env python3
"""
Automated backtest runner for both markets.
Useful for CI/CD and automated validation.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from backtest.backtester import Backtester
from utils.market_selector import (
    Market,
    get_market_config_path,
    get_market_name,
)


def run_backtest_for_market(market: Market) -> dict:
    """Run a backtest for a specific market and return results."""
    market_name = get_market_name(market)
    config_path = get_market_config_path(market)
    
    print(f"\n{'='*70}")
    print(f"Running Backtest: {market_name}")
    print('='*70)
    print(f"Config: {config_path}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load configuration
        config = Config.load_from_file(config_path)
        print(f"✓ Configuration loaded")
        
        # Load data adapter
        data_adapter = CSVAdapter(config.data.csv_path)
        print(f"✓ Data adapter initialized")
        print(f"  Data file: {config.data.csv_path}")
        print(f"  Symbols: {', '.join(config.data.symbols)}")
        
        # Initialize backtester
        backtester = Backtester(
            data_adapter=data_adapter,
            initial_capital=100000.0,
            slippage_bps=config.execution.slippage_bps,
            spread_bps=config.execution.spread_bps,
        )
        print(f"✓ Backtester initialized")
        
        # Run backtest
        print(f"\nRunning backtest...")
        results = backtester.run(
            symbols=config.data.symbols,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            timeframe="1d",
        )
        print(f"✓ Backtest completed successfully")
        
        # Get metrics
        metrics = results.get('metrics', {})
        print(f"\nResults:")
        print(f"  Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"  Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        return {
            "market": market.value,
            "success": True,
            "error": None,
            "results": results,
        }
    except Exception as e:
        print(f"✗ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return {
            "market": market.value,
            "success": False,
            "error": str(e),
            "results": None,
        }


def main():
    """Run backtests for both markets."""
    print("\n" + "="*70)
    print("AUTOMATED BACKTEST RUNNER - MULTI-MARKET")
    print("="*70)
    
    results = []
    
    # Run backtest for each market
    for market in [Market.USA, Market.NSE]:
        result = run_backtest_for_market(market)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    for result in results:
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"{status} - {result['market'].upper()}")
        if not result["success"]:
            print(f"       Error: {result['error']}")
        else:
            metrics = result["results"].get('metrics', {})
            print(f"       Return: {metrics.get('total_return_pct', 0):.2f}%  |  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    print(f"\nTotal: {successful} passed, {failed} failed")
    print('='*70 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

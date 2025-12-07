#!/usr/bin/env python3
"""
Main entry point for live trading with market selection.
Supports both USA and Indian NSE markets.
WARNING: Only run this with appropriate safety checks and risk limits enabled!
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from data.adapters import CSVAdapter
from orchestrator.orchestrator import Orchestrator
from utils.market_selector import select_market, get_market_config_path


def main():
    """Run live trading with market selection."""
    # Select market
    market = select_market()
    config_path = get_market_config_path(market)
    
    print(f"\n=== Algorithmic Trading System - LIVE MODE ({market.value.upper()}) ===")
    print("WARNING: Only enable live trading after thorough testing!")
    print(f"Selected Market: {market.value.upper()}")
    print()
    
    # Load configuration
    try:
        config = Config.load_from_file(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = Config()
    
    # Verify operation mode
    if config.orchestrator.operation_mode != "LIVE":
        print("ERROR: operation_mode is not set to LIVE")
        print(f"Current mode: {config.orchestrator.operation_mode}")
        sys.exit(1)
    
    # Verify safety limits
    print("Safety Checks:")
    print(f"  - Max position risk: {config.risk.max_position_risk_pct}%")
    print(f"  - Max daily drawdown: {config.risk.max_daily_drawdown_pct}%")
    print(f"  - Kill-switch enabled: {config.risk.kill_switch_enabled}")
    print(f"  - Kill-switch level: {config.risk.kill_switch_drawdown_pct}%")
    print()
    
    # Load data adapter
    data_adapter = CSVAdapter(config.data.csv_path)
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config, data_adapter)
    
    # Run
    try:
        orchestrator.run_continuous(
            run_frequency=config.orchestrator.run_frequency,
            symbols=config.data.symbols,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        orchestrator.stop()


if __name__ == "__main__":
    main()

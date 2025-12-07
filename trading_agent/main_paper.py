#!/usr/bin/env python3
"""
Main entry point for paper trading with market selection.
Supports both USA and Indian NSE markets.
Safe mode for testing and development.
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
    """Run paper trading with market selection."""
    # Select market
    market = select_market()
    config_path = get_market_config_path(market)
    
    print(f"\n=== Algorithmic Trading System - PAPER TRADING ({market.value.upper()}) ===")
    print("This is a simulated trading environment. No real capital at risk.")
    print()
    
    # Load configuration
    try:
        config = Config.load_from_file(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = Config()
    
    print(f"Operation Mode: {config.orchestrator.operation_mode}")
    print(f"Symbols: {', '.join(config.data.symbols)}")
    print(f"Run Frequency: {config.orchestrator.run_frequency}")
    print()
    
    # Load data adapter
    try:
        data_adapter = CSVAdapter(config.data.csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {config.data.csv_path}")
        print("Please configure data.csv_path in config file")
        sys.exit(1)
    
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
        
        # Print final metrics
        print("\n=== Final Metrics ===")
        metrics = orchestrator.get_metrics_summary()
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()

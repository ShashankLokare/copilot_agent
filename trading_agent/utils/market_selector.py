"""
Market selector utility for choosing between USA and Indian NSE markets.
"""

from enum import Enum
from pathlib import Path


class Market(Enum):
    """Supported markets."""
    USA = "usa"
    NSE = "nse"


def get_market_config_path(market: Market) -> str:
    """Get configuration file path for a market."""
    if market == Market.USA:
        return "config/usa_config.yaml"
    elif market == Market.NSE:
        return "config/nse_config.yaml"
    else:
        raise ValueError(f"Unsupported market: {market}")


def get_market_symbols(market: Market) -> list:
    """Get default symbols for a market."""
    if market == Market.USA:
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
    elif market == Market.NSE:
        return ["TCS", "INFY", "RELIANCE", "HDFC", "BAJAJ-AUTO"]
    else:
        raise ValueError(f"Unsupported market: {market}")


def get_market_name(market: Market) -> str:
    """Get human-readable market name."""
    if market == Market.USA:
        return "USA Stock Market (NASDAQ/NYSE)"
    elif market == Market.NSE:
        return "Indian Stock Market (NSE NIFTY50)"
    else:
        return "Unknown Market"


def select_market() -> Market:
    """Interactive market selection prompt."""
    print("\n" + "="*60)
    print("ALGORITHMIC TRADING SYSTEM - MARKET SELECTION")
    print("="*60 + "\n")
    
    print("Please select your trading market:\n")
    print("  1) USA Stock Market (NASDAQ/NYSE)")
    print("     - Symbols: AAPL, GOOGL, MSFT, AMZN, NVDA")
    print("     - Trading Hours: 9:30 AM - 4:00 PM EST")
    print("     - Slippage: ~1 bps\n")
    
    print("  2) Indian Stock Market (NSE NIFTY50)")
    print("     - Symbols: TCS, INFY, RELIANCE, HDFC, BAJAJ-AUTO")
    print("     - Trading Hours: 9:15 AM - 3:30 PM IST")
    print("     - Slippage: ~3 bps\n")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print(f"\n✓ Selected: {get_market_name(Market.USA)}")
                return Market.USA
            elif choice == "2":
                print(f"\n✓ Selected: {get_market_name(Market.NSE)}")
                return Market.NSE
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def get_sample_data_path(market: Market) -> str:
    """Get sample data file path for a market."""
    if market == Market.USA:
        return "data/usa_ohlcv.csv"
    elif market == Market.NSE:
        return "data/nse_ohlcv.csv"
    else:
        raise ValueError(f"Unsupported market: {market}")

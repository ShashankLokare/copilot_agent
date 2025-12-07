#!/usr/bin/env python3
"""
Generate sample NSE (Indian stock market) OHLCV data for backtesting.
Creates realistic data for NIFTY50 stocks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def generate_nse_sample_data(
    output_path: str = "data/nse_ohlcv.csv",
    start_date: datetime = None,
    end_date: datetime = None,
    symbols: list = None,
):
    """
    Generate realistic NSE market data.
    
    Args:
        output_path: Where to save the CSV
        start_date: Start date for generation
        end_date: End date for generation
        symbols: List of symbols to generate
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime(2023, 12, 31)
    if symbols is None:
        symbols = ["TCS", "INFY", "RELIANCE", "HDFC", "BAJAJ-AUTO"]
    
    # Generate date range (excluding weekends and holidays)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business days
    
    data = []
    np.random.seed(42)
    
    # Starting prices for NSE stocks (realistic)
    starting_prices = {
        "TCS": 3500,
        "INFY": 1800,
        "RELIANCE": 2600,
        "HDFC": 2400,
        "BAJAJ-AUTO": 6200,
    }
    
    for symbol in symbols:
        price = starting_prices.get(symbol, 2000)
        
        for date in all_dates:
            # Generate realistic OHLCV data
            # Daily return: mean ~0.05%, std ~1.5%
            daily_return = np.random.normal(0.0005, 0.015)
            
            # Intraday volatility
            open_price = price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price * (1 + daily_return)
            
            # Ensure OHLC constraints
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume: 1-10 million shares typically
            volume = np.random.uniform(1000000, 10000000)
            
            data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
            
            price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} rows of NSE sample data")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    generate_nse_sample_data()

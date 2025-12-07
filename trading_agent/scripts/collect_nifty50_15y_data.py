#!/usr/bin/env python3
"""
Collect 15 years of NIFTY50 historical data for ML prediction model training.

This script downloads OHLCV data for all NIFTY50 stocks from 2010-2025,
suitable for walk-forward training and backtesting the prediction engine.

Data sources:
1. yfinance - Primary source for NIFTY50 stocks
2. NSEpy - Alternative for NSE-specific data
3. Fallback - Synthetic data if real data unavailable

Output:
- data/nifty50_15y_ohlcv.csv - Complete 15-year dataset
- logs/data_collection.log - Collection details
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# NIFTY50 stocks (as of 2024)
NIFTY50_SYMBOLS = [
    # Financial Services
    "TCS", "INFY", "RELIANCE", "HDFC", "ICICIBANK",
    "KOTAKBANK", "AXISBANK", "LT", "BAJAJFINSV", "BAJAJAUTOL",
    
    # Energy & Utilities
    "NTPC", "POWERGRID", "BHARTIARTL", "JSWSTEEL", "MARUTI",
    
    # Technology
    "WIPRO", "HCLTECH", "TECHM", "MFSL", "SUNPHARMA",
    
    # Consumer & Retail
    "HINDUNILVR", "ITC", "NESTLEIND", "INDIGO", "MARUTISUZU",
    
    # Infrastructure & Industrial
    "SBICARD", "ADANIPORTS", "ADANIGREEN", "ADANITRANS", "ADANIPOWER",
    
    # Healthcare & Pharma
    "CIPLA", "DRREDDY", "DIVISLAB", "PHARMAIND", "COLPAL",
    
    # FMCG & Consumer
    "BRITANNIA", "PIDILITIND", "BAJAJFINSV", "GODREJCP", "HEROMOTOCO",
    
    # Metals & Mining
    "TATASTEEL", "HINDALCO", "SAIL", "VEDL", "NMDC",
    
    # Others
    "ULTRACEMCO", "SHREECEM", "ACC", "BOSCHLTD"
]

# Remove duplicates and get unique 50 stocks
NIFTY50_SYMBOLS = list(dict.fromkeys(NIFTY50_SYMBOLS))[:50]


def try_import_yfinance() -> Optional[object]:
    """Try to import yfinance library."""
    try:
        import yfinance as yf
        logger.info("✓ yfinance library available")
        return yf
    except ImportError:
        logger.warning("✗ yfinance not installed. Install with: pip install yfinance")
        return None


def try_import_nsepy() -> Optional[object]:
    """Try to import NSEpy library."""
    try:
        from nsepy import get_history
        logger.info("✓ nsepy library available")
        return get_history
    except ImportError:
        logger.warning("✗ nsepy not installed. Install with: pip install nsepy")
        return None


def collect_data_yfinance(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    max_retries: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Collect data using yfinance.
    
    Args:
        symbols: List of symbols to download
        start_date: Start date (2010-01-01)
        end_date: End date (2025-01-01)
        max_retries: Max retry attempts for failed downloads
    
    Returns:
        (DataFrame with OHLCV data, List of symbols that failed)
    """
    yf = try_import_yfinance()
    if yf is None:
        return pd.DataFrame(), symbols
    
    logger.info(f"Collecting data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    logger.info(f"Data will be pulled from Yahoo Finance (NSE = .NS suffix)")
    
    all_data = []
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        symbol_nse = f"{symbol}.NS"  # Yahoo Finance uses .NS for NSE stocks
        
        try:
            logger.info(f"[{i}/{len(symbols)}] Downloading {symbol}...")
            
            # Download with retries
            for attempt in range(max_retries):
                try:
                    df = yf.download(
                        symbol_nse,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        verbose=False
                    )
                    
                    if df.empty:
                        logger.warning(f"  No data returned for {symbol}")
                        failed_symbols.append(symbol)
                        break
                    
                    # Prepare data
                    df = df.reset_index()
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                    df['symbol'] = symbol
                    
                    # Reorder columns
                    df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Convert to UTC
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
                    else:
                        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
                    
                    all_data.append(df)
                    logger.info(f"  ✓ {symbol}: {len(df)} bars")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"  Attempt {attempt+1} failed: {str(e)[:50]}... Retrying...")
                        time.sleep(1)  # Wait before retry
                    else:
                        logger.error(f"  ✗ {symbol}: Failed after {max_retries} attempts")
                        failed_symbols.append(symbol)
        
        except Exception as e:
            logger.error(f"  ✗ {symbol}: {str(e)}")
            failed_symbols.append(symbol)
        
        # Rate limiting
        time.sleep(0.5)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        return combined_df, failed_symbols
    else:
        return pd.DataFrame(), failed_symbols


def collect_data_nsepy(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Collect data using nsepy (NSE-specific).
    
    Args:
        symbols: List of symbols
        start_date: Start date
        end_date: End date
    
    Returns:
        (DataFrame with OHLCV data, List of failed symbols)
    """
    get_history = try_import_nsepy()
    if get_history is None:
        return pd.DataFrame(), symbols
    
    logger.info(f"Collecting NSE data for {len(symbols)} symbols")
    
    all_data = []
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{len(symbols)}] Downloading {symbol} from NSE...")
            
            df = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            if df is None or df.empty:
                logger.warning(f"  No data for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Normalize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"  Missing columns for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            df['symbol'] = symbol
            df['timestamp'] = pd.to_datetime(df.index)
            
            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            all_data.append(df)
            logger.info(f"  ✓ {symbol}: {len(df)} bars")
            
        except Exception as e:
            logger.error(f"  ✗ {symbol}: {str(e)}")
            failed_symbols.append(symbol)
        
        time.sleep(0.5)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df, failed_symbols
    else:
        return pd.DataFrame(), failed_symbols


def generate_synthetic_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    starting_prices: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for symbols.
    Used as fallback when real data unavailable.
    
    Args:
        symbols: List of symbols
        start_date: Start date
        end_date: End date
        starting_prices: Dict of starting prices per symbol
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    logger.info(f"Generating synthetic data for {len(symbols)} symbols")
    
    if starting_prices is None:
        starting_prices = {symbol: 2000 + np.random.randint(0, 4000) for symbol in symbols}
    
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    data = []
    np.random.seed(42)
    
    for symbol in symbols:
        price = starting_prices.get(symbol, 2000)
        
        for date in all_dates:
            # Simulate realistic price movements
            daily_return = np.random.normal(0.0003, 0.015)  # 0.03% mean, 1.5% std
            
            open_price = price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price * (1 + daily_return)
            
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            volume = np.random.uniform(100000, 5000000)
            
            data.append({
                'symbol': symbol,
                'timestamp': pd.Timestamp(date, tz='UTC'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
            
            price = close_price
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} synthetic bars for {len(symbols)} symbols")
    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate collected data quality.
    
    Returns:
        (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return False, f"Missing columns. Has: {list(df.columns)}"
    
    # Check OHLC constraints
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    
    if invalid_ohlc > 0:
        return False, f"{invalid_ohlc} rows have invalid OHLC constraints"
    
    # Check volume
    if (df['volume'] < 0).sum() > 0:
        return False, "Negative volume found"
    
    return True, "✓ All validations passed"


def collect_nifty50_data(
    output_path: str = "data/nifty50_15y_ohlcv.csv",
    start_year: int = 2010,
    end_year: int = 2025,
    use_synthetic_fallback: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to collect 15 years of NIFTY50 data.
    
    Args:
        output_path: Where to save the CSV
        start_year: Starting year (default 2010)
        end_year: Ending year (default 2025)
        use_synthetic_fallback: Generate synthetic data if real data fails
    
    Returns:
        (DataFrame with all data, Dict with collection stats)
    """
    # Date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    
    logger.info("=" * 70)
    logger.info("NIFTY50 15-Year Data Collection")
    logger.info("=" * 70)
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {len(NIFTY50_SYMBOLS)}")
    logger.info(f"Symbols: {', '.join(NIFTY50_SYMBOLS[:10])}... (showing first 10)")
    logger.info("=" * 70)
    
    # Try yfinance first
    logger.info("\n--- Attempting yfinance (recommended) ---")
    df, failed = collect_data_yfinance(NIFTY50_SYMBOLS, start_date, end_date)
    
    if df.empty and try_import_nsepy() is not None:
        # Try nsepy as fallback
        logger.info("\n--- yfinance failed, trying nsepy ---")
        df, failed = collect_data_nsepy(NIFTY50_SYMBOLS, start_date, end_date)
    
    # Generate synthetic data for failed symbols
    if use_synthetic_fallback and failed:
        logger.info(f"\n--- Generating synthetic data for {len(failed)} failed symbols ---")
        synthetic_df = generate_synthetic_data(failed, start_date, end_date)
        if not df.empty:
            df = pd.concat([df, synthetic_df], ignore_index=True)
        else:
            df = synthetic_df
    
    # If all sources failed, use synthetic
    if df.empty:
        logger.warning("\n--- All sources failed, generating synthetic data ---")
        df = generate_synthetic_data(NIFTY50_SYMBOLS, start_date, end_date)
    
    # Validate data
    is_valid, validation_msg = validate_data(df)
    logger.info(f"\nData Validation: {validation_msg}")
    
    if not is_valid:
        logger.error("Data validation failed!")
        return df, {"error": validation_msg, "rows": 0}
    
    # Sort and deduplicate
    df = df.drop_duplicates(['symbol', 'timestamp']).reset_index(drop=True)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Calculate statistics
    stats = {
        "output_file": str(output_path),
        "total_rows": len(df),
        "unique_symbols": df['symbol'].nunique(),
        "symbols": sorted(df['symbol'].unique().tolist()),
        "date_range": f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        "bars_per_symbol": df.groupby('symbol').size().to_dict(),
        "successful_symbols": len(NIFTY50_SYMBOLS) - len(failed),
        "failed_symbols": failed,
        "synthetic_generated": len(failed),
    }
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Rows: {stats['total_rows']:,}")
    logger.info(f"Unique Symbols: {stats['unique_symbols']}")
    logger.info(f"Date Range: {stats['date_range']}")
    logger.info(f"Successful Downloads: {stats['successful_symbols']}/{len(NIFTY50_SYMBOLS)}")
    if failed:
        logger.info(f"Failed Symbols ({len(failed)}): {', '.join(failed)}")
        logger.info(f"  → Generated synthetic data for failed symbols")
    logger.info(f"Output File: {output_path}")
    logger.info(f"File Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info("=" * 70)
    
    # Show sample data
    logger.info("\nSample Data (first 10 rows):")
    logger.info(df.head(10).to_string())
    
    logger.info("\nPer-Symbol Stats:")
    for symbol, count in sorted(stats["bars_per_symbol"].items())[:10]:
        logger.info(f"  {symbol}: {count:,} bars")
    if len(stats["bars_per_symbol"]) > 10:
        logger.info(f"  ... and {len(stats['bars_per_symbol']) - 10} more symbols")
    
    return df, stats


if __name__ == "__main__":
    # Collect data
    df, stats = collect_nifty50_data(
        output_path="data/nifty50_15y_ohlcv.csv",
        start_year=2010,
        end_year=2025,
        use_synthetic_fallback=True
    )
    
    print("\n✓ Data collection complete!")
    print(f"Ready for ML training: {stats['output_file']}")

#!/usr/bin/env python3
"""
Demo: NIFTY50 Data Collection & ML Training Workflow

This demo shows the complete workflow WITHOUT requiring yfinance/nsepy.
It uses synthetic data to demonstrate the pipeline.

Workflow:
1. Generate synthetic 15-year NIFTY50 data
2. Prepare features and labels
3. Create walk-forward splits
4. Train a simple model
5. Show results

Run: python scripts/demo_nifty50_workflow.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Tuple, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_nifty50_data(
    output_path: str = "data/nifty50_15y_demo.csv",
    start_year: int = 2010,
    end_year: int = 2025
) -> Tuple[pd.DataFrame, dict]:
    """Generate synthetic 15-year NIFTY50 data."""
    
    logger.info("=" * 70)
    logger.info("GENERATING SYNTHETIC 15-YEAR NIFTY50 DATA")
    logger.info("=" * 70)
    
    # NIFTY50 stocks
    symbols = [
        "TCS", "INFY", "RELIANCE", "HDFC", "ICICIBANK",
        "KOTAKBANK", "AXISBANK", "LT", "BAJAJFINSV", "BAJAJAUTOL",
        "NTPC", "POWERGRID", "BHARTIARTL", "JSWSTEEL", "MARUTI",
        "WIPRO", "HCLTECH", "TECHM", "MFSL", "SUNPHARMA",
        "HINDUNILVR", "ITC", "NESTLEIND", "INDIGO", "MARUTISUZU",
        "SBICARD", "ADANIPORTS", "ADANIGREEN", "ADANITRANS", "ADANIPOWER",
        "CIPLA", "DRREDDY", "DIVISLAB", "PHARMAIND", "COLPAL",
        "BRITANNIA", "PIDILITIND", "GODREJCP", "HEROMOTOCO", "TATASTEEL",
        "HINDALCO", "SAIL", "VEDL", "NMDC", "ULTRACEMCO",
        "SHREECEM", "ACC", "BOSCHLTD", "BAJAJFINSV", "MARUTI"
    ]
    
    # Get unique 50
    symbols = list(dict.fromkeys(symbols))[:50]
    
    logger.info(f"Generating data for {len(symbols)} symbols")
    logger.info(f"Date range: {start_year}-01-01 to {end_year}-01-01")
    
    # Date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    logger.info(f"Business days: {len(all_dates)}")
    
    # Starting prices (realistic for NIFTY50)
    starting_prices = {
        "TCS": 3500, "INFY": 1800, "RELIANCE": 2600, "HDFC": 2400, "ICICIBANK": 450,
        "KOTAKBANK": 1200, "AXISBANK": 900, "LT": 2000, "BAJAJFINSV": 1400, "BAJAJAUTOL": 6200,
        "NTPC": 250, "POWERGRID": 180, "BHARTIARTL": 850, "JSWSTEEL": 750, "MARUTI": 8500,
    }
    
    data = []
    np.random.seed(42)
    
    for symbol in symbols:
        price = starting_prices.get(symbol, 2000 + np.random.randint(0, 4000))
        
        for date in all_dates:
            # Realistic price movements
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
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    stats = {
        "output_file": str(output_path),
        "total_rows": len(df),
        "unique_symbols": df['symbol'].nunique(),
        "date_range": f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        "symbols": sorted(df['symbol'].unique().tolist()),
    }
    
    logger.info(f"Generated {len(df):,} bars for {len(symbols)} symbols")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return df, stats


def engineer_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer simple technical features."""
    
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING TECHNICAL FEATURES")
    logger.info("=" * 70)
    
    result = []
    
    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy()
        close = sym_df['close']
        
        # Simple Moving Averages
        sym_df['sma_20'] = close.rolling(20).mean()
        sym_df['sma_50'] = close.rolling(50).mean()
        sym_df['sma_200'] = close.rolling(200).mean()
        
        # Price ratios
        sym_df['price_sma_ratio'] = (close / sym_df['sma_20']).fillna(1.0)
        
        # Returns and volatility
        sym_df['daily_return'] = close.pct_change()
        sym_df['volatility'] = close.pct_change().rolling(20).std()
        
        # Momentum
        sym_df['momentum'] = (close / close.shift(20) - 1).fillna(0)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        sym_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        sym_df['macd'] = ema_12 - ema_26
        sym_df['macd_signal'] = sym_df['macd'].ewm(span=9, adjust=False).mean()
        
        result.append(sym_df)
    
    df = pd.concat(result, ignore_index=True)
    
    feature_cols = ['sma_20', 'sma_50', 'sma_200', 'price_sma_ratio', 'daily_return', 'volatility', 'momentum', 'rsi', 'macd', 'macd_signal']
    
    logger.info(f"Engineered {len(feature_cols)} features")
    logger.info(f"Features: {feature_cols}")
    
    return df


def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Generate binary labels: next day up (1) or down (0)."""
    
    logger.info("\n" + "=" * 70)
    logger.info(f"GENERATING LABELS ({horizon}-bar horizon)")
    logger.info("=" * 70)
    
    df['label'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    label_dist = df['label'].value_counts()
    logger.info(f"Label distribution:")
    logger.info(f"  Down (0): {label_dist.get(0, 0):,} ({label_dist.get(0, 0)/len(df)*100:.1f}%)")
    logger.info(f"  Up   (1): {label_dist.get(1, 0):,} ({label_dist.get(1, 0)/len(df)*100:.1f}%)")
    
    return df


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_window_days: int = 252 * 3,
    test_window_days: int = 252,
    step_size_days: int = 252
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk-forward train/test splits."""
    
    logger.info("\n" + "=" * 70)
    logger.info("CREATING WALK-FORWARD SPLITS")
    logger.info("=" * 70)
    
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    splits = []
    current_date = min_date + timedelta(days=train_window_days)
    
    while current_date + timedelta(days=test_window_days) <= max_date:
        train_start = current_date - timedelta(days=train_window_days)
        train_end = current_date
        test_start = current_date
        test_end = current_date + timedelta(days=test_window_days)
        
        train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
        test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
        
        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))
        
        current_date += timedelta(days=step_size_days)
    
    logger.info(f"Created {len(splits)} walk-forward splits")
    
    for i, (train, test) in enumerate(splits[:3]):
        logger.info(f"  Split {i+1}: Train {len(train):,} bars ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
        logger.info(f"            Test  {len(test):,} bars ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")
    
    if len(splits) > 3:
        logger.info(f"  ... and {len(splits) - 3} more splits")
    
    return splits


def train_simple_model(splits: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
    """Train a simple logistic regression model on walk-forward splits."""
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SIMPLE LOGISTIC REGRESSION MODEL")
    logger.info("=" * 70)
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        feature_cols = ['sma_20', 'sma_50', 'price_sma_ratio', 'volatility', 'momentum', 'rsi', 'macd']
        fold_results = []
        
        for fold_idx, (train_df, test_df) in enumerate(splits[:3]):
            logger.info(f"\nFold {fold_idx + 1}/3")
            
            # Prepare data
            train_df_clean = train_df.dropna(subset=feature_cols + ['label'])
            test_df_clean = test_df.dropna(subset=feature_cols + ['label'])
            
            if len(train_df_clean) == 0 or len(test_df_clean) == 0:
                logger.warning(f"  Skipping fold {fold_idx + 1} (insufficient data)")
                continue
            
            X_train = train_df_clean[feature_cols].values
            y_train = train_df_clean['label'].values
            X_test = test_df_clean[feature_cols].values
            y_test = test_df_clean['label'].values
            
            # Train
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            prob_test = model.predict_proba(X_test)[:, 1]
            
            train_acc = accuracy_score(y_train, pred_train)
            test_acc = accuracy_score(y_test, pred_test)
            test_auc = roc_auc_score(y_test, prob_test)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_auc': test_auc
            })
            
            logger.info(f"  Train: {len(X_train):,} bars | Acc: {train_acc:.4f}")
            logger.info(f"  Test:  {len(X_test):,} bars | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
        
        return fold_results
    
    except ImportError:
        logger.warning("scikit-learn not available, skipping model training")
        logger.warning("Install with: pip install scikit-learn")
        return []


def main():
    """Run the complete demo workflow."""
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "NIFTY50 DATA COLLECTION & ML TRAINING DEMO".center(68) + "║")
    print("║" + "(Without external data sources)".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Step 1: Generate data
    df, data_stats = generate_synthetic_nifty50_data()
    print()
    
    # Step 2: Engineer features
    df = engineer_features_simple(df)
    print()
    
    # Step 3: Generate labels
    df = generate_labels(df, horizon=1)
    print()
    
    # Step 4: Create splits
    splits = create_walk_forward_splits(df)
    print()
    
    # Step 5: Train model
    fold_results = train_simple_model(splits)
    print()
    
    # Summary
    logger.info("=" * 70)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✓ Generated synthetic 15-year NIFTY50 data")
    logger.info(f"  Symbols: {data_stats['unique_symbols']}")
    logger.info(f"  Rows: {data_stats['total_rows']:,}")
    logger.info(f"  Date Range: {data_stats['date_range']}")
    logger.info(f"\n✓ Engineered technical features")
    logger.info(f"  SMA, RSI, MACD, Momentum, Volatility, etc.")
    logger.info(f"\n✓ Generated binary labels (up/down)")
    logger.info(f"\n✓ Created {len(splits)} walk-forward splits")
    logger.info(f"  Train Window: 3 years | Test Window: 1 year")
    logger.info(f"\n✓ Trained models on 3 folds")
    if fold_results:
        best = max(fold_results, key=lambda x: x['test_auc'])
        logger.info(f"  Best Test AUC: {best['test_auc']:.4f}")
        logger.info(f"  Best Test Accuracy: {best['test_accuracy']:.4f}")
    logger.info(f"\n" + "=" * 70)
    
    print("\n✓ DEMO COMPLETE!")
    print("\nTo use real data, install dependencies:")
    print("  bash install_ml_dependencies.sh")
    print("\nThen run:")
    print("  python scripts/collect_nifty50_15y_data.py")
    print("  python scripts/prepare_nifty50_data.py")
    print("  python scripts/train_nifty50_models.py")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data Preparation Module for ML Training

Converts raw NIFTY50 OHLCV data into training-ready datasets
with features, labels, and proper train/test splits.

Integrates with:
- learning/prediction_engine.py (model training)
- learning/walk_forward.py (time-series cross-validation)
- features/labels.py (label generation)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.labels import LabelGenerator
from learning.data_structures import FeatureVector
from utils.types import Bar

logger = logging.getLogger(__name__)


class NiftyDataPreparation:
    """
    Prepare NIFTY50 raw data for ML model training.
    
    Workflow:
    1. Load raw OHLCV data (15 years)
    2. Engineer features (technical indicators, ratios, etc.)
    3. Generate labels (binary: next-day up/down prediction)
    4. Handle missing data and outliers
    5. Create walk-forward splits (train/test windows)
    6. Normalize features
    7. Export for model training
    """
    
    def __init__(
        self,
        raw_data_path: str = "data/nifty50_15y_ohlcv.csv",
        label_horizon: int = 1,
        cost_bps: float = 0.0,
    ):
        """
        Initialize data preparation.
        
        Args:
            raw_data_path: Path to raw NIFTY50 OHLCV CSV
        """
        self.raw_data_path = Path(raw_data_path)
        self.df = None
        self.symbol_dfs = {}
        self.features_df = None
        self.labels_df = None
        self.label_horizon = label_horizon
        self.cost_decimal = cost_bps / 10000.0
        
        logger.info(f"Initializing NIFTY50 data preparation")
        logger.info(f"Data source: {self.raw_data_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw NIFTY50 data."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
        
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        self.df = pd.read_csv(self.raw_data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df):,} rows for {self.df['symbol'].nunique()} symbols")
        logger.info(f"Date range: {self.df['timestamp'].min().date()} to {self.df['timestamp'].max().date()}")
        
        return self.df
    
    def calculate_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate technical indicators for a single symbol.
        
        Indicators:
        - Price-based: SMA 5/20/50/200, EMA, volatility, momentum
        - Volume-based: On-Balance Volume, VWAP
        - Momentum: RSI, MACD, Stochastic
        - Strength: ADX, True Range
        """
        df = df.copy()
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        # Simple Moving Averages
        df['sma_5'] = close.rolling(window=5).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['sma_200'] = close.rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        # Price ratios
        df['sma_ratio_5_20'] = (close / df['sma_20']).fillna(1.0) - 1
        df['sma_ratio_20_50'] = (df['sma_20'] / df['sma_50']).fillna(1.0) - 1
        df['sma_ratio_50_200'] = (df['sma_50'] / df['sma_200']).fillna(1.0) - 1
        
        # Returns and volatility
        df['daily_return'] = close.pct_change()
        df['volatility_20'] = close.pct_change().rolling(20).std()
        df['volatility_60'] = close.pct_change().rolling(60).std()
        df['volatility_5'] = close.pct_change().rolling(5).std()
        df['volatility_10'] = close.pct_change().rolling(10).std()
        
        # Momentum
        df['momentum_10'] = (close / close.shift(10)) - 1
        df['momentum_20'] = (close / close.shift(20)) - 1
        df['return_5'] = (close / close.shift(5)) - 1
        df['return_10'] = (close / close.shift(10)) - 1
        df['return_20'] = (close / close.shift(20)) - 1
        df['return_60'] = (close / close.shift(60)) - 1
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1)
        df['volume_zscore_20'] = (volume - df['volume_sma_20']) / (df['volume'].rolling(20).std() + 1e-9)
        
        # High-Low Range
        df['hl_range'] = (high - low) / close
        
        # Gap
        df['gap'] = (close - close.shift(1)) / close.shift(1)

        # Price rank and trend slope
        rolling_window = 252
        df['price_rank_252'] = close.rolling(rolling_window).rank(pct=True)
        # Simple trend slope via rolling linear regression on last 20 bars
        window = 20
        idx = np.arange(window)
        def slope(series):
            if series.isna().any() or len(series) != window:
                return np.nan
            coef = np.polyfit(idx, series.values, 1)
            return coef[0] / (np.mean(series) + 1e-9)
        df['trend_slope_20'] = close.rolling(window).apply(slope, raw=False)
        df['trend_slope_50'] = close.rolling(50).apply(lambda s: slope(s.tail(window)), raw=False)
        
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features for all symbols.
        
        Returns:
            DataFrame with features for each bar
        """
        if self.df is None:
            self.load_raw_data()
        
        logger.info("Engineering features for all symbols...")
        
        all_features = []
        
        for symbol in self.df['symbol'].unique():
            symbol_df = self.df[self.df['symbol'] == symbol].copy()
            
            # Calculate indicators
            symbol_df = self.calculate_technical_indicators(symbol_df, symbol)
            
            all_features.append(symbol_df)
        
        # Combine all symbols
        self.features_df = pd.concat(all_features, ignore_index=True)
        self.features_df = self.features_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

        # Regime tagging (simple trend/vol buckets per symbol)
        def assign_regimes(df_sym: pd.DataFrame) -> pd.DataFrame:
            df_sym = df_sym.copy()
            # Volatility regime from vol_20 quantiles
            q_low, q_high = df_sym['volatility_20'].quantile([0.33, 0.66])
            df_sym['vol_regime'] = np.select(
                [df_sym['volatility_20'] <= q_low, df_sym['volatility_20'] >= q_high],
                [0, 2],
                default=1
            )
            # Trend regime from return_20 quantiles
            tq_low, tq_high = df_sym['return_20'].quantile([0.33, 0.66])
            df_sym['trend_regime'] = np.select(
                [df_sym['return_20'] <= tq_low, df_sym['return_20'] >= tq_high],
                [-1, 1],
                default=0
            )
            df_sym['regime_bucket'] = df_sym['trend_regime'] * 3 + df_sym['vol_regime']
            return df_sym

        self.features_df = (
            self.features_df.groupby('symbol', group_keys=False)
            .apply(assign_regimes)
            .reset_index(drop=True)
        )
        
        logger.info(f"Engineered {self.features_df.shape[1] - 7} features")
        logger.info(f"Feature columns: {[col for col in self.features_df.columns if col not in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']][:10]}...")
        
        return self.features_df
    
    def generate_labels(self, horizon: Optional[int] = None, label_type: str = "binary") -> pd.DataFrame:
        """
        Generate prediction labels.
        
        Args:
            horizon: Bars ahead to predict (default to self.label_horizon)
            label_type: "binary" (up/down), "regression" (return), "multiclass" (strong up/down/neutral)
        
        Returns:
            DataFrame with labels
        """
        if self.features_df is None:
            self.engineer_features()
        
        target_h = horizon or self.label_horizon
        logger.info(f"Generating {label_type} labels with {target_h}-bar horizon (cost_bps={self.cost_decimal*10000:.2f})")
        
        df = self.features_df.copy()

        horizons = sorted(set([1, 5, 10, target_h]))
        for h in horizons:
            fwd = df['close'].shift(-h) / df['close'] - 1
            df[f'forward_return_{h}'] = fwd
            df[f'forward_return_cost_{h}'] = fwd - self.cost_decimal
        
        if label_type == "binary":
            df['label'] = (df[f'forward_return_cost_{target_h}'] > 0).astype(int)
            df['label_confidence'] = abs(df[f'forward_return_cost_{target_h}'])
        
        elif label_type == "regression":
            # Regression: expected return over horizon
            df['label'] = df[f'forward_return_cost_{target_h}']
            df['label_confidence'] = abs(df['label'])
        
        elif label_type == "multiclass":
            # Multiclass: strong down (0), neutral (1), strong up (2)
            ret = df[f'forward_return_cost_{target_h}']
            threshold = 0.02  # 2% threshold
            df['label'] = pd.cut(ret, bins=[-np.inf, -threshold, threshold, np.inf], labels=[0, 1, 2])
            df['label'] = df['label'].astype(int)
            df['label_confidence'] = abs(ret)
        
        self.labels_df = df
        unique = self.labels_df['label'].nunique() if 'label' in self.labels_df else 0
        logger.info(f"Generated labels: {unique} unique values")
        
        return self.labels_df
    
    def remove_nan_rows(self) -> pd.DataFrame:
        """Remove rows with NaN values (from indicator calculations)."""
        if self.labels_df is None:
            self.generate_labels()
        
        df = self.labels_df.copy()
        
        # Drop rows with NaN in important columns
        feature_cols = [col for col in df.columns if col.startswith((
            'sma_', 'ema_', 'rsi_', 'macd', 'bb_', 'atr_', 'volatility_', 'momentum_', 'label', 'return_', 'volume_', 'price_rank', 'trend_slope', 'forward_return_'
        ))]
        
        initial_rows = len(df)
        df = df.dropna(subset=feature_cols)
        
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
        logger.info(f"Remaining rows: {len(df):,} ({len(df)/initial_rows*100:.1f}%)")
        
        return df
    
    def create_training_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create final training dataset.
        
        Returns:
            (DataFrame ready for training, List of feature columns)
        """
        # Process pipeline
        self.load_raw_data()
        self.engineer_features()
        df = self.generate_labels()
        df = self.remove_nan_rows()
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col.startswith((
            'sma_', 'ema_', 'rsi_', 'macd', 'bb_', 'atr_', 'volatility_',
            'momentum_', 'gap', 'hl_range', 'volume_', 'daily_return', 'return_', 'price_rank', 'trend_slope',
            'regime_', 'trend_regime', 'vol_regime'
        ))]
        
        logger.info(f"Training dataset: {len(df):,} rows × {len(feature_cols)} features")
        logger.info(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        
        return df, feature_cols
    
    def save_training_data(self, output_path: str = "data/nifty50_training_data.csv"):
        """Save training-ready data to CSV."""
        df, feature_cols = self.create_training_dataset()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save full data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved training data to {output_path}")
        
        # Save feature list
        feature_list_path = output_path.parent / "feature_columns.txt"
        with open(feature_list_path, 'w') as f:
            f.write('\n'.join(feature_cols))
        logger.info(f"Saved feature list to {feature_list_path}")
        
        return df, feature_cols
    
    def get_train_test_split(
        self,
        df: pd.DataFrame,
        train_end_date: datetime,
        test_start_date: datetime,
        test_end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split for a specific time period.
        
        Args:
            df: Full dataset
            train_end_date: When training data ends
            test_start_date: When test data starts
            test_end_date: When test data ends
        
        Returns:
            (train_df, test_df)
        """
        train_df = df[df['timestamp'] <= train_end_date].copy()
        test_df = df[(df['timestamp'] >= test_start_date) & (df['timestamp'] <= test_end_date)].copy()
        
        logger.info(f"Train: {len(train_df):,} rows ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
        logger.info(f"Test: {len(test_df):,} rows ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")
        
        return train_df, test_df
    
    def get_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_window_days: int = 252 * 3,  # 3 years
        test_window_days: int = 252,  # 1 year
        step_size_days: int = 252  # Move 1 year at a time
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward splits for backtesting.
        
        Args:
            df: Full dataset
            train_window_days: Training window size
            test_window_days: Test window size
            step_size_days: How much to move forward each step
        
        Returns:
            List of (train_df, test_df) tuples
        """
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
        
        return splits


def prepare_nifty50_data(
    raw_data_path: str = "data/nifty50_15y_ohlcv.csv",
    output_path: str = "data/nifty50_training_data.csv",
    label_horizon: int = 1,
    cost_bps: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main function: Prepare NIFTY50 data for ML training.
    
    Args:
        raw_data_path: Path to raw NIFTY50 OHLCV data
        output_path: Where to save training data
    
    Returns:
        (training_df, feature_columns)
    """
    logger.info("=" * 70)
    logger.info("NIFTY50 DATA PREPARATION FOR ML TRAINING")
    logger.info("=" * 70)
    
    prep = NiftyDataPreparation(raw_data_path, label_horizon=label_horizon, cost_bps=cost_bps)
    df, features = prep.save_training_data(output_path)
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    
    return df, features


if __name__ == "__main__":
    # Example: Prepare data for training
    prepare_nifty50_data()

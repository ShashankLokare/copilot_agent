"""
Label Generator for ML Training

Generates forward-looking labels from price data.
Ensures strict no-leakage compliance: only uses past/present data.
"""

from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from learning.data_structures import Label


class LabelGenerator:
    """
    Generates classification and regression targets for model training.
    
    Key principles:
    - Only uses data up to and including time t to generate features.
    - Only uses data after time t to generate labels (k-bar forward returns).
    - Strictly prevents look-ahead bias.
    """
    
    def __init__(
        self,
        horizons: List[int] = None,
        direction_threshold_pct: float = 0.5,
        neutral_band_pct: float = 0.2,
        verbose: bool = False,
    ):
        """
        Initialize label generator.
        
        Args:
            horizons: List of forward-looking horizons in bars. Default: [5, 20]
            direction_threshold_pct: Threshold for classifying as up/down move (%).
                                      E.g., 0.5 means ±0.5% is the threshold.
            neutral_band_pct: Half-width of neutral band around 0 return (%).
                             E.g., 0.2 means [-0.1%, +0.1%] is neutral.
            verbose: Whether to print progress messages.
        """
        self.horizons = horizons or [5, 20]
        self.direction_threshold_pct = direction_threshold_pct
        self.neutral_band_pct = neutral_band_pct
        self.verbose = verbose
    
    def generate_labels(
        self,
        price_df: pd.DataFrame,
        symbol: str = None,
        horizon: int = None,
    ) -> pd.DataFrame:
        """
        Generate labels from price data.
        
        Args:
            price_df: DataFrame with columns ['timestamp', 'close'] or 
                     ['timestamp', 'symbol', 'close'] if multiple symbols.
            symbol: If provided, filter to this symbol only.
            horizon: If provided, use only this horizon. Otherwise, use all configured horizons.
            
        Returns:
            DataFrame with columns:
                - timestamp
                - symbol (if input had it)
                - horizon_bars
                - target_return_k (float)
                - target_direction_k (int: -1, 0, +1)
        """
        # Validate input
        if len(price_df) < 2:
            raise ValueError("price_df must have at least 2 rows")
        
        if 'close' not in price_df.columns:
            raise ValueError("price_df must contain 'close' column")
        
        # Make a copy to avoid modifying input
        df = price_df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df['timestamp'] = df.index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to symbol if specified
        if symbol is not None and 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select horizons to use
        horizons_to_use = [horizon] if horizon is not None else self.horizons
        
        # Generate labels
        labels = []
        
        for h in horizons_to_use:
            if h >= len(df):
                if self.verbose:
                    print(f"Warning: horizon {h} >= data length {len(df)}, skipping")
                continue
            
            for i in range(len(df) - h):
                # Current time step: i (we generate features at time i)
                # Label horizon: h bars forward (from time i+1 to i+h)
                
                current_timestamp = df.iloc[i]['timestamp']
                current_close = df.iloc[i]['close']
                
                # Future price at i+h
                future_close = df.iloc[i + h]['close']
                
                # Calculate k-bar return
                ret_k = (future_close / current_close) - 1.0
                
                # Determine direction
                ret_pct = ret_k * 100
                threshold = self.direction_threshold_pct
                neutral_band = self.neutral_band_pct
                
                if ret_pct > threshold:
                    direction = 1  # Up
                elif ret_pct < -threshold:
                    direction = -1  # Down
                elif abs(ret_pct) <= neutral_band:
                    direction = 0  # Neutral
                else:
                    # In threshold band but outside neutral band
                    # Assign based on sign
                    direction = 1 if ret_pct > 0 else -1
                
                # Create label
                label = Label(
                    timestamp=current_timestamp,
                    symbol=df.iloc[i].get('symbol', 'UNKNOWN'),
                    horizon_bars=h,
                    target_return_k=ret_k,
                    target_direction_k=direction,
                )
                
                labels.append(label)
        
        # Convert to DataFrame
        labels_list = []
        for label in labels:
            if label.is_valid():
                labels_list.append({
                    'timestamp': label.timestamp,
                    'symbol': label.symbol,
                    'horizon_bars': label.horizon_bars,
                    'target_return_k': label.target_return_k,
                    'target_direction_k': label.target_direction_k,
                })
        
        if not labels_list:
            raise ValueError("No valid labels generated")
        
        result_df = pd.DataFrame(labels_list)
        
        if self.verbose:
            print(f"Generated {len(result_df)} labels for {len(horizons_to_use)} horizon(s)")
        
        return result_df
    
    def generate_labels_from_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        use_close: bool = True,
    ) -> pd.DataFrame:
        """
        Generate labels from OHLCV data.
        
        Uses close price by default (more realistic for daily data).
        
        Args:
            ohlcv_df: DataFrame with columns like ['timestamp', 'symbol', 'close', ...]
            use_close: If True, use 'close' price. Otherwise use 'open'.
            
        Returns:
            DataFrame with labels.
        """
        # Validate required columns
        required_cols = {'timestamp', 'close'}
        if not required_cols.issubset(ohlcv_df.columns):
            raise ValueError(f"ohlcv_df must contain columns: {required_cols}")
        
        return self.generate_labels(ohlcv_df)
    
    def split_by_horizon(
        self,
        labels_df: pd.DataFrame,
    ) -> Dict[int, pd.DataFrame]:
        """
        Split labels dataframe by horizon.
        
        Args:
            labels_df: DataFrame with 'horizon_bars' column.
            
        Returns:
            Dict mapping horizon -> filtered DataFrame.
        """
        result = {}
        for horizon in self.horizons:
            result[horizon] = labels_df[labels_df['horizon_bars'] == horizon].copy()
        return result
    
    def split_by_regime(
        self,
        labels_df: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Join labels with regime information and split by regime.
        
        Args:
            labels_df: DataFrame with ['timestamp', 'symbol', ...]
            regime_df: DataFrame with ['timestamp', 'symbol', 'regime', ...]
            
        Returns:
            Dict mapping regime -> filtered DataFrame.
        """
        # Merge on timestamp and symbol
        merged = labels_df.merge(
            regime_df[['timestamp', 'symbol', 'regime']],
            on=['timestamp', 'symbol'],
            how='left',
        )
        
        # Split by regime
        result = {}
        for regime in merged['regime'].unique():
            if pd.notna(regime):
                result[regime] = merged[merged['regime'] == regime].copy()
        
        return result
    
    @staticmethod
    def balance_classes(
        labels_df: pd.DataFrame,
        target_col: str = 'target_direction_k',
        method: str = 'undersample',
    ) -> pd.DataFrame:
        """
        Balance classification targets (optional, for training).
        
        Args:
            labels_df: DataFrame with target column.
            target_col: Name of target column (default: 'target_direction_k').
            method: 'undersample' (default) or 'oversample'.
            
        Returns:
            Balanced DataFrame.
        """
        if method == 'undersample':
            # Find minority class count
            value_counts = labels_df[target_col].value_counts()
            min_count = value_counts.min()
            
            # Undersample each class
            balanced_dfs = []
            for class_val in labels_df[target_col].unique():
                class_df = labels_df[labels_df[target_col] == class_val]
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            
            return pd.concat(balanced_dfs, ignore_index=True).sort_values('timestamp')
        
        else:
            raise NotImplementedError(f"Method {method} not implemented")


# Example usage and testing
if __name__ == "__main__":
    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    price_df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'close': prices,
    })
    
    # Generate labels
    generator = LabelGenerator(horizons=[5, 20], verbose=True)
    labels_df = generator.generate_labels(price_df)
    
    print("\nFirst 10 labels:")
    print(labels_df.head(10))
    print(f"\nLabel distribution:")
    print(labels_df['target_direction_k'].value_counts())

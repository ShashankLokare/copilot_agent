"""
Walk-Forward Training & Retraining

Implements time-based cross-validation and walk-forward evaluation
to avoid overfitting and test model generalization.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from learning.data_structures import WalkForwardFold


logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """
    Generates walk-forward cross-validation folds.
    
    Key principle: never look into the future. Training data is always
    before validation data, with optional gap between them.
    """
    
    def __init__(
        self,
        train_window_days: int = 252 * 2,  # 2 years
        valid_window_days: int = 252,      # 1 year
        step_days: int = 63,                # ~quarterly
        gap_days: int = 0,                  # No gap by default
        verbose: bool = False,
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            train_window_days: Length of training window (calendar days).
            valid_window_days: Length of validation window (calendar days).
            step_days: How far to step forward for next fold.
            gap_days: Gap between train and validation (to avoid leakage).
            verbose: Whether to print progress.
        """
        self.train_window_days = train_window_days
        self.valid_window_days = valid_window_days
        self.step_days = step_days
        self.gap_days = gap_days
        self.verbose = verbose
    
    def generate_folds(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        min_train_samples: int = 100,
        min_valid_samples: int = 50,
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds from a DataFrame.
        
        Args:
            df: DataFrame with timestamp column.
            timestamp_col: Name of timestamp column.
            min_train_samples: Minimum samples required for training.
            min_valid_samples: Minimum samples required for validation.
            
        Returns:
            List of WalkForwardFold objects.
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()
        
        folds = []
        fold_id = 0
        
        current_train_start = min_date
        
        while True:
            # Define train window
            train_start = current_train_start
            train_end = train_start + timedelta(days=self.train_window_days)
            
            # Define gap (optional)
            valid_start = train_end + timedelta(days=self.gap_days)
            
            # Define validation window
            valid_end = valid_start + timedelta(days=self.valid_window_days)
            
            # Check if valid_end exceeds data
            if valid_end > max_date:
                if self.verbose:
                    logger.info(f"Reached end of data at fold {fold_id}")
                break
            
            # Filter data
            train_mask = (
                (df[timestamp_col] >= train_start) &
                (df[timestamp_col] < train_end)
            )
            valid_mask = (
                (df[timestamp_col] >= valid_start) &
                (df[timestamp_col] < valid_end)
            )
            
            train_data = df[train_mask]
            valid_data = df[valid_mask]
            
            # Check minimum samples
            if len(train_data) < min_train_samples or len(valid_data) < min_valid_samples:
                if self.verbose:
                    logger.warning(
                        f"Fold {fold_id} has insufficient samples: "
                        f"train={len(train_data)}, valid={len(valid_data)}"
                    )
                # Move forward and try again
                current_train_start += timedelta(days=self.step_days)
                continue
            
            # Create fold
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                train_df=train_data,
                valid_df=valid_data,
            )
            
            folds.append(fold)
            
            if self.verbose:
                logger.info(f"Fold {fold_id}: {fold}")
            
            # Move to next fold
            fold_id += 1
            current_train_start += timedelta(days=self.step_days)
        
        if self.verbose:
            logger.info(f"Generated {len(folds)} walk-forward folds")
        
        return folds
    
    def plot_folds(
        self,
        folds: List[WalkForwardFold],
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        Visualize walk-forward folds (requires matplotlib).
        
        Args:
            folds: List of WalkForwardFold objects.
            figsize: Figure size.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for fold in folds:
            train_start = fold.train_start
            train_end = fold.train_end
            valid_start = fold.valid_start
            valid_end = fold.valid_end
            
            # Draw train and valid ranges
            ax.barh(
                fold.fold_id,
                (train_end - train_start).days,
                left=(train_start - folds[0].train_start).days,
                height=0.4,
                color='blue',
                alpha=0.6,
                label='train' if fold.fold_id == 0 else '',
            )
            ax.barh(
                fold.fold_id,
                (valid_end - valid_start).days,
                left=(valid_start - folds[0].train_start).days,
                height=0.4,
                color='orange',
                alpha=0.6,
                label='valid' if fold.fold_id == 0 else '',
            )
        
        ax.set_xlabel('Days from first fold start')
        ax.set_ylabel('Fold ID')
        ax.set_title('Walk-Forward Cross-Validation Folds')
        ax.legend()
        plt.tight_layout()
        return fig


class TimeBasedSplitter:
    """
    Simple time-based train/valid/test split (non-overlapping).
    """
    
    @staticmethod
    def split(
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2,
        timestamp_col: str = 'timestamp',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into non-overlapping train/valid/test by time.
        
        Args:
            df: DataFrame to split.
            train_ratio: Fraction for training (default: 0.6).
            valid_ratio: Fraction for validation (default: 0.2).
            test_ratio: Fraction for testing (default: 0.2).
            timestamp_col: Name of timestamp column.
            
        Returns:
            Tuple of (train_df, valid_df, test_df).
        """
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        n = len(df)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        
        train_df = df[:n_train]
        valid_df = df[n_train:n_train + n_valid]
        test_df = df[n_train + n_valid:]
        
        return train_df, valid_df, test_df


class DataPreparer:
    """
    Utility class to prepare data for ML pipeline.
    
    Handles:
    - Merging features, labels, and regimes
    - Filtering by symbol and date range
    - Handling missing values
    """
    
    @staticmethod
    def prepare_datasets(
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        regime_df: Optional[pd.DataFrame] = None,
        symbols: Optional[List[str]] = None,
        min_valid_target: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare aligned dataset with features, labels, and regimes.
        
        Args:
            features_df: DataFrame with features (columns: timestamp, symbol, [features...]).
            labels_df: DataFrame with labels (columns: timestamp, symbol, horizon_bars, targets...).
            regime_df: DataFrame with regimes (columns: timestamp, symbol, regime).
            symbols: List of symbols to include. If None, use all.
            min_valid_target: If True, filter out rows with NaN targets.
            
        Returns:
            Merged DataFrame with all aligned data.
        """
        # Merge features and labels
        merged = features_df.merge(
            labels_df,
            on=['timestamp', 'symbol'],
            how='inner',
        )
        
        # Merge regime if provided
        if regime_df is not None:
            regime_df = regime_df[['timestamp', 'symbol', 'regime']].drop_duplicates()
            merged = merged.merge(
                regime_df,
                on=['timestamp', 'symbol'],
                how='left',
            )
        else:
            # Add default regime
            merged['regime'] = 'UNKNOWN'
        
        # Filter by symbols
        if symbols is not None:
            merged = merged[merged['symbol'].isin(symbols)]
        
        # Remove NaN targets if requested
        if min_valid_target:
            merged = merged[
                merged['target_return_k'].notna() &
                merged['target_direction_k'].notna()
            ]
        
        # Sort by timestamp
        merged = merged.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Prepared dataset with {len(merged)} samples, {len(merged['symbol'].unique())} symbols")
        
        return merged
    
    @staticmethod
    def handle_missing_features(
        df: pd.DataFrame,
        feature_columns: List[str],
        method: str = 'drop',
        fillna_value: float = 0.0,
    ) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            df: DataFrame with features.
            feature_columns: List of feature columns.
            method: 'drop' (remove rows) or 'fillna' (fill with value).
            fillna_value: Value to fill NaNs with (if method='fillna').
            
        Returns:
            DataFrame with missing values handled.
        """
        df = df.copy()
        
        if method == 'drop':
            df = df.dropna(subset=feature_columns)
        elif method == 'fillna':
            df[feature_columns] = df[feature_columns].fillna(fillna_value)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'value': np.random.randn(1000).cumsum(),
    })
    
    # Generate folds
    splitter = WalkForwardSplitter(
        train_window_days=252,
        valid_window_days=100,
        step_days=50,
    )
    folds = splitter.generate_folds(df, verbose=True)
    
    print(f"\nGenerated {len(folds)} folds")
    for fold in folds[:3]:
        print(fold)

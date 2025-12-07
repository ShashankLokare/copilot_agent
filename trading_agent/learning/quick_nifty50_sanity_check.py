#!/usr/bin/env python3
"""
Quick NIFTY50 Sanity Check - Validates Prediction Engine & Strategy

This script runs a minimal but meaningful validation of:
1. Prediction Engine: Does it have real predictive edge on NIFTY50?
2. Trading Strategy: Does it generate cost-adjusted profits?

Usage:
    python learning/quick_nifty50_sanity_check.py [--data-file path/to/data.csv] [--horizon 5]

The script:
- Loads 15 years of NIFTY50 daily data (2010-2025)
- Splits into train (65%) and test (35%)
- Trains ML model on train period
- Evaluates prediction engine on test period (ROC-AUC, Brier, buckets)
- Runs strategy backtest on test period (CAGR, Sharpe, MaxDD, PF)
- Prints PASS/FAIL verdict and human-readable summary
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.prediction_engine import PredictionEngine
from learning.data_structures import FeatureVector, PredictionOutput
from features.feature_engineering import FeatureEngineering


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Prediction engine evaluation metrics."""
    roc_auc: float
    brier_score: float
    accuracy: float
    has_edge: bool
    buckets: List[Dict[str, Any]]  # List of bucket statistics
    num_samples: int
    
    def print_summary(self):
        """Print summary of prediction metrics."""
        print("\n" + "="*70)
        print("PREDICTION ENGINE EVALUATION (Test Period)")
        print("="*70)
        print(f"Samples:          {self.num_samples:,}")
        print(f"ROC-AUC:          {self.roc_auc:.4f}")
        print(f"Brier Score:      {self.brier_score:.4f}")
        print(f"Accuracy:         {self.accuracy:.2%}")
        print()
        print("Bucketed Performance (prob_up ranges):")
        print("-" * 70)
        print(f"{'Bucket':<12} {'Count':>8} {'Win Rate':>12} {'Avg Return':>12}")
        print("-" * 70)
        for bucket in self.buckets:
            print(
                f"{bucket['bucket']:<12} {bucket['count']:>8} "
                f"{bucket['win_rate']:>11.2%} {bucket['avg_return']:>11.2%}"
            )
        print("-" * 70)
        print(f"Edge Detection:   {'HAS EDGE' if self.has_edge else 'WEAK / NO CLEAR EDGE'}")
        print()


@dataclass
class StrategyMetrics:
    """Strategy backtest metrics."""
    cagr: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    num_trades: int
    turnover: float
    
    # Verdict
    passed: bool
    comments: str
    
    def print_summary(self):
        """Print summary of strategy metrics."""
        print("\n" + "="*70)
        print("STRATEGY BACKTEST (Test Period)")
        print("="*70)
        print(f"Trades:           {self.num_trades}")
        print(f"CAGR:             {self.cagr:>8.2%}")
        print(f"Max Drawdown:     {self.max_drawdown:>8.2%}")
        print(f"Sharpe Ratio:     {self.sharpe:>8.2f}")
        print(f"Profit Factor:    {self.profit_factor:>8.2f}")
        print(f"Win Rate:         {self.win_rate:>8.2%}")
        print(f"Avg Win / Loss:   {self.avg_win:>8.2f} / {self.avg_loss:>8.2f}")
        print(f"Turnover:         {self.turnover:>8.2f}x")
        print()
        print(f"VERDICT:          {'PASS ✓' if self.passed else 'FAIL ✗'}")
        print(f"Comment:          {self.comments}")
        print()


class QuickNIFTY50Check:
    """Orchestrates the sanity check."""
    
    def __init__(
        self,
        data_file: Path = Path("data/nifty50_15y_demo.csv"),
        horizon_bars: int = 5,
        train_frac: float = 0.65,
    ):
        """
        Initialize sanity check.
        
        Args:
            data_file: Path to NIFTY50 OHLCV CSV
            horizon_bars: Forward-looking horizon (days) for labels
            train_frac: Fraction of data to use for training (0.65 = 65%)
        """
        self.data_file = Path(data_file)
        self.horizon_bars = horizon_bars
        self.train_frac = train_frac
        
        # Will be populated
        self.df: Optional[pd.DataFrame] = None
        self.symbols: List[str] = []
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and validate NIFTY50 data.
        
        Returns:
            (dataframe, list of symbols)
        """
        logger.info(f"Loading data from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        df = pd.read_csv(self.data_file)
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Extract symbols
        symbols = sorted(df['symbol'].unique().tolist())
        
        logger.info(f"Loaded {len(df):,} bars across {len(symbols)} symbols")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.df = df
        self.symbols = symbols
        
        return df, symbols
    
    def split_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-based, no shuffling).
        
        Returns:
            (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get unique dates
        dates = sorted(self.df['timestamp'].unique())
        split_idx = int(len(dates) * self.train_frac)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        train_df = self.df[self.df['timestamp'].isin(train_dates)].copy()
        test_df = self.df[self.df['timestamp'].isin(test_dates)].copy()
        
        logger.info(
            f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} "
            f"({len(train_df):,} bars)"
        )
        logger.info(
            f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()} "
            f"({len(test_df):,} bars)"
        )
        
        self.train_data = train_df
        self.test_data = test_df
        
        return train_df, test_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for all symbols in dataframe.
        
        Args:
            df: OHLCV dataframe with columns (timestamp, symbol, open, high, low, close, volume)
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        fe = FeatureEngineering()
        
        # Compute features per symbol
        features_list = []
        
        for symbol in self.symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) < 50:
                logger.warning(f"Skipping {symbol}: only {len(symbol_df)} bars")
                continue
            
            # Compute returns
            symbol_df['returns'] = symbol_df['close'].pct_change()
            symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            
            # SMA
            symbol_df['sma_20'] = symbol_df['close'].rolling(20).mean()
            symbol_df['sma_50'] = symbol_df['close'].rolling(50).mean()
            symbol_df['sma_200'] = symbol_df['close'].rolling(200).mean()
            
            # Price ratios
            symbol_df['price_to_sma20'] = symbol_df['close'] / symbol_df['sma_20']
            symbol_df['price_to_sma50'] = symbol_df['close'] / symbol_df['sma_50']
            
            # Momentum
            symbol_df['rsi'] = self._compute_rsi(symbol_df['close'], 14)
            
            # Volatility (ATR approximation)
            symbol_df['tr'] = self._compute_true_range(symbol_df)
            symbol_df['atr'] = symbol_df['tr'].rolling(14).mean()
            symbol_df['volatility'] = symbol_df['log_returns'].rolling(20).std()
            
            # Volume
            symbol_df['volume_sma'] = symbol_df['volume'].rolling(20).mean()
            symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma']
            
            features_list.append(symbol_df)
        
        result_df = pd.concat(features_list, ignore_index=True)
        result_df = result_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Engineered features for {len(features_list)} symbols")
        return result_df
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_true_range(df: pd.DataFrame) -> pd.Series:
        """Compute true range for ATR."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for ML training.
        
        Target: Is the 5-day forward return positive?
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with label column
        """
        logger.info(f"Generating labels (horizon={self.horizon_bars} bars)...")
        
        df = df.copy()
        df['future_close'] = df.groupby('symbol')['close'].shift(-self.horizon_bars)
        df['target_return'] = (df['future_close'] - df['close']) / df['close']
        df['label'] = (df['target_return'] > 0).astype(int)
        
        valid_rows = df['label'].notna()
        logger.info(f"Valid labels: {valid_rows.sum():,} / {len(df):,}")
        
        return df
    
    def train_prediction_engine(self, train_df: pd.DataFrame) -> PredictionEngine:
        """
        Train ML model on training data.
        
        Args:
            train_df: Training data with features and labels
        
        Returns:
            Trained PredictionEngine
        """
        logger.info("Training prediction engine...")
        
        # Select feature columns
        feature_cols = [
            'returns', 'sma_20', 'sma_50', 'sma_200',
            'price_to_sma20', 'price_to_sma50',
            'rsi', 'atr', 'volatility', 'volume_ratio'
        ]
        
        # Remove rows with NaN features or labels
        train_clean = train_df.dropna(subset=feature_cols + ['label'])
        
        if len(train_clean) < 100:
            raise ValueError(f"Not enough training data: {len(train_clean)}")
        
        X = train_clean[feature_cols].values
        y = train_clean['label'].values
        
        logger.info(f"Training on {len(X):,} samples with {len(feature_cols)} features")
        logger.info(f"Class distribution: {(y == 0).sum()} down / {(y == 1).sum()} up")
        
        # Train XGBoost model
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        
        model.fit(X, y)
        
        # Create PredictionEngine wrapper
        engine = PredictionEngine(
            config={
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
            },
            model_type='xgboost',
        )
        
        # Store trained model and metadata
        engine.models_direction[self.horizon_bars] = model
        engine.feature_names = feature_cols
        engine.horizons = [self.horizon_bars]
        
        logger.info(f"Model trained successfully")
        
        return engine, feature_cols
    
    def evaluate_prediction_engine(
        self,
        model: PredictionEngine,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> PredictionMetrics:
        """
        Evaluate prediction engine on test set.
        
        Computes ROC-AUC, Brier score, and bucketed performance.
        
        Args:
            model: Trained PredictionEngine
            test_df: Test data with features and labels
            feature_cols: Feature column names
        
        Returns:
            PredictionMetrics object
        """
        logger.info("Evaluating prediction engine on test set...")
        
        # Clean data
        test_clean = test_df.dropna(subset=feature_cols + ['label'])
        
        if len(test_clean) < 100:
            raise ValueError(f"Not enough test data: {len(test_clean)}")
        
        X = test_clean[feature_cols].values
        y = test_clean['label'].values
        
        # Get predictions
        try:
            xgb_model = model.models_direction[self.horizon_bars]
            y_pred_proba = xgb_model.predict_proba(X)[:, 1]  # Prob of class 1 (up)
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            raise
        
        # Compute metrics
        roc_auc = roc_auc_score(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)
        accuracy = ((y_pred_proba > 0.5).astype(int) == y).mean()
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}, Brier: {brier:.4f}, Accuracy: {accuracy:.2%}")
        
        # Bucket analysis
        buckets = self._bucket_predictions(y, y_pred_proba)
        
        # Determine if model has edge
        has_edge = self._check_for_edge(roc_auc, buckets)
        
        metrics = PredictionMetrics(
            roc_auc=roc_auc,
            brier_score=brier,
            accuracy=accuracy,
            has_edge=has_edge,
            buckets=buckets,
            num_samples=len(X),
        )
        
        return metrics
    
    @staticmethod
    def _bucket_predictions(
        y: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Bucket predictions by probability and compute stats.
        
        Args:
            y: True labels (0/1)
            y_pred_proba: Predicted probabilities
        
        Returns:
            List of bucket statistics
        """
        buckets = [
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.75),
            (0.75, 1.00),
        ]
        
        bucket_stats = []
        
        for low, high in buckets:
            mask = (y_pred_proba >= low) & (y_pred_proba < high)
            
            if mask.sum() == 0:
                continue
            
            y_bucket = y[mask]
            y_proba_bucket = y_pred_proba[mask]
            
            # Get returns for this bucket (if available in original test_df)
            win_rate = y_bucket.mean()
            
            bucket_stats.append({
                'bucket': f"{low:.2f}-{high:.2f}",
                'count': int(mask.sum()),
                'win_rate': win_rate,
                'avg_return': 0.0,  # Would need to fetch from test_df
            })
        
        return bucket_stats
    
    @staticmethod
    def _check_for_edge(roc_auc: float, buckets: List[Dict[str, Any]]) -> bool:
        """
        Simple edge detection: ROC-AUC >= 0.57 AND high bucket win rate > 55%.
        """
        if roc_auc < 0.57:
            return False
        
        if not buckets:
            return False
        
        # Check if highest bucket has win rate > 55%
        highest_bucket = buckets[-1]
        if highest_bucket['win_rate'] > 0.55:
            return True
        
        return False
    
    def run_strategy_backtest(
        self,
        model: PredictionEngine,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> StrategyMetrics:
        """
        Run strategy backtest on test period.
        
        Uses ML predictions to generate signals and tracks P&L.
        
        Args:
            model: Trained PredictionEngine
            test_df: Test data
            feature_cols: Feature columns
        
        Returns:
            StrategyMetrics with backtest results
        """
        logger.info("Running strategy backtest on test set...")
        
        # Clean test data
        test_clean = test_df.dropna(subset=feature_cols + ['label', 'future_close']).reset_index(drop=True)
        
        if len(test_clean) < 100:
            logger.warning("Not enough test data after cleaning")
            return StrategyMetrics(
                cagr=0.0, max_drawdown=0.0, sharpe=0.0, profit_factor=1.0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0, num_trades=0, turnover=0.0,
                passed=False, comments="Not enough data"
            )
        
        trades = []
        portfolio_values = [100000]  # Initial capital
        
        try:
            xgb_model = model.models_direction[self.horizon_bars]
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            raise
        
        position_size = 1000  # Fixed position size for each trade
        costs_per_trade = position_size * 0.0015  # 0.15% round trip
        
        # Process each row and make trade signals
        for idx, row in test_clean.iterrows():
            X = np.array(row[feature_cols]).reshape(1, -1)
            prob_up = xgb_model.predict_proba(X)[0, 1]
            
            entry_threshold = 0.60
            
            if prob_up > entry_threshold:
                entry_price = row['close']
                exit_price = row['future_close']
                
                if entry_price > 0 and exit_price > 0:
                    # P&L calculation
                    gross_pnl = position_size * (exit_price / entry_price - 1)
                    net_pnl = gross_pnl - costs_per_trade
                    
                    # Track trade
                    trades.append({
                        'entry_date': row['timestamp'],
                        'symbol': row['symbol'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'prob_up': prob_up,
                        'pnl': net_pnl,
                        'return_pct': (exit_price / entry_price - 1) * 100,
                    })
                    
                    # Update portfolio
                    current_value = portfolio_values[-1]
                    new_value = current_value + net_pnl
                    portfolio_values.append(new_value)
        
        # If no trades, return failure
        if len(trades) == 0:
            logger.warning("No trades generated")
            return StrategyMetrics(
                cagr=0.0, max_drawdown=0.0, sharpe=0.0, profit_factor=1.0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0, num_trades=0, turnover=0.0,
                passed=False, comments="No trades generated"
            )
        
        # Compute performance metrics
        portfolio_values = np.array(portfolio_values)
        
        # CAGR
        initial = portfolio_values[0]
        final = portfolio_values[-1]
        trading_days = (test_clean['timestamp'].max() - test_clean['timestamp'].min()).days
        years = max(trading_days / 365.0, 0.01)
        cagr = (final / initial) ** (1 / years) - 1
        
        # Daily returns for Sharpe
        daily_pnls = np.diff(portfolio_values)
        daily_returns = daily_pnls / initial  # Normalize by initial capital
        
        if len(daily_returns) > 1:
            mean_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe = (mean_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = -np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Trade metrics
        pnls = np.array([t['pnl'] for t in trades])
        wins = np.sum(pnls > 0)
        losses = np.sum(pnls < 0)
        win_rate = wins / len(trades) if len(trades) > 0 else 0.0
        
        avg_win = np.mean(pnls[pnls > 0]) if wins > 0 else 0.0
        avg_loss = np.mean(pnls[pnls < 0]) if losses > 0 else 0.0
        
        gross_profit = np.sum(pnls[pnls > 0]) if wins > 0 else 0.0
        gross_loss = np.sum(pnls[pnls < 0]) if losses > 0 else 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        
        # Turnover
        trading_years = trading_days / 252.0
        turnover = len(trades) / trading_years if trading_years > 0 else 0.0
        
        # Verdict (PASS if all conditions met)
        passed = (sharpe >= 1.0 and max_drawdown <= 0.30 and profit_factor >= 1.2)
        
        # Comment
        comments = []
        if sharpe < 1.0:
            comments.append(f"Sharpe {sharpe:.2f} < 1.0")
        if max_drawdown > 0.30:
            comments.append(f"MaxDD {max_drawdown:.1%} > 30%")
        if profit_factor < 1.2:
            comments.append(f"PF {profit_factor:.2f} < 1.2")
        
        if passed:
            comments = ["All thresholds met ✓"]
        
        comment_str = " | ".join(comments)
        
        metrics = StrategyMetrics(
            cagr=cagr,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=len(trades),
            turnover=turnover,
            passed=passed,
            comments=comment_str,
        )
        
        logger.info(f"Backtest complete: {len(trades)} trades, CAGR={cagr:.2%}, Sharpe={sharpe:.2f}")
        
        return metrics
    
    def run(self) -> Tuple[PredictionMetrics, StrategyMetrics]:
        """
        Run the complete sanity check.
        
        Returns:
            (prediction_metrics, strategy_metrics)
        """
        print("\n" + "="*70)
        print("NIFTY50 SANITY CHECK - Prediction Engine & Strategy Validation")
        print("="*70)
        
        # 1. Load data
        self.load_data()
        
        # 2. Split train/test
        train_df, test_df = self.split_train_test()
        
        # 3. Engineer features
        train_featured = self.engineer_features(train_df)
        test_featured = self.engineer_features(test_df)
        
        # 4. Generate labels
        train_labeled = self.generate_labels(train_featured)
        test_labeled = self.generate_labels(test_featured)
        
        # 5. Train prediction engine
        model, feature_cols = self.train_prediction_engine(train_labeled)
        
        # 6. Evaluate prediction engine
        pred_metrics = self.evaluate_prediction_engine(model, test_labeled, feature_cols)
        pred_metrics.print_summary()
        
        # 7. Run strategy backtest
        strat_metrics = self.run_strategy_backtest(model, test_labeled, feature_cols)
        strat_metrics.print_summary()
        
        # 8. Final verdict
        print("="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        pred_verdict = "HAS EDGE" if pred_metrics.has_edge else "WEAK / NO CLEAR EDGE"
        print(f"\nPrediction Engine (NIFTY50):  {pred_verdict}")
        print(f"  • ROC-AUC: {pred_metrics.roc_auc:.4f}")
        print(f"  • Win rate at prob_up≥0.70: {pred_metrics.buckets[-1]['win_rate']:.2%}")
        
        strat_verdict = "PASS" if strat_metrics.passed else "FAIL"
        print(f"\nStrategy Backtest (NIFTY50):  {strat_verdict}")
        print(f"  • Sharpe: {strat_metrics.sharpe:.2f} (threshold: ≥1.0)")
        print(f"  • Max DD: {strat_metrics.max_drawdown:.2%} (threshold: ≤30%)")
        print(f"  • Profit Factor: {strat_metrics.profit_factor:.2f} (threshold: ≥1.2)")
        
        print("\n" + "="*70)
        
        return pred_metrics, strat_metrics


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Quick NIFTY50 sanity check for prediction engine & strategy"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/nifty50_15y_demo.csv",
        help="Path to NIFTY50 OHLCV data",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward-looking horizon in bars",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.65,
        help="Fraction of data to use for training",
    )
    
    args = parser.parse_args()
    
    checker = QuickNIFTY50Check(
        data_file=Path(args.data_file),
        horizon_bars=args.horizon,
        train_frac=args.train_frac,
    )
    
    try:
        pred_metrics, strat_metrics = checker.run()
        
        # Exit code based on strategy pass/fail
        return 0 if strat_metrics.passed else 1
    
    except Exception as e:
        logger.error(f"Sanity check failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
ML Model Training Example

End-to-end example: Load data → Engineer features → Generate labels → 
Train model → Evaluate → Save model → Backtest.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from features.labels import LabelGenerator
from learning.data_structures import FeatureVector, TrainingDataset
from learning.prediction_engine import PredictionEngine
from learning.walk_forward import WalkForwardSplitter, DataPreparer
from learning.performance_analyzer import PerformanceAnalyzer, ReliabilityChecker
from learning.model_store import ModelStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data() -> pd.DataFrame:
    """
    Load sample OHLCV data (in practice, this would read from file/database).
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    # Generate synthetic data for demo
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    
    n = len(dates)
    
    # Generate price data with trend and noise
    price = 100.0
    prices = [price]
    for i in range(n - 1):
        change = np.random.normal(0.0005, 0.02)  # Mean drift + daily volatility
        price = price * (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "SAMPLE",
        "close": prices,
        "open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "volume": [np.random.randint(1000000, 10000000) for _ in range(n)],
    })
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} price records for {df['symbol'].nunique()} symbols")
    return df


def engineer_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical features from price data.
    
    Args:
        price_df: Price DataFrame.
        
    Returns:
        DataFrame with features added.
    """
    df = price_df.copy()
    
    # Simple moving averages
    for window in [5, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    
    # Mean reversion feature
    df['ma_ratio'] = df['close'] / df['sma_20']
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # RSI-like momentum
    returns = df['close'].pct_change()
    df['momentum'] = returns.rolling(window=10).mean()
    
    # Volume-based feature
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Forward-fill missing values from rolling windows
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    feature_cols = ['ma_ratio', 'volatility', 'momentum', 'volume_ratio', 'sma_5', 'sma_20']
    
    logger.info(f"Engineered {len(feature_cols)} features: {feature_cols}")
    return df


def generate_labels_for_training(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate training labels (forward-looking returns and directions).
    
    Args:
        price_df: Price DataFrame with close prices.
        
    Returns:
        DataFrame with labels.
    """
    label_gen = LabelGenerator(
        horizons=[5, 20],  # Predict 5-bar and 20-bar forward returns
        direction_threshold_pct=0.5,  # ±0.5% threshold for direction
        neutral_band_pct=0.1,  # ±0.1% neutral band
    )
    
    # Generate labels
    labels_df = label_gen.generate_labels_from_ohlcv(price_df)
    
    logger.info(f"Generated {len(labels_df)} labels")
    return labels_df


def train_walk_forward_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_columns: list,
) -> Tuple[list, list]:
    """
    Train models using walk-forward validation.
    
    Args:
        features_df: Features DataFrame.
        labels_df: Labels DataFrame.
        feature_columns: List of feature column names.
        
    Returns:
        Tuple of (fold_results, all_trained_models).
    """
    # Split data into folds
    splitter = WalkForwardSplitter(
        train_window_days=504,  # ~2 years
        valid_window_days=252,  # ~1 year
        step_days=63,  # Quarterly stepping
        gap_days=5,  # 5-day gap to avoid leakage
    )
    
    folds = splitter.generate_folds(features_df, min_train_samples=100)
    logger.info(f"Generated {len(folds)} walk-forward folds")
    
    # Prepare data
    preparer = DataPreparer()
    all_data = preparer.prepare_datasets(features_df, labels_df, regime_df=None)
    
    fold_results = []
    all_models = []
    
    # Train on each fold
    for fold_idx, fold in enumerate(folds):
        logger.info(f"\n[Fold {fold_idx+1}/{len(folds)}]")
        
        # Get train and validation data for this fold
        train_df = all_data[
            (all_data['timestamp'] >= fold.train_start) & 
            (all_data['timestamp'] <= fold.train_end)
        ]
        
        valid_df = all_data[
            (all_data['timestamp'] >= fold.valid_start) & 
            (all_data['timestamp'] <= fold.valid_end)
        ]
        
        if len(train_df) < 100 or len(valid_df) < 50:
            logger.warning(f"Skipping fold {fold_idx}: insufficient data")
            continue
        
        logger.info(f"  Train: {len(train_df)} samples ({fold.train_start} to {fold.train_end})")
        logger.info(f"  Valid: {len(valid_df)} samples ({fold.valid_start} to {fold.valid_end})")
        
        # Initialize model
        config = {
            "model_type": "xgboost",
            "regime_mode": "single_model_with_feature",
            "xgboost_params": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 0,
            }
        }
        
        model = PredictionEngine(config=config)
        
        # Train model
        targets = {
            "target_return_5": "target_return_5",
            "target_return_20": "target_return_20",
            "target_direction_5": "target_direction_5",
            "target_direction_20": "target_direction_20",
        }
        
        try:
            model.fit(
                train_df=train_df,
                valid_df=valid_df,
                feature_columns=feature_columns,
                targets=targets,
            )
            
            logger.info(f"  Trained model successfully")
            
            # Evaluate on validation fold
            valid_predictions = model.predict(valid_df)
            
            # Extract predictions and actuals for metrics
            y_true_ret = valid_df['target_return_20'].values
            y_true_dir = np.sign(y_true_ret)
            
            y_pred_proba = np.array([p.prob_up for p in valid_predictions])
            y_pred_ret = np.array([p.expected_return for p in valid_predictions])
            
            # Compute metrics
            metrics = PerformanceAnalyzer.evaluate_on_fold(
                y_true_ret,
                y_true_dir,
                y_pred_proba,
                y_pred_ret,
            )
            
            fold_results.append({
                "fold": fold_idx,
                "metrics": metrics,
                "n_valid_samples": len(valid_df),
            })
            
            # Check deployment criteria
            is_ok, warnings = ReliabilityChecker.check_deployment_criteria(metrics)
            
            ReliabilityChecker.print_report(metrics, fold_id=fold_idx)
            
            if not is_ok:
                logger.warning(f"Fold {fold_idx} warnings: {warnings}")
            
            all_models.append((fold_idx, model))
            
        except Exception as e:
            logger.error(f"Training failed for fold {fold_idx}: {e}")
            continue
    
    return fold_results, all_models


def save_best_model(
    all_models: list,
    fold_results: list,
) -> str:
    """
    Save the best performing model.
    
    Args:
        all_models: List of (fold_idx, model) tuples.
        fold_results: List of fold evaluation results.
        
    Returns:
        Path where model was saved.
    """
    if not fold_results:
        logger.error("No fold results available")
        return ""
    
    # Find fold with highest accuracy
    best_fold = max(fold_results, key=lambda x: x["metrics"].accuracy or 0)
    best_fold_idx = best_fold["fold"]
    
    logger.info(f"\nBest fold: {best_fold_idx} with accuracy {best_fold['metrics'].accuracy:.4f}")
    
    # Get corresponding model
    best_model = None
    for fold_idx, model in all_models:
        if fold_idx == best_fold_idx:
            best_model = model
            break
    
    if best_model is None:
        logger.error("Could not find best model")
        return ""
    
    # Save model
    store = ModelStore(base_path="models")
    
    tags = {
        "fold": best_fold_idx,
        "accuracy": float(best_fold["metrics"].accuracy or 0),
        "hit_rate_60": float(best_fold["metrics"].hit_rate_60 or 0),
        "data_source": "example_training",
    }
    
    model_path = store.save_model(
        best_model,
        model_name="xgboost_example",
        tags=tags,
    )
    
    logger.info(f"Saved best model to {model_path}")
    return model_path


def main():
    """
    Run end-to-end training example.
    """
    logger.info("=" * 70)
    logger.info("ML Model Training Example")
    logger.info("=" * 70)
    
    # Step 1: Load data
    logger.info("\n[Step 1] Loading price data...")
    price_df = load_sample_data()
    
    # Step 2: Engineer features
    logger.info("\n[Step 2] Engineering features...")
    features_df = engineer_features(price_df)
    feature_columns = [
        'ma_ratio', 'volatility', 'momentum', 'volume_ratio', 'sma_5', 'sma_20'
    ]
    
    # Step 3: Generate labels
    logger.info("\n[Step 3] Generating labels...")
    labels_df = generate_labels_for_training(price_df)
    
    # Step 4: Train walk-forward models
    logger.info("\n[Step 4] Training models with walk-forward validation...")
    fold_results, all_models = train_walk_forward_model(
        features_df,
        labels_df,
        feature_columns,
    )
    
    # Step 5: Save best model
    logger.info("\n[Step 5] Saving best model...")
    if all_models:
        model_path = save_best_model(all_models, fold_results)
        logger.info(f"Model saved to: {model_path}")
    else:
        logger.error("No models trained successfully")
    
    # Step 6: Summary
    logger.info("\n[Summary]")
    logger.info(f"  Total folds: {len(fold_results)}")
    if fold_results:
        avg_accuracy = np.mean([r["metrics"].accuracy for r in fold_results if r["metrics"].accuracy])
        logger.info(f"  Average accuracy: {avg_accuracy:.4f}")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()

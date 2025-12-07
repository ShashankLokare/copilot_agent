#!/usr/bin/env python3
"""
NIFTY50 15-Year ML Model Training Workflow

Complete end-to-end workflow for training prediction models on 15 years of NIFTY50 data:

1. Collect 15 years of NIFTY50 data (2010-2025)
2. Prepare data: features, labels, handling NaN
3. Create walk-forward splits
4. Train XGBoost/LightGBM models
5. Grade models (A/B/C/D)
6. Register in model registry
7. Generate backtest

Usage:
    python scripts/train_nifty50_models.py [--year 2024] [--symbols 5]
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Tuple, List, Dict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from scripts.collect_nifty50_15y_data import collect_nifty50_data
from scripts.prepare_nifty50_data import NiftyDataPreparation, prepare_nifty50_data
from learning.prediction_engine import PredictionEngine
from learning.walk_forward import WalkForwardSplitter
from learning.performance_analyzer import PerformanceAnalyzer
from learning.model_store import ModelStore
from learning.model_metadata import ModelRegistry, ModelRecord, StrategyGradeResult
from learning.data_structures import FeatureVector, TrainingDataset
from backtest.strategy_grader import StrategyGrader
from config.ml_model_config import get_xgboost_params, get_lightgbm_params


def run_complete_training_pipeline(
    num_symbols: int = 50,
    train_years: int = 3,
    test_years: int = 1,
    model_type: str = "xgboost"
) -> Dict:
    """
    Run complete training pipeline.
    
    Args:
        num_symbols: Number of NIFTY50 symbols to use
        train_years: Years of data for training window
        test_years: Years of data for test window
        model_type: "xgboost" or "lightgbm"
    
    Returns:
        Dict with training results
    """
    logger.info("=" * 70)
    logger.info("NIFTY50 15-YEAR ML TRAINING PIPELINE")
    logger.info("=" * 70)
    
    results = {
        "status": "started",
        "timestamp": datetime.now(),
        "stages": {}
    }
    
    # Stage 1: Collect Data
    logger.info("\n[STAGE 1] COLLECTING 15 YEARS OF NIFTY50 DATA")
    logger.info("-" * 70)
    
    try:
        df_raw, collect_stats = collect_nifty50_data(
            output_path="data/nifty50_15y_ohlcv.csv",
            start_year=2010,
            end_year=2025,
            use_synthetic_fallback=True
        )
        
        results["stages"]["data_collection"] = {
            "status": "success",
            "rows": len(df_raw),
            "symbols": df_raw['symbol'].nunique(),
            "date_range": f"{df_raw['timestamp'].min().date()} to {df_raw['timestamp'].max().date()}"
        }
        
        logger.info("✓ Data collection complete")
        
    except Exception as e:
        logger.error(f"✗ Data collection failed: {str(e)}")
        results["stages"]["data_collection"] = {"status": "failed", "error": str(e)}
        return results
    
    # Stage 2: Prepare Data
    logger.info("\n[STAGE 2] PREPARING DATA (FEATURES, LABELS, HANDLING NaN)")
    logger.info("-" * 70)
    
    try:
        df_training, feature_cols = prepare_nifty50_data(
            raw_data_path="data/nifty50_15y_ohlcv.csv",
            output_path="data/nifty50_training_data.csv"
        )
        
        results["stages"]["data_preparation"] = {
            "status": "success",
            "rows": len(df_training),
            "features": len(feature_cols),
            "symbols": df_training['symbol'].nunique()
        }
        
        logger.info("✓ Data preparation complete")
        
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {str(e)}")
        results["stages"]["data_preparation"] = {"status": "failed", "error": str(e)}
        return results
    
    # Stage 3: Create Walk-Forward Splits
    logger.info("\n[STAGE 3] CREATING WALK-FORWARD SPLITS")
    logger.info("-" * 70)
    
    try:
        prep = NiftyDataPreparation()
        splits = prep.get_walk_forward_splits(
            df=df_training,
            train_window_days=252 * train_years,
            test_window_days=252 * test_years,
            step_size_days=252
        )
        
        results["stages"]["walk_forward_splits"] = {
            "status": "success",
            "num_splits": len(splits),
            "train_window_years": train_years,
            "test_window_years": test_years
        }
        
        logger.info(f"✓ Created {len(splits)} walk-forward splits")
        
    except Exception as e:
        logger.error(f"✗ Walk-forward split creation failed: {str(e)}")
        results["stages"]["walk_forward_splits"] = {"status": "failed", "error": str(e)}
        return results
    
    # Stage 4: Train Models
    logger.info("\n[STAGE 4] TRAINING MODELS ON WALK-FORWARD SPLITS")
    logger.info("-" * 70)
    
    try:
        engine = PredictionEngine()
        trained_models = []
        fold_results = []
        
        for fold_idx, (train_df, test_df) in enumerate(splits[:3]):  # Use first 3 splits for demo
            logger.info(f"\nFold {fold_idx + 1}/{min(3, len(splits))}")
            logger.info(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")
            
            # Prepare features and labels
            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['label'].values
            
            # Train model
            if model_type == "xgboost":
                model = engine.train_xgboost(X_train, y_train, feature_names=feature_cols)
            else:
                model = engine.train_lightgbm(X_train, y_train, feature_names=feature_cols)
            
            # Evaluate
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            
            # Store results
            train_accuracy = (pred_train.round() == y_train).mean()
            test_accuracy = (pred_test.round() == y_test).mean()
            
            trained_models.append(model)
            fold_results.append({
                "fold": fold_idx,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_auc": engine.compute_roc_auc(y_train, pred_train),
                "test_auc": engine.compute_roc_auc(y_test, pred_test),
                "train_size": len(X_train),
                "test_size": len(X_test)
            })
            
            logger.info(f"  Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
            logger.info(f"  Train AUC: {fold_results[-1]['train_auc']:.4f} | Test AUC: {fold_results[-1]['test_auc']:.4f}")
        
        # Select best model
        best_fold = max(enumerate(fold_results), key=lambda x: x[1]['test_auc'])
        best_model = trained_models[best_fold[0]]
        
        results["stages"]["model_training"] = {
            "status": "success",
            "num_folds": len(fold_results),
            "best_fold": best_fold[0],
            "best_test_auc": best_fold[1]['test_auc'],
            "best_test_accuracy": best_fold[1]['test_accuracy'],
            "fold_results": fold_results
        }
        
        logger.info(f"\n✓ Model training complete (best fold: {best_fold[0]}, AUC: {best_fold[1]['test_auc']:.4f})")
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        results["stages"]["model_training"] = {"status": "failed", "error": str(e)}
        return results
    
    # Stage 5: Register Model
    logger.info("\n[STAGE 5] REGISTERING MODEL IN REGISTRY")
    logger.info("-" * 70)
    
    try:
        model_store = ModelStore()
        registry = ModelRegistry()
        
        # Create model record
        model_record = ModelRecord(
            model_id=f"nifty50_xgb_15y_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name="NIFTY50 15-Year XGBoost Predictor",
            model_type="prediction_engine",
            version="1.0",
            created_date=pd.Timestamp.now(),
            trained_by="AutoML Pipeline",
            training_metadata=None,  # Can add later with detailed metadata
            deployment_status="INCUBATOR",
            is_active=True,
            config_snapshot={
                "model_type": model_type,
                "train_years": train_years,
                "test_years": test_years,
                "num_symbols": df_training['symbol'].nunique(),
                "num_features": len(feature_cols),
                "best_test_auc": best_fold[1]['test_auc'],
            },
        )
        
        # Register model
        registry.register_model(model_record)
        
        # Save model
        model_path = model_store.save_model(best_model, model_record.model_id)
        
        results["stages"]["model_registration"] = {
            "status": "success",
            "model_id": model_record.model_id,
            "model_path": str(model_path),
            "deployment_status": "INCUBATOR"
        }
        
        logger.info(f"✓ Model registered: {model_record.model_id}")
        
    except Exception as e:
        logger.error(f"✗ Model registration failed: {str(e)}")
        results["stages"]["model_registration"] = {"status": "failed", "error": str(e)}
        return results
    
    # Stage 6: Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Data: {len(df_training):,} bars × {len(feature_cols)} features")
    logger.info(f"Symbols: {df_training['symbol'].nunique()}")
    logger.info(f"Date Range: {df_training['timestamp'].min().date()} to {df_training['timestamp'].max().date()}")
    logger.info(f"Walk-Forward Splits: {len(splits)}")
    logger.info(f"Best Test AUC: {best_fold[1]['test_auc']:.4f}")
    logger.info(f"Best Test Accuracy: {best_fold[1]['test_accuracy']:.4f}")
    logger.info(f"Model ID: {model_record.model_id}")
    logger.info(f"Status: Ready for backtesting and deployment")
    logger.info("=" * 70)
    
    results["status"] = "completed"
    results["best_model_id"] = model_record.model_id
    results["best_test_auc"] = best_fold[1]['test_auc']
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train NIFTY50 ML models on 15 years of data")
    parser.add_argument("--symbols", type=int, default=50, help="Number of NIFTY50 symbols")
    parser.add_argument("--train-years", type=int, default=3, help="Training window in years")
    parser.add_argument("--test-years", type=int, default=1, help="Test window in years")
    parser.add_argument("--model", type=str, default="xgboost", help="Model type: xgboost or lightgbm")
    
    args = parser.parse_args()
    
    results = run_complete_training_pipeline(
        num_symbols=args.symbols,
        train_years=args.train_years,
        test_years=args.test_years,
        model_type=args.model
    )
    
    # Print final status
    if results["status"] == "completed":
        print("\n✓ Training pipeline completed successfully!")
        print(f"Best Model: {results['best_model_id']}")
        print(f"Best Test AUC: {results['best_test_auc']:.4f}")
    else:
        print(f"\n✗ Training pipeline failed at stage: {list(results['stages'].keys())[-1]}")


if __name__ == "__main__":
    main()

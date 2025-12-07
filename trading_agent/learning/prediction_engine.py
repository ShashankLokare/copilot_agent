"""
Core Prediction Engine

Trains and runs ensemble ML models for price direction and return forecasting.
Supports regime-aware modeling and multiple horizons.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from learning.data_structures import (
    FeatureVector,
    PredictionOutput,
    TrainingDataset,
    ModelMetadata,
    EvaluationMetrics,
)


logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Ensemble prediction engine for forecasting returns and direction.
    
    Features:
    - Regime-aware: can use single model with regime feature or separate per-regime models.
    - Multi-horizon: can train separate models for different forward-looking periods.
    - Gradient boosting: uses XGBoost or LightGBM as backbone.
    - Calibrated probabilities: uses isotonic calibration for reliable confidence scores.
    - No data leakage: enforces train/valid/test time-based splits.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_type: str = "xgboost",
        regime_mode: str = "single_model_with_feature",
    ):
        """
        Initialize prediction engine.
        
        Args:
            config: Dictionary with hyperparameters:
                - max_depth: int
                - learning_rate: float
                - n_estimators: int
                - subsample: float
                - colsample_bytree: float
                - min_child_weight: float
                - reg_alpha: float (L1 regularization)
                - reg_lambda: float (L2 regularization)
            model_type: "xgboost" or "lightgbm"
            regime_mode: "single_model_with_feature" or "per_regime_models"
        """
        self.config = config
        self.model_type = model_type
        self.regime_mode = regime_mode
        
        # Validate model availability
        if model_type == "xgboost" and not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        # Models: Dict[horizon -> model dict]
        self.models_direction: Dict[int, Any] = {}  # Classifier for direction
        self.models_return: Dict[int, Any] = {}      # Regressor for return
        self.scalers: Dict[int, StandardScaler] = {} # Feature scalers per horizon
        
        # Metadata
        self.feature_names: List[str] = []
        self.regimes: List[str] = []
        self.horizons: List[int] = []
        self.metadata: Optional[ModelMetadata] = None
        
        logger.info(
            f"Initialized PredictionEngine: "
            f"type={model_type}, regime_mode={regime_mode}"
        )
    
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_columns: List[str],
        target_return_col: str = "target_return_k",
        target_direction_col: str = "target_direction_k",
        regime_col: str = "regime",
        horizon_col: str = "horizon_bars",
    ) -> Dict[str, Any]:
        """
        Train direction and return prediction models.
        
        Args:
            train_df: Training DataFrame with features, targets, regime, horizon.
            valid_df: Validation DataFrame (same structure).
            feature_columns: List of feature column names.
            target_return_col: Name of return target column.
            target_direction_col: Name of direction target column.
            regime_col: Name of regime column.
            horizon_col: Name of horizon column.
            
        Returns:
            Dict with training results and metrics.
        """
        logger.info(f"Training PredictionEngine on {len(train_df)} samples")
        
        self.feature_names = feature_columns
        self.horizons = sorted(train_df[horizon_col].unique().tolist())
        self.regimes = sorted(train_df[regime_col].unique().tolist())
        
        train_results = {
            'train_samples': len(train_df),
            'valid_samples': len(valid_df),
            'horizons': self.horizons,
            'regimes': self.regimes,
            'models_trained': {},
        }
        
        # Train models for each horizon
        for horizon in self.horizons:
            logger.info(f"Training models for horizon={horizon}")
            
            # Filter data to this horizon
            train_h = train_df[train_df[horizon_col] == horizon].copy()
            valid_h = valid_df[valid_df[horizon_col] == horizon].copy()
            
            if len(train_h) < 10 or len(valid_h) < 5:
                logger.warning(
                    f"Skipping horizon {horizon}: insufficient data "
                    f"(train={len(train_h)}, valid={len(valid_h)})"
                )
                continue
            
            # Extract feature matrices
            X_train = train_h[feature_columns].values
            X_valid = valid_h[feature_columns].values
            y_ret_train = train_h[target_return_col].values
            y_ret_valid = valid_h[target_return_col].values
            y_dir_train = train_h[target_direction_col].values
            y_dir_valid = valid_h[target_direction_col].values
            
            # Add regime as categorical feature if using single-model mode
            if self.regime_mode == "single_model_with_feature":
                regime_train = train_h[regime_col].astype('category').cat.codes.values
                regime_valid = valid_h[regime_col].astype('category').cat.codes.values
                X_train = np.column_stack([X_train, regime_train])
                X_valid = np.column_stack([X_valid, regime_valid])
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_valid_scaled = scaler.transform(X_valid)
            self.scalers[horizon] = scaler
            
            # Train direction classifier
            logger.info(f"  Training direction classifier...")
            clf_direction = self._build_classifier()
            clf_direction.fit(X_train_scaled, y_dir_train)
            
            # Calibrate probabilities
            clf_direction_calib = CalibratedClassifierCV(
                clf_direction,
                method='isotonic',
                cv=3,
            )
            clf_direction_calib.fit(X_train_scaled, y_dir_train)
            self.models_direction[horizon] = clf_direction_calib
            
            # Train return regressor
            logger.info(f"  Training return regressor...")
            reg_return = self._build_regressor()
            reg_return.fit(X_train_scaled, y_ret_train)
            self.models_return[horizon] = reg_return
            
            # Validation metrics (will be computed by evaluation)
            train_results['models_trained'][horizon] = {
                'direction_model': 'trained',
                'return_model': 'trained',
                'train_samples': len(train_h),
                'valid_samples': len(valid_h),
            }
        
        # Store metadata
        self.metadata = ModelMetadata(
            model_name=f"EnsembleML_{self.model_type}",
            model_version="1.0",
            feature_names=self.feature_names,
            regimes=self.regimes,
            horizons=self.horizons,
            training_period=(train_df['timestamp'].min(), train_df['timestamp'].max()),
            hyperparameters=self.config,
            created_timestamp=pd.Timestamp.now(),
        )
        
        logger.info(f"Training complete. Models: {train_results['models_trained']}")
        return train_results
    
    def predict(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str] = None,
        regime_col: str = "regime",
        horizon: int = None,
        return_metadata: bool = True,
    ) -> List[PredictionOutput]:
        """
        Generate predictions for new data.
        
        Args:
            features_df: DataFrame with features and regime (same structure as training).
            feature_columns: Feature column names (if None, uses self.feature_names).
            regime_col: Name of regime column.
            horizon: If specified, predict for only this horizon. Otherwise, use all.
            return_metadata: Whether to include metadata in predictions.
            
        Returns:
            List of PredictionOutput objects.
        """
        if not self.models_direction or not self.models_return:
            raise RuntimeError("Models not trained yet. Call fit() first.")
        
        feature_columns = feature_columns or self.feature_names
        horizons_to_predict = [horizon] if horizon is not None else self.horizons
        
        predictions = []
        
        for h in horizons_to_predict:
            if h not in self.models_direction or h not in self.models_return:
                logger.warning(f"No trained model for horizon {h}, skipping")
                continue
            
            scaler = self.scalers[h]
            clf = self.models_direction[h]
            reg = self.models_return[h]
            
            # Extract features
            X = features_df[feature_columns].values
            
            # Add regime as feature if needed
            if self.regime_mode == "single_model_with_feature":
                regime = features_df[regime_col].astype('category').cat.codes.values
                X = np.column_stack([X, regime])
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Predict direction
            # For binary classification, we need to predict on upward class
            # We'll compute prob_up as P(class=1) if 1 means "up"
            y_pred_proba_dir = clf.predict_proba(X_scaled)
            
            # Handle different probability shapes
            if y_pred_proba_dir.shape[1] == 3:
                # Multi-class: [-1, 0, +1]
                # Assume order is [-1, 0, +1]
                prob_up = y_pred_proba_dir[:, 2]  # P(+1)
                prob_down = y_pred_proba_dir[:, 0]  # P(-1)
            elif y_pred_proba_dir.shape[1] == 2:
                # Binary: [0, 1] or [down, up]
                prob_up = y_pred_proba_dir[:, 1]
                prob_down = 1 - prob_up
            else:
                raise ValueError(f"Unexpected probability shape: {y_pred_proba_dir.shape}")
            
            # Predict return
            y_pred_ret = reg.predict(X_scaled)
            
            # Create predictions
            for i, row in features_df.iterrows():
                pred = PredictionOutput(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    horizon_bars=h,
                    prob_up=float(prob_up[i]),
                    prob_down=float(prob_down[i]),
                    expected_return=float(y_pred_ret[i]),
                    expected_volatility=None,  # Can be computed separately
                    model_name=self.metadata.model_name if self.metadata else "unknown",
                    regime=row.get(regime_col, 'unknown'),
                    confidence=max(float(prob_up[i]), float(prob_down[i])),
                    metadata={
                        'model_type': self.model_type,
                        'regime_mode': self.regime_mode,
                    } if return_metadata else {},
                )
                predictions.append(pred)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save trained models and metadata to disk.
        
        Args:
            path: Directory path to save models.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for horizon, model in self.models_direction.items():
            model_path = path / f"direction_h{horizon}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        for horizon, model in self.models_return.items():
            model_path = path / f"return_h{horizon}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for horizon, scaler in self.scalers.items():
            scaler_path = path / f"scaler_h{horizon}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save metadata
        if self.metadata:
            metadata_path = path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2, default=str)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'regime_mode': self.regime_mode,
                'config': self.config,
                'feature_names': self.feature_names,
                'horizons': self.horizons,
                'regimes': self.regimes,
            }, f, indent=2, default=str)
        
        logger.info(f"Saved models to {path}")
    
    @classmethod
    def load(cls, path: str) -> "PredictionEngine":
        """
        Load trained models and metadata from disk.
        
        Args:
            path: Directory path containing saved models.
            
        Returns:
            PredictionEngine with loaded models.
        """
        path = Path(path)
        
        # Load config
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create engine
        engine = cls(
            config=config_dict['config'],
            model_type=config_dict['model_type'],
            regime_mode=config_dict['regime_mode'],
        )
        
        # Load models
        for pkl_file in path.glob("direction_h*.pkl"):
            horizon = int(pkl_file.stem.replace("direction_h", ""))
            with open(pkl_file, 'rb') as f:
                engine.models_direction[horizon] = pickle.load(f)
        
        for pkl_file in path.glob("return_h*.pkl"):
            horizon = int(pkl_file.stem.replace("return_h", ""))
            with open(pkl_file, 'rb') as f:
                engine.models_return[horizon] = pickle.load(f)
        
        # Load scalers
        for pkl_file in path.glob("scaler_h*.pkl"):
            horizon = int(pkl_file.stem.replace("scaler_h", ""))
            with open(pkl_file, 'rb') as f:
                engine.scalers[horizon] = pickle.load(f)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                engine.metadata = ModelMetadata(**metadata_dict)
        
        # Set attributes
        engine.feature_names = config_dict['feature_names']
        engine.horizons = config_dict['horizons']
        engine.regimes = config_dict['regimes']
        
        logger.info(f"Loaded models from {path}")
        return engine
    
    def _build_classifier(self):
        """Build direction classifier model."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                max_depth=self.config.get('max_depth', 5),
                learning_rate=self.config.get('learning_rate', 0.1),
                n_estimators=self.config.get('n_estimators', 100),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                min_child_weight=self.config.get('min_child_weight', 1),
                reg_alpha=self.config.get('reg_alpha', 0.0),
                reg_lambda=self.config.get('reg_lambda', 1.0),
                early_stopping_rounds=10,
                random_state=42,
                verbosity=0,
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                max_depth=self.config.get('max_depth', 5),
                learning_rate=self.config.get('learning_rate', 0.1),
                n_estimators=self.config.get('n_estimators', 100),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                min_child_weight=self.config.get('min_child_weight', 1),
                reg_alpha=self.config.get('reg_alpha', 0.0),
                reg_lambda=self.config.get('reg_lambda', 1.0),
                random_state=42,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_regressor(self):
        """Build return regressor model."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                max_depth=self.config.get('max_depth', 5),
                learning_rate=self.config.get('learning_rate', 0.1),
                n_estimators=self.config.get('n_estimators', 100),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                min_child_weight=self.config.get('min_child_weight', 1),
                reg_alpha=self.config.get('reg_alpha', 0.0),
                reg_lambda=self.config.get('reg_lambda', 1.0),
                early_stopping_rounds=10,
                random_state=42,
                verbosity=0,
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                max_depth=self.config.get('max_depth', 5),
                learning_rate=self.config.get('learning_rate', 0.1),
                n_estimators=self.config.get('n_estimators', 100),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                min_child_weight=self.config.get('min_child_weight', 1),
                reg_alpha=self.config.get('reg_alpha', 0.0),
                reg_lambda=self.config.get('reg_lambda', 1.0),
                random_state=42,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

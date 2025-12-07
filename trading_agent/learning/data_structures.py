"""
ML Pipeline Data Structures

Defines core dataclasses for feature vectors, labels, and predictions.
These serve as interfaces between feature engineering, model training,
and signal generation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import pandas as pd


@dataclass
class FeatureVector:
    """
    Represents a complete feature vector for a single symbol at a single timestamp.
    
    Used as input to the prediction engine.
    """
    timestamp: pd.Timestamp
    symbol: str
    features: Dict[str, float]  # Feature name -> value mapping
    regime: str  # From regime detector (e.g., "TREND_HIGH_VOL")
    metadata: Dict = field(default_factory=dict)  # Optional metadata
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """
        Convert to numpy array in specified feature order.
        
        Args:
            feature_names: Ordered list of feature names to extract.
            
        Returns:
            np.ndarray of shape (len(feature_names),)
            
        Raises:
            KeyError if any required feature is missing.
        """
        return np.array([self.features[name] for name in feature_names])


@dataclass
class Label:
    """
    Represents a label (target) for model training.
    
    Generated from future price movements relative to a timestamp.
    """
    timestamp: pd.Timestamp
    symbol: str
    horizon_bars: int  # Forward-looking horizon (e.g., 5, 20 bars)
    
    # Return-based target
    target_return_k: float  # k-bar forward return (e.g., -0.05, 0.02)
    
    # Direction-based target (discrete)
    target_direction_k: int  # -1 (down), 0 (neutral), +1 (up)
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if label is valid (non-NaN, reasonable values)."""
        return (
            not np.isnan(self.target_return_k) and
            self.target_direction_k in [-1, 0, 1] and
            self.horizon_bars > 0
        )


@dataclass
class PredictionOutput:
    """
    Represents a single prediction from the prediction engine.
    
    Used as output and passed to signal generation.
    """
    timestamp: pd.Timestamp
    symbol: str
    horizon_bars: int
    
    # Directional prediction
    prob_up: float  # P(next k-bar return > 0), typically in [0, 1]
    prob_down: float  # P(next k-bar return < 0), typically in [0, 1]
    
    # Return prediction
    expected_return: float  # E[k-bar return], e.g., -0.02, +0.03
    expected_volatility: Optional[float] = None  # Optional vol forecast
    
    # Model metadata
    model_name: str = ""
    regime: str = ""
    confidence: float = 0.5  # Model-specific confidence in [0, 1]
    
    # Additional info for debugging/analysis
    metadata: Dict = field(default_factory=dict)
    
    def direction(self) -> int:
        """
        Get most likely direction based on probabilities.
        
        Returns:
            1 if prob_up > 0.5, -1 if prob_down > 0.5, else 0
        """
        if self.prob_up > 0.5:
            return 1
        elif self.prob_down > 0.5:
            return -1
        else:
            return 0
    
    def is_confident_signal(self, prob_threshold: float = 0.6) -> bool:
        """
        Check if prediction is confident enough for trading.
        
        Args:
            prob_threshold: Minimum probability for direction confidence
            
        Returns:
            True if |prob_up - 0.5| > threshold
        """
        return abs(self.prob_up - 0.5) > (prob_threshold - 0.5)


@dataclass
class TrainingDataset:
    """
    Container for training/validation/test feature and label data.
    
    Ensures alignment between features and labels via timestamp/symbol.
    """
    features_df: pd.DataFrame  # Columns: timestamp, symbol, [features...]
    labels_df: pd.DataFrame    # Columns: timestamp, symbol, target_return_k, target_direction_k
    horizon_bars: int
    regimes: Optional[Dict[str, str]] = None  # timestamp -> regime mapping
    metadata: Dict = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Check that features and labels are aligned.
        
        Returns:
            True if valid, False otherwise.
        """
        if len(self.features_df) == 0 or len(self.labels_df) == 0:
            return False
        
        # Check required columns
        required_feature_cols = {'timestamp', 'symbol'}
        required_label_cols = {'timestamp', 'symbol', 'target_return_k', 'target_direction_k'}
        
        if not required_feature_cols.issubset(self.features_df.columns):
            return False
        if not required_label_cols.issubset(self.labels_df.columns):
            return False
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names (excluding timestamp/symbol)."""
        exclude = {'timestamp', 'symbol'}
        return [col for col in self.features_df.columns if col not in exclude]


@dataclass
class ModelMetadata:
    """
    Metadata about a trained model.
    
    Used for saving/loading and versioning.
    """
    model_name: str
    model_version: str
    feature_names: List[str]
    regimes: List[str]  # Regimes the model was trained on
    horizons: List[int]  # Forward-looking horizons
    training_period: tuple  # (start_date, end_date)
    
    # Evaluation metrics (populated after training)
    train_metrics: Dict = field(default_factory=dict)
    valid_metrics: Dict = field(default_factory=dict)
    
    # Hyperparameters (for reproducibility)
    hyperparameters: Dict = field(default_factory=dict)
    
    # Build info
    created_timestamp: Optional[pd.Timestamp] = None
    python_version: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'regimes': self.regimes,
            'horizons': self.horizons,
            'training_period': self.training_period,
            'train_metrics': self.train_metrics,
            'valid_metrics': self.valid_metrics,
            'hyperparameters': self.hyperparameters,
            'created_timestamp': str(self.created_timestamp) if self.created_timestamp else None,
            'python_version': self.python_version,
        }


@dataclass
class WalkForwardFold:
    """
    Represents a single train/validation split in walk-forward testing.
    """
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    
    train_df: Optional[pd.DataFrame] = None  # Features + labels for training
    valid_df: Optional[pd.DataFrame] = None  # Features + labels for validation
    
    test_results: Dict = field(default_factory=dict)  # Populated after evaluation
    
    def __str__(self) -> str:
        return (
            f"WalkForwardFold({self.fold_id}): "
            f"train=[{self.train_start.date()}, {self.train_end.date()}], "
            f"valid=[{self.valid_start.date()}, {self.valid_end.date()}]"
        )


@dataclass
class EvaluationMetrics:
    """
    Container for ML model evaluation metrics.
    """
    # Classification metrics (for direction prediction)
    accuracy: Optional[float] = None
    precision_up: Optional[float] = None
    recall_up: Optional[float] = None
    f1_up: Optional[float] = None
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Regression metrics (for return prediction)
    mae: Optional[float] = None  # Mean absolute error
    rmse: Optional[float] = None  # Root mean squared error
    r2_score: Optional[float] = None
    
    # Trading-relevant metrics
    hit_rate_60: Optional[float] = None  # % correct when prob > 0.6
    hit_rate_70: Optional[float] = None  # % correct when prob > 0.7
    avg_return_if_signal: Optional[float] = None  # Avg actual return when model signals
    avg_return_random: Optional[float] = None  # Baseline average return
    
    # Per-regime metrics (regimes -> metric dicts)
    per_regime_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        result = {}
        for key, value in self.__dict__.items():
            if key != 'per_regime_metrics':
                result[key] = value
        result['per_regime_metrics'] = self.per_regime_metrics
        return result

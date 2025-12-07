"""
Regime detection module.
Identifies market regimes (trend/range + volatility) to inform strategy decisions.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from features.feature_engineering import Features


class Regime(Enum):
    """Market regime labels."""
    TREND_HIGH_VOL = "TREND_HIGH_VOL"
    TREND_LOW_VOL = "TREND_LOW_VOL"
    RANGE_HIGH_VOL = "RANGE_HIGH_VOL"
    RANGE_LOW_VOL = "RANGE_LOW_VOL"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    """Current regime state."""
    regime: Regime
    confidence: float  # 0.0 to 1.0
    trend_strength: float  # -1 to 1 (negative=down, positive=up)
    volatility_percentile: float  # 0 to 100


class RegimeDetector(ABC):
    """Abstract base class for regime detection."""
    
    @abstractmethod
    def detect(self, features: Features) -> RegimeState:
        """Detect current regime from features."""
        pass


class SimpleRulesRegimeDetector(RegimeDetector):
    """
    Simple rule-based regime detector.
    Uses price position, moving averages, and volatility.
    """
    
    def __init__(
        self,
        trend_threshold: float = 0.5,
        volatility_threshold: float = 1.0,
        historical_vol_percentile: float = 50.0,
    ):
        """
        Initialize detector.
        
        Args:
            trend_threshold: % change threshold for trend vs range (0-1)
            volatility_threshold: Annualized vol threshold for high vs low
            historical_vol_percentile: Percentile for vol comparison
        """
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.historical_vol_percentile = historical_vol_percentile
    
    def detect(self, features: Features) -> RegimeState:
        """
        Detect regime using simple rules.
        
        Rules:
        - Trend: Price is above/below the midpoint of SMA20-SMA50 range
        - Range: Price is within the middle of the range
        - High Vol: ATR > threshold
        - Low Vol: ATR < threshold
        """
        if features.sma_20 == 0 or features.sma_50 == 0:
            return RegimeState(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=50.0
            )
        
        # Detect trend
        sma_mid = (features.sma_20 + features.sma_50) / 2
        price_pct_above_mid = (features.price - sma_mid) / sma_mid
        
        is_uptrend = price_pct_above_mid > self.trend_threshold
        is_downtrend = price_pct_above_mid < -self.trend_threshold
        is_range = not (is_uptrend or is_downtrend)
        
        trend_strength = max(-1, min(1, price_pct_above_mid / self.trend_threshold))
        
        # Detect volatility level
        is_high_vol = features.atr > self.volatility_threshold
        
        # Determine regime
        if is_uptrend:
            regime = Regime.TREND_HIGH_VOL if is_high_vol else Regime.TREND_LOW_VOL
        elif is_downtrend:
            regime = Regime.TREND_HIGH_VOL if is_high_vol else Regime.TREND_LOW_VOL
        else:  # range
            regime = Regime.RANGE_HIGH_VOL if is_high_vol else Regime.RANGE_LOW_VOL
        
        # Confidence based on how strong the signals are
        trend_confidence = min(1.0, abs(price_pct_above_mid) / (self.trend_threshold * 2))
        vol_confidence = min(1.0, features.atr / (self.volatility_threshold * 2))
        confidence = (trend_confidence + vol_confidence) / 2
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_percentile=50.0 + (vol_confidence * 50)
        )


class MLRegimeDetector(RegimeDetector):
    """
    Machine learning-based regime detector.
    Uses a pre-trained classifier (e.g., RandomForest).
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML detector.
        
        Args:
            model_path: Path to saved model (pkl format)
        """
        self.model_path = model_path
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained model."""
        import pickle
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def detect(self, features: Features) -> RegimeState:
        """
        Detect regime using ML model.
        
        Feature vector: [sma_20, sma_50, rsi, atr, volatility, returns]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Provide model_path in __init__")
        
        # Build feature vector
        feature_vector = [
            features.sma_20,
            features.sma_50,
            features.rsi,
            features.atr,
            features.price,
            features.returns,
        ]
        
        # Predict
        prediction = self.model.predict([feature_vector])[0]
        confidence = max(self.model.predict_proba([feature_vector])[0])
        
        # Map prediction to regime
        regime_map = {
            0: Regime.TREND_HIGH_VOL,
            1: Regime.TREND_LOW_VOL,
            2: Regime.RANGE_HIGH_VOL,
            3: Regime.RANGE_LOW_VOL,
        }
        regime = regime_map.get(prediction, Regime.UNKNOWN)
        
        return RegimeState(
            regime=regime,
            confidence=float(confidence),
            trend_strength=0.0,  # Not provided by classifier
            volatility_percentile=50.0
        )


class RegimeManager:
    """
    Manages regime detection and caching.
    """
    
    def __init__(self, detector: RegimeDetector):
        """
        Initialize regime manager.
        
        Args:
            detector: RegimeDetector implementation
        """
        self.detector = detector
        self._last_regime: Optional[RegimeState] = None
    
    def get_regime(self, features: Features) -> RegimeState:
        """Get current regime."""
        regime_state = self.detector.detect(features)
        self._last_regime = regime_state
        return regime_state
    
    def get_last_regime(self) -> Optional[RegimeState]:
        """Get cached regime state."""
        return self._last_regime

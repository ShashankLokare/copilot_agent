"""
ML-Based Alpha Generation

Integrates PredictionEngine with signal generation for live trading.
Converts model predictions to tradeable signals.
"""

from typing import Optional, List
import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np

from alpha.base_alpha import AlphaModel, Signal
from learning.prediction_engine import PredictionEngine
from learning.data_structures import PredictionOutput


logger = logging.getLogger(__name__)


@dataclass
class MLAlphaConfig:
    """Configuration for ML-based alpha."""
    
    # Probability thresholds for signal generation
    long_prob_threshold: float = 0.60  # p(up) > threshold → LONG
    short_prob_threshold: float = 0.40  # p(up) < threshold → SHORT
    
    # Position sizing
    max_position_size: float = 1.0  # Max exposure per symbol
    position_size_mode: str = "fixed"  # "fixed" or "prob_weighted"
    
    # Expected return threshold for entry
    min_expected_return: float = 0.0001  # 0.01% minimum expected return
    
    # Regime-based filters
    filter_by_regime: bool = False
    allowed_regimes: Optional[List[str]] = None  # Only trade in these regimes
    
    # Holding period
    holding_bars: int = 20  # Hold position for N bars
    
    # Confidence requirement
    min_confidence: float = 0.5  # Minimum confidence for signal


class MLAlpha(AlphaModel):
    """
    Machine learning-based alpha model.
    
    Generates trading signals from ML model predictions.
    """
    
    def __init__(
        self,
        config: MLAlphaConfig,
        model: PredictionEngine,
    ):
        """
        Initialize ML alpha model.
        
        Args:
            config: MLAlphaConfig object.
            model: Trained PredictionEngine model.
        """
        super().__init__(name="MLAlpha")
        self.config = config
        self.model = model
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_signal(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        features: dict,
        regime: Optional[str] = None,
        **kwargs
    ) -> Optional[Signal]:
        """
        Generate trading signal from ML model.
        
        Args:
            timestamp: Current timestamp.
            symbol: Stock symbol.
            features: Feature dict with feature_name -> value.
            regime: Current market regime (optional).
            
        Returns:
            Signal object or None if no signal.
        """
        # Filter by regime if configured
        if self.config.filter_by_regime:
            if regime is None:
                self.logger.warning(f"Regime filter enabled but regime is None for {symbol}")
                return None
            
            if self.config.allowed_regimes and regime not in self.config.allowed_regimes:
                return None
        
        # Create feature vector
        from learning.data_structures import FeatureVector
        
        feature_vector = FeatureVector(
            timestamp=timestamp,
            symbol=symbol,
            features=features,
            regime=regime or "unknown",
        )
        
        # Get prediction from model
        try:
            predictions = self.model.predict(
                pd.DataFrame([{
                    **features,
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'regime': regime or 'unknown',
                }])
            )
            
            if not predictions:
                return None
            
            pred: PredictionOutput = predictions[0]
        except Exception as e:
            self.logger.error(f"Error getting prediction for {symbol}: {e}")
            return None
        
        # Check confidence
        if pred.confidence < self.config.min_confidence:
            return None
        
        # Check expected return
        if pred.expected_return < self.config.min_expected_return:
            return None
        
        # Generate signal based on probabilities
        signal = None
        
        # Long signal
        if pred.prob_up > self.config.long_prob_threshold:
            # Position sizing
            if self.config.position_size_mode == "prob_weighted":
                # Size proportional to confidence
                quantity = self.config.max_position_size * pred.confidence
            else:
                # Fixed size
                quantity = self.config.max_position_size
            
            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                action="LONG",
                quantity=quantity,
                confidence=pred.confidence,
                expected_return=pred.expected_return,
                holding_bars=self.config.holding_bars,
                reason=f"ML: p(up)={pred.prob_up:.3f}, exp_ret={pred.expected_return:.4f}",
            )
            
            self.logger.debug(f"LONG signal: {symbol} @ {timestamp}, confidence={pred.confidence:.3f}")
        
        # Short signal
        elif pred.prob_up < self.config.short_prob_threshold:
            # Position sizing
            if self.config.position_size_mode == "prob_weighted":
                quantity = self.config.max_position_size * (1 - pred.confidence)
            else:
                quantity = self.config.max_position_size
            
            signal = Signal(
                timestamp=timestamp,
                symbol=symbol,
                action="SHORT",
                quantity=quantity,
                confidence=1 - pred.confidence,
                expected_return=-pred.expected_return,
                holding_bars=self.config.holding_bars,
                reason=f"ML: p(down)={1-pred.prob_up:.3f}, exp_ret={-pred.expected_return:.4f}",
            )
            
            self.logger.debug(f"SHORT signal: {symbol} @ {timestamp}, confidence={1-pred.confidence:.3f}")
        
        return signal
    
    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        regime_column: Optional[str] = None,
        timestamp_column: str = "timestamp",
        symbol_column: str = "symbol",
    ) -> List[Signal]:
        """
        Generate signals for a batch of data.
        
        Args:
            df: DataFrame with features, symbols, timestamps.
            feature_columns: List of feature column names.
            regime_column: Optional regime column name.
            timestamp_column: Timestamp column name.
            symbol_column: Symbol column name.
            
        Returns:
            List of Signal objects.
        """
        signals = []
        
        # Get predictions from model
        try:
            predictions = self.model.predict(df)
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return []
        
        if not predictions:
            return []
        
        # Generate signal for each row
        for idx, row in df.iterrows():
            if idx >= len(predictions):
                break
            
            pred = predictions[idx]
            timestamp = row.get(timestamp_column)
            symbol = row.get(symbol_column)
            regime = row.get(regime_column) if regime_column else None
            
            # Filter by regime if configured
            if self.config.filter_by_regime:
                if self.config.allowed_regimes and regime not in self.config.allowed_regimes:
                    continue
            
            # Check confidence
            if pred.confidence < self.config.min_confidence:
                continue
            
            # Check expected return
            if pred.expected_return < self.config.min_expected_return:
                continue
            
            # Generate signal
            signal = None
            
            if pred.prob_up > self.config.long_prob_threshold:
                if self.config.position_size_mode == "prob_weighted":
                    quantity = self.config.max_position_size * pred.confidence
                else:
                    quantity = self.config.max_position_size
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    action="LONG",
                    quantity=quantity,
                    confidence=pred.confidence,
                    expected_return=pred.expected_return,
                    holding_bars=self.config.holding_bars,
                    reason=f"ML: p(up)={pred.prob_up:.3f}",
                )
            
            elif pred.prob_up < self.config.short_prob_threshold:
                if self.config.position_size_mode == "prob_weighted":
                    quantity = self.config.max_position_size * (1 - pred.confidence)
                else:
                    quantity = self.config.max_position_size
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    action="SHORT",
                    quantity=quantity,
                    confidence=1 - pred.confidence,
                    expected_return=-pred.expected_return,
                    holding_bars=self.config.holding_bars,
                    reason=f"ML: p(down)={1-pred.prob_up:.3f}",
                )
            
            if signal:
                signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} signals from {len(df)} rows")
        return signals
    
    def update_model(self, model: PredictionEngine) -> None:
        """
        Update the prediction engine model (for retraining).
        
        Args:
            model: New trained PredictionEngine.
        """
        self.model = model
        self.logger.info("Model updated")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/shashanklokare/Documents/copilot_agent/trading_agent")
    
    # Example usage (requires a trained model)
    # config = MLAlphaConfig(
    #     long_prob_threshold=0.60,
    #     short_prob_threshold=0.40,
    #     position_size_mode="prob_weighted",
    # )
    #
    # model = PredictionEngine.load("models/xgboost_model_latest")
    # alpha = MLAlpha(config=config, model=model)
    #
    # # Generate signal for single row
    # signal = alpha.generate_signal(
    #     timestamp=pd.Timestamp("2024-01-01"),
    #     symbol="AAPL",
    #     features={"rsi": 0.65, "ma_ratio": 1.02},
    #     regime="uptrend",
    # )
    # print(f"Signal: {signal}")
    
    print("ML Alpha model loaded. Use generate_signal() or generate_signals_batch().")

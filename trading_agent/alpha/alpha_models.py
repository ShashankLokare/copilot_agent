"""
Alpha models (strategies) for signal generation.
Each alpha produces trading signals based on market conditions.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from utils.types import Signal, SignalDirection, MarketState
from features.feature_engineering import Features
from regime.regime_detector import RegimeState


class AlphaModel(ABC):
    """Abstract base class for alpha models."""
    
    def __init__(self, name: str):
        """
        Initialize alpha model.
        
        Args:
            name: Unique name for this alpha
        """
        self.name = name
    
    @abstractmethod
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate trading signals.
        
        Args:
            market_state: Current market state
            features: Computed features
            regime: Current market regime
        
        Returns:
            List of Signal objects (may be empty)
        """
        pass


class MomentumAlpha(AlphaModel):
    """
    Momentum-based alpha.
    Trades in direction of recent price momentum.
    """
    
    def __init__(
        self,
        name: str = "momentum",
        rsi_threshold: float = 70.0,
        macd_threshold: float = 0.0,
        min_strength: float = 0.6,
    ):
        """
        Initialize momentum alpha.
        
        Args:
            name: Name of alpha
            rsi_threshold: RSI level for overbought/oversold
            macd_threshold: MACD threshold for signal
            min_strength: Minimum signal strength to emit
        """
        super().__init__(name)
        self.rsi_threshold = rsi_threshold
        self.macd_threshold = macd_threshold
        self.min_strength = min_strength
    
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate momentum signals based on RSI and MACD.
        """
        signals = []
        
        # Check MACD for direction
        macd_positive = features.macd > self.macd_threshold
        
        # Check RSI for confirmation
        rsi_strong = features.rsi > self.rsi_threshold or features.rsi < (100 - self.rsi_threshold)
        
        # Long signal: RSI > threshold AND MACD positive
        if features.rsi > self.rsi_threshold and macd_positive:
            strength = min(1.0, (features.rsi - self.rsi_threshold) / (100 - self.rsi_threshold))
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="RSI overbought + positive MACD"
                ))
        
        # Short signal: RSI < threshold AND MACD negative
        elif features.rsi < (100 - self.rsi_threshold) and not macd_positive:
            strength = min(1.0, (100 - self.rsi_threshold - features.rsi) / (100 - self.rsi_threshold))
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="RSI oversold + negative MACD"
                ))
        
        return signals


class MeanReversionAlpha(AlphaModel):
    """
    Mean reversion alpha.
    Trades against extremes, expecting reversion to moving averages.
    """
    
    def __init__(
        self,
        name: str = "mean_reversion",
        bollinger_threshold: float = 0.8,
        min_strength: float = 0.6,
    ):
        """
        Initialize mean reversion alpha.
        
        Args:
            name: Name of alpha
            bollinger_threshold: Bollinger band threshold (0-1, where 1 = at band)
            min_strength: Minimum signal strength
        """
        super().__init__(name)
        self.bollinger_threshold = bollinger_threshold
        self.min_strength = min_strength
    
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate mean reversion signals based on Bollinger Bands.
        """
        signals = []
        
        # Skip if bollinger bands not calculated
        if features.bollinger_width == 0:
            return signals
        
        # Calculate position within bollinger bands
        if features.bollinger_width > 0:
            bb_position = (features.price - features.bollinger_lower) / features.bollinger_width
        else:
            return signals
        
        # Short signal: Price near upper band
        if bb_position > (1.0 - self.bollinger_threshold):
            strength = min(1.0, (bb_position - (1.0 - self.bollinger_threshold)) / self.bollinger_threshold)
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="Price near upper Bollinger Band"
                ))
        
        # Long signal: Price near lower band
        elif bb_position < self.bollinger_threshold:
            strength = min(1.0, (self.bollinger_threshold - bb_position) / self.bollinger_threshold)
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="Price near lower Bollinger Band"
                ))
        
        return signals


class BreakoutAlpha(AlphaModel):
    """
    Breakout alpha.
    Trades breakouts above/below moving average resistance/support.
    """
    
    def __init__(
        self,
        name: str = "breakout",
        breakout_threshold: float = 0.02,  # 2% above SMA
        min_strength: float = 0.6,
    ):
        """
        Initialize breakout alpha.
        
        Args:
            name: Name of alpha
            breakout_threshold: Percent above/below SMA for breakout
            min_strength: Minimum signal strength
        """
        super().__init__(name)
        self.breakout_threshold = breakout_threshold
        self.min_strength = min_strength
    
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate breakout signals based on SMA crossovers.
        """
        signals = []
        
        # Use SMA50 as primary support/resistance
        if features.sma_50 == 0:
            return signals
        
        distance_from_sma = (features.price - features.sma_50) / features.sma_50
        
        # Long signal: Price breaks above SMA50
        if distance_from_sma > self.breakout_threshold:
            strength = min(1.0, distance_from_sma / (self.breakout_threshold * 2))
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="Breakout above SMA50"
                ))
        
        # Short signal: Price breaks below SMA50
        elif distance_from_sma < -self.breakout_threshold:
            strength = min(1.0, abs(distance_from_sma) / (self.breakout_threshold * 2))
            if strength >= self.min_strength:
                signals.append(Signal(
                    symbol=market_state.symbol,
                    direction=SignalDirection.SHORT,
                    strength=strength,
                    timestamp=market_state.timestamp,
                    alpha_name=self.name,
                    reasoning="Breakout below SMA50"
                ))
        
        return signals


class MLAlphaXGBoost(AlphaModel):
    """
    Machine learning alpha using XGBoost.
    Placeholder for extensibility - implement actual ML model.
    """
    
    def __init__(
        self,
        name: str = "ml_xgboost",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize ML alpha.
        
        Args:
            name: Name of alpha
            model_path: Path to saved XGBoost model
            confidence_threshold: Min model confidence for signal
        """
        super().__init__(name)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load XGBoost model."""
        try:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
        except ImportError:
            pass  # XGBoost not installed
    
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate signals from ML model.
        """
        if self.model is None:
            return []
        
        # Build feature vector for model
        feature_vector = [
            features.sma_20,
            features.sma_50,
            features.rsi,
            features.atr,
            features.returns,
            features.macd,
            features.bollinger_width,
        ]
        
        # This is a placeholder - would need actual XGBoost implementation
        # In practice, you'd call self.model.predict(feature_vector)
        
        return []


class AlphaEngine:
    """
    Engine that manages multiple alpha models and aggregates signals.
    """
    
    def __init__(self, alphas: Optional[List[AlphaModel]] = None):
        """
        Initialize alpha engine.
        
        Args:
            alphas: List of alpha models to use
        """
        self.alphas = alphas or []
    
    def add_alpha(self, alpha: AlphaModel):
        """Add an alpha model."""
        self.alphas.append(alpha)
    
    def generate_all_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate signals from all alphas.
        
        Returns:
            Aggregated list of signals from all alphas
        """
        all_signals = []
        
        for alpha in self.alphas:
            signals = alpha.generate_signals(market_state, features, regime)
            all_signals.extend(signals)
        
        return all_signals

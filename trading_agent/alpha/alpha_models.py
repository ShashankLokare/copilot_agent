"""
Alpha models (strategies) for signal generation.
Each alpha produces trading signals based on market conditions.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
from utils.types import Signal, SignalDirection, MarketState
from features.feature_engineering import Features
from regime.regime_detector import RegimeState
from learning.model_store import ModelStore


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
    Machine learning alpha using the trained XGBoost classifier saved by the
    training pipeline. Applies regime gating, expected-edge filtering, and
    high conviction probability thresholds.
    """
    
    def __init__(
        self,
        name: str = "ml_xgboost",
        model_id: str = "nifty50_xgboost_adv_20251207_155931",
        model_version: Optional[str] = None,
        model_store_path: str = "models",
        prob_long: float = 0.99,
        prob_short: float = 0.01,
        min_expected_edge: float = 0.001,
        allow_regime_buckets: Optional[List[int]] = None,
        allow_shorts: bool = False,
    ):
        """
        Initialize ML alpha.
        
        Args:
            name: Name of alpha
            model_id: Model registry id (folder under models/)
            model_version: Optional version folder to load (latest if None)
            model_store_path: Base models directory
            prob_long: Probability threshold for long signals
            prob_short: Probability threshold for short signals
            min_expected_edge: Minimum (prob-0.5) edge to accept a trade
            allow_regime_buckets: Optional list of allowed regime buckets (-3..5)
            allow_shorts: If False, discard short trades
        """
        super().__init__(name)
        self.model_id = model_id
        self.model_version = model_version
        self.model_store_path = model_store_path
        self.prob_long = prob_long
        self.prob_short = prob_short
        self.min_expected_edge = min_expected_edge
        self.allow_regime_buckets = allow_regime_buckets
        self.allow_shorts = allow_shorts
        
        self.model = None
        self.feature_names: List[str] = []
        
        self._load_model()
    
    def _load_model(self):
        """Load calibrated classifier and feature names from registry if present."""
        try:
            store = ModelStore(base_path=self.model_store_path)
            self.model = store.load_model(self.model_id, version=self.model_version)
        except Exception:
            self.model = None
        
        # Load feature names from registry.json if available
        registry_path = Path(self.model_store_path) / "registry.json"
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
                if self.model_id in registry and registry[self.model_id]:
                    meta = registry[self.model_id][0]
                    self.feature_names = meta.get("training_features", [])
            except Exception:
                self.feature_names = []
        
        # Fallback: empty feature list will trigger short-circuit in generate_signals
    
    @staticmethod
    def _trend_slope(series: pd.Series, window: int = 20) -> float:
        """Rolling slope normalized by mean price."""
        if series.isna().any() or len(series) < window:
            return 0.0
        idx = np.arange(window)
        coef = np.polyfit(idx, series.iloc[-window:].values, 1)
        denom = np.mean(series.iloc[-window:]) + 1e-9
        return float(coef[0] / denom)
    
    def _compute_feature_row(self, bars) -> Optional[np.ndarray]:
        """Compute the feature vector aligned to training schema."""
        if not self.feature_names:
            return None
        
        df = pd.DataFrame(
            {
                "timestamp": [b.timestamp for b in bars],
                "close": [b.close for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "volume": [b.volume for b in bars],
            }
        ).sort_values("timestamp")
        
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]
        
        # Core rolling stats
        df["sma_5"] = close.rolling(5).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()
        df["sma_ratio_5_20"] = (close / df["sma_20"]).fillna(1.0) - 1
        df["sma_ratio_20_50"] = (df["sma_20"] / df["sma_50"]).fillna(1.0) - 1
        df["sma_ratio_50_200"] = (df["sma_50"] / df["sma_200"]).fillna(1.0) - 1
        
        df["daily_return"] = close.pct_change()
        df["volatility_20"] = close.pct_change().rolling(20).std()
        df["volatility_60"] = close.pct_change().rolling(60).std()
        df["volatility_5"] = close.pct_change().rolling(5).std()
        df["volatility_10"] = close.pct_change().rolling(10).std()
        
        df["momentum_10"] = (close / close.shift(10)) - 1
        df["momentum_20"] = (close / close.shift(20)) - 1
        df["return_5"] = (close / close.shift(5)) - 1
        df["return_10"] = (close / close.shift(10)) - 1
        df["return_20"] = (close / close.shift(20)) - 1
        df["return_60"] = (close / close.shift(60)) - 1
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]
        
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_lower"] = sma - (std * 2)
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / (df["volume_sma_20"] + 1)
        df["volume_zscore_20"] = (volume - df["volume_sma_20"]) / (volume.rolling(20).std() + 1e-9)
        df["hl_range"] = (high - low) / close
        df["gap"] = (close - close.shift(1)) / close.shift(1)
        df["price_rank_252"] = close.rolling(252).rank(pct=True)
        df["trend_slope_20"] = close.rolling(20).apply(lambda s: self._trend_slope(s, 20), raw=False)
        df["trend_slope_50"] = close.rolling(50).apply(lambda s: self._trend_slope(s, 20), raw=False)
        
        # Regime buckets (trend x vol)
        if len(df) >= 30:
            q_low, q_high = df["volatility_20"].quantile([0.33, 0.66])
            df["vol_regime"] = np.select(
                [df["volatility_20"] <= q_low, df["volatility_20"] >= q_high],
                [0, 2],
                default=1,
            )
            tq_low, tq_high = df["return_20"].quantile([0.33, 0.66])
            df["trend_regime"] = np.select(
                [df["return_20"] <= tq_low, df["return_20"] >= tq_high],
                [-1, 1],
                default=0,
            )
            df["regime_bucket"] = df["trend_regime"] * 3 + df["vol_regime"]
        else:
            df["vol_regime"] = 1
            df["trend_regime"] = 0
            df["regime_bucket"] = 0
        
        latest = df.iloc[-1].fillna(0.0)
        vector = [float(latest.get(col, 0.0)) for col in self.feature_names]
        return np.array(vector, dtype=float)
    
    def generate_signals(
        self,
        market_state: MarketState,
        features: Features,
        regime: RegimeState
    ) -> List[Signal]:
        """
        Generate signals from ML model with regime and edge gating.
        """
        if self.model is None or not self.feature_names:
            return []
        
        # Require enough bars to compute longer-horizon features
        if len(market_state.bars) < max(60, len(self.feature_names)):
            return []
        
        feature_vector = self._compute_feature_row(market_state.bars)
        if feature_vector is None:
            return []
        
        try:
            proba = float(self.model.predict_proba(feature_vector.reshape(1, -1))[:, 1][0])
        except Exception:
            return []
        
        # Regime gate using derived regime_bucket if present in features
        regime_bucket = None
        try:
            df_tmp = pd.DataFrame(
                {
                    "timestamp": [b.timestamp for b in market_state.bars],
                    "close": [b.close for b in market_state.bars],
                    "high": [b.high for b in market_state.bars],
                    "low": [b.low for b in market_state.bars],
                    "volume": [b.volume for b in market_state.bars],
                }
            )
            if len(df_tmp) >= 30:
                vol20 = df_tmp["close"].pct_change().rolling(20).std()
                q_low, q_high = vol20.quantile([0.33, 0.66])
                vol_regime = (
                    0 if vol20.iloc[-1] <= q_low else 2 if vol20.iloc[-1] >= q_high else 1
                )
                ret20 = (df_tmp["close"] / df_tmp["close"].shift(20)) - 1
                tq_low, tq_high = ret20.quantile([0.33, 0.66])
                trend_regime = (
                    -1 if ret20.iloc[-1] <= tq_low else 1 if ret20.iloc[-1] >= tq_high else 0
                )
                regime_bucket = trend_regime * 3 + vol_regime
            else:
                regime_bucket = 0
        except Exception:
            regime_bucket = None
        
        if self.allow_regime_buckets is not None and regime_bucket is not None:
            if regime_bucket not in self.allow_regime_buckets:
                return []
        
        direction = SignalDirection.LONG if proba >= 0.5 else SignalDirection.SHORT
        expected_edge = (proba - 0.5) if direction == SignalDirection.LONG else (0.5 - proba)
        
        if not self.allow_shorts and direction == SignalDirection.SHORT:
            return []
        # Probability thresholds to enforce sparse high-conviction signals
        if direction == SignalDirection.LONG and proba < self.prob_long:
            return []
        if direction == SignalDirection.SHORT and proba > self.prob_short:
            return []
        if expected_edge < self.min_expected_edge:
            return []
        
        strength = float(max(0.0, min(1.0, proba)))
        
        return [
            Signal(
                symbol=market_state.symbol,
                direction=direction,
                strength=strength,
                timestamp=market_state.timestamp,
                alpha_name=self.name,
                reasoning=f"ML prob={proba:.3f}, edge={expected_edge:.4f}, regime={regime_bucket}"
            )
        ]


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

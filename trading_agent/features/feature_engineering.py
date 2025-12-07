"""
Feature engineering and technical indicator computation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.types import Bar, MarketState


@dataclass
class Features:
    """Container for computed features."""
    timestamp: pd.Timestamp
    symbol: str
    price: float
    returns: float = 0.0
    log_returns: float = 0.0
    
    # Moving averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    
    # Momentum indicators
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Volatility
    atr: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_width: float = 0.0
    
    # Volume-based
    volume_sma: float = 0.0
    on_balance_volume: float = 0.0
    
    # Custom fields for extensibility
    custom: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom is None:
            self.custom = {}


class FeatureEngineering:
    """
    Compute technical features from market data.
    Extensible design allows adding new indicators easily.
    """
    
    # Default lookback periods
    DEFAULT_LOOKBACKS = {
        "price_lookback": 252,
        "sma_short": 20,
        "sma_mid": 50,
        "sma_long": 200,
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "bollinger_period": 20,
        "volume_period": 20,
    }
    
    def __init__(self, lookback_config: Optional[Dict[str, int]] = None):
        """
        Initialize feature engineering.
        
        Args:
            lookback_config: Override default lookback periods
        """
        self.lookbacks = self.DEFAULT_LOOKBACKS.copy()
        if lookback_config:
            self.lookbacks.update(lookback_config)
    
    def compute_features(
        self,
        bars: List[Bar],
        enabled_indicators: Optional[List[str]] = None
    ) -> Features:
        """
        Compute all features for the most recent bar.
        
        Args:
            bars: List of Bar objects (must be sorted by timestamp)
            enabled_indicators: List of indicators to compute (None = compute all)
        
        Returns:
            Features object with computed values
        """
        if not bars:
            raise ValueError("bars list is empty")
        
        # Convert to DataFrame for easier computation
        df = self._bars_to_dataframe(bars)
        
        if enabled_indicators is None:
            enabled_indicators = [
                "sma_20", "sma_50", "sma_200",
                "rsi", "macd", "atr", "bollinger"
            ]
        
        features = Features(
            timestamp=df.iloc[-1]['timestamp'],
            symbol=bars[-1].symbol,
            price=float(df.iloc[-1]['close'])
        )
        
        # Basic price features
        if len(df) > 1:
            features.returns = (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
            features.log_returns = np.log(df.iloc[-1]['close'] / df.iloc[-2]['close'])
        
        # Compute enabled indicators
        if "sma_20" in enabled_indicators:
            features.sma_20 = self._compute_sma(df['close'], self.lookbacks["sma_short"])
        
        if "sma_50" in enabled_indicators:
            features.sma_50 = self._compute_sma(df['close'], self.lookbacks["sma_mid"])
        
        if "sma_200" in enabled_indicators:
            features.sma_200 = self._compute_sma(df['close'], self.lookbacks["sma_long"])
        
        if "rsi" in enabled_indicators:
            features.rsi = self._compute_rsi(df['close'], self.lookbacks["rsi_period"])
        
        if "macd" in enabled_indicators:
            macd_values = self._compute_macd(
                df['close'],
                self.lookbacks["macd_fast"],
                self.lookbacks["macd_slow"],
                self.lookbacks["macd_signal"]
            )
            features.macd = macd_values[0]
            features.macd_signal = macd_values[1]
            features.macd_histogram = macd_values[2]
        
        if "atr" in enabled_indicators:
            features.atr = self._compute_atr(
                df['high'],
                df['low'],
                df['close'],
                self.lookbacks["atr_period"]
            )
        
        if "bollinger" in enabled_indicators:
            bb_values = self._compute_bollinger(
                df['close'],
                self.lookbacks["bollinger_period"]
            )
            features.bollinger_upper = bb_values[0]
            features.bollinger_lower = bb_values[1]
            features.bollinger_width = bb_values[2]
        
        if "volume" in enabled_indicators:
            features.volume_sma = self._compute_sma(
                df['volume'],
                self.lookbacks["volume_period"]
            )
        
        return features
    
    @staticmethod
    def _bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
        """Convert Bar objects to DataFrame."""
        data = {
            'timestamp': [bar.timestamp for bar in bars],
            'symbol': [bar.symbol for bar in bars],
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars],
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def _compute_sma(prices: pd.Series, period: int) -> float:
        """Compute Simple Moving Average."""
        if len(prices) < period:
            return float(prices.iloc[-1])
        return float(prices.iloc[-period:].mean())
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int) -> float:
        """Compute Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = 100.0 - 100.0 / (1.0 + rs)
        
        # Smooth RSI for remaining bars
        for d in deltas[period + 1:]:
            if d > 0:
                up = (up * (period - 1) + d) / period
                down = (down * (period - 1)) / period
            else:
                up = (up * (period - 1)) / period
                down = (down * (period - 1) - d) / period
            rs = up / down if down != 0 else 0
            rsi = 100.0 - 100.0 / (1.0 + rs)
        
        return float(rsi)
    
    @staticmethod
    def _compute_macd(
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> tuple:
        """Compute MACD, Signal, and Histogram."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return (
            float(macd.iloc[-1]),
            float(macd_signal.iloc[-1]),
            float(macd_histogram.iloc[-1])
        )
    
    @staticmethod
    def _compute_atr(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        period: int
    ) -> float:
        """Compute Average True Range."""
        if len(highs) < period:
            return 0.0
        
        tr_list = []
        for i in range(len(highs)):
            high_low = highs.iloc[i] - lows.iloc[i]
            high_close = abs(highs.iloc[i] - closes.iloc[i - 1]) if i > 0 else high_low
            low_close = abs(lows.iloc[i] - closes.iloc[i - 1]) if i > 0 else high_low
            tr_list.append(max(high_low, high_close, low_close))
        
        tr = pd.Series(tr_list)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return float(atr.iloc[-1])
    
    @staticmethod
    def _compute_bollinger(
        prices: pd.Series,
        period: int,
        std_dev: float = 2.0
    ) -> tuple:
        """Compute Bollinger Bands (upper, lower, width)."""
        if len(prices) < period:
            return float(prices.iloc[-1]), float(prices.iloc[-1]), 0.0
        
        sma = prices.iloc[-period:].mean()
        std = prices.iloc[-period:].std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = upper - lower
        
        return float(upper), float(lower), float(width)

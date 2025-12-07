import random
import unittest
from datetime import datetime, timedelta

import pytz

from alpha.alpha_models import BreakoutAlpha, MeanReversionAlpha, MomentumAlpha
from config.config import Config
from data.adapters import DataAdapter
from features.feature_engineering import FeatureEngineering, Features
from orchestrator.orchestrator import Orchestrator
from regime.regime_detector import Regime, RegimeState
from signals.signal_processor import (
    SignalFilter,
    SignalProcessor,
    SignalScorer,
    SignalValidator,
    StrategyPerformance,
)
from utils.types import Bar, MarketState, Signal, SignalDirection


def generate_trending_bars(symbol: str = "TEST", start_price: float = 100.0, count: int = 60):
    """Create deterministic trending bars for testing."""
    bars = []
    timestamp = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    price = start_price

    for i in range(count):
        price += 2.0
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=timestamp + timedelta(days=i),
                open=price - 1.0,
                high=price + 1.0,
                low=price - 2.0,
                close=price,
                volume=1_000 + i,
            )
        )

    return bars


class FakeDataAdapter(DataAdapter):
    """Minimal data adapter that serves deterministic bars."""

    def __init__(self, bars):
        self._bars = bars

    def fetch_bars(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "1d"):
        return [bar for bar in self._bars if bar.symbol == symbol and bar.timestamp <= end_date]

    def fetch_latest_price(self, symbol: str) -> float:
        return [bar for bar in self._bars if bar.symbol == symbol][-1].close

    def get_symbols(self):
        return list({bar.symbol for bar in self._bars})


class TestPredictionLogic(unittest.TestCase):
    """Validate feature computation and alpha signal generation."""

    def setUp(self):
        self.bars = generate_trending_bars(count=40)
        self.market_state = MarketState(
            symbol="TEST",
            timestamp=self.bars[-1].timestamp,
            current_price=self.bars[-1].close,
            bid=self.bars[-1].close * 0.999,
            ask=self.bars[-1].close * 1.001,
            volume=10_000,
            volatility=0.2,
            bars=self.bars,
        )
        self.regime = RegimeState(
            regime=Regime.TREND_HIGH_VOL,
            confidence=0.8,
            trend_strength=0.7,
            volatility_percentile=70.0,
        )

    def test_indicator_case_normalization(self):
        fe = FeatureEngineering()
        features = fe.compute_features(
            self.bars,
            enabled_indicators=["SMA_20", "ATR", "RSI", "MACD", "BOLLINGER"],
        )

        self.assertGreater(features.sma_20, 0.0)
        self.assertGreater(features.atr, 0.0)
        self.assertGreater(features.bollinger_width, 0.0)
        self.assertGreaterEqual(features.rsi, 0.0)
        self.assertLessEqual(features.rsi, 100.0)

    def test_alpha_models_directional_bias(self):
        features = Features(
            timestamp=self.market_state.timestamp,
            symbol="TEST",
            price=self.market_state.current_price,
            sma_20=102.0,
            sma_50=100.0,
            rsi=90.0,
            macd=0.6,
            atr=1.2,
            bollinger_upper=115.0,
            bollinger_lower=95.0,
            bollinger_width=20.0,
        )

        momentum_signals = MomentumAlpha().generate_signals(self.market_state, features, self.regime)
        self.assertTrue(any(sig.direction is SignalDirection.LONG for sig in momentum_signals))

        features.price = 114.0
        mean_reversion_signals = MeanReversionAlpha().generate_signals(
            self.market_state, features, self.regime
        )
        self.assertTrue(any(sig.direction is SignalDirection.SHORT for sig in mean_reversion_signals))

        features.price = 103.0
        breakout_signals = BreakoutAlpha().generate_signals(self.market_state, features, self.regime)
        self.assertTrue(any(sig.direction is SignalDirection.LONG for sig in breakout_signals))

    def test_signal_scoring_uses_performance_weighting(self):
        timestamp = self.market_state.timestamp
        signals = [
            Signal("TEST", SignalDirection.LONG, 0.9, timestamp, "alpha_high"),
            Signal("TEST", SignalDirection.LONG, 0.7, timestamp, "alpha_mid"),
        ]

        validator = SignalValidator(min_signal_strength=0.5)
        scorer = SignalScorer(base_confidence=0.5)
        scorer.update_performance(
            "alpha_high",
            StrategyPerformance(win_rate=0.8, profit_factor=1.5, average_return_pct=5.0, sharpe_ratio=1.2, sample_count=50),
        )
        scorer.update_performance(
            "alpha_mid",
            StrategyPerformance(win_rate=0.65, profit_factor=1.2, average_return_pct=3.0, sharpe_ratio=0.8, sample_count=50),
        )
        signal_filter = SignalFilter(min_confidence=0.6, min_edge=0.05)
        processor = SignalProcessor(validator=validator, scorer=scorer, filter=signal_filter)

        processed = processor.process(signals)
        self.assertEqual(len(processed), 1)
        scored_signal = processed[0]
        expected_edge = scored_signal.confidence * ((0.9 + 0.7) / 2)
        self.assertAlmostEqual(scored_signal.edge, expected_edge)
        self.assertGreaterEqual(scored_signal.confidence, 0.9)


class TestTradingOrchestrator(unittest.TestCase):
    """Integration test that runs a full orchestrator iteration."""

    def test_run_iteration_updates_portfolio(self):
        random.seed(42)
        bars = generate_trending_bars()
        adapter = FakeDataAdapter(bars)
        config = Config()
        config.data.symbols = ["TEST"]
        config.features.enabled_indicators = ["SMA_20", "SMA_50", "RSI", "MACD", "ATR", "BOLLINGER"]
        config.alpha.enabled_models = ["breakout"]

        orchestrator = Orchestrator(config=config, data_adapter=adapter)
        orchestrator.run_iteration(timestamp=bars[-1].timestamp, symbols=["TEST"])

        self.assertIsNotNone(orchestrator.portfolio_state)
        self.assertIn("TEST", orchestrator.positions)
        self.assertLess(orchestrator.portfolio_state.cash, 100000.0)
        self.assertGreater(orchestrator.portfolio_state.total_value, 0.0)
        position = orchestrator.positions["TEST"]
        reconstructed_value = (
            orchestrator.portfolio_state.cash + position.quantity * position.current_price
        )
        self.assertAlmostEqual(orchestrator.portfolio_state.total_value, reconstructed_value, places=2)


if __name__ == "__main__":
    unittest.main()

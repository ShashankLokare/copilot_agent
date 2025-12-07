"""
Unit tests for core trading system components.
This is a template - expand these tests as needed.
"""

import unittest
from datetime import datetime, timedelta
import pytz

# Test imports
from utils.types import (
    Bar, Signal, SignalDirection, MarketState, Position,
    Order, OrderType, Trade, PortfolioState
)
from features.feature_engineering import FeatureEngineering
from risk.risk_engine import PositionSizer, RiskEngine
from alpha.alpha_models import MomentumAlpha, MeanReversionAlpha, BreakoutAlpha
from signals.signal_processor import SignalValidator, SignalScorer, SignalFilter


class TestDataTypes(unittest.TestCase):
    """Test core data types."""
    
    def test_bar_creation(self):
        """Test Bar dataclass."""
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime.now(pytz.UTC),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000000.0
        )
        
        self.assertEqual(bar.symbol, "AAPL")
        self.assertEqual(bar.close, 150.5)
        self.assertIsNotNone(bar.timestamp.tzinfo)
    
    def test_signal_creation(self):
        """Test Signal dataclass."""
        signal = Signal(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            timestamp=datetime.now(pytz.UTC),
            alpha_name="test_alpha",
            reasoning="Test signal"
        )
        
        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertEqual(signal.strength, 0.8)
    
    def test_position_creation(self):
        """Test Position dataclass."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now(pytz.UTC)
        )
        
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.pnl, 500.0)  # 100 * (155 - 150)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature computation."""
    
    def setUp(self):
        """Create sample bars."""
        self.fe = FeatureEngineering()
        self.bars = []

        # Generate 50 bars of data
        base_price = 100.0
        for i in range(50):
            price_change = 1.0 + (i * 0.01)  # Slight uptrend
            self.bars.append(Bar(
                symbol="TEST",
                timestamp=datetime(2024, 1, 1, tzinfo=pytz.UTC) + timedelta(days=i),
                open=base_price,
                high=base_price * 1.01,
                low=base_price * 0.99,
                close=base_price * price_change,
                volume=1000000.0
            ))
            base_price = base_price * price_change
    
    def test_sma_computation(self):
        """Test SMA calculation."""
        features = self.fe.compute_features(
            self.bars,
            enabled_indicators=["sma_20"]
        )
        
        self.assertGreater(features.sma_20, 0)
    
    def test_rsi_computation(self):
        """Test RSI calculation."""
        features = self.fe.compute_features(
            self.bars,
            enabled_indicators=["rsi"]
        )
        
        # RSI should be between 0 and 100
        self.assertGreaterEqual(features.rsi, 0)
        self.assertLessEqual(features.rsi, 100)


class TestRiskEngine(unittest.TestCase):
    """Test risk management."""
    
    def setUp(self):
        """Create risk engine."""
        self.sizer = PositionSizer(
            risk_pct_per_trade=1.0,
            atr_stop_multiple=2.0
        )
        self.risk_engine = RiskEngine(
            position_sizer=self.sizer,
            max_position_risk_pct=1.0,
            max_concurrent_positions=10
        )
    
    def test_position_sizing(self):
        """Test position size calculation."""
        size = self.sizer.calculate_position_size(
            entry_price=100.0,
            stop_loss_price=95.0,
            portfolio_equity=10000.0
        )
        
        # Risk = $100 (1% of $10k), Distance = $5
        # Size = $100 / $5 = 20 shares
        self.assertEqual(size, 20.0)
    
    def test_stop_loss_calculation(self):
        """Test ATR-based stop loss."""
        stop = self.sizer.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            direction=SignalDirection.LONG
        )
        
        # Stop = 100 - (2 * 2.0) = 96
        self.assertEqual(stop, 96.0)


class TestAlphaModels(unittest.TestCase):
    """Test alpha models."""
    
    def setUp(self):
        """Create test data."""
        self.market_state = MarketState(
            symbol="TEST",
            timestamp=datetime.now(pytz.UTC),
            current_price=100.0,
            bid=99.9,
            ask=100.1,
            volume=1000000.0
        )
        
        from features.feature_engineering import Features
        from regime.regime_detector import RegimeState, Regime
        
        self.features = Features(
            timestamp=datetime.now(pytz.UTC),
            symbol="TEST",
            price=100.0,
            sma_20=98.0,
            sma_50=99.0,
            rsi=95.0,  # Overbought
            macd=0.5,
            atr=2.0,
            bollinger_upper=105.0,
            bollinger_lower=95.0
        )
        self.features.bollinger_width = self.features.bollinger_upper - self.features.bollinger_lower
        
        self.regime = RegimeState(
            regime=Regime.TREND_HIGH_VOL,
            confidence=0.8,
            trend_strength=0.5,
            volatility_percentile=75.0
        )
    
    def test_momentum_alpha(self):
        """Test momentum alpha."""
        alpha = MomentumAlpha()
        signals = alpha.generate_signals(
            self.market_state,
            self.features,
            self.regime
        )
        
        # Should generate LONG signal (RSI overbought + positive MACD)
        self.assertGreater(len(signals), 0)
    
    def test_mean_reversion_alpha(self):
        """Test mean reversion alpha."""
        # Set price near upper band
        self.features.price = 104.5
        
        alpha = MeanReversionAlpha()
        signals = alpha.generate_signals(
            self.market_state,
            self.features,
            self.regime
        )
        
        # Should generate SHORT signal
        self.assertGreater(len(signals), 0)


class TestSignalProcessing(unittest.TestCase):
    """Test signal validation and scoring."""
    
    def setUp(self):
        """Create sample signals."""
        self.signals = [
            Signal(
                symbol="TEST",
                direction=SignalDirection.LONG,
                strength=0.8,
                timestamp=datetime.now(pytz.UTC),
                alpha_name="alpha1"
            ),
            Signal(
                symbol="TEST",
                direction=SignalDirection.LONG,
                strength=0.7,
                timestamp=datetime.now(pytz.UTC),
                alpha_name="alpha2"
            ),
        ]
    
    def test_signal_validator(self):
        """Test signal validation."""
        validator = SignalValidator(min_signal_strength=0.5)
        validated = validator.validate(self.signals)
        
        # Both signals should pass
        self.assertEqual(len(validated), 2)
    
    def test_signal_scorer(self):
        """Test signal scoring."""
        scorer = SignalScorer(base_confidence=0.5)
        validated_signals = self.signals
        scored = scorer.score_signals(validated_signals)
        
        # Should produce scored signals
        self.assertGreater(len(scored), 0)
        self.assertGreater(scored[0].confidence, 0)
    
    def test_signal_filter(self):
        """Test signal filtering."""
        scorer = SignalScorer()
        scored = scorer.score_signals(self.signals)
        
        filter = SignalFilter(min_confidence=0.5, min_edge=0.01)
        filtered = filter.filter(scored)
        
        # Filtering should work
        self.assertLessEqual(len(filtered), len(scored))


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full signal pipeline."""
        from signals.signal_processor import SignalProcessor
        
        # Create sample signals
        signals = [
            Signal(
                symbol="TEST",
                direction=SignalDirection.LONG,
                strength=0.8,
                timestamp=datetime.now(pytz.UTC),
                alpha_name="test"
            )
        ]
        
        # Process through full pipeline
        processor = SignalProcessor()
        result = processor.process(signals)
        
        # Should produce scored signals
        self.assertGreaterEqual(len(result), 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()

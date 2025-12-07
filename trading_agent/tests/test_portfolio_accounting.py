import unittest
from datetime import datetime

import pytz

from config.config import Config
from orchestrator.orchestrator import Orchestrator
from utils.types import (
    MarketState,
    PortfolioState,
    Order,
    OrderType,
    SignalDirection,
    ScoredSignal,
)
from risk.risk_engine import RiskAssessment, RiskAction
from data.adapters import DataAdapter
from features.feature_engineering import Features


class _NoopAdapter(DataAdapter):
    def fetch_bars(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "1d"):
        return []

    def fetch_latest_price(self, symbol: str) -> float:
        return 0.0

    def get_symbols(self):
        return []


class _FakeRiskEngine:
    def __init__(self, assessment: RiskAssessment):
        self.assessment = assessment
        self.last_atr = None

    def assess_trade(self, signal, entry_price, atr, portfolio_state, current_positions):
        self.last_atr = atr
        return self.assessment


def _feature(atr: float) -> Features:
    return Features(
        timestamp=datetime.now(pytz.UTC),
        symbol="TEST",
        price=100.0,
        atr=atr,
    )


class TestRiskAndSizingFallbacks(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.orchestrator = Orchestrator(config=self.config, data_adapter=_NoopAdapter())

    def test_uses_atr_when_present(self):
        assessment = RiskAssessment(action=RiskAction.ACCEPT, approved_quantity=5, reasoning="ok")
        self.orchestrator.risk_engine = _FakeRiskEngine(assessment)

        market_state = MarketState(
            symbol="TEST",
            timestamp=datetime.now(pytz.UTC),
            current_price=25.0,
            bid=24.9,
            ask=25.1,
            volume=1_000,
            volatility=0.3,
        )
        scored_signal = ScoredSignal(
            symbol="TEST",
            direction=SignalDirection.LONG,
            confidence=0.9,
            edge=0.1,
            timestamp=market_state.timestamp,
            alpha_names=["alpha"],
        )

        orders = self.orchestrator._build_orders_from_signals(
            [scored_signal],
            {"TEST": _feature(atr=2.5)},
            {"TEST": market_state},
            market_state.timestamp,
        )

        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(self.orchestrator.risk_engine.last_atr, 2.5)

    def test_falls_back_to_volatility(self):
        assessment = RiskAssessment(action=RiskAction.ACCEPT, approved_quantity=10, reasoning="vol fallback")
        self.orchestrator.risk_engine = _FakeRiskEngine(assessment)

        market_state = MarketState(
            symbol="TEST",
            timestamp=datetime.now(pytz.UTC),
            current_price=50.0,
            bid=49.9,
            ask=50.1,
            volume=1_000,
            volatility=0.2,
        )
        signal = ScoredSignal(
            symbol="TEST",
            direction=SignalDirection.LONG,
            confidence=0.8,
            edge=0.1,
            timestamp=market_state.timestamp,
            alpha_names=["alpha"],
        )

        orders = self.orchestrator._build_orders_from_signals(
            [signal],
            {"TEST": _feature(atr=0.0)},
            {"TEST": market_state},
            market_state.timestamp,
        )

        expected_atr = (market_state.volatility * market_state.current_price) / (252 ** 0.5)
        self.assertAlmostEqual(self.orchestrator.risk_engine.last_atr, expected_atr)
        self.assertEqual(len(orders), 1)

    def test_skips_when_no_volatility(self):
        assessment = RiskAssessment(action=RiskAction.ACCEPT, approved_quantity=10, reasoning="should skip")
        self.orchestrator.risk_engine = _FakeRiskEngine(assessment)

        market_state = MarketState(
            symbol="TEST",
            timestamp=datetime.now(pytz.UTC),
            current_price=50.0,
            bid=49.9,
            ask=50.1,
            volume=1_000,
            volatility=0.0,
        )
        signal = ScoredSignal(
            symbol="TEST",
            direction=SignalDirection.LONG,
            confidence=0.8,
            edge=0.1,
            timestamp=market_state.timestamp,
            alpha_names=["alpha"],
        )

        orders = self.orchestrator._build_orders_from_signals(
            [signal],
            {"TEST": _feature(atr=0.0)},
            {"TEST": market_state},
            market_state.timestamp,
        )

        self.assertEqual(len(orders), 0)


class TestPortfolioAccounting(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.orchestrator = Orchestrator(config=self.config, data_adapter=_NoopAdapter())
        self.orchestrator.portfolio_state = PortfolioState(
            timestamp=datetime.now(pytz.UTC),
            cash=100_000.0,
            equity=100_000.0,
            total_value=100_000.0,
        )

    def _market_state(self, price: float) -> MarketState:
        return MarketState(
            symbol="TEST",
            timestamp=datetime.now(pytz.UTC),
            current_price=price,
            bid=price * 0.999,
            ask=price * 1.001,
            volume=10_000,
            volatility=0.1,
        )

    def _order(self, direction: SignalDirection, qty: float, price: float) -> Order:
        return Order(
            symbol="TEST",
            quantity=qty,
            order_type=OrderType.MARKET,
            direction=direction,
            limit_price=price,
            filled_quantity=qty,
            avg_fill_price=price,
            timestamp=datetime.now(pytz.UTC),
        )

    def test_open_and_add_to_long(self):
        market_states = {"TEST": self._market_state(105.0)}
        buy_order = self._order(SignalDirection.LONG, 10, 100.0)
        add_order = self._order(SignalDirection.LONG, 5, 110.0)

        self.orchestrator._update_portfolio_with_fills([buy_order], market_states, datetime.now(pytz.UTC))
        self.orchestrator._update_portfolio_with_fills([add_order], market_states, datetime.now(pytz.UTC))

        position = self.orchestrator.positions["TEST"]
        self.assertEqual(position.quantity, 15)
        expected_entry = ((10 * 100.0) + (5 * 110.0)) / 15
        self.assertAlmostEqual(position.entry_price, expected_entry)

    def test_partial_and_full_close_long(self):
        market_states = {"TEST": self._market_state(95.0)}
        buy_order = self._order(SignalDirection.LONG, 10, 100.0)
        self.orchestrator._update_portfolio_with_fills([buy_order], market_states, datetime.now(pytz.UTC))

        partial_close = self._order(SignalDirection.SHORT, 4, 105.0)
        self.orchestrator._update_portfolio_with_fills([partial_close], market_states, datetime.now(pytz.UTC))
        position = self.orchestrator.positions["TEST"]
        self.assertEqual(position.quantity, 6)
        self.assertEqual(position.entry_price, 100.0)

        full_close = self._order(SignalDirection.SHORT, 6, 90.0)
        self.orchestrator._update_portfolio_with_fills([full_close], market_states, datetime.now(pytz.UTC))
        self.assertNotIn("TEST", self.orchestrator.positions)

    def test_flip_from_long_to_short(self):
        market_states = {"TEST": self._market_state(90.0)}
        open_long = self._order(SignalDirection.LONG, 10, 100.0)
        self.orchestrator._update_portfolio_with_fills([open_long], market_states, datetime.now(pytz.UTC))

        flip = self._order(SignalDirection.SHORT, 15, 90.0)
        self.orchestrator._update_portfolio_with_fills([flip], market_states, datetime.now(pytz.UTC))

        position = self.orchestrator.positions["TEST"]
        self.assertEqual(position.quantity, -5)
        self.assertEqual(position.entry_price, 90.0)

    def test_mark_to_market_updates_equity(self):
        market_states = {"TEST": self._market_state(105.0)}
        open_long = self._order(SignalDirection.LONG, 10, 100.0)
        self.orchestrator._update_portfolio_with_fills([open_long], market_states, datetime.now(pytz.UTC))

        market_states["TEST"] = self._market_state(120.0)
        self.orchestrator._recalculate_portfolio_values(market_states)

        position = self.orchestrator.positions["TEST"]
        self.assertAlmostEqual(position.pnl, 20.0 * 10)
        self.assertAlmostEqual(
            self.orchestrator.portfolio_state.total_value,
            self.orchestrator.portfolio_state.cash + position.quantity * position.current_price,
        )


if __name__ == "__main__":
    unittest.main()

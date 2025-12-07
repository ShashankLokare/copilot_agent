"""
Backtesting engine.
Replays historical data through the full trading pipeline.
"""

from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from utils.types import Bar, PortfolioState, Position, SignalDirection
from data.adapters import DataAdapter
from features.feature_engineering import FeatureEngineering
from regime.regime_detector import RegimeManager, SimpleRulesRegimeDetector
from alpha.alpha_models import AlphaEngine, MomentumAlpha, MeanReversionAlpha, BreakoutAlpha
from signals.signal_processor import SignalProcessor
from risk.risk_engine import RiskEngine
from portfolio.portfolio_engine import PortfolioBuilder
from execution.execution_engine import SimulatedExecutor, ExecutionEngine
from monitoring.metrics import PerformanceTracker, MetricsCalculator


class Backtester:
    """
    Full backtesting engine.
    Runs historical simulation of trading system.
    """
    
    def __init__(
        self,
        data_adapter: DataAdapter,
        initial_capital: float = 100000.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
    ):
        """
        Initialize backtester.
        
        Args:
            data_adapter: Data source adapter
            initial_capital: Starting capital
            slippage_bps: Slippage in basis points
            spread_bps: Bid-ask spread in basis points
        """
        self.data_adapter = data_adapter
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        
        # Core components
        self.feature_engine = FeatureEngineering()
        self.regime_manager = RegimeManager(SimpleRulesRegimeDetector())
        
        self.alpha_engine = AlphaEngine([
            MomentumAlpha(),
            MeanReversionAlpha(),
            BreakoutAlpha(),
        ])
        
        self.signal_processor = SignalProcessor()
        self.risk_engine = RiskEngine()
        self.portfolio_builder = PortfolioBuilder()
        
        executor = SimulatedExecutor(slippage_bps=slippage_bps, spread_bps=spread_bps)
        self.execution_engine = ExecutionEngine(executor)
        
        self.tracker = PerformanceTracker()
    
    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            symbols: List of symbols to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Bar timeframe
        
        Returns:
            Dictionary with results
        """
        # Fetch all data
        all_bars = {}
        for symbol in symbols:
            bars = self.data_adapter.fetch_bars(symbol, start_date, end_date, timeframe)
            if bars:
                all_bars[symbol] = bars
        
        if not all_bars:
            return {"error": "No data found"}
        
        # Get unique timestamps
        all_timestamps = set()
        for bars in all_bars.values():
            for bar in bars:
                all_timestamps.add(bar.timestamp)
        
        sorted_timestamps = sorted(all_timestamps)
        
        # Portfolio state
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions: Dict[str, Position] = {}
        
        # Run simulation
        for timestamp in sorted_timestamps:
            # Get current bars for this timestamp
            current_bars = {}
            for symbol in symbols:
                if symbol in all_bars:
                    # Find bar at or before this timestamp
                    matching_bars = [b for b in all_bars[symbol] if b.timestamp <= timestamp]
                    if matching_bars:
                        current_bars[symbol] = matching_bars[-1]
            
            if not current_bars:
                continue
            
            # Process each symbol
            all_signals = []
            
            for symbol, bar in current_bars.items():
                # Get recent bars for feature computation (last 200 bars)
                recent_bars = [b for b in all_bars[symbol] if b.timestamp <= timestamp][-200:]
                
                if len(recent_bars) < 20:
                    continue  # Not enough data
                
                # Compute features
                features = self.feature_engine.compute_features(recent_bars)
                
                # Detect regime
                regime = self.regime_manager.get_regime(features)
                
                # Generate signals
                from utils.types import MarketState
                market_state = MarketState(
                    symbol=symbol,
                    timestamp=timestamp,
                    current_price=bar.close,
                    bid=bar.close * 0.9999,
                    ask=bar.close * 1.0001,
                    volume=bar.volume,
                    volatility=features.atr / bar.close if bar.close > 0 else 0,
                    bars=recent_bars
                )
                
                signals = self.alpha_engine.generate_all_signals(market_state, features, regime)
                all_signals.extend(signals)
            
            # Process signals
            if all_signals:
                scored_signals = self.signal_processor.process(all_signals)
                
                # Apply risk management
                approved_signals = []
                portfolio_state = PortfolioState(
                    timestamp=timestamp,
                    cash=cash,
                    equity=sum([p.quantity * p.current_price for p in positions.values()]),
                    total_value=portfolio_value,
                    positions=positions
                )
                
                for signal in scored_signals:
                    if signal.symbol in current_bars:
                        current_price = current_bars[signal.symbol].close
                        
                        # Get ATR for risk sizing
                        recent_bars = [b for b in all_bars[signal.symbol] if b.timestamp <= timestamp][-20:]
                        if recent_bars:
                            from features.feature_engineering import FeatureEngineering
                            features = self.feature_engine.compute_features(recent_bars)
                            atr = features.atr
                        else:
                            atr = current_price * 0.02  # Default 2%
                        
                        assessment = self.risk_engine.assess_trade(
                            signal,
                            current_price,
                            atr,
                            portfolio_state,
                            list(positions.values())
                        )
                        
                        if assessment.action.value == "ACCEPT":
                            # Create and execute order
                            from utils.types import Order, OrderType
                            order = Order(
                                symbol=signal.symbol,
                                quantity=int(assessment.approved_quantity),
                                order_type=OrderType.MARKET,
                                direction=signal.direction,
                                timestamp=timestamp,
                                limit_price=current_price,  # Set limit price for fill price calculation
                            )
                            
                            # Execute order
                            try:
                                order_id = self.execution_engine.execute_order(order)
                                if order_id:
                                    # Get the filled order from executor
                                    executor = self.execution_engine.adapter
                                    if hasattr(executor, 'orders') and order_id in executor.orders:
                                        filled_order = executor.orders[order_id]
                                        
                                        # Update cash based on direction
                                        trade_cost = filled_order.quantity * filled_order.avg_fill_price
                                        if signal.direction == SignalDirection.LONG:
                                            cash -= trade_cost
                                            # Update or create position
                                            if signal.symbol in positions:
                                                pos = positions[signal.symbol]
                                                pos.quantity += filled_order.quantity
                                            else:
                                                from utils.types import Position
                                                positions[signal.symbol] = Position(
                                                    symbol=signal.symbol,
                                                    quantity=filled_order.quantity,
                                                    entry_price=filled_order.avg_fill_price,
                                                    current_price=current_price,
                                                    timestamp=timestamp,
                                                )
                                        elif signal.direction == SignalDirection.SHORT:
                                            cash += trade_cost
                                            if signal.symbol in positions:
                                                pos = positions[signal.symbol]
                                                pos.quantity -= filled_order.quantity
                                            else:
                                                from utils.types import Position
                                                positions[signal.symbol] = Position(
                                                    symbol=signal.symbol,
                                                    quantity=-filled_order.quantity,
                                                    entry_price=filled_order.avg_fill_price,
                                                    current_price=current_price,
                                                    timestamp=timestamp,
                                                )
                            except Exception as e:
                                pass  # Order execution failed, skip
            
            # Update all positions' current prices
            for symbol, pos in positions.items():
                if symbol in current_bars:
                    pos.current_price = current_bars[symbol].close
            
            # Record portfolio state
            portfolio_value = cash + sum(
                [p.quantity * p.current_price for p in positions.values()]
            )
            self.tracker.record_equity(timestamp, portfolio_value)
        
        # Calculate metrics
        metrics = self.tracker.calculate_current_metrics()
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": portfolio_value,
            "metrics": {
                "total_return_pct": metrics.total_return_pct,
                "annual_return_pct": metrics.annual_return_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
            },
            "equity_curve": self.tracker.equity_history,
            "timestamps": self.tracker.timestamp_history,
        }

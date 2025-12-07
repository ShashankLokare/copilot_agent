"""
Orchestrator - Main control loop for the trading system.
Coordinates all components and executes the full trading pipeline.
"""

from typing import List, Optional, Dict
from datetime import datetime
import time

from config.config import Config
from data.adapters import DataAdapter, MarketDataProvider
from features.feature_engineering import FeatureEngineering
from regime.regime_detector import RegimeManager, SimpleRulesRegimeDetector, MLRegimeDetector
from alpha.alpha_models import AlphaEngine, MomentumAlpha, MeanReversionAlpha, BreakoutAlpha
from signals.signal_processor import SignalProcessor, SignalValidator, SignalScorer, SignalFilter
from risk.risk_engine import RiskEngine
from portfolio.portfolio_engine import PortfolioBuilder, PortfolioRebalancer
from execution.execution_engine import ExecutionEngine, SimulatedExecutor
from monitoring.metrics import PerformanceTracker, Logger
from utils.types import PortfolioState, Position, SignalDirection


class Orchestrator:
    """
    Main trading system orchestrator.
    Implements the full trading pipeline:
    Data -> Features -> Regime -> Alphas -> Signals -> Risk -> Portfolio -> Execution
    """
    
    def __init__(
        self,
        config: Config,
        data_adapter: DataAdapter,
    ):
        """
        Initialize orchestrator with full stack of components.
        
        Args:
            config: Configuration object
            data_adapter: Data source adapter
        """
        self.config = config
        self.logger = Logger(config.monitoring.log_path)
        
        # Data layer
        self.data_provider = MarketDataProvider(data_adapter)
        
        # Feature engineering
        self.feature_engine = FeatureEngineering(
            config.features.lookback_periods
        )
        
        # Regime detection
        if config.regime.detector_type == "ml":
            regime_detector = MLRegimeDetector(config.regime.ml_model_path)
        else:
            regime_detector = SimpleRulesRegimeDetector(
                trend_threshold=config.regime.trend_threshold,
                volatility_threshold=config.regime.volatility_threshold,
            )
        self.regime_manager = RegimeManager(regime_detector)
        
        # Alpha models
        self.alpha_engine = AlphaEngine()
        self._initialize_alphas(config.alpha.enabled_models)
        
        # Signal processing
        signal_validator = SignalValidator(
            min_signal_strength=0.3,
            require_confirmation=False,
        )
        signal_scorer = SignalScorer(base_confidence=0.5)
        signal_filter = SignalFilter(
            min_confidence=config.signals.min_confidence,
            min_edge=config.signals.min_edge,
        )
        self.signal_processor = SignalProcessor(
            validator=signal_validator,
            scorer=signal_scorer,
            filter=signal_filter,
        )
        
        # Risk management
        self.risk_engine = RiskEngine(
            max_position_risk_pct=config.risk.max_position_risk_pct,
            max_daily_drawdown_pct=config.risk.max_daily_drawdown_pct,
            max_weekly_drawdown_pct=config.risk.max_weekly_drawdown_pct,
            max_concurrent_positions=config.risk.max_concurrent_positions,
            kill_switch_enabled=config.risk.kill_switch_enabled,
            kill_switch_drawdown_pct=config.risk.kill_switch_drawdown_pct,
        )
        
        # Portfolio construction
        self.portfolio_builder = PortfolioBuilder(
            method=config.portfolio.diversification_method,
            max_position_pct=config.portfolio.max_single_position_pct,
            max_sector_exposure_pct=config.portfolio.max_sector_exposure_pct,
            use_volatility_targeting=True,
            target_volatility=config.portfolio.target_volatility,
        )
        
        self.portfolio_rebalancer = PortfolioRebalancer(rebalance_threshold=0.05)
        
        # Execution
        executor = SimulatedExecutor(
            slippage_bps=config.execution.slippage_bps,
            spread_bps=config.execution.spread_bps,
        )
        self.execution_engine = ExecutionEngine(
            adapter=executor,
            max_retries=config.execution.max_retries,
            retry_backoff_seconds=config.execution.retry_backoff_seconds,
        )
        
        # Monitoring
        self.tracker = PerformanceTracker()
        
        # State
        self.portfolio_state: Optional[PortfolioState] = None
        self.positions: Dict[str, Position] = {}
        self.is_running = False
    
    def _initialize_alphas(self, enabled_models: List[str]):
        """Initialize configured alpha models."""
        alpha_map = {
            "momentum": MomentumAlpha(),
            "mean_reversion": MeanReversionAlpha(),
            "breakout": BreakoutAlpha(),
        }
        
        for model_name in enabled_models:
            if model_name in alpha_map:
                self.alpha_engine.add_alpha(alpha_map[model_name])
                self.logger.log_info(f"Initialized alpha: {model_name}")
    
    def run_iteration(
        self,
        timestamp: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
    ):
        """
        Run one iteration of the trading pipeline.
        
        Args:
            timestamp: Current timestamp (defaults to now)
            symbols: List of symbols to trade (defaults to config)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbols is None:
            symbols = self.config.data.symbols
        
        self.logger.log_info(f"=== Iteration {timestamp} ===")
        
        try:
            # Step 1: Get market data
            self.logger.log_info("Step 1: Fetching market data...")
            market_states = {}
            for symbol in symbols:
                try:
                    market_states[symbol] = self.data_provider.get_market_state(
                        symbol, timestamp, lookback_bars=200
                    )
                except Exception as e:
                    self.logger.log_error(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if not market_states:
                self.logger.log_warning("No market data available")
                return
            
            # Step 2: Compute features and detect regime
            self.logger.log_info("Step 2: Computing features and detecting regime...")
            feature_data = {}
            regime_state = None
            
            for symbol, market_state in market_states.items():
                if not market_state.bars:
                    continue
                
                features = self.feature_engine.compute_features(
                    market_state.bars,
                    self.config.features.enabled_indicators
                )
                feature_data[symbol] = features
                
                # Detect regime (once per iteration)
                if regime_state is None:
                    regime_state = self.regime_manager.get_regime(features)
            
            if regime_state is None:
                self.logger.log_warning("Could not detect regime")
                return
            
            self.logger.log_info(f"Detected regime: {regime_state.regime.value}")
            
            # Step 3: Generate signals from alphas
            self.logger.log_info("Step 3: Generating alpha signals...")
            all_signals = []
            
            for symbol, features in feature_data.items():
                if symbol not in market_states:
                    continue
                
                market_state = market_states[symbol]
                signals = self.alpha_engine.generate_all_signals(
                    market_state, features, regime_state
                )
                all_signals.extend(signals)
                
                if signals:
                    self.logger.log_info(
                        f"Generated {len(signals)} signals for {symbol}"
                    )
            
            if not all_signals:
                self.logger.log_info("No signals generated")
                return
            
            # Step 4: Score and filter signals
            self.logger.log_info("Step 4: Scoring and filtering signals...")
            scored_signals = self.signal_processor.process(all_signals)
            
            self.logger.log_info(f"Approved {len(scored_signals)} signals")
            
            if not scored_signals:
                return
            
            # Step 5: Apply risk management
            self.logger.log_info("Step 5: Applying risk management...")

            if self.portfolio_state is None:
                starting_cash = 100000.0
                self.portfolio_state = PortfolioState(
                    timestamp=timestamp,
                    cash=starting_cash,
                    equity=starting_cash,
                    total_value=starting_cash,
                    positions=self.positions,
                )

            approved_orders = []

            for signal in scored_signals:
                if signal.symbol not in market_states:
                    continue

                current_price = market_states[signal.symbol].current_price
                atr = feature_data[signal.symbol].atr if signal.symbol in feature_data else 0.0

                if atr <= 0:
                    volatility = market_states[signal.symbol].volatility
                    atr = (volatility * current_price) / (252 ** 0.5) if volatility > 0 else 0.0

                if atr <= 0:
                    self.logger.log_warning(
                        f"Skipping {signal.symbol}: unable to compute ATR/volatility for sizing"
                    )
                    continue

                assessment = self.risk_engine.assess_trade(
                    signal,
                    current_price,
                    atr,
                    self.portfolio_state,
                    list(self.positions.values())
                )

                if assessment.action.value == "ACCEPT":
                    self.logger.log_info(f"Trade approved: {signal.symbol} {assessment.reasoning}")
                    
                    # Create order
                    from utils.types import Order, OrderType
                    order = Order(
                        symbol=signal.symbol,
                        quantity=assessment.approved_quantity,
                        order_type=OrderType.MARKET,
                        direction=signal.direction,
                        limit_price=current_price,
                        timestamp=timestamp,
                    )
                    approved_orders.append(order)
                else:
                    self.logger.log_info(f"Trade rejected: {signal.symbol} {assessment.reasoning}")
            
            # Step 6: Portfolio construction (optional rebalancing)
            self.logger.log_info("Step 6: Constructing portfolio...")
            
            if approved_orders:
                # Execute orders
                order_ids = self.execution_engine.execute_orders(approved_orders)
                self.logger.log_info(f"Executed {len(order_ids)} orders")

                filled_orders = self.execution_engine.get_filled_orders()
                if filled_orders:
                    self._update_portfolio_with_fills(filled_orders, market_states, timestamp)

            # Step 7: Record metrics
            self.logger.log_info("Step 7: Recording metrics...")
            self.tracker.record_equity(timestamp, self.portfolio_state.total_value)
            
            self.logger.log_info("=== Iteration Complete ===\n")
        
        except Exception as e:
            self.logger.log_error(f"Error in iteration: {e}")
            import traceback
            traceback.print_exc()
            self.stop()

    def _update_portfolio_with_fills(self, filled_orders, market_states, timestamp):
        """Update cash, positions, and portfolio value based on filled orders."""
        for order in filled_orders:
            fill_value = order.avg_fill_price * order.filled_quantity
            quantity = order.filled_quantity if order.direction == SignalDirection.LONG else -order.filled_quantity

            if order.direction == SignalDirection.LONG:
                self.portfolio_state.cash -= fill_value
            else:
                self.portfolio_state.cash += fill_value

            existing_position = self.positions.get(order.symbol)
            market_price = market_states.get(order.symbol).current_price if order.symbol in market_states else order.avg_fill_price

            if existing_position:
                new_quantity = existing_position.quantity + quantity
                if new_quantity == 0:
                    del self.positions[order.symbol]
                else:
                    weighted_entry = (
                        existing_position.entry_price * existing_position.quantity +
                        order.avg_fill_price * quantity
                    ) / new_quantity
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=new_quantity,
                        entry_price=weighted_entry,
                        current_price=market_price,
                        timestamp=timestamp,
                    )
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=quantity,
                    entry_price=order.avg_fill_price,
                    current_price=market_price,
                    timestamp=timestamp,
                )

        self._recalculate_portfolio_values(market_states)

    def _recalculate_portfolio_values(self, market_states):
        """Recompute equity and total value from current positions and cash."""
        equity_value = self.portfolio_state.cash
        for symbol, position in self.positions.items():
            market_price = market_states.get(symbol).current_price if symbol in market_states else position.current_price
            position.current_price = market_price
            position._update_pnl()
            equity_value += position.quantity * market_price

        self.portfolio_state.equity = equity_value
        self.portfolio_state.total_value = equity_value
    
    def run_continuous(
        self,
        run_frequency: str = "daily",
        symbols: Optional[List[str]] = None,
    ):
        """
        Run orchestrator continuously.
        
        Args:
            run_frequency: "minute", "hourly", "daily"
            symbols: List of symbols to trade
        """
        self.is_running = True
        self.logger.log_info(f"Starting orchestrator (mode: {self.config.orchestrator.operation_mode})")
        
        # Calculate sleep interval
        if run_frequency == "minute":
            sleep_interval = 60
        elif run_frequency == "hourly":
            sleep_interval = 3600
        else:  # daily
            sleep_interval = 86400
        
        try:
            while self.is_running:
                self.run_iteration(symbols=symbols)
                
                if self.is_running:
                    time.sleep(sleep_interval)
        
        except KeyboardInterrupt:
            self.logger.log_info("Orchestrator stopped by user")
            self.stop()
    
    def stop(self):
        """Stop the orchestrator."""
        self.is_running = False
        self.logger.log_info("Orchestrator stopped")
    
    def get_metrics_summary(self) -> Dict:
        """Get current performance metrics."""
        metrics = self.tracker.calculate_current_metrics()
        
        return {
            "total_return_pct": metrics.total_return_pct,
            "annual_return_pct": metrics.annual_return_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "win_rate": metrics.win_rate,
            "total_trades": metrics.total_trades,
        }

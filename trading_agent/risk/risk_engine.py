"""
Risk management engine.
Enforces position sizing, drawdown limits, and portfolio risk constraints.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from utils.types import (
    ScoredSignal, Position, PortfolioState, Order, OrderType, SignalDirection
)


class RiskAction(Enum):
    """Action recommended by risk engine."""
    ACCEPT = "ACCEPT"
    REDUCE_SIZE = "REDUCE_SIZE"
    REJECT = "REJECT"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class RiskAssessment:
    """Risk assessment for a potential trade."""
    action: RiskAction
    approved_quantity: float  # 0 if rejected
    reasoning: str


class PositionSizer:
    """
    Calculates position size based on risk parameters.
    """
    
    def __init__(
        self,
        risk_pct_per_trade: float = 1.0,
        atr_stop_multiple: float = 2.0,
    ):
        """
        Initialize position sizer.
        
        Args:
            risk_pct_per_trade: % of equity at risk per trade (0-100)
            atr_stop_multiple: Stop loss distance in ATR multiples
        """
        self.risk_pct_per_trade = risk_pct_per_trade / 100.0
        self.atr_stop_multiple = atr_stop_multiple
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio_equity: float,
    ) -> float:
        """
        Calculate position size given entry, stop loss, and account equity.
        
        Args:
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            portfolio_equity: Current equity in portfolio
        
        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0
        
        # Calculate max risk in dollars
        max_risk_dollars = portfolio_equity * self.risk_pct_per_trade
        
        # Calculate position size
        position_size = max_risk_dollars / risk_per_share
        
        return max(0, position_size)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: SignalDirection
    ) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: LONG or SHORT
        
        Returns:
            Stop loss price
        """
        stop_distance = atr * self.atr_stop_multiple
        
        if direction == SignalDirection.LONG:
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance


class RiskEngine:
    """
    Risk management engine.
    Validates trades against risk constraints.
    """
    
    def __init__(
        self,
        position_sizer: Optional[PositionSizer] = None,
        max_position_risk_pct: float = 1.0,
        max_daily_drawdown_pct: float = 5.0,
        max_weekly_drawdown_pct: float = 10.0,
        max_concurrent_positions: int = 10,
        kill_switch_enabled: bool = True,
        kill_switch_drawdown_pct: float = 20.0,
    ):
        """
        Initialize risk engine.
        
        Args:
            position_sizer: PositionSizer instance
            max_position_risk_pct: Max % equity at risk per trade
            max_daily_drawdown_pct: Max daily drawdown %
            max_weekly_drawdown_pct: Max weekly drawdown %
            max_concurrent_positions: Max number of open positions
            kill_switch_enabled: Enable automated kill-switch on severe drawdown
            kill_switch_drawdown_pct: Drawdown % to trigger kill-switch
        """
        self.position_sizer = position_sizer or PositionSizer(
            risk_pct_per_trade=max_position_risk_pct,
            atr_stop_multiple=2.0
        )
        self.max_position_risk_pct = max_position_risk_pct / 100.0
        self.max_daily_drawdown_pct = max_daily_drawdown_pct / 100.0
        self.max_weekly_drawdown_pct = max_weekly_drawdown_pct / 100.0
        self.max_concurrent_positions = max_concurrent_positions
        self.kill_switch_enabled = kill_switch_enabled
        self.kill_switch_drawdown_pct = kill_switch_drawdown_pct / 100.0
        
        self.peak_equity = None
        self.daily_start_equity = None
        self.weekly_start_equity = None
    
    def assess_trade(
        self,
        signal: ScoredSignal,
        entry_price: float,
        atr: float,
        portfolio_state: PortfolioState,
        current_positions: List[Position],
    ) -> RiskAssessment:
        """
        Assess a potential trade for risk.
        
        Args:
            signal: Scored signal
            entry_price: Entry price
            atr: Average True Range for position sizing
            portfolio_state: Current portfolio state
            current_positions: List of open positions
        
        Returns:
            RiskAssessment with action and approved size
        """
        # Check kill-switch first
        if self._check_kill_switch(portfolio_state):
            return RiskAssessment(
                action=RiskAction.KILL_SWITCH,
                approved_quantity=0,
                reasoning="Kill-switch activated: excessive drawdown"
            )
        
        # Check daily drawdown
        if self._check_daily_drawdown(portfolio_state):
            return RiskAssessment(
                action=RiskAction.REJECT,
                approved_quantity=0,
                reasoning="Daily drawdown limit exceeded"
            )
        
        # Check weekly drawdown
        if self._check_weekly_drawdown(portfolio_state):
            return RiskAssessment(
                action=RiskAction.REJECT,
                approved_quantity=0,
                reasoning="Weekly drawdown limit exceeded"
            )
        
        # Check concurrent positions
        if len(current_positions) >= self.max_concurrent_positions:
            return RiskAssessment(
                action=RiskAction.REJECT,
                approved_quantity=0,
                reasoning=f"Max concurrent positions ({self.max_concurrent_positions}) reached"
            )
        
        # Calculate position size
        stop_loss = self.position_sizer.calculate_stop_loss(
            entry_price,
            atr,
            signal.direction
        )
        
        position_size = self.position_sizer.calculate_position_size(
            entry_price,
            stop_loss,
            portfolio_state.total_value
        )
        
        if position_size <= 0:
            return RiskAssessment(
                action=RiskAction.REJECT,
                approved_quantity=0,
                reasoning="Position size calculation resulted in 0 shares"
            )
        
        return RiskAssessment(
            action=RiskAction.ACCEPT,
            approved_quantity=position_size,
            reasoning=f"Trade approved: {position_size:.0f} shares, stop @{stop_loss:.2f}"
        )
    
    def _check_kill_switch(self, portfolio_state: PortfolioState) -> bool:
        """Check if kill-switch should be triggered."""
        if not self.kill_switch_enabled:
            return False
        
        if self.peak_equity is None:
            self.peak_equity = portfolio_state.total_value
        
        # Update peak equity
        if portfolio_state.total_value > self.peak_equity:
            self.peak_equity = portfolio_state.total_value
        
        # Check drawdown
        current_drawdown = (self.peak_equity - portfolio_state.total_value) / self.peak_equity
        
        return current_drawdown >= self.kill_switch_drawdown_pct
    
    def _check_daily_drawdown(self, portfolio_state: PortfolioState) -> bool:
        """Check if daily drawdown limit exceeded."""
        if self.daily_start_equity is None:
            self.daily_start_equity = portfolio_state.total_value
            return False
        
        current_drawdown = (self.daily_start_equity - portfolio_state.total_value) / self.daily_start_equity
        
        return current_drawdown >= self.max_daily_drawdown_pct
    
    def _check_weekly_drawdown(self, portfolio_state: PortfolioState) -> bool:
        """Check if weekly drawdown limit exceeded."""
        if self.weekly_start_equity is None:
            self.weekly_start_equity = portfolio_state.total_value
            return False
        
        current_drawdown = (self.weekly_start_equity - portfolio_state.total_value) / self.weekly_start_equity
        
        return current_drawdown >= self.max_weekly_drawdown_pct
    
    def reset_daily_counter(self, portfolio_state: PortfolioState):
        """Reset daily drawdown counter (call at market open)."""
        self.daily_start_equity = portfolio_state.total_value
    
    def reset_weekly_counter(self, portfolio_state: PortfolioState):
        """Reset weekly drawdown counter (call at week start)."""
        self.weekly_start_equity = portfolio_state.total_value
    
    def create_exit_order(
        self,
        position: Position,
        stop_loss_price: float,
    ) -> Order:
        """
        Create a stop-loss exit order for a position.
        
        Args:
            position: Position to exit
            stop_loss_price: Stop loss price
        
        Returns:
            Order object
        """
        # Determine exit direction (opposite of position direction)
        exit_direction = SignalDirection.SHORT if position.quantity > 0 else SignalDirection.LONG
        
        order = Order(
            symbol=position.symbol,
            quantity=abs(position.quantity),
            order_type=OrderType.STOP,
            direction=exit_direction,
            stop_price=stop_loss_price,
        )
        
        return order

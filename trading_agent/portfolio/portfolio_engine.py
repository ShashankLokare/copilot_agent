"""
Portfolio construction and optimization.
Builds optimal portfolio from risk-approved signals.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from utils.types import (
    ScoredSignal, Position, PortfolioState, Order, OrderType, SignalDirection
)


@dataclass
class PortfolioWeights:
    """Target weights for portfolio."""
    weights: Dict[str, float]  # symbol -> weight (0-1)
    cash_weight: float = 0.0
    
    def __post_init__(self):
        """Ensure weights sum to 1."""
        total = sum(self.weights.values()) + self.cash_weight
        if abs(total - 1.0) > 0.01:
            # Normalize
            if total > 0:
                scale = 1.0 / total
                self.weights = {k: v * scale for k, v in self.weights.items()}
                self.cash_weight *= scale


class PortfolioBuilder:
    """
    Constructs portfolio from approved signals.
    """
    
    def __init__(
        self,
        method: str = "equal_weight",
        max_position_pct: float = 10.0,
        max_sector_exposure_pct: float = 30.0,
        use_volatility_targeting: bool = False,
        target_volatility: float = 0.15,
    ):
        """
        Initialize portfolio builder.
        
        Args:
            method: Construction method ("equal_weight", "volatility_target", "risk_parity")
            max_position_pct: Max % per individual position
            max_sector_exposure_pct: Max % per sector
            use_volatility_targeting: Whether to target portfolio volatility
            target_volatility: Target portfolio volatility (annual)
        """
        self.method = method
        self.max_position_pct = max_position_pct / 100.0
        self.max_sector_exposure_pct = max_sector_exposure_pct / 100.0
        self.use_volatility_targeting = use_volatility_targeting
        self.target_volatility = target_volatility
    
    def build_weights(
        self,
        approved_signals: List[ScoredSignal],
        current_positions: Dict[str, Position],
        portfolio_value: float,
    ) -> PortfolioWeights:
        """
        Build target portfolio weights from approved signals.
        
        Args:
            approved_signals: Risk-approved signals
            current_positions: Current open positions
            portfolio_value: Total portfolio value
        
        Returns:
            PortfolioWeights object
        """
        if not approved_signals:
            # No new signals - maintain current positions
            weights = {}
            for symbol, position in current_positions.items():
                weight = (position.quantity * position.current_price) / portfolio_value
                if weight != 0:
                    weights[symbol] = weight
            return PortfolioWeights(weights=weights)
        
        if self.method == "equal_weight":
            return self._build_equal_weights(approved_signals, current_positions)
        elif self.method == "volatility_target":
            return self._build_volatility_target_weights(approved_signals)
        elif self.method == "risk_parity":
            return self._build_risk_parity_weights(approved_signals)
        else:
            raise ValueError(f"Unknown construction method: {self.method}")
    
    def _build_equal_weights(
        self,
        signals: List[ScoredSignal],
        current_positions: Dict[str, Position]
    ) -> PortfolioWeights:
        """Equal weight across all signals."""
        n_signals = len(signals)
        if n_signals == 0:
            return PortfolioWeights(weights={})
        
        # Base weight per signal
        base_weight = 1.0 / n_signals
        
        # Apply position limit
        weight_per_signal = min(base_weight, self.max_position_pct)
        
        weights = {}
        for signal in signals:
            weights[signal.symbol] = weight_per_signal
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return PortfolioWeights(weights=weights)
    
    def _build_volatility_target_weights(
        self,
        signals: List[ScoredSignal]
    ) -> PortfolioWeights:
        """
        Build weights with volatility targeting.
        
        Higher volatility assets get lower weight.
        """
        weights = {}
        
        # First pass: assign weights inversely to signal volatility
        for signal in signals:
            # Use confidence as proxy for expected volatility
            # Lower confidence -> assume higher volatility
            volatility_proxy = 1.0 - signal.confidence
            inv_vol = 1.0 / max(0.1, volatility_proxy)
            weights[signal.symbol] = inv_vol
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Apply position limit
        weights = {k: min(v, self.max_position_pct) for k, v in weights.items()}
        
        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return PortfolioWeights(weights=weights)
    
    def _build_risk_parity_weights(
        self,
        signals: List[ScoredSignal]
    ) -> PortfolioWeights:
        """
        Build risk-parity weights.
        
        Each position contributes equally to portfolio risk.
        """
        weights = {}
        
        for signal in signals:
            # Risk parity: inverse confidence (higher confidence = lower risk = lower weight)
            # This is a simplified approach
            weights[signal.symbol] = 1.0 / len(signals)
        
        # Apply position limit
        weights = {k: min(v, self.max_position_pct) for k, v in weights.items()}
        
        # Re-normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return PortfolioWeights(weights=weights)


class PortfolioRebalancer:
    """
    Rebalances portfolio from current state to target weights.
    """
    
    def __init__(self, rebalance_threshold: float = 0.05):
        """
        Initialize rebalancer.
        
        Args:
            rebalance_threshold: % drift before rebalancing
        """
        self.rebalance_threshold = rebalance_threshold
    
    def generate_rebalance_orders(
        self,
        current_positions: Dict[str, Position],
        target_weights: PortfolioWeights,
        portfolio_value: float,
        current_price: float,
    ) -> List[Order]:
        """
        Generate orders to rebalance from current to target weights.
        
        Args:
            current_positions: Current open positions
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            current_price: Current prices per symbol (optional, use position.current_price)
        
        Returns:
            List of Order objects
        """
        orders = []
        
        # Build current weights
        current_weights = {}
        for symbol, position in current_positions.items():
            current_value = position.quantity * position.current_price
            current_weights[symbol] = current_value / portfolio_value
        
        # Check which positions need rebalancing
        all_symbols = set(current_weights.keys()) | set(target_weights.weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.weights.get(symbol, 0.0)
            
            weight_diff = abs(current_weight - target_weight)
            
            # Only rebalance if drift exceeds threshold
            if weight_diff < self.rebalance_threshold:
                continue
            
            target_value = target_weight * portfolio_value
            current_value = current_weight * portfolio_value
            value_diff = target_value - current_value
            
            if value_diff == 0:
                continue
            
            # Get current price
            if symbol in current_positions:
                price = current_positions[symbol].current_price
            else:
                continue  # Skip if no price available
            
            quantity = abs(value_diff) / price
            
            # Determine order direction
            if value_diff > 0:  # Need to buy
                direction = SignalDirection.LONG
            else:  # Need to sell
                direction = SignalDirection.SHORT
            
            order = Order(
                symbol=symbol,
                quantity=quantity,
                order_type=OrderType.MARKET,
                direction=direction,
            )
            orders.append(order)
        
        return orders

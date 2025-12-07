"""
Execution engine.
Handles order execution with simulated and live broker interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import random
import pytz

from utils.types import Order, OrderStatus, Trade


class ExecutionAdapter(ABC):
    """Abstract base class for execution adapters."""
    
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """
        Submit order for execution.
        
        Args:
            order: Order object
        
        Returns:
            Order ID
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID
        
        Returns:
            True if cancelled, False if not found
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        pass
    
    @abstractmethod
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        pass


class SimulatedExecutor(ExecutionAdapter):
    """
    Simulated execution for backtesting and paper trading.
    Models fill probability, slippage, and spread.
    """
    
    def __init__(
        self,
        slippage_bps: float = 2.0,
        spread_bps: float = 1.0,
        fill_probability: float = 0.95,
    ):
        """
        Initialize simulated executor.
        
        Args:
            slippage_bps: Slippage in basis points
            spread_bps: Bid-ask spread in basis points
            fill_probability: Probability of order fill (0-1)
        """
        self.slippage_bps = slippage_bps / 10000.0
        self.spread_bps = spread_bps / 10000.0
        self.fill_probability = fill_probability
        
        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.next_order_id = 1
    
    def submit_order(self, order: Order) -> str:
        """Submit order for simulated execution."""
        order_id = f"SIM_{self.next_order_id}"
        self.next_order_id += 1
        
        order.order_id = order_id
        order.status = OrderStatus.SUBMITTED
        self.orders[order_id] = order
        
        # Simulate fill
        self._process_fill(order_id)
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order if still pending."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id not in self.orders:
            return OrderStatus.REJECTED
        return self.orders[order_id].status
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return self.filled_orders
    
    def _process_fill(self, order_id: str):
        """Process simulated order fill."""
        order = self.orders[order_id]
        
        # Determine if order fills
        if random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            return
        
        # Calculate fill price with slippage and spread
        fill_price = self._calculate_fill_price(order)
        
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        
        self.filled_orders.append(order)
    
    def _calculate_fill_price(self, order: Order) -> float:
        """Calculate fill price including slippage and spread."""
        base_price = order.limit_price if order.limit_price else 100.0  # Default if not set
        
        # Add slippage
        slippage = base_price * self.slippage_bps
        
        # Add spread (depends on direction)
        spread = base_price * (self.spread_bps / 2)
        
        fill_price = base_price + slippage + spread
        
        return fill_price


class LiveBrokerAdapter(ExecutionAdapter):
    """
    Live broker adapter (placeholder).
    Implement specific broker integrations by subclassing.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize live broker adapter.
        
        Args:
            api_key: Broker API key
            api_secret: Broker API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
    
    def submit_order(self, order: Order) -> str:
        """
        Submit order to broker.
        
        This is a placeholder. Implement actual broker API calls.
        """
        raise NotImplementedError("Subclass and implement submit_order()")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order with broker."""
        raise NotImplementedError("Subclass and implement cancel_order()")
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from broker."""
        raise NotImplementedError("Subclass and implement get_order_status()")
    
    def get_filled_orders(self) -> List[Order]:
        """Get filled orders from broker."""
        return self.filled_orders


class ExecutionEngine:
    """
    Main execution engine.
    Routes orders through adapter and tracks fills.
    """
    
    def __init__(
        self,
        adapter: ExecutionAdapter,
        max_retries: int = 3,
        retry_backoff_seconds: int = 5,
    ):
        """
        Initialize execution engine.
        
        Args:
            adapter: ExecutionAdapter implementation
            max_retries: Max order submission retries
            retry_backoff_seconds: Seconds between retries
        """
        self.adapter = adapter
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        
        self.submitted_orders: Dict[str, Order] = {}
        self.completed_trades: List[Trade] = []
    
    def execute_order(self, order: Order) -> Optional[str]:
        """
        Execute an order with retry logic.
        
        Args:
            order: Order to execute
        
        Returns:
            Order ID if successful, None if failed
        """
        for attempt in range(self.max_retries):
            try:
                order_id = self.adapter.submit_order(order)
                self.submitted_orders[order_id] = order
                return order_id
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return None
                # Would sleep(self.retry_backoff_seconds) here in real implementation
        
        return None
    
    def execute_orders(self, orders: List[Order]) -> List[str]:
        """
        Execute multiple orders.
        
        Args:
            orders: List of orders
        
        Returns:
            List of successful order IDs
        """
        order_ids = []
        for order in orders:
            order_id = self.execute_order(order)
            if order_id:
                order_ids.append(order_id)
        
        return order_ids
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of a submitted order."""
        return self.adapter.get_order_status(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a submitted order."""
        return self.adapter.cancel_order(order_id)
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        return self.adapter.get_filled_orders()
    
    def process_fills(self):
        """
        Process filled orders into trades.
        Should be called periodically to match entry/exit fills.
        """
        filled = self.adapter.get_filled_orders()
        
        # Simple approach: match fills chronologically
        # In production, this would track entry and exit fills separately
        for order in filled:
            if order not in self.submitted_orders.values():
                # New fill
                trade = Trade(
                    symbol=order.symbol,
                    entry_price=order.avg_fill_price,
                    entry_timestamp=order.timestamp,
                    quantity=order.filled_quantity,
                )
                self.completed_trades.append(trade)
    
    def get_completed_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.completed_trades

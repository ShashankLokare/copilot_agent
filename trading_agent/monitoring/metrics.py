"""
Monitoring, metrics, and performance analysis.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from utils.types import Trade, PortfolioState


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading system."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    profit_factor: float = 0.0  # Avg win / Avg loss
    payoff_ratio: float = 0.0  # Avg win / Avg loss magnitude
    
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    expectancy: float = 0.0  # Expected return per trade
    
    # Per-symbol metrics
    symbol_metrics: Dict[str, "PerformanceMetrics"] = None


class MetricsCalculator:
    """
    Calculates performance metrics from trades and portfolio data.
    """
    
    @staticmethod
    def calculate_metrics(
        trades: List[Trade],
        equity_curve: List[float],
        timestamps: Optional[List[datetime]] = None,
        annual_risk_free_rate: float = 0.02,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of completed trades
            equity_curve: List of equity values over time
            timestamps: Corresponding timestamps for equity curve
            annual_risk_free_rate: Risk-free rate for Sharpe/Sortino
        
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if not equity_curve:
            return metrics
        
        # Basic trade metrics (can be empty)
        metrics.total_trades = len(trades)
        
        pnls = []
        returns_pct = []
        
        for trade in trades:
            if trade.pnl > 0:
                metrics.winning_trades += 1
                pnls.append(trade.pnl)
                returns_pct.append(trade.pnl_pct)
            elif trade.pnl < 0:
                metrics.losing_trades += 1
                pnls.append(trade.pnl)
                returns_pct.append(trade.pnl_pct)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # Return metrics
        if equity_curve:
            initial_equity = equity_curve[0]
            final_equity = equity_curve[-1]
            metrics.total_return_pct = (final_equity - initial_equity) / initial_equity * 100
            
            # Annualize if we have timestamps
            if timestamps and len(timestamps) > 1:
                days_elapsed = (timestamps[-1] - timestamps[0]).days
                if days_elapsed > 0:
                    metrics.annual_return_pct = metrics.total_return_pct * (365 / days_elapsed)
        
        # Drawdown metrics
        max_equity = equity_curve[0]
        current_drawdown = 0
        max_drawdown = 0
        drawdown_start = None
        drawdown_duration = 0
        max_drawdown_duration = 0
        
        for i, equity in enumerate(equity_curve):
            if equity > max_equity:
                max_equity = equity
            
            current_drawdown = (max_equity - equity) / max_equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                if drawdown_start is not None and timestamps:
                    current_duration = (timestamps[i] - drawdown_start).days
                    if current_duration > max_drawdown_duration:
                        max_drawdown_duration = current_duration
                drawdown_start = timestamps[i] if timestamps else None
        
        metrics.max_drawdown_pct = max_drawdown * 100
        metrics.max_drawdown_duration_days = max_drawdown_duration
        
        # Risk-adjusted metrics
        if returns_pct:
            returns = np.array(returns_pct)
            returns_annual = returns * 252  # Assuming daily returns
            
            sharpe = MetricsCalculator._calculate_sharpe(
                returns_annual,
                annual_risk_free_rate
            )
            metrics.sharpe_ratio = sharpe
            
            sortino = MetricsCalculator._calculate_sortino(
                returns_annual,
                annual_risk_free_rate
            )
            metrics.sortino_ratio = sortino
            
            # Profit factor
            if metrics.losing_trades > 0:
                total_wins = sum([p for p in pnls if p > 0])
                total_losses = abs(sum([p for p in pnls if p < 0]))
                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses
            
            # Average metrics
            winning_pnls = [p for p in returns_pct if p > 0]
            losing_pnls = [p for p in returns_pct if p < 0]
            
            if winning_pnls:
                metrics.avg_win_pct = np.mean(winning_pnls) * 100
            if losing_pnls:
                metrics.avg_loss_pct = np.mean(losing_pnls) * 100
            
            # Expectancy
            metrics.expectancy = np.mean(returns_pct) if returns_pct else 0
        
        return metrics
    
    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(returns) > 0 else 0.0
    
    @staticmethod
    def _calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = np.minimum(returns, risk_free_rate)
        downside_dev = np.std(downside_returns)
        
        return float(np.mean(excess_returns) / downside_dev) if downside_dev > 0 else 0.0


class PerformanceTracker:
    """
    Tracks system performance over time.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.equity_history: List[float] = []
        self.timestamp_history: List[datetime] = []
        self.trades: List[Trade] = []
        self.metrics_history: Dict[datetime, PerformanceMetrics] = {}
    
    def record_equity(self, timestamp: datetime, equity: float):
        """Record equity value at a point in time."""
        self.timestamp_history.append(timestamp)
        self.equity_history.append(equity)
    
    def record_trade(self, trade: Trade):
        """Record a completed trade."""
        self.trades.append(trade)
    
    def calculate_current_metrics(self) -> PerformanceMetrics:
        """Calculate metrics for current state."""
        return MetricsCalculator.calculate_metrics(
            self.trades,
            self.equity_history,
            self.timestamp_history
        )
    
    def save_metrics_snapshot(self, timestamp: datetime):
        """Save metrics snapshot at a point in time."""
        metrics = self.calculate_current_metrics()
        self.metrics_history[timestamp] = metrics
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamp_history,
            'equity': self.equity_history
        })
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'symbol': trade.symbol,
                'entry_timestamp': trade.entry_timestamp,
                'entry_price': trade.entry_price,
                'exit_timestamp': trade.exit_timestamp,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'alpha_name': trade.alpha_name,
            })
        
        return pd.DataFrame(trade_data)


class Logger:
    """
    Simple logging utility.
    """
    
    def __init__(self, log_path: str = "logs/"):
        """Initialize logger."""
        import os
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        # Optionally write to file
        # with open(f"{self.log_path}/trading.log", "a") as f:
        #     f.write(log_message + "\n")
    
    def log_info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def log_error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")

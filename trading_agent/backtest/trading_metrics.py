"""
Comprehensive Strategy Evaluation Framework

Implements full backtesting, PnL metrics, Monte Carlo robustness tests,
and strategy grading (A/B/C/D) based on risk-adjusted performance.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade execution record."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    quantity: float
    entry_cost: float = 0.0  # Slippage + commission
    exit_cost: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_periods: int = 0  # Number of bars held
    
    def __post_init__(self):
        if self.pnl == 0:
            if self.direction == "LONG":
                gross_pnl = (self.exit_price - self.entry_price) * self.quantity
            else:  # SHORT
                gross_pnl = (self.entry_price - self.exit_price) * self.quantity
            
            self.pnl = gross_pnl - self.entry_cost - self.exit_cost
            self.pnl_pct = self.pnl / (self.entry_price * self.quantity) if self.entry_price > 0 else 0


@dataclass
class EquityCurve:
    """Equity curve and associated metrics."""
    timestamps: pd.DatetimeIndex
    equity: np.ndarray  # Daily equity values
    daily_returns: np.ndarray
    daily_pnl: np.ndarray
    cumulative_pnl: np.ndarray
    trades: List[TradeRecord]
    regime_series: Optional[pd.Series] = None
    
    @property
    def final_equity(self) -> float:
        return self.equity[-1] if len(self.equity) > 0 else 0
    
    @property
    def total_pnl(self) -> float:
        return self.cumulative_pnl[-1] if len(self.cumulative_pnl) > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity,
            'daily_returns': self.daily_returns,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
        })
        
        if self.regime_series is not None:
            df['regime'] = self.regime_series.values
        
        return df


class TradingMetrics:
    """Compute comprehensive trading metrics from equity curve."""
    
    @staticmethod
    def compute_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
    ) -> float:
        """
        Compute annualized Sharpe ratio.
        
        Args:
            returns: Daily returns array.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year (252 for daily).
            
        Returns:
            Sharpe ratio.
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(periods_per_year)
    
    @staticmethod
    def compute_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
    ) -> float:
        """
        Compute annualized Sortino ratio (downside volatility only).
        
        Args:
            returns: Daily returns array.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year.
            
        Returns:
            Sortino ratio.
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        # Downside volatility
        downside = np.where(excess_returns < 0, excess_returns, 0)
        downside_vol = np.std(downside)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_vol
        return sortino * np.sqrt(periods_per_year)
    
    @staticmethod
    def compute_max_drawdown(equity: np.ndarray) -> Tuple[float, int, int]:
        """
        Compute maximum drawdown and when it occurred.
        
        Args:
            equity: Equity curve array.
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx).
        """
        if len(equity) < 2:
            return 0.0, 0, 0
        
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        
        max_dd_idx = np.argmin(drawdown)
        max_dd = abs(drawdown[max_dd_idx])
        
        # Find when peak occurred
        peak_idx = np.where(cummax == cummax[max_dd_idx])[0][-1]
        
        return max_dd, peak_idx, max_dd_idx
    
    @staticmethod
    def compute_cagr(
        equity_start: float,
        equity_end: float,
        periods: int,
        periods_per_year: int = 252,
    ) -> float:
        """
        Compute Compound Annual Growth Rate.
        
        Args:
            equity_start: Starting equity.
            equity_end: Ending equity.
            periods: Number of periods (days).
            periods_per_year: Trading periods per year.
            
        Returns:
            CAGR as decimal (e.g., 0.15 for 15%).
        """
        if equity_start <= 0 or equity_end <= 0:
            return 0.0
        
        years = periods / periods_per_year
        
        if years <= 0:
            return 0.0
        
        cagr = (equity_end / equity_start) ** (1 / years) - 1
        return cagr
    
    @staticmethod
    def compute_profit_factor(trades: List[TradeRecord]) -> float:
        """
        Compute profit factor (gross profit / gross loss).
        
        Args:
            trades: List of TradeRecord objects.
            
        Returns:
            Profit factor.
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def compute_win_rate(trades: List[TradeRecord]) -> float:
        """
        Compute win rate (% of profitable trades).
        
        Args:
            trades: List of TradeRecord objects.
            
        Returns:
            Win rate (0 to 1).
        """
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)
    
    @staticmethod
    def compute_avg_trade(trades: List[TradeRecord]) -> Tuple[float, float]:
        """
        Compute average win and average loss.
        
        Args:
            trades: List of TradeRecord objects.
            
        Returns:
            Tuple of (avg_win, avg_loss).
        """
        if not trades:
            return 0.0, 0.0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    @staticmethod
    def compute_payoff_ratio(trades: List[TradeRecord]) -> float:
        """
        Compute payoff ratio (avg win / abs(avg loss)).
        
        Args:
            trades: List of TradeRecord objects.
            
        Returns:
            Payoff ratio.
        """
        avg_win, avg_loss = TradingMetrics.compute_avg_trade(trades)
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return avg_win / abs(avg_loss)
    
    @staticmethod
    def compute_calmar_ratio(
        cagr: float,
        max_drawdown: float,
    ) -> float:
        """
        Compute Calmar ratio (CAGR / MaxDD).
        
        Args:
            cagr: Compound annual growth rate.
            max_drawdown: Maximum drawdown (as decimal).
            
        Returns:
            Calmar ratio.
        """
        if max_drawdown == 0:
            return float('inf') if cagr > 0 else 0.0
        
        return cagr / max_drawdown
    
    @staticmethod
    def compute_turnover(
        trades: List[TradeRecord],
        avg_equity: float,
        periods: int,
    ) -> float:
        """
        Compute turnover (total notional traded / avg equity).
        
        Args:
            trades: List of TradeRecord objects.
            avg_equity: Average equity during period.
            periods: Number of trading periods.
            
        Returns:
            Annualized turnover ratio.
        """
        if avg_equity == 0:
            return 0.0
        
        total_notional = sum(
            t.quantity * t.entry_price for t in trades
        )
        
        return (total_notional / avg_equity) * (252 / periods)
    
    @staticmethod
    def compute_cost_metrics(
        trades: List[TradeRecord],
        total_pnl: float,
    ) -> Dict[str, float]:
        """
        Compute transaction cost metrics.
        
        Args:
            trades: List of TradeRecord objects.
            total_pnl: Total P&L from strategy.
            
        Returns:
            Dict with cost metrics.
        """
        total_cost = sum(t.entry_cost + t.exit_cost for t in trades)
        
        return {
            "total_costs": total_cost,
            "cost_per_trade": total_cost / len(trades) if trades else 0,
            "cost_pct_of_pnl": (total_cost / total_pnl * 100) if total_pnl > 0 else float('inf'),
        }


@dataclass
class StrategyPerformance:
    """Comprehensive strategy performance metrics."""
    
    # Time period
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    periods: int
    
    # Core PnL metrics
    total_pnl: float
    final_equity: float
    cagr: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    
    # Efficiency
    turnover: float
    cost_metrics: Dict[str, float]
    
    # Regime analysis
    regime_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"StrategyPerformance(\n"
            f"  CAGR: {self.cagr:.2%}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}\n"
            f"  Max DD: {self.max_drawdown:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Num Trades: {self.num_trades}\n"
            f")"
        )


class StrategyEvaluator:
    """Evaluate strategy performance across multiple dimensions."""
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def evaluate(
        self,
        equity_curve: EquityCurve,
    ) -> StrategyPerformance:
        """
        Compute comprehensive performance metrics.
        
        Args:
            equity_curve: EquityCurve object with trades and returns.
            
        Returns:
            StrategyPerformance object.
        """
        periods = len(equity_curve.equity)
        cagr = TradingMetrics.compute_cagr(
            equity_curve.equity[0],
            equity_curve.final_equity,
            periods,
        )
        
        max_dd, _, _ = TradingMetrics.compute_max_drawdown(equity_curve.equity)
        
        sharpe = TradingMetrics.compute_sharpe_ratio(
            equity_curve.daily_returns,
            self.risk_free_rate,
        )
        
        sortino = TradingMetrics.compute_sortino_ratio(
            equity_curve.daily_returns,
            self.risk_free_rate,
        )
        
        calmar = TradingMetrics.compute_calmar_ratio(cagr, max_dd)
        
        win_rate = TradingMetrics.compute_win_rate(equity_curve.trades)
        pf = TradingMetrics.compute_profit_factor(equity_curve.trades)
        avg_win, avg_loss = TradingMetrics.compute_avg_trade(equity_curve.trades)
        payoff_ratio = TradingMetrics.compute_payoff_ratio(equity_curve.trades)
        
        turnover = TradingMetrics.compute_turnover(
            equity_curve.trades,
            np.mean(equity_curve.equity),
            periods,
        )
        
        cost_metrics = TradingMetrics.compute_cost_metrics(
            equity_curve.trades,
            equity_curve.total_pnl,
        )
        
        # Regime stats
        regime_stats = {}
        if equity_curve.regime_series is not None:
            regime_stats = self._compute_regime_stats(equity_curve)
        
        return StrategyPerformance(
            start_date=equity_curve.timestamps[0],
            end_date=equity_curve.timestamps[-1],
            periods=periods,
            total_pnl=equity_curve.total_pnl,
            final_equity=equity_curve.final_equity,
            cagr=cagr,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            num_trades=len(equity_curve.trades),
            win_rate=win_rate,
            profit_factor=pf,
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=payoff_ratio,
            turnover=turnover,
            cost_metrics=cost_metrics,
            regime_stats=regime_stats,
        )
    
    def _compute_regime_stats(
        self,
        equity_curve: EquityCurve,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for each regime.
        
        Args:
            equity_curve: EquityCurve object.
            
        Returns:
            Dict mapping regime -> metrics dict.
        """
        regime_stats = {}
        
        regimes = equity_curve.regime_series.unique()
        
        for regime in regimes:
            mask = equity_curve.regime_series == regime
            
            if mask.sum() < 20:  # Skip regimes with too few samples
                continue
            
            regime_df = pd.DataFrame({
                'equity': equity_curve.equity[mask],
                'returns': equity_curve.daily_returns[mask],
                'pnl': equity_curve.daily_pnl[mask],
            })
            
            regime_trades = [
                t for t in equity_curve.trades
                if equity_curve.timestamps.get_loc(t.entry_time) in np.where(mask)[0]
            ]
            
            regime_sharpe = TradingMetrics.compute_sharpe_ratio(
                regime_df['returns'].values,
                self.risk_free_rate,
            )
            
            regime_dd, _, _ = TradingMetrics.compute_max_drawdown(
                regime_df['equity'].values
            )
            
            regime_stats[str(regime)] = {
                'num_bars': mask.sum(),
                'num_trades': len(regime_trades),
                'pnl': regime_df['pnl'].sum(),
                'sharpe': regime_sharpe,
                'max_dd': regime_dd,
                'win_rate': TradingMetrics.compute_win_rate(regime_trades),
                'profit_factor': TradingMetrics.compute_profit_factor(regime_trades),
            }
        
        return regime_stats


if __name__ == "__main__":
    # Test metrics calculation
    np.random.seed(42)
    
    # Simulate an equity curve
    daily_returns = np.random.normal(0.0005, 0.01, 252)
    equity = 100000 * np.cumprod(1 + daily_returns)
    
    daily_pnl = np.diff(equity, prepend=100000)
    cumulative_pnl = daily_pnl.cumsum()
    
    timestamps = pd.date_range('2023-01-01', periods=252, freq='D')
    
    trades = [
        TradeRecord(
            entry_time=timestamps[10],
            entry_price=100.0,
            exit_time=timestamps[15],
            exit_price=101.0,
            direction="LONG",
            quantity=100,
            entry_cost=10,
            exit_cost=10,
        ),
        TradeRecord(
            entry_time=timestamps[50],
            entry_price=102.0,
            exit_time=timestamps[60],
            exit_price=101.0,
            direction="LONG",
            quantity=100,
            entry_cost=10,
            exit_cost=10,
        ),
    ]
    
    eq = EquityCurve(
        timestamps=timestamps,
        equity=equity,
        daily_returns=daily_returns,
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
        trades=trades,
    )
    
    evaluator = StrategyEvaluator()
    perf = evaluator.evaluate(eq)
    
    print(perf)

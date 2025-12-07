"""
Monte Carlo & Robustness Testing

Implements trade reshuffling, daily returns bootstrap, and parameter sensitivity tests
to assess strategy robustness and overfitting risk.
"""

import logging
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from backtest.trading_metrics import TradeRecord, TradingMetrics


logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    
    num_iterations: int
    metrics_distribution: Dict[str, np.ndarray]  # metric_name -> array of values
    
    # Summary statistics
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Percentiles
    percentiles: Dict[str, Dict[int, float]] = field(default_factory=dict)
    
    # Probability calculations
    prob_profitable: float = 0.0
    prob_positive_sharpe: float = 0.0
    
    def compute_summary(self) -> None:
        """Compute summary statistics from distributions."""
        for metric_name, values in self.metrics_distribution.items():
            self.summary_stats[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
            }
            
            # Percentiles
            self.percentiles[metric_name] = {
                5: np.percentile(values, 5),
                25: np.percentile(values, 25),
                50: np.percentile(values, 50),
                75: np.percentile(values, 75),
                95: np.percentile(values, 95),
            }


class MonteCarloTester:
    """Implement various Monte Carlo robustness tests."""
    
    @staticmethod
    def trade_shuffling_mc(
        trades: List[TradeRecord],
        num_iterations: int = 1000,
        metrics_to_compute: Optional[List[str]] = None,
    ) -> MonteCarloResult:
        """
        Shuffle trade order (preserving individual trade PnLs) and compute metrics.
        
        This tests whether the strategy's returns come from consistent strategy logic
        or from luck in the timing of large wins/losses.
        
        Args:
            trades: List of TradeRecord objects.
            num_iterations: Number of shuffles to perform.
            metrics_to_compute: List of metrics to compute ('sharpe', 'max_dd', 'final_pnl', etc).
            
        Returns:
            MonteCarloResult object with distributions.
        """
        if metrics_to_compute is None:
            metrics_to_compute = ['sharpe_ratio', 'max_drawdown', 'final_pnl', 'win_rate']
        
        if not trades:
            raise ValueError("No trades to shuffle")
        
        result = MonteCarloResult(
            num_iterations=num_iterations,
            metrics_distribution={m: [] for m in metrics_to_compute},
        )
        
        # Original trade PnLs
        trade_pnls = np.array([t.pnl for t in trades])
        
        for iteration in range(num_iterations):
            # Shuffle trade order
            shuffled_pnls = np.random.permutation(trade_pnls)
            
            # Build equity curve from shuffled trades
            cumulative_pnl = np.concatenate([[0], np.cumsum(shuffled_pnls)])
            equity = 100000 + cumulative_pnl
            
            # Compute metrics
            for metric in metrics_to_compute:
                if metric == 'sharpe_ratio':
                    # Use daily PnL (assume trades happen daily on average)
                    daily_pnl = shuffled_pnls / len(shuffled_pnls)
                    sharpe = TradingMetrics.compute_sharpe_ratio(daily_pnl)
                    result.metrics_distribution[metric].append(sharpe)
                
                elif metric == 'max_drawdown':
                    dd, _, _ = TradingMetrics.compute_max_drawdown(equity)
                    result.metrics_distribution[metric].append(dd)
                
                elif metric == 'final_pnl':
                    result.metrics_distribution[metric].append(equity[-1])
                
                elif metric == 'win_rate':
                    win_rate = np.sum(shuffled_pnls > 0) / len(shuffled_pnls)
                    result.metrics_distribution[metric].append(win_rate)
                
                elif metric == 'profit_factor':
                    gross_profit = np.sum(shuffled_pnls[shuffled_pnls > 0])
                    gross_loss = np.abs(np.sum(shuffled_pnls[shuffled_pnls < 0]))
                    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
                    result.metrics_distribution[metric].append(pf)
        
        # Convert lists to arrays
        for metric in metrics_to_compute:
            result.metrics_distribution[metric] = np.array(result.metrics_distribution[metric])
        
        # Compute summary stats
        result.compute_summary()
        
        # Compute probabilities
        if 'final_pnl' in result.metrics_distribution:
            result.prob_profitable = np.mean(
                result.metrics_distribution['final_pnl'] > 100000
            )
        
        if 'sharpe_ratio' in result.metrics_distribution:
            result.prob_positive_sharpe = np.mean(
                result.metrics_distribution['sharpe_ratio'] > 0
            )
        
        return result
    
    @staticmethod
    def daily_returns_bootstrap(
        daily_returns: np.ndarray,
        num_iterations: int = 1000,
        sample_size: Optional[int] = None,
        metrics_to_compute: Optional[List[str]] = None,
        risk_free_rate: float = 0.04,
    ) -> MonteCarloResult:
        """
        Bootstrap daily returns with replacement to generate synthetic equity curves.
        
        This tests robustness to different orderings of returns and can highlight
        period-specific luck (e.g., clustering of good days).
        
        Args:
            daily_returns: Array of daily returns.
            num_iterations: Number of bootstrap samples.
            sample_size: Size of each bootstrap sample (default: len(daily_returns)).
            metrics_to_compute: Metrics to compute on each bootstrap sample.
            risk_free_rate: Risk-free rate for Sharpe calculation.
            
        Returns:
            MonteCarloResult object.
        """
        if metrics_to_compute is None:
            metrics_to_compute = ['sharpe_ratio', 'max_drawdown', 'final_pnl', 'cagr']
        
        if sample_size is None:
            sample_size = len(daily_returns)
        
        result = MonteCarloResult(
            num_iterations=num_iterations,
            metrics_distribution={m: [] for m in metrics_to_compute},
        )
        
        initial_equity = 100000
        
        for iteration in range(num_iterations):
            # Bootstrap sample
            bootstrap_returns = np.random.choice(daily_returns, size=sample_size, replace=True)
            
            # Generate equity curve
            equity = initial_equity * np.cumprod(1 + bootstrap_returns)
            
            # Compute metrics
            for metric in metrics_to_compute:
                if metric == 'sharpe_ratio':
                    sharpe = TradingMetrics.compute_sharpe_ratio(
                        bootstrap_returns,
                        risk_free_rate,
                    )
                    result.metrics_distribution[metric].append(sharpe)
                
                elif metric == 'max_drawdown':
                    dd, _, _ = TradingMetrics.compute_max_drawdown(equity)
                    result.metrics_distribution[metric].append(dd)
                
                elif metric == 'final_pnl':
                    result.metrics_distribution[metric].append(equity[-1] - initial_equity)
                
                elif metric == 'cagr':
                    cagr = TradingMetrics.compute_cagr(
                        initial_equity,
                        equity[-1],
                        len(equity),
                    )
                    result.metrics_distribution[metric].append(cagr)
                
                elif metric == 'sortino_ratio':
                    sortino = TradingMetrics.compute_sortino_ratio(
                        bootstrap_returns,
                        risk_free_rate,
                    )
                    result.metrics_distribution[metric].append(sortino)
        
        # Convert to arrays
        for metric in metrics_to_compute:
            result.metrics_distribution[metric] = np.array(result.metrics_distribution[metric])
        
        result.compute_summary()
        
        if 'final_pnl' in result.metrics_distribution:
            result.prob_profitable = np.mean(result.metrics_distribution['final_pnl'] > 0)
        
        if 'sharpe_ratio' in result.metrics_distribution:
            result.prob_positive_sharpe = np.mean(
                result.metrics_distribution['sharpe_ratio'] > 0
            )
        
        return result


@dataclass
class ParameterRobustnessResult:
    """Results from parameter robustness testing."""
    
    base_params: Dict[str, float]
    parameter_results: Dict[str, Dict[str, float]]  # param_name -> {perturb_val -> metric}
    base_metric_value: float
    metric_name: str
    
    num_profitable_variations: int = 0
    num_total_variations: int = 0
    
    def __post_init__(self):
        self.num_total_variations = sum(
            len(v) for v in self.parameter_results.values()
        )
    
    @property
    def robustness_score(self) -> float:
        """Fraction of parameter variations that remain profitable."""
        if self.num_total_variations == 0:
            return 0.0
        return self.num_profitable_variations / self.num_total_variations


class ParameterRobustnessTester:
    """Test strategy robustness to parameter perturbations."""
    
    @staticmethod
    def test_parameter_robustness(
        backtest_fn: Callable[[Dict[str, float]], Dict[str, float]],
        base_params: Dict[str, float],
        parameter_perturbations: Dict[str, List[float]],
        metric_name: str = 'sharpe_ratio',
        min_metric_profitable: float = 0.5,
    ) -> ParameterRobustnessResult:
        """
        Test strategy performance across parameter variations.
        
        Args:
            backtest_fn: Function that takes params dict and returns metrics dict.
            base_params: Base parameters to perturb.
            parameter_perturbations: Dict of {param_name: [perturbation_values]}.
            metric_name: Metric to track (e.g., 'sharpe_ratio').
            min_metric_profitable: Minimum metric value to count as profitable.
            
        Returns:
            ParameterRobustnessResult object.
        """
        # Compute base metrics
        base_metrics = backtest_fn(base_params)
        base_metric_value = base_metrics.get(metric_name, 0.0)
        
        result = ParameterRobustnessResult(
            base_params=base_params,
            parameter_results={},
            base_metric_value=base_metric_value,
            metric_name=metric_name,
        )
        
        num_profitable = 0
        num_total = 0
        
        # Test each parameter with its perturbations
        for param_name, perturbations in parameter_perturbations.items():
            result.parameter_results[param_name] = {}
            
            for perturb_value in perturbations:
                # Create perturbed params
                test_params = base_params.copy()
                test_params[param_name] = perturb_value
                
                # Run backtest
                test_metrics = backtest_fn(test_params)
                test_metric_value = test_metrics.get(metric_name, 0.0)
                
                result.parameter_results[param_name][perturb_value] = test_metric_value
                
                # Count as profitable if above threshold
                if test_metric_value >= min_metric_profitable:
                    num_profitable += 1
                
                num_total += 1
        
        result.num_profitable_variations = num_profitable
        result.num_total_variations = num_total
        
        return result


if __name__ == "__main__":
    # Test trade shuffling MC
    trades = [
        TradeRecord(
            entry_time=pd.Timestamp('2023-01-01'),
            entry_price=100.0,
            exit_time=pd.Timestamp('2023-01-02'),
            exit_price=101.0,
            direction="LONG",
            quantity=100,
            entry_cost=10,
            exit_cost=10,
            pnl=80,
        ),
        TradeRecord(
            entry_time=pd.Timestamp('2023-01-03'),
            entry_price=101.0,
            exit_time=pd.Timestamp('2023-01-04'),
            exit_price=102.0,
            direction="LONG",
            quantity=100,
            entry_cost=10,
            exit_cost=10,
            pnl=80,
        ),
    ]
    
    mc_result = MonteCarloTester.trade_shuffling_mc(
        trades,
        num_iterations=100,
        metrics_to_compute=['sharpe_ratio', 'max_drawdown', 'final_pnl'],
    )
    
    print("Trade Shuffling MC Results:")
    print(f"Prob profitable: {mc_result.prob_profitable:.2%}")
    print(f"Summary: {mc_result.summary_stats}")

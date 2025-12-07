"""
Strategy Evaluation & Grading

Comprehensive P&L analysis and grading of full trading strategies.
Grades strategies A/B/C/D based on risk-adjusted returns, Sharpe, drawdown, and robustness.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Complete performance metrics for a strategy."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades_count: int
    annual_trades: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk


@dataclass
class StrategyGradeResult:
    """Complete grading result for trading strategy."""
    grade: str  # "A" | "B" | "C" | "D"
    primary_metrics: StrategyMetrics
    regime_breakdown: Dict[str, StrategyMetrics]
    cross_fold_stability: float  # Consistency across folds
    regime_consistency: float  # Worst / best regime Sharpe ratio
    out_of_sample_degradation: float  # In-sample vs out-of-sample difference
    monte_carlo_results: Dict[str, float]  # MC confidence intervals
    comments: str
    timestamp: str
    deployment_recommendation: str  # "LIVE", "PAPER", "INCUBATION", "BLOCK"


class StrategyGrader:
    """
    Grades trading strategies based on comprehensive risk and return analysis.
    """
    
    def __init__(self, config_path: str = "config/strategy_grading.yaml"):
        """
        Initialize grader with config.
        
        Args:
            config_path: Path to strategy grading configuration YAML.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compute_strategy_metrics(
        self,
        returns: np.ndarray,
        trades: pd.DataFrame,  # columns: entry_price, exit_price, direction, entry_date, exit_date
        annual_factor: float = 252,  # Trading days per year
    ) -> StrategyMetrics:
        """
        Compute comprehensive strategy metrics.
        
        Args:
            returns: Daily returns array (equity curve).
            trades: DataFrame with trade details.
            annual_factor: Annualization factor (252 for daily).
            
        Returns:
            StrategyMetrics object.
        """
        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (annual_factor / len(returns)) - 1
        
        # Volatility & Sharpe
        daily_vol = np.std(returns)
        sharpe_ratio = np.mean(returns) / (daily_vol + 1e-10) * np.sqrt(annual_factor)
        
        # Drawdown analysis
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Trade analysis
        if len(trades) > 0:
            trades_copy = trades.copy()
            trades_copy['pnl'] = np.where(
                trades_copy['direction'] == 1,
                (trades_copy['exit_price'] - trades_copy['entry_price']) / trades_copy['entry_price'],
                (trades_copy['entry_price'] - trades_copy['exit_price']) / trades_copy['entry_price'],
            )
            
            win_rate = np.mean(trades_copy['pnl'] > 0)
            
            winning_trades = trades_copy[trades_copy['pnl'] > 0]
            losing_trades = trades_copy[trades_copy['pnl'] <= 0]
            
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / (gross_loss + 1e-10)
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            largest_win = trades_copy['pnl'].max()
            largest_loss = trades_copy['pnl'].min()
            
            # Consecutive wins/losses
            pnl_signs = np.sign(trades_copy['pnl'].values)
            consecutive_wins = self._max_consecutive(pnl_signs == 1)
            consecutive_losses = self._max_consecutive(pnl_signs <= 0)
            
            trades_count = len(trades)
            annual_trades = trades_count * (annual_factor / len(returns))
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            largest_win = 0
            largest_loss = 0
            consecutive_wins = 0
            consecutive_losses = 0
            trades_count = 0
            annual_trades = 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return StrategyMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_count=trades_count,
            annual_trades=annual_trades,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            var_95=var_95,
            cvar_95=cvar_95,
        )
    
    def _max_consecutive(self, arr: np.ndarray) -> int:
        """Find maximum consecutive True values in array."""
        if len(arr) == 0:
            return 0
        
        counts = np.diff(np.concatenate(([False], arr, [False])).astype(int))
        starts = np.where(counts == 1)[0]
        ends = np.where(counts == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        return np.max(ends - starts)
    
    def evaluate_per_regime(
        self,
        returns_by_regime: Dict[str, np.ndarray],
        trades_by_regime: Dict[str, pd.DataFrame],
    ) -> Dict[str, StrategyMetrics]:
        """
        Compute metrics separately for each regime.
        
        Args:
            returns_by_regime: Dict mapping regime -> returns array.
            trades_by_regime: Dict mapping regime -> trades DataFrame.
            
        Returns:
            Dict mapping regime -> StrategyMetrics.
        """
        regime_metrics = {}
        
        min_samples = 50  # Minimum samples to evaluate regime
        
        for regime in returns_by_regime.keys():
            returns = returns_by_regime[regime]
            trades = trades_by_regime.get(regime, pd.DataFrame())
            
            if len(returns) < min_samples:
                self.logger.warning(
                    f"Regime {regime}: only {len(returns)} samples, skipping"
                )
                continue
            
            metrics = self.compute_strategy_metrics(returns, trades)
            regime_metrics[regime] = metrics
        
        return regime_metrics
    
    def compute_regime_consistency(
        self,
        regime_metrics: Dict[str, StrategyMetrics],
    ) -> float:
        """
        Compute consistency across regimes (worst / best Sharpe).
        
        Args:
            regime_metrics: Per-regime metrics.
            
        Returns:
            Ratio of worst to best Sharpe ratio.
        """
        if not regime_metrics:
            return 1.0
        
        sharpe_ratios = [m.sharpe_ratio for m in regime_metrics.values()]
        
        worst = min(sharpe_ratios)
        best = max(sharpe_ratios)
        
        # Clamp to prevent division issues
        if best <= 0:
            return 0.0
        
        return max(worst / best, 0.0)
    
    def compute_cross_fold_stability(
        self,
        fold_metrics_list: List[StrategyMetrics],
    ) -> float:
        """
        Compute stability of metrics across folds.
        
        Args:
            fold_metrics_list: List of StrategyMetrics from each fold.
            
        Returns:
            Stability score [0, 1] based on consistency.
        """
        if len(fold_metrics_list) < 2:
            return 1.0
        
        # Extract key metrics
        sharpe_ratios = [m.sharpe_ratio for m in fold_metrics_list]
        returns = [m.annual_return for m in fold_metrics_list]
        
        # Compute coefficient of variation
        metrics_to_check = [sharpe_ratios, returns]
        cvs = []
        
        for values in metrics_to_check:
            values_clean = [v for v in values if np.isfinite(v)]
            if len(values_clean) >= 2:
                cv = np.std(values_clean) / (np.mean(values_clean) + 1e-10)
                cvs.append(cv)
        
        if not cvs:
            return 1.0
        
        avg_cv = np.mean(cvs)
        stability = 1.0 / (1.0 + avg_cv)
        
        return stability
    
    def compute_is_oos_degradation(
        self,
        in_sample_metrics: StrategyMetrics,
        out_of_sample_metrics: StrategyMetrics,
    ) -> float:
        """
        Measure degradation from in-sample to out-of-sample.
        
        Args:
            in_sample_metrics: IS metrics.
            out_of_sample_metrics: OOS metrics.
            
        Returns:
            Degradation score (lower is better, 0 = no degradation).
        """
        # Compare Sharpe ratios (primary metric)
        sharpe_is = in_sample_metrics.sharpe_ratio
        sharpe_oos = out_of_sample_metrics.sharpe_ratio
        
        if sharpe_is <= 0:
            return 1.0  # Can't evaluate
        
        degradation = max(0, (sharpe_is - sharpe_oos) / sharpe_is)
        
        return degradation
    
    def monte_carlo_analysis(
        self,
        returns: np.ndarray,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Run Monte Carlo analysis on strategy returns.
        
        Args:
            returns: Daily returns.
            n_simulations: Number of simulations.
            confidence_level: Confidence level for confidence intervals.
            
        Returns:
            Dict with MC results.
        """
        results = {}
        
        # Block bootstrap for return paths
        block_size = self.config['monte_carlo']['block_size']
        n_blocks = len(returns) // block_size + 1
        
        annual_returns_mc = []
        sharpe_mc = []
        max_dd_mc = []
        
        np.random.seed(42)  # Reproducibility
        
        for _ in range(n_simulations):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
            sampled_returns = []
            
            for idx in block_indices:
                start = idx * block_size
                end = min(start + block_size, len(returns))
                sampled_returns.extend(returns[start:end])
            
            sampled_returns = np.array(sampled_returns[:len(returns)])
            
            # Compute metrics on sampled path
            cumulative = np.cumprod(1 + sampled_returns)
            total_ret = np.prod(1 + sampled_returns) - 1
            annual_ret = (1 + total_ret) ** (252 / len(sampled_returns)) - 1
            
            daily_vol = np.std(sampled_returns)
            sharpe = np.mean(sampled_returns) / (daily_vol + 1e-10) * np.sqrt(252)
            
            running_max = np.maximum.accumulate(cumulative)
            dd = (cumulative - running_max) / running_max
            max_dd = np.min(dd)
            
            annual_returns_mc.append(annual_ret)
            sharpe_mc.append(sharpe)
            max_dd_mc.append(max_dd)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        
        results['return_ci_lower'] = np.percentile(annual_returns_mc, alpha/2 * 100)
        results['return_ci_upper'] = np.percentile(annual_returns_mc, (1 - alpha/2) * 100)
        results['return_ci_mean'] = np.mean(annual_returns_mc)
        
        results['sharpe_ci_lower'] = np.percentile(sharpe_mc, alpha/2 * 100)
        results['sharpe_ci_upper'] = np.percentile(sharpe_mc, (1 - alpha/2) * 100)
        results['sharpe_ci_mean'] = np.mean(sharpe_mc)
        
        results['dd_ci_lower'] = np.percentile(max_dd_mc, alpha/2 * 100)
        results['dd_ci_upper'] = np.percentile(max_dd_mc, (1 - alpha/2) * 100)
        results['dd_ci_mean'] = np.mean(max_dd_mc)
        
        return results
    
    def assign_grade(
        self,
        metrics: StrategyMetrics,
        regime_consistency: float,
        cross_fold_stability: float,
        is_oos_degradation: float,
    ) -> str:
        """
        Assign letter grade based on metrics.
        
        Args:
            metrics: Primary strategy metrics.
            regime_consistency: Worst/best regime Sharpe.
            cross_fold_stability: Fold stability score.
            is_oos_degradation: IS to OOS degradation.
            
        Returns:
            Grade string: "A", "B", "C", or "D".
        """
        grades_config = self.config['grades']
        
        # Grade A checks
        a_checks = [
            metrics.annual_return >= grades_config['A']['annual_return_min'],
            metrics.sharpe_ratio >= grades_config['A']['sharpe_ratio_min'],
            metrics.max_drawdown >= grades_config['A']['max_drawdown_max'],
            metrics.win_rate >= grades_config['A']['win_rate_min'],
            metrics.profit_factor >= grades_config['A']['profit_factor_min'],
            regime_consistency >= grades_config['A']['regime_consistency_min'],
            cross_fold_stability >= grades_config['A']['cross_fold_stability_min'],
            metrics.annual_trades <= grades_config['A']['transactions_per_year_max'],
        ]
        
        if all(a_checks):
            return "A"
        
        # Grade B checks
        b_checks = [
            metrics.annual_return >= grades_config['B']['annual_return_min'],
            metrics.sharpe_ratio >= grades_config['B']['sharpe_ratio_min'],
            metrics.max_drawdown >= grades_config['B']['max_drawdown_max'],
            metrics.win_rate >= grades_config['B']['win_rate_min'],
            metrics.profit_factor >= grades_config['B']['profit_factor_min'],
            regime_consistency >= grades_config['B']['regime_consistency_min'],
            cross_fold_stability >= grades_config['B']['cross_fold_stability_min'],
            metrics.annual_trades <= grades_config['B']['transactions_per_year_max'],
        ]
        
        if all(b_checks):
            return "B"
        
        # Grade C checks
        c_checks = [
            metrics.annual_return >= grades_config['C']['annual_return_min'],
            metrics.sharpe_ratio >= grades_config['C']['sharpe_ratio_min'],
            metrics.max_drawdown >= grades_config['C']['max_drawdown_max'],
            regime_consistency >= grades_config['C']['regime_consistency_min'],
            cross_fold_stability >= grades_config['C']['cross_fold_stability_min'],
        ]
        
        if all(c_checks):
            return "C"
        
        return "D"
    
    def recommend_deployment(
        self,
        grade: str,
        regime_consistency: float,
    ) -> str:
        """
        Recommend deployment mode based on grade and stability.
        
        Args:
            grade: Letter grade.
            regime_consistency: Regime consistency score.
            
        Returns:
            Deployment recommendation: "LIVE", "PAPER", "INCUBATION", "BLOCK".
        """
        gates_config = self.config['deployment_gates']
        pred_grade_min = gates_config['prediction_grade_min']
        
        if grade == "A":
            if regime_consistency >= 0.80:
                return "LIVE"
            else:
                return "PAPER"
        elif grade == "B":
            return gates_config['live_mode']  # Usually "PAPER"
        elif grade == "C":
            return "INCUBATION"  # Very limited trading
        else:
            return "BLOCK"  # Do not trade
    
    def grade_strategy(
        self,
        primary_metrics: StrategyMetrics,
        fold_metrics: List[StrategyMetrics],
        regime_metrics: Dict[str, StrategyMetrics],
        monte_carlo_results: Dict[str, float],
        in_sample_metrics: Optional[StrategyMetrics] = None,
    ) -> StrategyGradeResult:
        """
        Comprehensive grading of trading strategy.
        
        Args:
            primary_metrics: Overall metrics on test set.
            fold_metrics: Metrics from each walk-forward fold.
            regime_metrics: Per-regime metrics.
            monte_carlo_results: MC simulation results.
            in_sample_metrics: Optional IS metrics for degradation analysis.
            
        Returns:
            StrategyGradeResult.
        """
        # Compute stability and consistency
        cross_fold_stability = self.compute_cross_fold_stability(fold_metrics)
        regime_consistency = self.compute_regime_consistency(regime_metrics)
        
        # IS to OOS degradation
        if in_sample_metrics:
            is_oos_degradation = self.compute_is_oos_degradation(
                in_sample_metrics,
                primary_metrics,
            )
        else:
            is_oos_degradation = 0.0
        
        # Assign grade
        grade = self.assign_grade(
            primary_metrics,
            regime_consistency,
            cross_fold_stability,
            is_oos_degradation,
        )
        
        # Deployment recommendation
        deployment = self.recommend_deployment(grade, regime_consistency)
        
        # Generate comments
        comments = self._generate_grade_comments(
            grade,
            primary_metrics,
            regime_consistency,
            cross_fold_stability,
            is_oos_degradation,
        )
        
        result = StrategyGradeResult(
            grade=grade,
            primary_metrics=primary_metrics,
            regime_breakdown=regime_metrics,
            cross_fold_stability=cross_fold_stability,
            regime_consistency=regime_consistency,
            out_of_sample_degradation=is_oos_degradation,
            monte_carlo_results=monte_carlo_results,
            comments=comments,
            timestamp=pd.Timestamp.now().isoformat(),
            deployment_recommendation=deployment,
        )
        
        return result
    
    def _generate_grade_comments(
        self,
        grade: str,
        metrics: StrategyMetrics,
        regime_consistency: float,
        cross_fold_stability: float,
        is_oos_degradation: float,
    ) -> str:
        """Generate human-readable comments about the grade."""
        comments = []
        
        if grade == "A":
            comments.append("Excellent risk-adjusted returns with strong stability.")
            comments.append("Recommended for live trading with full position size.")
        elif grade == "B":
            comments.append("Good risk-adjusted returns with acceptable stability.")
            comments.append("Recommended for paper trading before live deployment.")
        elif grade == "C":
            comments.append("Positive returns but marginal risk metrics.")
            comments.append("Recommended for incubation with small position size.")
        else:  # Grade D
            comments.append("Insufficient risk-adjusted returns or high instability.")
            comments.append("DO NOT deploy. Requires significant strategy redesign.")
        
        if metrics.sharpe_ratio < 0.5:
            comments.append(f"Warning: Sharpe ratio {metrics.sharpe_ratio:.2f} is weak.")
        
        if metrics.max_drawdown < -0.50:
            comments.append(f"Warning: Max drawdown {metrics.max_drawdown:.2%} is severe.")
        
        if regime_consistency < 0.60:
            comments.append(f"Warning: Poor regime consistency (ratio {regime_consistency:.2f}).")
        
        if cross_fold_stability < 0.70:
            comments.append(f"Warning: Unstable across folds (stability {cross_fold_stability:.2f}).")
        
        if is_oos_degradation > 0.30:
            comments.append(f"Warning: Significant IS/OOS degradation ({is_oos_degradation:.1%}).")
        
        if metrics.annual_trades > 1000:
            comments.append(f"Warning: High trading frequency ({metrics.annual_trades:.0f} trades/year).")
        
        return " ".join(comments)
    
    def report_grade(self, result: StrategyGradeResult) -> str:
        """
        Generate formatted report string.
        
        Args:
            result: StrategyGradeResult.
            
        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 80)
        report.append("STRATEGY GRADING REPORT")
        report.append("=" * 80)
        report.append(f"\nGrade: {result.grade}")
        report.append(f"Deployment Recommendation: {result.deployment_recommendation}")
        report.append(f"Timestamp: {result.timestamp}")
        
        m = result.primary_metrics
        report.append(f"\n[Primary Metrics]")
        report.append(f"  Annual Return:       {m.annual_return:>10.2%}")
        report.append(f"  Sharpe Ratio:        {m.sharpe_ratio:>10.2f}")
        report.append(f"  Max Drawdown:        {m.max_drawdown:>10.2%}")
        report.append(f"  Calmar Ratio:        {m.calmar_ratio:>10.2f}")
        report.append(f"  Win Rate:            {m.win_rate:>10.2%}")
        report.append(f"  Profit Factor:       {m.profit_factor:>10.2f}")
        report.append(f"  Total Trades:        {m.trades_count:>10.0f}")
        report.append(f"  Annual Trades:       {m.annual_trades:>10.1f}")
        
        report.append(f"\n[Stability & Consistency]")
        report.append(f"  Cross-Fold Stability:    {result.cross_fold_stability:>6.2f}")
        report.append(f"  Regime Consistency:      {result.regime_consistency:>6.2f}")
        report.append(f"  IS/OOS Degradation:      {result.out_of_sample_degradation:>6.1%}")
        
        if result.monte_carlo_results:
            mc = result.monte_carlo_results
            report.append(f"\n[Monte Carlo (95% CI)]")
            if 'return_ci_lower' in mc:
                report.append(
                    f"  Return Range:     [{mc['return_ci_lower']:>6.2%}, {mc['return_ci_upper']:>6.2%}]"
                )
            if 'sharpe_ci_lower' in mc:
                report.append(
                    f"  Sharpe Range:     [{mc['sharpe_ci_lower']:>6.2f}, {mc['sharpe_ci_upper']:>6.2f}]"
                )
            if 'dd_ci_lower' in mc:
                report.append(
                    f"  Drawdown Range:   [{mc['dd_ci_lower']:>6.2%}, {mc['dd_ci_upper']:>6.2%}]"
                )
        
        if result.regime_breakdown:
            report.append(f"\n[Per-Regime Analysis]")
            report.append(f"  {'Regime':<15} {'Return':<12} {'Sharpe':<12} {'Max DD':<12}")
            report.append("-" * 55)
            for regime_name, metrics in result.regime_breakdown.items():
                report.append(
                    f"  {regime_name:<15} {metrics.annual_return:>11.2%} "
                    f"{metrics.sharpe_ratio:>11.2f} {metrics.max_drawdown:>11.2%}"
                )
        
        report.append(f"\n[Comments]")
        report.append(result.comments)
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    grader = StrategyGrader()
    
    # Simulate returns and trades
    returns = np.random.normal(0.0005, 0.01, 252)  # Daily returns
    trades_df = pd.DataFrame({
        'entry_price': np.random.uniform(100, 110, 50),
        'exit_price': np.random.uniform(100, 110, 50),
        'direction': np.random.choice([1, -1], 50),
        'entry_date': pd.date_range('2023-01-01', periods=50),
        'exit_date': pd.date_range('2023-01-10', periods=50),
    })
    
    metrics = grader.compute_strategy_metrics(returns, trades_df)
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Annual Return: {metrics.annual_return:.2%}")

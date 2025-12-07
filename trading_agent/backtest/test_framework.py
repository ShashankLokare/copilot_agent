"""
Integrated Test & Grading Framework

Orchestrates prediction engine evaluation and strategy backtesting with grading.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from learning.prediction_grader import PredictionGrader, PredictionGradeResult
from backtest.strategy_grader import StrategyGrader, StrategyGradeResult, StrategyMetrics
from learning.model_metadata import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)


class TestFramework:
    """
    Integrated testing framework for predictions and full strategies.
    """
    
    def __init__(
        self,
        pred_config: str = "config/prediction_grading.yaml",
        strat_config: str = "config/strategy_grading.yaml",
        registry_path: str = "models/registry.json",
    ):
        """
        Initialize test framework.
        
        Args:
            pred_config: Path to prediction grading config.
            strat_config: Path to strategy grading config.
            registry_path: Path to model registry.
        """
        self.pred_grader = PredictionGrader(pred_config)
        self.strat_grader = StrategyGrader(strat_config)
        self.registry = ModelRegistry(registry_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def evaluate_prediction_engine_walk_forward(
        self,
        model_name: str,
        model_version: str,
        fold_results: List[Dict],
    ) -> PredictionGradeResult:
        """
        Evaluate prediction engine across walk-forward folds.
        
        Args:
            model_name: Name of model.
            model_version: Version of model.
            fold_results: List of test results from each fold.
                         Each fold should contain:
                         - y_true, y_pred_proba, y_pred_return, y_true_return, metrics, regimes
            
        Returns:
            PredictionGradeResult.
        """
        # Aggregate metrics across folds
        y_true_all = np.concatenate([r['y_true'] for r in fold_results])
        y_pred_proba_all = np.concatenate([r['y_pred_proba'] for r in fold_results])
        y_true_return_all = np.concatenate([r['y_true_return'] for r in fold_results])
        y_pred_return_all = np.concatenate([r['y_pred_return'] for r in fold_results])
        
        # Compute overall metrics
        overall_metrics = self.pred_grader.evaluate_single_test_set(
            y_true_all,
            y_pred_proba_all,
            y_pred_return_all,
            y_true_return_all,
        )
        
        # Compute grade
        grade_result = self.pred_grader.grade_prediction_engine(
            fold_results,
            overall_metrics,
        )
        
        # Register in model registry
        self.registry.register_model(
            ModelMetadata(
                model_name=model_name,
                model_version=model_version,
                model_type="prediction_engine",
                training_date=pd.Timestamp.now().isoformat(),
                training_features=[],  # Filled in by caller if needed
                training_horizons=[],  # Filled in by caller if needed
                training_data_period={"start": "", "end": ""},
                training_n_samples=len(y_true_all),
                hyperparameters={},
            )
        )
        
        # Update with grading results
        self.registry.update_grading(
            model_name,
            model_version,
            grade_result.grade,
            {
                "roc_auc": grade_result.roc_auc,
                "brier_score": grade_result.brier_score,
                "accuracy": grade_result.accuracy,
                "cross_fold_stability": grade_result.cross_fold_stability,
                "regime_consistency": grade_result.regime_consistency,
            },
            comments=grade_result.comments,
        )
        
        self.logger.info(f"Graded {model_name} v{model_version}: {grade_result.grade}")
        
        return grade_result
    
    def evaluate_strategy_walk_forward(
        self,
        strategy_name: str,
        strategy_version: str,
        fold_results: List[Tuple[np.ndarray, pd.DataFrame]],  # (returns, trades) per fold
        regime_mapping: Optional[Dict[str, List[int]]] = None,  # Indices for each regime
    ) -> StrategyGradeResult:
        """
        Evaluate full trading strategy across walk-forward folds.
        
        Args:
            strategy_name: Name of strategy.
            strategy_version: Version of strategy.
            fold_results: List of (returns, trades_df) tuples from each fold.
            regime_mapping: Optional dict mapping regime_name -> list of fold indices.
            
        Returns:
            StrategyGradeResult.
        """
        # Aggregate results across folds
        all_returns = []
        all_trades = []
        fold_metrics_list = []
        
        for returns, trades_df in fold_results:
            all_returns.extend(returns)
            all_trades.append(trades_df)
            
            # Compute metrics for this fold
            fold_metrics = self.strat_grader.compute_strategy_metrics(returns, trades_df)
            fold_metrics_list.append(fold_metrics)
        
        all_returns = np.array(all_returns)
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        
        # Overall metrics
        primary_metrics = self.strat_grader.compute_strategy_metrics(
            all_returns,
            all_trades_df,
        )
        
        # Per-regime metrics
        regime_metrics = {}
        if regime_mapping:
            for regime_name, fold_indices in regime_mapping.items():
                regime_returns_list = []
                regime_trades_list = []
                
                for fold_idx in fold_indices:
                    if fold_idx < len(fold_results):
                        returns, trades_df = fold_results[fold_idx]
                        regime_returns_list.extend(returns)
                        regime_trades_list.append(trades_df)
                
                if regime_returns_list:
                    regime_returns = np.array(regime_returns_list)
                    regime_trades = pd.concat(regime_trades_list, ignore_index=True)
                    
                    regime_metrics[regime_name] = self.strat_grader.compute_strategy_metrics(
                        regime_returns,
                        regime_trades,
                    )
        
        # Monte Carlo analysis
        monte_carlo_results = self.strat_grader.monte_carlo_analysis(all_returns)
        
        # Compute grade
        grade_result = self.strat_grader.grade_strategy(
            primary_metrics,
            fold_metrics_list,
            regime_metrics,
            monte_carlo_results,
        )
        
        self.logger.info(f"Graded {strategy_name} v{strategy_version}: {grade_result.grade}")
        
        return grade_result
    
    def generate_test_report(
        self,
        pred_grade: PredictionGradeResult,
        strat_grade: StrategyGradeResult,
        output_dir: str = "reports/grading",
    ) -> None:
        """
        Generate comprehensive test report.
        
        Args:
            pred_grade: Prediction grading result.
            strat_grade: Strategy grading result.
            output_dir: Output directory for reports.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prediction report
        pred_report = self.pred_grader.report_grade(pred_grade)
        pred_report_path = output_path / "prediction_grade_report.txt"
        with open(pred_report_path, 'w') as f:
            f.write(pred_report)
        self.logger.info(f"Saved prediction report to {pred_report_path}")
        
        # Strategy report
        strat_report = self.strat_grader.report_grade(strat_grade)
        strat_report_path = output_path / "strategy_grade_report.txt"
        with open(strat_report_path, 'w') as f:
            f.write(strat_report)
        self.logger.info(f"Saved strategy report to {strat_report_path}")
        
        # Summary JSON
        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "prediction_grade": pred_grade.grade,
            "strategy_grade": strat_grade.grade,
            "deployment_recommendation": strat_grade.deployment_recommendation,
            "prediction_metrics": {
                "roc_auc": pred_grade.roc_auc,
                "brier_score": pred_grade.brier_score,
                "accuracy": pred_grade.accuracy,
            },
            "strategy_metrics": {
                "annual_return": primary_metrics.annual_return,
                "sharpe_ratio": primary_metrics.sharpe_ratio,
                "max_drawdown": primary_metrics.max_drawdown,
                "win_rate": primary_metrics.win_rate,
            },
            "stability": {
                "cross_fold_stability": strat_grade.cross_fold_stability,
                "regime_consistency": strat_grade.regime_consistency,
            },
        }
        
        summary_path = output_path / "grading_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Saved summary to {summary_path}")
    
    def check_deployment_gates(
        self,
        pred_grade: str,
        strat_grade: str,
    ) -> Tuple[bool, str]:
        """
        Check if model/strategy can be deployed.
        
        Args:
            pred_grade: Prediction engine grade.
            strat_grade: Strategy grade.
            
        Returns:
            Tuple of (is_approvable, recommendation).
        """
        gates = self._load_gates_config()
        
        pred_min = gates.get('prediction_grade_min', 'B')
        strat_min = gates.get('strategy_grade_min', 'B')
        
        grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
        
        pred_ok = grade_rank.get(pred_grade, 3) <= grade_rank.get(pred_min, 1)
        strat_ok = grade_rank.get(strat_grade, 3) <= grade_rank.get(strat_min, 1)
        
        if not (pred_ok and strat_ok):
            return False, "Grade requirements not met"
        
        if gates.get('requires_manual_approval', False):
            return False, "Manual approval required"
        
        return True, "Approved for deployment"
    
    def _load_gates_config(self) -> Dict:
        """Load deployment gates config."""
        try:
            with open("config/strategy_grading.yaml", 'r') as f:
                config = yaml.safe_load(f)
                return config.get('deployment_gates', {})
        except Exception as e:
            self.logger.error(f"Failed to load gates config: {e}")
            return {}


class StrategyBacktestEvaluator:
    """
    Evaluates strategies through backtesting with metrics and grading.
    """
    
    def __init__(self, test_framework: TestFramework):
        """
        Initialize evaluator.
        
        Args:
            test_framework: TestFramework instance.
        """
        self.framework = test_framework
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_walk_forward_backtest(
        self,
        strategy_func,  # Function that takes (train_df, test_df) -> (returns, trades)
        data: pd.DataFrame,
        train_years: int = 2,
        test_years: int = 1,
        step_years: int = 1,
    ) -> List[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Run walk-forward backtest.
        
        Args:
            strategy_func: Function taking (train_df, test_df) and returning (returns, trades).
            data: Full dataset with DatetimeIndex.
            train_years: Training window in years.
            test_years: Testing window in years.
            step_years: Rolling window step in years.
            
        Returns:
            List of (returns, trades) tuples for each fold.
        """
        fold_results = []
        
        # Generate folds
        start_date = data.index.min()
        end_date = data.index.max()
        
        train_days = int(train_years * 252)
        test_days = int(test_years * 252)
        step_days = int(step_years * 252)
        
        fold_start = start_date
        
        while True:
            train_end = fold_start + pd.Timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_days)
            
            if test_end > end_date:
                break
            
            train_df = data.loc[fold_start:train_end]
            test_df = data.loc[test_start:test_end]
            
            if len(train_df) < 100 or len(test_df) < 50:
                break
            
            # Run strategy
            returns, trades = strategy_func(train_df, test_df)
            fold_results.append((returns, trades))
            
            self.logger.info(
                f"Fold: Train {fold_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )
            
            fold_start = fold_start + pd.Timedelta(days=step_days)
        
        return fold_results


if __name__ == "__main__":
    # Example usage
    framework = TestFramework()
    
    # Example prediction evaluation
    fold_results = [
        {
            'y_true': np.random.binomial(1, 0.5, 100),
            'y_pred_proba': np.random.uniform(0, 1, 100),
            'y_pred_return': np.random.normal(0, 0.01, 100),
            'y_true_return': np.random.normal(0, 0.015, 100),
            'metrics': {},
            'regimes': None,
        }
        for _ in range(3)
    ]
    
    pred_result = framework.evaluate_prediction_engine_walk_forward(
        "test_model",
        "1.0.0",
        fold_results,
    )
    
    print(f"Prediction Grade: {pred_result.grade}")

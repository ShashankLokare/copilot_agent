"""
Performance Analysis & Metrics Computation

Computes ML metrics (classification, regression, trading-specific)
and reliability checks for model deployment.
"""

from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import asdict

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from learning.data_structures import EvaluationMetrics


logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Computes comprehensive ML and trading-specific metrics.
    """
    
    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        pos_label: int = 1,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC).
            pos_label: Positive class label.
            
        Returns:
            Dict with metric names and values.
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # For binary classification
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label)
            metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label)
            metrics['f1'] = f1_score(y_true, y_pred, pos_label=pos_label)
            
            # ROC-AUC
            if y_pred_proba is not None:
                if y_pred_proba.ndim == 2:
                    # Multi-output, take second column
                    y_pred_proba = y_pred_proba[:, 1]
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                except ValueError:
                    metrics['roc_auc'] = np.nan
        
        # Brier score (probability calibration)
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba
            
            brier = np.mean((y_pred_proba_pos - y_true) ** 2)
            metrics['brier_score'] = brier
        
        return metrics
    
    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dict with metric names and values.
        """
        metrics = {}
        
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # R² (only if variance in y_true)
        if np.var(y_true) > 1e-10:
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            metrics['r2'] = np.nan
        
        return metrics
    
    @staticmethod
    def compute_trading_metrics(
        y_true_return: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_return: np.ndarray,
        prob_thresholds: List[float] = None,
    ) -> Dict[str, float]:
        """
        Compute trading-specific metrics.
        
        Args:
            y_true_return: True forward returns.
            y_pred_proba: Predicted probabilities for up move.
            y_pred_return: Predicted returns.
            prob_thresholds: Probability thresholds to compute hit rates.
            
        Returns:
            Dict with trading metrics.
        """
        prob_thresholds = prob_thresholds or [0.55, 0.6, 0.65, 0.7]
        
        metrics = {}
        
        # Determine true direction from returns
        y_true_dir = np.sign(y_true_return)  # -1, 0, +1
        
        # Determine predicted direction from probabilities
        y_pred_dir = np.where(y_pred_proba > 0.5, 1, -1)
        
        # Hit rate: % correct when probability exceeds threshold
        for threshold in prob_thresholds:
            mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
            if np.sum(mask) > 0:
                # Among high-confidence predictions, how many correct?
                hit_rate = np.mean(y_true_dir[mask] == y_pred_dir[mask])
                metrics[f'hit_rate_{int(threshold*100)}'] = hit_rate
        
        # Average return conditional on signal
        mask_signal = y_pred_dir == 1  # Long signals
        if np.sum(mask_signal) > 0:
            metrics['avg_return_if_signal'] = np.mean(y_true_return[mask_signal])
        
        mask_no_signal = y_pred_dir == -1  # Short signals
        if np.sum(mask_no_signal) > 0:
            metrics['avg_return_if_short'] = np.mean(y_true_return[mask_no_signal])
        
        # Baseline: average return random
        metrics['avg_return_all'] = np.mean(y_true_return)
        
        # Correlation between predicted and actual returns
        correlation = np.corrcoef(y_pred_return, y_true_return)[0, 1]
        metrics['return_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return metrics
    
    @staticmethod
    def compute_per_regime_metrics(
        df: pd.DataFrame,
        regimes: List[str],
        regime_col: str = 'regime',
        y_true_ret_col: str = 'target_return_k',
        y_pred_proba_col: str = 'pred_proba',
        y_pred_ret_col: str = 'pred_return',
    ) -> Dict[str, Dict]:
        """
        Compute metrics separately for each regime.
        
        Args:
            df: DataFrame with predictions and actuals.
            regimes: List of regime names.
            regime_col: Regime column name.
            y_true_ret_col: True return column name.
            y_pred_proba_col: Predicted probability column name.
            y_pred_ret_col: Predicted return column name.
            
        Returns:
            Dict mapping regime -> metrics dict.
        """
        per_regime = {}
        
        for regime in regimes:
            regime_df = df[df[regime_col] == regime]
            
            if len(regime_df) < 10:
                logger.warning(f"Skipping regime {regime}: only {len(regime_df)} samples")
                continue
            
            # Extract data for this regime
            y_ret_true = regime_df[y_true_ret_col].values
            y_prob_pred = regime_df[y_pred_proba_col].values
            y_ret_pred = regime_df[y_pred_ret_col].values
            
            # Compute metrics
            trading_metrics = PerformanceAnalyzer.compute_trading_metrics(
                y_ret_true,
                y_prob_pred,
                y_ret_pred,
            )
            
            regression_metrics = PerformanceAnalyzer.compute_regression_metrics(
                y_ret_true,
                y_ret_pred,
            )
            
            per_regime[regime] = {**trading_metrics, **regression_metrics}
        
        return per_regime
    
    @staticmethod
    def evaluate_on_fold(
        y_true_ret: np.ndarray,
        y_true_dir: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_ret: np.ndarray,
        regimes: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate predictions on a validation fold.
        
        Args:
            y_true_ret: True forward returns.
            y_true_dir: True directions (-1, 0, +1).
            y_pred_proba: Predicted probability of up move.
            y_pred_ret: Predicted forward return.
            regimes: Optional regime labels.
            
        Returns:
            EvaluationMetrics object.
        """
        # Direction classification metrics
        y_pred_dir = np.where(y_pred_proba > 0.5, 1, -1)
        
        class_metrics = PerformanceAnalyzer.compute_classification_metrics(
            y_true_dir,
            y_pred_dir,
            y_pred_proba=y_pred_proba,
            pos_label=1,
        )
        
        # Return regression metrics
        reg_metrics = PerformanceAnalyzer.compute_regression_metrics(
            y_true_ret,
            y_pred_ret,
        )
        
        # Trading metrics
        trading_metrics = PerformanceAnalyzer.compute_trading_metrics(
            y_true_ret,
            y_pred_proba,
            y_pred_ret,
        )
        
        # Per-regime metrics
        per_regime = {}
        if regimes is not None:
            unique_regimes = np.unique(regimes)
            for regime in unique_regimes:
                mask = regimes == regime
                regime_metrics = PerformanceAnalyzer.compute_trading_metrics(
                    y_true_ret[mask],
                    y_pred_proba[mask],
                    y_pred_ret[mask],
                )
                per_regime[str(regime)] = regime_metrics
        
        # Build result
        result = EvaluationMetrics(
            accuracy=class_metrics.get('accuracy'),
            precision_up=class_metrics.get('precision'),
            recall_up=class_metrics.get('recall'),
            f1_up=class_metrics.get('f1'),
            roc_auc=class_metrics.get('roc_auc'),
            brier_score=class_metrics.get('brier_score'),
            mae=reg_metrics.get('mae'),
            rmse=reg_metrics.get('rmse'),
            r2_score=reg_metrics.get('r2'),
            hit_rate_60=trading_metrics.get('hit_rate_60'),
            hit_rate_70=trading_metrics.get('hit_rate_70'),
            avg_return_if_signal=trading_metrics.get('avg_return_if_signal'),
            avg_return_random=trading_metrics.get('avg_return_all'),
            per_regime_metrics=per_regime,
        )
        
        return result


class ReliabilityChecker:
    """
    Checks whether a model is suitable for deployment.
    """
    
    @staticmethod
    def check_deployment_criteria(
        metrics: EvaluationMetrics,
        min_accuracy: float = 0.50,
        min_brier: float = 0.25,
        min_hit_rate_60: float = 0.52,
        max_regime_variance: float = 0.10,
    ) -> Tuple[bool, List[str]]:
        """
        Check if model meets deployment criteria.
        
        Args:
            metrics: EvaluationMetrics object.
            min_accuracy: Minimum accuracy required.
            min_brier: Maximum Brier score (lower is better).
            min_hit_rate_60: Minimum hit rate at 60% confidence.
            max_regime_variance: Max variance in hit rate across regimes.
            
        Returns:
            Tuple of (is_deployable, list of warnings).
        """
        warnings = []
        
        # Check accuracy
        if metrics.accuracy is not None and metrics.accuracy < min_accuracy:
            warnings.append(
                f"Accuracy {metrics.accuracy:.3f} < threshold {min_accuracy}"
            )
        
        # Check Brier score (probability calibration)
        if metrics.brier_score is not None and metrics.brier_score > min_brier:
            warnings.append(
                f"Brier score {metrics.brier_score:.3f} > threshold {min_brier} "
                "(probabilities not well-calibrated)"
            )
        
        # Check hit rate at 60%
        if metrics.hit_rate_60 is not None and metrics.hit_rate_60 < min_hit_rate_60:
            warnings.append(
                f"Hit rate at 60% confidence {metrics.hit_rate_60:.3f} "
                f"< threshold {min_hit_rate_60} (worse than random)"
            )
        
        # Check regime consistency
        if metrics.per_regime_metrics:
            hit_rates = [
                m.get('hit_rate_60', np.nan)
                for m in metrics.per_regime_metrics.values()
            ]
            hit_rates = [h for h in hit_rates if not np.isnan(h)]
            
            if len(hit_rates) > 1:
                hit_rate_std = np.std(hit_rates)
                if hit_rate_std > max_regime_variance:
                    warnings.append(
                        f"High variance in hit rate across regimes "
                        f"(std={hit_rate_std:.3f}), model may not generalize"
                    )
        
        is_deployable = len(warnings) == 0
        
        return is_deployable, warnings
    
    @staticmethod
    def print_report(
        metrics: EvaluationMetrics,
        fold_id: Optional[int] = None,
    ) -> None:
        """
        Print a formatted metrics report.
        
        Args:
            metrics: EvaluationMetrics object.
            fold_id: Optional fold ID for reporting.
        """
        fold_str = f" (Fold {fold_id})" if fold_id is not None else ""
        
        print(f"\n{'='*60}")
        print(f"Performance Metrics{fold_str}")
        print('='*60)
        
        print("\n[Classification Metrics]")
        print(f"  Accuracy:          {metrics.accuracy:.4f}" if metrics.accuracy else "  Accuracy:         N/A")
        print(f"  Precision (Up):    {metrics.precision_up:.4f}" if metrics.precision_up else "  Precision (Up):   N/A")
        print(f"  Recall (Up):       {metrics.recall_up:.4f}" if metrics.recall_up else "  Recall (Up):      N/A")
        print(f"  F1-Score (Up):     {metrics.f1_up:.4f}" if metrics.f1_up else "  F1-Score (Up):    N/A")
        print(f"  ROC-AUC:           {metrics.roc_auc:.4f}" if metrics.roc_auc else "  ROC-AUC:          N/A")
        print(f"  Brier Score:       {metrics.brier_score:.4f}" if metrics.brier_score else "  Brier Score:      N/A")
        
        print("\n[Regression Metrics]")
        print(f"  MAE:               {metrics.mae:.6f}" if metrics.mae else "  MAE:              N/A")
        print(f"  RMSE:              {metrics.rmse:.6f}" if metrics.rmse else "  RMSE:             N/A")
        print(f"  R² Score:          {metrics.r2_score:.4f}" if metrics.r2_score else "  R² Score:         N/A")
        
        print("\n[Trading Metrics]")
        print(f"  Hit Rate @ 60%:    {metrics.hit_rate_60:.4f}" if metrics.hit_rate_60 else "  Hit Rate @ 60%:   N/A")
        print(f"  Hit Rate @ 70%:    {metrics.hit_rate_70:.4f}" if metrics.hit_rate_70 else "  Hit Rate @ 70%:   N/A")
        print(f"  Avg Ret if Signal: {metrics.avg_return_if_signal:.4f}" if metrics.avg_return_if_signal else "  Avg Ret if Signal: N/A")
        print(f"  Baseline Avg Ret:  {metrics.avg_return_random:.4f}" if metrics.avg_return_random else "  Baseline Avg Ret:  N/A")
        
        # Per-regime
        if metrics.per_regime_metrics:
            print("\n[Per-Regime Metrics]")
            for regime, regime_metrics in metrics.per_regime_metrics.items():
                hit_rate = regime_metrics.get('hit_rate_60', np.nan)
                print(f"  {regime:20s}: {hit_rate:.4f}" if not np.isnan(hit_rate) else f"  {regime:20s}: N/A")


if __name__ == "__main__":
    # Test metrics computation
    y_true_ret = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.03, -0.015])
    y_true_dir = np.sign(y_true_ret)
    y_pred_proba = np.array([0.65, 0.35, 0.70, 0.40, 0.75, 0.30, 0.80, 0.25])
    y_pred_ret = np.array([0.015, -0.018, 0.020, -0.008, 0.025, -0.012, 0.032, -0.020])
    
    metrics = PerformanceAnalyzer.evaluate_on_fold(
        y_true_ret,
        y_true_dir,
        y_pred_proba,
        y_pred_ret,
    )
    
    ReliabilityChecker.print_report(metrics, fold_id=0)
    is_ok, warnings = ReliabilityChecker.check_deployment_criteria(metrics)
    print(f"\nDeployable: {is_ok}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

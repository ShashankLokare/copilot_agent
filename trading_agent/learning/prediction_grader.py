"""
Prediction Engine Evaluation & Grading

Comprehensive analysis of ML model quality out-of-sample.
Grades models A/B/C/D based on classification metrics, calibration, and cross-regime stability.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityBucketStats:
    """Statistics for a probability bucket."""
    bucket_label: str  # e.g., "0.60-0.65"
    prob_min: float
    prob_max: float
    n_samples: int
    actual_hit_rate: float  # Fraction of actual up moves
    expected_hit_rate: float  # Average probability in bucket
    avg_realized_return: float
    calibration_error: float  # |actual_hit_rate - expected_hit_rate|


@dataclass
class RegimeMetrics:
    """Metrics for a single regime."""
    regime_name: str
    n_samples: int
    accuracy: float
    precision_up: float
    recall_up: float
    roc_auc: Optional[float]
    brier_score: float
    hit_rate_60: float
    hit_rate_70: float


@dataclass
class PredictionGradeResult:
    """Complete grading result for prediction engine."""
    grade: str  # "A" | "B" | "C" | "D"
    roc_auc: float
    brier_score: float
    accuracy: float
    bucket_stats: List[ProbabilityBucketStats]
    regime_breakdown: Dict[str, RegimeMetrics]
    cross_fold_stability: float  # Correlation of metrics across folds
    regime_consistency: float  # Hit rate ratio: worst / best
    comments: str
    timestamp: str


class PredictionGrader:
    """
    Grades prediction engine based on multiple criteria.
    """
    
    def __init__(self, config_path: str = "config/prediction_grading.yaml"):
        """
        Initialize grader with config.
        
        Args:
            config_path: Path to grading configuration YAML.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def evaluate_single_test_set(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_return: np.ndarray,
        y_true_return: np.ndarray,
        regimes: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions on a single test set.
        
        Args:
            y_true: Binary labels (0=down, 1=up).
            y_pred_proba: Predicted probability of up move.
            y_pred_return: Predicted forward return.
            y_true_return: Realized forward return.
            regimes: Optional regime labels.
            
        Returns:
            Dict with all metrics.
        """
        metrics = {}
        
        # Accuracy
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision & Recall for UP class
        if len(np.unique(y_true)) == 2:
            metrics['precision_up'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall_up'] = recall_score(y_true, y_pred, zero_division=0)
            
            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = np.nan
        
        # Brier Score (calibration)
        metrics['brier_score'] = np.mean((y_pred_proba - y_true) ** 2)
        
        # Return correlation
        correlation = np.corrcoef(y_pred_return, y_true_return)[0, 1]
        metrics['return_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Hit rates at specific thresholds
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            mask = (y_pred_proba > threshold) | (y_pred_proba < (1 - threshold))
            if np.sum(mask) > 5:
                predicted_dir = (y_pred_proba[mask] > 0.5).astype(int)
                actual_dir = y_true[mask]
                hit_rate = accuracy_score(actual_dir, predicted_dir)
                metrics[f'hit_rate_{int(threshold*100)}'] = hit_rate
        
        return metrics
    
    def compute_probability_buckets(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_true_return: np.ndarray,
    ) -> List[ProbabilityBucketStats]:
        """
        Bucket predictions by probability and analyze calibration.
        
        Args:
            y_true: Binary labels (0=down, 1=up).
            y_pred_proba: Predicted probabilities.
            y_true_return: Realized returns.
            
        Returns:
            List of ProbabilityBucketStats.
        """
        bucket_edges = self.config['probability_buckets']['edges']
        bucket_stats = []
        
        for i in range(len(bucket_edges) - 1):
            prob_min = bucket_edges[i]
            prob_max = bucket_edges[i + 1]
            
            # Get samples in this bucket
            mask = (y_pred_proba >= prob_min) & (y_pred_proba < prob_max)
            
            if np.sum(mask) < 5:
                continue
            
            bucket_true = y_true[mask]
            bucket_proba = y_pred_proba[mask]
            bucket_return = y_true_return[mask]
            
            # Actual hit rate
            actual_hit_rate = np.mean(bucket_true)
            
            # Expected hit rate (average probability in bucket)
            expected_hit_rate = np.mean(bucket_proba)
            
            # Calibration error
            calibration_error = abs(actual_hit_rate - expected_hit_rate)
            
            stats = ProbabilityBucketStats(
                bucket_label=f"{prob_min:.2f}-{prob_max:.2f}",
                prob_min=prob_min,
                prob_max=prob_max,
                n_samples=np.sum(mask),
                actual_hit_rate=actual_hit_rate,
                expected_hit_rate=expected_hit_rate,
                avg_realized_return=np.mean(bucket_return),
                calibration_error=calibration_error,
            )
            bucket_stats.append(stats)
        
        return bucket_stats
    
    def check_bucket_monotonicity(
        self,
        bucket_stats: List[ProbabilityBucketStats],
    ) -> bool:
        """
        Check if hit-rate increases monotonically with probability.
        
        Args:
            bucket_stats: List of bucket statistics.
            
        Returns:
            True if monotonic (or mostly monotonic).
        """
        if len(bucket_stats) < 2:
            return False
        
        hit_rates = [b.actual_hit_rate for b in bucket_stats]
        
        # Check if mostly increasing (allow some noise)
        increasing_pairs = sum(
            1 for i in range(len(hit_rates) - 1)
            if hit_rates[i + 1] >= hit_rates[i]
        )
        
        return increasing_pairs >= len(hit_rates) - 2
    
    def evaluate_per_regime(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_return: np.ndarray,
        y_true_return: np.ndarray,
        regimes: np.ndarray,
    ) -> Dict[str, RegimeMetrics]:
        """
        Evaluate predictions separately for each regime.
        
        Args:
            y_true: Binary labels.
            y_pred_proba: Predicted probabilities.
            y_pred_return: Predicted returns.
            y_true_return: Realized returns.
            regimes: Regime labels.
            
        Returns:
            Dict mapping regime -> RegimeMetrics.
        """
        regime_metrics = {}
        min_samples = self.config['regime_analysis']['min_samples_per_regime']
        
        for regime in np.unique(regimes):
            mask = regimes == regime
            
            if np.sum(mask) < min_samples:
                self.logger.warning(
                    f"Regime {regime}: only {np.sum(mask)} samples, skipping"
                )
                continue
            
            y_true_regime = y_true[mask]
            y_pred_proba_regime = y_pred_proba[mask]
            y_true_return_regime = y_true_return[mask]
            
            # Compute metrics
            y_pred = (y_pred_proba_regime > 0.5).astype(int)
            accuracy = accuracy_score(y_true_regime, y_pred)
            precision = precision_score(y_true_regime, y_pred, zero_division=0)
            recall = recall_score(y_true_regime, y_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_true_regime, y_pred_proba_regime)
            except ValueError:
                roc_auc = np.nan
            
            brier = np.mean((y_pred_proba_regime - y_true_regime) ** 2)
            
            # Hit rates
            hit_rate_60 = np.mean(
                y_pred[(y_pred_proba_regime > 0.60) | (y_pred_proba_regime < 0.40)] ==
                y_true_regime[(y_pred_proba_regime > 0.60) | (y_pred_proba_regime < 0.40)]
            )
            hit_rate_70 = np.mean(
                y_pred[(y_pred_proba_regime > 0.70) | (y_pred_proba_regime < 0.30)] ==
                y_true_regime[(y_pred_proba_regime > 0.70) | (y_pred_proba_regime < 0.30)]
            )
            
            regime_metrics[str(regime)] = RegimeMetrics(
                regime_name=str(regime),
                n_samples=np.sum(mask),
                accuracy=accuracy,
                precision_up=precision,
                recall_up=recall,
                roc_auc=roc_auc,
                brier_score=brier,
                hit_rate_60=hit_rate_60,
                hit_rate_70=hit_rate_70,
            )
        
        return regime_metrics
    
    def compute_regime_consistency(
        self,
        regime_metrics: Dict[str, RegimeMetrics],
    ) -> float:
        """
        Compute consistency across regimes (worst / best).
        
        Args:
            regime_metrics: Per-regime metrics.
            
        Returns:
            Ratio of worst to best performance.
        """
        if not regime_metrics:
            return 1.0
        
        metric_name = self.config['regime_analysis']['consistency_metric']
        
        # Extract metric for each regime
        values = []
        for regime, metrics in regime_metrics.items():
            if metric_name == "hit_rate":
                value = metrics.hit_rate_60
            elif metric_name == "roc_auc":
                value = metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.5
            else:
                value = metrics.accuracy
            values.append(value)
        
        if not values:
            return 1.0
        
        worst = min(values)
        best = max(values)
        
        return worst / best if best > 0 else 0.0
    
    def compute_cross_fold_stability(
        self,
        fold_metrics_list: List[Dict[str, float]],
    ) -> float:
        """
        Compute stability of metrics across folds.
        
        Args:
            fold_metrics_list: List of metric dicts from each fold.
            
        Returns:
            Correlation of key metrics across folds (avg).
        """
        if len(fold_metrics_list) < 2:
            return 1.0
        
        # Extract common metrics from all folds
        key_metrics = ['accuracy', 'roc_auc', 'brier_score']
        
        metric_values = {}
        for metric_name in key_metrics:
            values = [
                m.get(metric_name, np.nan)
                for m in fold_metrics_list
                if metric_name in m
            ]
            if len(values) >= 2:
                metric_values[metric_name] = values
        
        if not metric_values:
            return 1.0
        
        # Compute CV of each metric
        cvs = []
        for metric_name, values in metric_values.items():
            values_clean = [v for v in values if not np.isnan(v)]
            if len(values_clean) >= 2:
                cv = np.std(values_clean) / (np.mean(values_clean) + 1e-10)
                cvs.append(cv)
        
        # Lower CV = higher stability
        # Convert to stability score [0, 1]
        if not cvs:
            return 1.0
        
        avg_cv = np.mean(cvs)
        stability = 1.0 / (1.0 + avg_cv)  # Sigmoid-like transformation
        
        return stability
    
    def assign_grade(
        self,
        roc_auc: float,
        brier_score: float,
        accuracy: float,
        bucket_monotonicity: bool,
        regime_consistency: float,
        cross_fold_stability: float,
    ) -> str:
        """
        Assign letter grade based on metrics.
        
        Args:
            roc_auc: Area under ROC curve.
            brier_score: Probability calibration score.
            accuracy: Classification accuracy.
            bucket_monotonicity: Whether buckets show monotonic improvement.
            regime_consistency: Worst/best performance across regimes.
            cross_fold_stability: Stability across folds.
            
        Returns:
            Grade string: "A", "B", "C", or "D".
        """
        grades_config = self.config['grades']
        
        # Check Grade A
        a_checks = [
            roc_auc >= grades_config['A']['roc_auc_min'],
            brier_score <= grades_config['A']['brier_max'],
            accuracy >= grades_config['A']['accuracy_min'],
            regime_consistency >= grades_config['A']['regime_consistency_min'],
            cross_fold_stability >= grades_config['A']['cross_fold_stability_min'],
        ]
        
        if grades_config['A']['bucket_monotonicity']:
            a_checks.append(bucket_monotonicity)
        
        if all(a_checks):
            return "A"
        
        # Check Grade B
        b_checks = [
            roc_auc >= grades_config['B']['roc_auc_min'],
            roc_auc < grades_config['A']['roc_auc_min'],
            brier_score <= grades_config['B']['brier_max'],
            accuracy >= grades_config['B']['accuracy_min'],
            regime_consistency >= grades_config['B']['regime_consistency_min'],
            cross_fold_stability >= grades_config['B']['cross_fold_stability_min'],
        ]
        
        if all(b_checks):
            return "B"
        
        # Check Grade C
        c_checks = [
            roc_auc >= grades_config['C']['roc_auc_min'],
            brier_score <= grades_config['C']['brier_max'],
            regime_consistency >= grades_config['C']['regime_consistency_min'],
            cross_fold_stability >= grades_config['C']['cross_fold_stability_min'],
        ]
        
        if all(c_checks):
            return "C"
        
        return "D"
    
    def grade_prediction_engine(
        self,
        fold_results: List[Dict],
        overall_metrics: Dict[str, float],
    ) -> PredictionGradeResult:
        """
        Comprehensive grading of prediction engine.
        
        Args:
            fold_results: List of test results from each walk-forward fold.
                         Each contains: y_true, y_pred_proba, y_true_return, regimes
            overall_metrics: Overall test set metrics.
            
        Returns:
            PredictionGradeResult.
        """
        roc_auc = overall_metrics.get('roc_auc', 0.5)
        brier_score = overall_metrics.get('brier_score', 0.25)
        accuracy = overall_metrics.get('accuracy', 0.5)
        
        # Compute probability bucket stats
        y_true_all = np.concatenate([r['y_true'] for r in fold_results])
        y_pred_proba_all = np.concatenate([r['y_pred_proba'] for r in fold_results])
        y_true_return_all = np.concatenate([r['y_true_return'] for r in fold_results])
        
        bucket_stats = self.compute_probability_buckets(
            y_true_all,
            y_pred_proba_all,
            y_true_return_all,
        )
        
        bucket_monotonicity = self.check_bucket_monotonicity(bucket_stats)
        
        # Per-regime analysis
        if fold_results[0].get('regimes') is not None:
            regimes_all = np.concatenate([r['regimes'] for r in fold_results])
            y_pred_return_all = np.concatenate([r['y_pred_return'] for r in fold_results])
            
            regime_breakdown = self.evaluate_per_regime(
                y_true_all,
                y_pred_proba_all,
                y_pred_return_all,
                y_true_return_all,
                regimes_all,
            )
        else:
            regime_breakdown = {}
        
        regime_consistency = self.compute_regime_consistency(regime_breakdown)
        
        # Cross-fold stability
        fold_metrics_list = [r.get('metrics', {}) for r in fold_results]
        cross_fold_stability = self.compute_cross_fold_stability(fold_metrics_list)
        
        # Assign grade
        grade = self.assign_grade(
            roc_auc=roc_auc,
            brier_score=brier_score,
            accuracy=accuracy,
            bucket_monotonicity=bucket_monotonicity,
            regime_consistency=regime_consistency,
            cross_fold_stability=cross_fold_stability,
        )
        
        # Generate comments
        comments = self._generate_grade_comments(
            grade,
            roc_auc,
            brier_score,
            accuracy,
            bucket_monotonicity,
            regime_consistency,
            cross_fold_stability,
        )
        
        result = PredictionGradeResult(
            grade=grade,
            roc_auc=roc_auc,
            brier_score=brier_score,
            accuracy=accuracy,
            bucket_stats=bucket_stats,
            regime_breakdown=regime_breakdown,
            cross_fold_stability=cross_fold_stability,
            regime_consistency=regime_consistency,
            comments=comments,
            timestamp=pd.Timestamp.now().isoformat(),
        )
        
        return result
    
    def _generate_grade_comments(
        self,
        grade: str,
        roc_auc: float,
        brier_score: float,
        accuracy: float,
        bucket_monotonicity: bool,
        regime_consistency: float,
        cross_fold_stability: float,
    ) -> str:
        """Generate human-readable comments about the grade."""
        comments = []
        
        if grade == "A":
            comments.append("Excellent predictive power out-of-sample.")
            comments.append("Suitable for live trading with confidence.")
        elif grade == "B":
            comments.append("Good predictive power with moderate calibration.")
            comments.append("Suitable for paper trading or incubation.")
        elif grade == "C":
            comments.append("Marginal predictive power, barely above random.")
            comments.append("Recommended: improve features or model architecture.")
        else:  # Grade D
            comments.append("Essentially random or worse.")
            comments.append("DO NOT deploy. Requires significant redesign.")
        
        if roc_auc < 0.55:
            comments.append(f"Warning: ROC-AUC {roc_auc:.3f} is weak.")
        
        if brier_score > 0.24:
            comments.append(f"Warning: Poor probability calibration (Brier {brier_score:.3f}).")
        
        if regime_consistency < 0.60:
            comments.append(f"Warning: Inconsistent across regimes (ratio {regime_consistency:.2f}).")
        
        if cross_fold_stability < 0.70:
            comments.append(f"Warning: Unstable across folds (stability {cross_fold_stability:.2f}).")
        
        if not bucket_monotonicity:
            comments.append("Note: Probability buckets don't show monotonic improvement.")
        
        return " ".join(comments)
    
    def report_grade(self, result: PredictionGradeResult) -> str:
        """
        Generate formatted report string.
        
        Args:
            result: PredictionGradeResult.
            
        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 70)
        report.append("PREDICTION ENGINE GRADING REPORT")
        report.append("=" * 70)
        report.append(f"\nGrade: {result.grade}")
        report.append(f"Timestamp: {result.timestamp}")
        
        report.append(f"\n[Key Metrics]")
        report.append(f"  ROC-AUC:          {result.roc_auc:.4f}")
        report.append(f"  Brier Score:      {result.brier_score:.4f}")
        report.append(f"  Accuracy:         {result.accuracy:.4f}")
        
        report.append(f"\n[Stability & Consistency]")
        report.append(f"  Cross-Fold Stability:  {result.cross_fold_stability:.2f}")
        report.append(f"  Regime Consistency:    {result.regime_consistency:.2f}")
        
        report.append(f"\n[Probability Bucket Analysis]")
        report.append(f"  {'Bucket':<12} {'N':<6} {'Actual':<10} {'Expected':<10} {'Calib Error':<12}")
        report.append("-" * 50)
        for bucket in result.bucket_stats:
            report.append(
                f"  {bucket.bucket_label:<12} {bucket.n_samples:<6} "
                f"{bucket.actual_hit_rate:<10.3f} {bucket.expected_hit_rate:<10.3f} "
                f"{bucket.calibration_error:<12.3f}"
            )
        
        if result.regime_breakdown:
            report.append(f"\n[Per-Regime Analysis]")
            report.append(f"  {'Regime':<15} {'Accuracy':<12} {'ROC-AUC':<12} {'Brier':<12}")
            report.append("-" * 50)
            for regime_name, metrics in result.regime_breakdown.items():
                roc_auc_str = f"{metrics.roc_auc:.3f}" if not np.isnan(metrics.roc_auc) else "N/A"
                report.append(
                    f"  {regime_name:<15} {metrics.accuracy:<12.3f} "
                    f"{roc_auc_str:<12} {metrics.brier_score:<12.3f}"
                )
        
        report.append(f"\n[Comments]")
        report.append(result.comments)
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    grader = PredictionGrader()
    
    # Simulate test results
    y_true = np.random.binomial(1, 0.5, 1000)
    y_pred_proba = y_true * 0.65 + (1 - y_true) * 0.35 + np.random.normal(0, 0.05, 1000)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    y_true_return = y_true * 0.02 - (1 - y_true) * 0.01 + np.random.normal(0, 0.01, 1000)
    y_pred_return = y_pred_proba * 0.02 - (1 - y_pred_proba) * 0.01
    
    metrics = grader.evaluate_single_test_set(
        y_true, y_pred_proba, y_pred_return, y_true_return
    )
    
    print(f"Metrics: {metrics}")
    
    bucket_stats = grader.compute_probability_buckets(y_true, y_pred_proba, y_true_return)
    print(f"Buckets: {len(bucket_stats)} computed")

"""
Signal validation and scoring.
Aggregates raw signals, filters them, and scores their quality.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from utils.types import Signal, ScoredSignal, SignalDirection


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy in current regime."""
    win_rate: float  # 0-1
    profit_factor: float  # avg win / avg loss
    average_return_pct: float
    sharpe_ratio: float
    sample_count: int


class SignalValidator:
    """
    Validates raw signals before scoring.
    Applies basic filters and checks.
    """
    
    def __init__(
        self,
        min_signal_strength: float = 0.3,
        require_confirmation: bool = False,
    ):
        """
        Initialize validator.
        
        Args:
            min_signal_strength: Minimum strength for raw signal (0-1)
            require_confirmation: If True, require multiple alphas to agree
        """
        self.min_signal_strength = min_signal_strength
        self.require_confirmation = require_confirmation
    
    def validate(self, signals: List[Signal]) -> List[Signal]:
        """
        Validate and filter signals.
        
        Args:
            signals: Raw signals from alphas
        
        Returns:
            Validated signals
        """
        # Filter by strength
        valid = [s for s in signals if s.strength >= self.min_signal_strength]
        
        # Group by symbol and direction
        grouped = self._group_signals(valid)
        
        # If requiring confirmation, keep only signals with agreement
        if self.require_confirmation:
            valid = self._filter_confirmed(grouped)
        else:
            valid = [s for symbols in grouped.values() for s in symbols]
        
        return valid
    
    @staticmethod
    def _group_signals(signals: List[Signal]) -> Dict[tuple, List[Signal]]:
        """Group signals by (symbol, direction)."""
        grouped = {}
        for signal in signals:
            key = (signal.symbol, signal.direction)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(signal)
        return grouped
    
    @staticmethod
    def _filter_confirmed(grouped: Dict[tuple, List[Signal]]) -> List[Signal]:
        """Keep only signals with multiple alphas agreeing."""
        confirmed = []
        for signals in grouped.values():
            if len(signals) > 1:  # Require at least 2 alphas
                # Return the strongest signal
                confirmed.append(max(signals, key=lambda s: s.strength))
        return confirmed


class SignalScorer:
    """
    Scores validated signals based on strategy performance and recent metrics.
    Higher score = higher confidence trade.
    """
    
    def __init__(
        self,
        base_confidence: float = 0.5,
        use_performance_metrics: bool = True,
    ):
        """
        Initialize scorer.
        
        Args:
            base_confidence: Base confidence for all signals
            use_performance_metrics: Whether to apply performance weighting
        """
        self.base_confidence = base_confidence
        self.use_performance_metrics = use_performance_metrics
        self.performance_cache: Dict[str, StrategyPerformance] = {}
    
    def update_performance(
        self,
        alpha_name: str,
        performance: StrategyPerformance
    ):
        """Update cached performance metrics for an alpha."""
        self.performance_cache[alpha_name] = performance
    
    def score_signals(self, signals: List[Signal]) -> List[ScoredSignal]:
        """
        Score validated signals.
        
        Args:
            signals: Validated raw signals
        
        Returns:
            List of ScoredSignal objects
        """
        # Group by symbol and direction
        grouped = self._group_by_symbol_direction(signals)
        
        scored_signals = []
        
        for (symbol, direction), signal_group in grouped.items():
            # Aggregate strength from multiple signals
            avg_strength = np.mean([s.strength for s in signal_group])
            
            # Compute confidence
            confidence = self._compute_confidence(signal_group, avg_strength)
            
            # Compute edge (expected return)
            edge = self._compute_edge(signal_group, confidence)
            
            scored = ScoredSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                edge=edge,
                timestamp=signal_group[0].timestamp,
                alpha_names=[s.alpha_name for s in signal_group],
                raw_signals=signal_group
            )
            
            scored_signals.append(scored)
        
        return scored_signals
    
    @staticmethod
    def _group_by_symbol_direction(signals: List[Signal]) -> Dict[tuple, List[Signal]]:
        """Group signals by (symbol, direction)."""
        grouped = {}
        for signal in signals:
            key = (signal.symbol, signal.direction)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(signal)
        return grouped
    
    def _compute_confidence(
        self,
        signal_group: List[Signal],
        avg_strength: float
    ) -> float:
        """
        Compute confidence score for a signal group.
        
        Factors:
        - Number of alphas agreeing
        - Individual signal strengths
        - Historical performance of alphas
        """
        # Base confidence from signal strengths
        confidence = self.base_confidence + (avg_strength * 0.3)
        
        # Boost for multiple alphas agreeing
        num_alphas = len(set(s.alpha_name for s in signal_group))
        agreement_boost = min(0.2, num_alphas * 0.1)
        confidence += agreement_boost
        
        # Apply performance weighting if enabled
        if self.use_performance_metrics:
            perf_weight = self._get_performance_weight(signal_group)
            confidence *= (0.5 + perf_weight)
        
        return min(1.0, confidence)
    
    def _compute_edge(self, signal_group: List[Signal], confidence: float) -> float:
        """
        Compute expected edge (return per unit risk).
        
        Simple approach: confidence * average strength
        More sophisticated: could use performance metrics
        """
        avg_strength = np.mean([s.strength for s in signal_group])
        edge = confidence * avg_strength
        
        return edge
    
    def _get_performance_weight(self, signal_group: List[Signal]) -> float:
        """
        Get performance-based weight for signal group.
        
        Returns weight in [0, 1] where 1 = strong historical performance
        """
        weights = []
        
        for signal in signal_group:
            if signal.alpha_name in self.performance_cache:
                perf = self.performance_cache[signal.alpha_name]
                # Simple weighting: higher win rate = higher weight
                weight = perf.win_rate * (1 + max(0, perf.sharpe_ratio) / 10)
                weights.append(min(1.0, weight))
            else:
                weights.append(0.5)  # Neutral if no history
        
        return np.mean(weights) if weights else 0.5


class SignalFilter:
    """
    Filters scored signals based on configuration thresholds.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        min_edge: float = 0.01,
    ):
        """
        Initialize filter.
        
        Args:
            min_confidence: Minimum confidence (0-1)
            min_edge: Minimum edge (0-1, representing % expected return)
        """
        self.min_confidence = min_confidence
        self.min_edge = min_edge
    
    def filter(self, scored_signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """
        Filter scored signals by thresholds.
        
        Returns:
            Filtered list of high-quality signals
        """
        filtered = []
        
        for signal in scored_signals:
            if signal.confidence >= self.min_confidence and signal.edge >= self.min_edge:
                filtered.append(signal)
        
        return filtered


class SignalProcessor:
    """
    Orchestrates the full signal pipeline:
    Raw -> Validated -> Scored -> Filtered
    """
    
    def __init__(
        self,
        validator: Optional[SignalValidator] = None,
        scorer: Optional[SignalScorer] = None,
        filter: Optional[SignalFilter] = None,
    ):
        """
        Initialize signal processor.
        
        Args:
            validator: Signal validator (if None, no validation)
            scorer: Signal scorer (if None, basic scoring)
            filter: Signal filter (if None, minimal filtering)
        """
        self.validator = validator or SignalValidator()
        self.scorer = scorer or SignalScorer()
        self.filter = filter or SignalFilter()
    
    def process(self, raw_signals: List[Signal]) -> List[ScoredSignal]:
        """
        Process raw signals through full pipeline.
        
        Returns:
            Filtered, scored signals ready for risk/portfolio engines
        """
        # Step 1: Validate
        validated = self.validator.validate(raw_signals)
        
        # Step 2: Score
        scored = self.scorer.score_signals(validated)
        
        # Step 3: Filter
        filtered = self.filter.filter(scored)
        
        return filtered

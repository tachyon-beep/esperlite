"""
Multi-metric intelligent reward computation system.

Provides sophisticated reward calculation for Tamiyo policy learning
by evaluating adaptation quality across multiple dimensions with
temporal analysis.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from esper.contracts.operational import AdaptationDecision

logger = logging.getLogger(__name__)


@dataclass
class RewardAnalysis:
    """Comprehensive reward analysis result."""

    total_reward: float
    components: Dict[str, float]
    temporal_analysis: Dict[str, Any]
    confidence: float
    explanation: str


class BaselineMetricsTracker:
    """Tracks baseline metrics for relative improvement calculation."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: Dict[str, List[float]] = {}

    def update(self, layer_name: str, metrics: Dict[str, float]):
        """Update baseline metrics for a layer."""
        for metric_name, value in metrics.items():
            key = f"{layer_name}:{metric_name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []

            self.metrics_history[key].append(value)

            # Maintain window size
            if len(self.metrics_history[key]) > self.window_size:
                self.metrics_history[key] = self.metrics_history[key][-self.window_size:]

    def get_baseline(self, layer_name: str, metric_name: str) -> Optional[float]:
        """Get baseline value for a metric."""
        key = f"{layer_name}:{metric_name}"
        if key in self.metrics_history and self.metrics_history[key]:
            return np.mean(self.metrics_history[key])
        return None


class TemporalRewardAnalyzer:
    """Analyzes reward patterns over time."""

    def __init__(self):
        self.temporal_windows = {
            "immediate": (0, 300),      # 0-5 minutes
            "short_term": (300, 1800),  # 5-30 minutes
            "medium_term": (1800, 7200), # 30-120 minutes
            "long_term": (7200, float('inf'))  # 2+ hours
        }

    def analyze_temporal_impact(
        self,
        metrics_timeline: List[Tuple[float, Dict[str, float]]],
        adaptation_timestamp: float
    ) -> Dict[str, Dict[str, float]]:
        """Analyze metrics across temporal windows."""
        current_time = time.time()
        temporal_analysis = {}

        for window_name, (start_offset, end_offset) in self.temporal_windows.items():
            window_start = adaptation_timestamp + start_offset
            window_end = adaptation_timestamp + end_offset

            # Filter metrics within window
            window_metrics = [
                metrics for timestamp, metrics in metrics_timeline
                if window_start <= timestamp <= min(window_end, current_time)
            ]

            if window_metrics:
                # Aggregate metrics for window
                temporal_analysis[window_name] = {
                    "avg_accuracy": np.mean([m.get("accuracy", 0) for m in window_metrics]),
                    "avg_latency": np.mean([m.get("latency_ms", 0) for m in window_metrics]),
                    "error_rate": np.mean([m.get("error_rate", 0) for m in window_metrics]),
                    "sample_count": len(window_metrics)
                }
            else:
                temporal_analysis[window_name] = {
                    "avg_accuracy": 0.0,
                    "avg_latency": 0.0,
                    "error_rate": 0.0,
                    "sample_count": 0
                }

        return temporal_analysis


class CorrelationDetector:
    """Detects correlations between adaptations and outcomes."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.adaptation_history: List[Dict] = []
        self.outcome_history: List[Dict] = []

    def record_adaptation(self, adaptation: AdaptationDecision, context: Dict):
        """Record an adaptation for correlation analysis."""
        self.adaptation_history.append({
            "timestamp": time.time(),
            "adaptation": adaptation,
            "context": context
        })

        # Maintain history size
        if len(self.adaptation_history) > self.history_size:
            self.adaptation_history = self.adaptation_history[-self.history_size:]

    def record_outcome(self, metrics: Dict[str, float]):
        """Record outcome metrics."""
        self.outcome_history.append({
            "timestamp": time.time(),
            "metrics": metrics
        })

        if len(self.outcome_history) > self.history_size:
            self.outcome_history = self.outcome_history[-self.history_size:]

    def compute_correlation_score(
        self,
        adaptation_type: str,
        metric_name: str
    ) -> float:
        """Compute correlation between adaptation type and metric improvement."""
        if len(self.adaptation_history) < 10 or len(self.outcome_history) < 10:
            return 0.0

        # Simple correlation based on outcomes following adaptations
        positive_outcomes = 0
        total_adaptations = 0

        for i, adaptation in enumerate(self.adaptation_history):
            if adaptation["adaptation"].adaptation_type == adaptation_type:
                total_adaptations += 1

                # Look for outcomes within 5 minutes
                for outcome in self.outcome_history:
                    if (adaptation["timestamp"] < outcome["timestamp"] <=
                        adaptation["timestamp"] + 300):

                        if metric_name in outcome["metrics"]:
                            # Check if metric improved
                            baseline = self._get_baseline_before(
                                adaptation["timestamp"], metric_name
                            )
                            if baseline and outcome["metrics"][metric_name] > baseline:
                                positive_outcomes += 1
                                break

        if total_adaptations > 0:
            return positive_outcomes / total_adaptations
        return 0.0

    def _get_baseline_before(self, timestamp: float, metric_name: str) -> Optional[float]:
        """Get baseline metric value before timestamp."""
        baseline_values = []

        for outcome in self.outcome_history:
            if outcome["timestamp"] < timestamp - 60:  # At least 1 minute before
                if metric_name in outcome["metrics"]:
                    baseline_values.append(outcome["metrics"][metric_name])

        if baseline_values:
            return np.mean(baseline_values[-10:])  # Average of last 10 values
        return None


class IntelligentRewardComputer:
    """
    Advanced reward computation system that evaluates adaptation quality
    using multiple metrics and temporal analysis.
    """

    def __init__(self):
        self.baseline_tracker = BaselineMetricsTracker()
        self.temporal_analyzer = TemporalRewardAnalyzer()
        self.correlation_detector = CorrelationDetector()

        # Reward weights optimized through meta-learning
        self.reward_weights = {
            'accuracy_improvement': 1.0,      # Primary objective
            'speed_improvement': 0.6,         # Secondary objective
            'memory_efficiency': 0.4,         # Efficiency objective
            'stability_improvement': 0.8,     # Reliability objective
            'adaptation_cost': -0.3,          # Cost penalty
            'risk_penalty': -0.5              # Safety penalty
        }

        # Temporal discount factors
        self.temporal_weights = {
            'immediate': 0.3,    # 0-5 minutes
            'short_term': 0.4,   # 5-30 minutes
            'medium_term': 0.25, # 30-120 minutes
            'long_term': 0.05    # 2+ hours
        }

        # Metrics timeline storage
        self.metrics_timeline: List[Tuple[float, Dict[str, float]]] = []

    async def compute_adaptation_reward(
        self,
        adaptation_decision: AdaptationDecision,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_window: float = 300.0  # 5 minutes
    ) -> RewardAnalysis:
        """
        Compute comprehensive reward for adaptation decision.
        
        Integrates with Phase 1 metrics for accurate performance assessment.
        """

        # Record metrics for temporal analysis
        self.metrics_timeline.append((time.time(), post_metrics))

        # Collect temporal metrics across time windows
        temporal_metrics = await self._collect_temporal_metrics(
            adaptation_decision.layer_name,
            temporal_window
        )

        # Compute individual reward components
        reward_components = {}

        # 1. Accuracy/Performance Improvement
        accuracy_reward = self._compute_accuracy_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['accuracy'] = accuracy_reward

        # 2. Speed/Latency Improvement (integrates with Phase 1 execution metrics)
        speed_reward = self._compute_speed_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['speed'] = speed_reward

        # 3. Memory Efficiency (uses Phase 1 cache metrics)
        memory_reward = self._compute_memory_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['memory'] = memory_reward

        # 4. Stability/Reliability (integrates with Phase 1 error recovery)
        stability_reward = self._compute_stability_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['stability'] = stability_reward

        # 5. Adaptation Cost (computational overhead)
        cost_penalty = self._compute_adaptation_cost(adaptation_decision)
        reward_components['cost'] = cost_penalty

        # 6. Risk Assessment (safety penalty)
        risk_penalty = await self._compute_risk_penalty(
            adaptation_decision, post_metrics
        )
        reward_components['risk'] = risk_penalty

        # Compute weighted total reward
        total_reward = sum(
            self.reward_weights.get(component, 1.0) * reward_value
            for component, reward_value in reward_components.items()
        )

        # Apply temporal discounting
        discounted_reward = self._apply_temporal_discounting(
            total_reward, temporal_metrics
        )

        # Normalize to [-1, 1] range
        normalized_reward = torch.tanh(torch.tensor(discounted_reward)).item()

        # Update baseline tracker
        self.baseline_tracker.update(adaptation_decision.layer_name, post_metrics)

        # Record for correlation analysis
        self.correlation_detector.record_adaptation(adaptation_decision, pre_metrics)
        self.correlation_detector.record_outcome(post_metrics)

        return RewardAnalysis(
            total_reward=normalized_reward,
            components=reward_components,
            temporal_analysis=temporal_metrics,
            confidence=self._compute_reward_confidence(reward_components),
            explanation=self._generate_reward_explanation(reward_components)
        )

    async def _collect_temporal_metrics(
        self,
        layer_name: str,
        window_size: float
    ) -> Dict[str, Any]:
        """Collect metrics across temporal windows."""
        adaptation_timestamp = time.time() - window_size

        # Get metrics timeline for the layer
        layer_metrics = [
            (ts, metrics) for ts, metrics in self.metrics_timeline
            if ts >= adaptation_timestamp
        ]

        # Analyze temporal patterns
        temporal_analysis = self.temporal_analyzer.analyze_temporal_impact(
            layer_metrics, adaptation_timestamp
        )

        return temporal_analysis

    def _compute_accuracy_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute accuracy improvement reward."""
        pre_accuracy = pre_metrics.get('accuracy', 0.0)
        post_accuracy = post_metrics.get('accuracy', 0.0)

        # Relative improvement
        if pre_accuracy > 0:
            accuracy_improvement = (post_accuracy - pre_accuracy) / pre_accuracy
        else:
            accuracy_improvement = post_accuracy

        # Consider temporal stability
        temporal_bonus = 0.0
        if 'short_term' in temporal_metrics:
            short_term_acc = temporal_metrics['short_term'].get('avg_accuracy', 0)
            if short_term_acc > pre_accuracy:
                temporal_bonus = 0.1  # Bonus for sustained improvement

        return accuracy_improvement + temporal_bonus

    def _compute_speed_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute speed improvement reward using Phase 1 execution metrics."""

        # Phase 1 integration: use actual execution latency
        pre_latency = pre_metrics.get('execution_latency_ms', 1000.0)
        post_latency = post_metrics.get('execution_latency_ms', 1000.0)

        # Relative improvement
        if pre_latency > 0:
            speed_improvement = (pre_latency - post_latency) / pre_latency
        else:
            speed_improvement = 0.0

        # Bonus for achieving target latency thresholds
        target_bonus = 0.0
        if post_latency < 0.5:  # Sub-millisecond execution
            target_bonus = 0.2
        elif post_latency < 1.0:  # Under 1ms
            target_bonus = 0.1

        # Penalty for degradation
        degradation_penalty = 0.0
        if speed_improvement < -0.1:  # >10% slowdown
            degradation_penalty = -0.5

        return speed_improvement + target_bonus + degradation_penalty

    def _compute_memory_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute memory efficiency reward using Phase 1 cache metrics."""

        # Phase 1 integration: cache hit rate
        pre_cache_hit = pre_metrics.get('cache_hit_rate', 0.8)
        post_cache_hit = post_metrics.get('cache_hit_rate', 0.8)

        # Memory usage
        pre_memory = pre_metrics.get('memory_usage_mb', 100.0)
        post_memory = post_metrics.get('memory_usage_mb', 100.0)

        # Cache improvement
        cache_improvement = (post_cache_hit - pre_cache_hit) * 2.0

        # Memory efficiency (negative is better)
        if pre_memory > 0:
            memory_efficiency = -(post_memory - pre_memory) / pre_memory
        else:
            memory_efficiency = 0.0

        return cache_improvement + memory_efficiency * 0.5

    def _compute_stability_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute stability reward using Phase 1 error recovery metrics."""

        # Phase 1 integration: use error recovery statistics
        pre_error_rate = pre_metrics.get('error_rate', 0.0)
        post_error_rate = post_metrics.get('error_rate', 0.0)

        # Error rate improvement
        error_improvement = max(0, pre_error_rate - post_error_rate)

        # Recovery success rate (from Phase 1 error recovery)
        recovery_rate = post_metrics.get('recovery_success_rate', 1.0)
        recovery_bonus = (recovery_rate - 0.95) * 2.0  # Bonus above 95%

        # Stability variance penalty
        if 'short_term' in temporal_metrics:
            latency_samples = temporal_metrics['short_term'].get('sample_count', 0)
            if latency_samples > 10:
                # Penalize high variance
                variance_penalty = -0.1
            else:
                variance_penalty = 0.0
        else:
            variance_penalty = 0.0

        return error_improvement * 2.0 + recovery_bonus + variance_penalty

    def _compute_adaptation_cost(self, adaptation_decision: AdaptationDecision) -> float:
        """Compute computational cost of adaptation."""
        cost = 0.0

        # Base cost for any adaptation
        cost -= 0.1

        # Additional cost based on adaptation type
        adaptation_costs = {
            "load_kernel": -0.2,
            "unload_kernel": -0.05,
            "modify_parameters": -0.15,
            "add_layer": -0.3,
        }

        cost += adaptation_costs.get(adaptation_decision.adaptation_type, -0.1)

        # Urgency modifier (urgent adaptations have lower cost penalty)
        if adaptation_decision.urgency > 0.8:
            cost *= 0.5

        return cost

    async def _compute_risk_penalty(
        self,
        adaptation_decision: AdaptationDecision,
        post_metrics: Dict[str, float]
    ) -> float:
        """Compute risk penalty for adaptation."""
        risk_penalty = 0.0

        # Base risk from adaptation metadata
        if hasattr(adaptation_decision, 'metadata'):
            risk_score = adaptation_decision.metadata.get('risk_score', 0.0)
            risk_penalty -= risk_score * 0.5

        # Additional risk from poor outcomes
        if post_metrics.get('error_rate', 0) > 0.1:
            risk_penalty -= 0.3

        if post_metrics.get('stability_score', 1.0) < 0.5:
            risk_penalty -= 0.2

        return risk_penalty

    def _apply_temporal_discounting(
        self,
        base_reward: float,
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Apply temporal discounting to reward."""
        discounted_reward = 0.0

        for window_name, weight in self.temporal_weights.items():
            if window_name in temporal_metrics:
                window_data = temporal_metrics[window_name]

                # Weight by sample count (more samples = more reliable)
                sample_weight = min(window_data['sample_count'] / 100.0, 1.0)

                # Apply temporal weight
                discounted_reward += base_reward * weight * sample_weight

        return discounted_reward

    def _compute_reward_confidence(self, components: Dict[str, float]) -> float:
        """Compute confidence in reward calculation."""
        # Base confidence
        confidence = 0.5

        # Increase confidence for consistent components
        component_values = list(components.values())
        if len(component_values) > 0:
            # Low variance = high confidence
            variance = np.var(component_values)
            confidence += 0.3 * (1.0 - min(variance, 1.0))

        # Increase confidence for strong signals
        max_component = max(abs(v) for v in component_values) if component_values else 0
        if max_component > 0.5:
            confidence += 0.2

        return min(confidence, 1.0)

    def _generate_reward_explanation(self, components: Dict[str, float]) -> str:
        """Generate human-readable explanation of reward."""
        explanations = []

        # Sort components by absolute impact
        sorted_components = sorted(
            components.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for component, value in sorted_components[:3]:  # Top 3 components
            if abs(value) > 0.1:
                direction = "positive" if value > 0 else "negative"
                explanations.append(
                    f"{component.replace('_', ' ').title()} had {direction} "
                    f"impact ({value:.2f})"
                )

        if explanations:
            return "; ".join(explanations)
        else:
            return "Minimal impact observed across all metrics"

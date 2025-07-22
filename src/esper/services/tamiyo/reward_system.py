"""
Multi-Metric Intelligent Reward System for Tamiyo Policy Training

This module implements a comprehensive reward computation system that evaluates
adaptation decisions across multiple dimensions: accuracy, speed, memory usage,
stability, and safety. It integrates with Phase 1 execution metrics and provides
sophisticated correlation analysis for policy learning.

Key Features:
- Multi-dimensional reward computation (accuracy, speed, memory, stability, safety)
- Temporal reward discounting with immediate to long-term horizons
- Correlation detection between actions and outcomes
- Phase 1 integration for real execution metrics
- Advanced statistical analysis and trend detection
"""

import logging
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from esper.contracts.operational import AdaptationDecision

from .health_collector import HealthSignal
from .model_graph_builder import ModelGraphState

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Components that contribute to the overall reward signal."""

    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    STABILITY = "stability"
    SAFETY = "safety"
    INNOVATION = "innovation"
    CONSISTENCY = "consistency"


@dataclass
class RewardMetrics:
    """Comprehensive metrics for reward computation."""

    # Primary performance metrics
    accuracy_improvement: float = 0.0
    speed_improvement: float = 0.0
    memory_efficiency: float = 0.0
    stability_score: float = 0.0
    safety_score: float = 1.0

    # Secondary metrics
    innovation_score: float = 0.0  # Novelty of the adaptation
    consistency_score: float = 0.0  # Consistency with past decisions

    # Temporal metrics
    immediate_impact: float = 0.0  # <1 minute impact
    short_term_impact: float = 0.0  # 1-10 minute impact
    medium_term_impact: float = 0.0  # 10-60 minute impact
    long_term_impact: float = 0.0  # >60 minute impact

    # Meta-metrics
    confidence_level: float = 0.0
    uncertainty: float = 0.0
    risk_assessment: float = 0.0

    # Context
    timestamp: float = field(default_factory=time.time)
    decision_context: Optional[Dict[str, Any]] = None


@dataclass
class RewardConfig:
    """Configuration for reward computation system."""

    # Component weights
    accuracy_weight: float = 0.25
    speed_weight: float = 0.20
    memory_weight: float = 0.15
    stability_weight: float = 0.20
    safety_weight: float = 0.15
    innovation_weight: float = 0.05

    # Temporal discounting
    immediate_discount: float = 1.0  # No discount for immediate effects
    short_term_discount: float = 0.9  # 10% discount for short-term
    medium_term_discount: float = 0.8  # 20% discount for medium-term
    long_term_discount: float = 0.7  # 30% discount for long-term

    # Thresholds
    min_improvement_threshold: float = 0.01  # Minimum improvement to consider positive
    safety_failure_penalty: float = -2.0  # Large penalty for safety violations
    stability_failure_penalty: float = -1.0  # Penalty for instability

    # Correlation analysis
    correlation_window_size: int = 100
    min_correlation_samples: int = 10
    correlation_significance_threshold: float = 0.05

    # Adaptive parameters
    enable_adaptive_weights: bool = True
    weight_adaptation_rate: float = 0.01
    weight_momentum: float = 0.9


class PerformanceTracker:
    """Tracks performance metrics over time for reward computation."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.baseline_metrics = {}
        self.trend_analyzer = TrendAnalyzer()

    def update_metrics(self, metrics: Dict[str, float], timestamp: float = None):
        """Update performance metrics with new measurements."""
        if timestamp is None:
            timestamp = time.time()

        self.metrics_history.append({"timestamp": timestamp, "metrics": metrics.copy()})

        # Update baseline if we have enough history
        if len(self.metrics_history) >= 50:
            self._update_baseline()

    def _update_baseline(self):
        """Update baseline metrics from recent history."""
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements

        # Compute baseline as median of recent measurements
        baseline = {}
        if recent_metrics:
            metric_keys = recent_metrics[0]["metrics"].keys()
            for key in metric_keys:
                values = [m["metrics"].get(key, 0.0) for m in recent_metrics]
                baseline[key] = np.median(values)

        self.baseline_metrics = baseline

    def get_improvement(self, metric_name: str, current_value: float) -> float:
        """Calculate improvement relative to baseline."""
        if metric_name not in self.baseline_metrics:
            return 0.0

        baseline = self.baseline_metrics[metric_name]
        if baseline == 0:
            return 0.0

        # Calculate relative improvement
        improvement = (current_value - baseline) / abs(baseline)
        return improvement

    def get_trend(self, metric_name: str, window_size: int = 20) -> float:
        """Get trend direction for a metric (-1 to 1)."""
        if len(self.metrics_history) < window_size:
            return 0.0

        recent_data = list(self.metrics_history)[-window_size:]
        values = [d["metrics"].get(metric_name, 0.0) for d in recent_data]

        return self.trend_analyzer.compute_trend(values)


class TrendAnalyzer:
    """Analyzes trends in time series data."""

    def compute_trend(self, values: List[float]) -> float:
        """Compute trend direction from -1 (decreasing) to 1 (increasing)."""
        if len(values) < 3:
            return 0.0

        # Use linear regression to find trend
        x = np.arange(len(values))
        y = np.array(values)

        # Handle constant values
        if np.std(y) < 1e-8:
            return 0.0

        try:
            slope, _, _, p_value, _ = stats.linregress(x, y)

            # Only consider significant trends
            if p_value > 0.05:
                return 0.0

            # Normalize slope to [-1, 1] range
            trend_strength = np.tanh(slope * len(values) / (np.std(y) + 1e-8))
            return float(trend_strength)

        except Exception:
            return 0.0


class CorrelationAnalyzer:
    """Analyzes correlations between decisions and outcomes."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.decision_outcome_pairs: deque = deque(maxlen=window_size)

    def add_decision_outcome(
        self,
        decision: AdaptationDecision,
        outcome_metrics: RewardMetrics,
        delay_minutes: float = 0.0,
    ):
        """Record a decision-outcome pair for correlation analysis."""
        decision_features = self._extract_decision_features(decision)
        outcome_features = self._extract_outcome_features(outcome_metrics)

        self.decision_outcome_pairs.append(
            {
                "decision": decision_features,
                "outcome": outcome_features,
                "delay": delay_minutes,
                "timestamp": time.time(),
            }
        )

    def _extract_decision_features(
        self, decision: AdaptationDecision
    ) -> Dict[str, float]:
        """Extract numerical features from adaptation decision."""
        return {
            "confidence": decision.confidence,
            "urgency": decision.urgency,
            "layer_depth": hash(decision.layer_name)
            % 100
            / 100.0,  # Crude layer encoding
            "adaptation_type": hash(str(decision.blueprint_request)) % 100 / 100.0,
        }

    def _extract_outcome_features(self, metrics: RewardMetrics) -> Dict[str, float]:
        """Extract numerical features from reward metrics."""
        return {
            "accuracy": metrics.accuracy_improvement,
            "speed": metrics.speed_improvement,
            "memory": metrics.memory_efficiency,
            "stability": metrics.stability_score,
            "safety": metrics.safety_score,
        }

    def compute_correlations(self) -> Dict[str, Dict[str, float]]:
        """Compute correlations between decision and outcome features."""
        if len(self.decision_outcome_pairs) < 10:
            return {}

        correlations = {}

        # Extract all decision and outcome features
        decision_data = defaultdict(list)
        outcome_data = defaultdict(list)

        for pair in self.decision_outcome_pairs:
            for key, value in pair["decision"].items():
                decision_data[key].append(value)

            for key, value in pair["outcome"].items():
                outcome_data[key].append(value)

        # Compute correlations between each decision feature and outcome feature
        for decision_feature in decision_data:
            correlations[decision_feature] = {}

            for outcome_feature in outcome_data:
                try:
                    correlation, p_value = stats.pearsonr(
                        decision_data[decision_feature], outcome_data[outcome_feature]
                    )

                    # Only record significant correlations
                    if p_value < 0.05 and not np.isnan(correlation):
                        correlations[decision_feature][outcome_feature] = correlation

                except Exception:
                    pass

        return correlations

    def get_mutual_information(self) -> Dict[str, float]:
        """Compute mutual information between decisions and outcomes."""
        if len(self.decision_outcome_pairs) < 20:
            return {}

        mutual_info = {}

        # Get decision confidence and outcome success
        confidences = []
        successes = []

        for pair in self.decision_outcome_pairs:
            confidences.append(pair["decision"]["confidence"])

            # Define success as positive overall outcome
            outcome = pair["outcome"]
            success = (
                outcome["accuracy"] > 0
                or outcome["speed"] > 0
                or outcome["stability"] > 0.7
            )
            successes.append(1 if success else 0)

        # Discretize confidence for mutual information
        confidence_bins = np.digitize(confidences, bins=[0.3, 0.5, 0.7, 0.9])

        try:
            # Simple mutual information approximation
            mi_score = self._compute_mutual_information(confidence_bins, successes)
            mutual_info["confidence_success"] = mi_score
        except Exception:
            pass

        return mutual_info

    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple mutual information computation."""
        try:
            # Convert to discrete values
            x_unique = np.unique(x)
            y_unique = np.unique(y)

            if len(x_unique) < 2 or len(y_unique) < 2:
                return 0.0

            # Compute joint and marginal distributions
            xy_counts = {}
            x_counts = {}
            y_counts = {}
            n = len(x)

            for i in range(n):
                xi, yi = x[i], y[i]
                xy_counts[(xi, yi)] = xy_counts.get((xi, yi), 0) + 1
                x_counts[xi] = x_counts.get(xi, 0) + 1
                y_counts[yi] = y_counts.get(yi, 0) + 1

            # Compute mutual information
            mi = 0.0
            for (xi, yi), count in xy_counts.items():
                pxy = count / n
                px = x_counts[xi] / n
                py = y_counts[yi] / n

                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * np.log(pxy / (px * py))

            return float(mi)

        except Exception:
            return 0.0


class MultiMetricRewardSystem:
    """
    Comprehensive reward system that evaluates adaptation decisions across
    multiple performance dimensions with temporal analysis and correlation detection.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.correlation_analyzer = CorrelationAnalyzer(
            window_size=self.config.correlation_window_size
        )

        # Reward history
        self.reward_history: deque = deque(maxlen=1000)
        self.decision_history: deque = deque(maxlen=1000)

        # Adaptive weight system
        if self.config.enable_adaptive_weights:
            self.weight_optimizer = AdaptiveWeightOptimizer(self.config)
        else:
            self.weight_optimizer = None

        # Statistics
        self.stats = {
            "total_rewards_computed": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "safety_violations": 0,
            "correlation_updates": 0,
        }

        logger.info("MultiMetricRewardSystem initialized")

    async def compute_reward(
        self,
        decision: AdaptationDecision,
        graph_state: ModelGraphState,
        execution_metrics: Optional[Dict[str, Any]] = None,
        health_signals: Optional[List[HealthSignal]] = None,
    ) -> Tuple[float, RewardMetrics]:
        """
        Compute comprehensive reward for an adaptation decision.

        Args:
            decision: The adaptation decision to evaluate
            graph_state: Current model graph state
            execution_metrics: Metrics from Phase 1 execution
            health_signals: Recent health signals for context

        Returns:
            (reward_value, detailed_metrics)
        """
        logger.debug("Computing reward for decision: %s", decision.layer_name)

        # Extract current performance metrics
        current_metrics = self._extract_performance_metrics(
            graph_state, execution_metrics, health_signals
        )

        # Update performance tracking
        self.performance_tracker.update_metrics(current_metrics)

        # Compute reward components
        reward_metrics = RewardMetrics(
            decision_context={
                "layer_name": decision.layer_name,
                "confidence": decision.confidence,
                "urgency": decision.urgency,
            }
        )

        # Primary metrics
        reward_metrics.accuracy_improvement = self._compute_accuracy_reward(
            current_metrics, decision
        )
        reward_metrics.speed_improvement = self._compute_speed_reward(
            current_metrics, decision
        )
        reward_metrics.memory_efficiency = self._compute_memory_reward(
            current_metrics, decision
        )
        reward_metrics.stability_score = self._compute_stability_reward(
            current_metrics, graph_state, decision
        )
        reward_metrics.safety_score = self._compute_safety_reward(decision, graph_state)

        # Secondary metrics
        reward_metrics.innovation_score = self._compute_innovation_reward(decision)
        reward_metrics.consistency_score = self._compute_consistency_reward(decision)

        # Temporal impact (estimated based on decision characteristics)
        temporal_impacts = self._estimate_temporal_impacts(decision, current_metrics)
        reward_metrics.immediate_impact = temporal_impacts["immediate"]
        reward_metrics.short_term_impact = temporal_impacts["short_term"]
        reward_metrics.medium_term_impact = temporal_impacts["medium_term"]
        reward_metrics.long_term_impact = temporal_impacts["long_term"]

        # Meta-metrics
        reward_metrics.confidence_level = decision.confidence
        reward_metrics.uncertainty = decision.metadata.get("uncertainty", 0.0)
        reward_metrics.risk_assessment = self._assess_risk(decision, graph_state)

        # Combine components into final reward
        final_reward = self._combine_reward_components(reward_metrics)

        # Apply safety and stability penalties
        final_reward = self._apply_safety_penalties(final_reward, reward_metrics)

        # Update correlation analysis
        self.correlation_analyzer.add_decision_outcome(decision, reward_metrics)

        # Store for history and learning
        self.reward_history.append(final_reward)
        self.decision_history.append(
            {
                "decision": decision,
                "reward": final_reward,
                "metrics": reward_metrics,
                "timestamp": time.time(),
            }
        )

        # Update adaptive weights
        if self.weight_optimizer:
            await self.weight_optimizer.update_weights(
                decision, reward_metrics, final_reward
            )

        # Update statistics
        self.stats["total_rewards_computed"] += 1
        if final_reward > 0:
            self.stats["positive_rewards"] += 1
        else:
            self.stats["negative_rewards"] += 1

        if reward_metrics.safety_score < 0.5:
            self.stats["safety_violations"] += 1

        logger.debug(
            f"Computed reward: {final_reward:.3f} "
            f"(acc:{reward_metrics.accuracy_improvement:.3f}, "
            f"speed:{reward_metrics.speed_improvement:.3f}, "
            f"safety:{reward_metrics.safety_score:.3f})"
        )

        return final_reward, reward_metrics

    def _extract_performance_metrics(
        self,
        graph_state: ModelGraphState,
        execution_metrics: Optional[Dict[str, Any]],
        health_signals: Optional[List[HealthSignal]],
    ) -> Dict[str, float]:
        """Extract current performance metrics from available data."""
        metrics = {}

        # From graph state
        if graph_state.global_metrics:
            metrics.update(
                {
                    "model_health": graph_state.global_metrics.get(
                        "overall_health", 0.5
                    ),
                    "stability": graph_state.global_metrics.get("stability", 0.5),
                    "error_rate": graph_state.global_metrics.get("error_rate", 0.0),
                    "gradient_health": graph_state.global_metrics.get("gradient_health", 0.5),
                    "training_stability": graph_state.global_metrics.get("training_stability", 0.5),
                    "gradient_norm": graph_state.global_metrics.get("avg_gradient_norm", 1.0),
                    "system_efficiency": graph_state.global_metrics.get("system_efficiency", 0.5),
                }
            )

        # From execution metrics (Phase 1 integration)
        if execution_metrics:
            metrics.update(
                {
                    "execution_latency": execution_metrics.get("avg_latency_ms", 0.0),
                    "cache_hit_rate": execution_metrics.get("cache_hit_rate", 0.0),
                    "memory_usage_mb": execution_metrics.get("memory_usage_mb", 0.0),
                    "success_rate": execution_metrics.get("success_rate", 1.0),
                }
            )

        # From health signals
        if health_signals:
            recent_signals = list(health_signals)[-100:]  # Last 100 signals
            if recent_signals:
                metrics["avg_health_score"] = np.mean([s.health_score for s in recent_signals])
                metrics["avg_error_count"] = np.mean([s.error_count for s in recent_signals])
                metrics["avg_latency"] = np.mean([s.execution_latency for s in recent_signals])
                
                # Gradient metrics from health signals
                metrics["avg_gradient_norm"] = np.mean([s.gradient_norm for s in recent_signals])
                metrics["avg_gradient_variance"] = np.mean([s.gradient_variance for s in recent_signals])
                metrics["avg_gradient_stability"] = np.mean([s.gradient_sign_stability for s in recent_signals])
                metrics["avg_param_norm_ratio"] = np.mean([s.param_norm_ratio for s in recent_signals])
                
                # Performance metrics
                metrics["avg_cache_hit_rate"] = np.mean([s.cache_hit_rate for s in recent_signals])
                total_exec = sum(s.total_executions for s in recent_signals)
                metrics["total_executions"] = total_exec

        return metrics

    def _compute_accuracy_reward(
        self, current_metrics: Dict[str, float], decision: AdaptationDecision
    ) -> float:
        """Compute accuracy-based reward component."""
        model_health = current_metrics.get("model_health", 0.5)
        success_rate = current_metrics.get("success_rate", 1.0)

        # Reward improvement in model health and success rate
        health_improvement = self.performance_tracker.get_improvement(
            "model_health", model_health
        )
        success_improvement = self.performance_tracker.get_improvement(
            "success_rate", success_rate
        )

        accuracy_reward = (health_improvement + success_improvement) / 2.0

        # Boost reward for confident decisions that target problematic areas
        if decision.confidence > 0.7 and decision.urgency > 0.5:
            accuracy_reward *= 1.2

        return float(np.clip(accuracy_reward, -1.0, 1.0))

    def _compute_speed_reward(
        self, current_metrics: Dict[str, float], _decision: AdaptationDecision
    ) -> float:
        """Compute speed-based reward component."""
        latency = current_metrics.get("execution_latency", 0.0)
        cache_hit_rate = current_metrics.get("cache_hit_rate", 0.0)

        # Lower latency and higher cache hit rate are better
        latency_improvement = -self.performance_tracker.get_improvement(
            "execution_latency", latency
        )
        cache_improvement = self.performance_tracker.get_improvement(
            "cache_hit_rate", cache_hit_rate
        )

        speed_reward = (latency_improvement + cache_improvement) / 2.0

        return float(np.clip(speed_reward, -1.0, 1.0))

    def _compute_memory_reward(
        self, current_metrics: Dict[str, float], _decision: AdaptationDecision
    ) -> float:
        """Compute memory efficiency reward component."""
        memory_usage = current_metrics.get("memory_usage_mb", 0.0)

        if memory_usage == 0.0:
            return 0.0  # No memory information available

        # Lower memory usage is better
        memory_improvement = -self.performance_tracker.get_improvement(
            "memory_usage_mb", memory_usage
        )

        return float(np.clip(memory_improvement, -1.0, 1.0))

    def _compute_stability_reward(
        self,
        current_metrics: Dict[str, float],
        graph_state: ModelGraphState,
        decision: AdaptationDecision,
    ) -> float:
        """Compute stability-based reward component with gradient awareness."""
        stability = current_metrics.get("stability", 0.5)
        error_rate = current_metrics.get("error_rate", 0.0)
        avg_errors = current_metrics.get("avg_error_count", 0.0)
        
        # Extract gradient metrics from graph state
        gradient_health = graph_state.global_metrics.get("gradient_health", 0.5)
        training_stability = graph_state.global_metrics.get("training_stability", 0.5)
        avg_gradient_stability = graph_state.global_metrics.get("avg_gradient_stability", 0.5)

        # Combine traditional stability with gradient stability
        overall_stability = (stability + training_stability + avg_gradient_stability) / 3.0
        
        # Higher stability and lower error rates are better
        stability_score = overall_stability
        error_penalty = min(error_rate + avg_errors, 1.0)  # Cap penalty at 1.0
        
        # Additional penalty for poor gradient health
        gradient_penalty = 0.0
        if gradient_health < 0.3:
            gradient_penalty = 0.3  # Significant penalty for unhealthy gradients
        elif gradient_health < 0.5:
            gradient_penalty = 0.1  # Moderate penalty

        stability_reward = stability_score - error_penalty - gradient_penalty
        
        # Bonus for improving gradient stability
        if decision.adaptation_type in ["add_gradient_clipping", "add_batch_normalization", "add_residual_connection"]:
            if gradient_health < 0.5 or training_stability < 0.5:
                stability_reward += 0.2  # Bonus for addressing gradient issues

        # Penalty for decisions that target stable layers without strong justification
        if (
            len(graph_state.problematic_layers) == 0
            and decision.layer_name not in graph_state.problematic_layers
            and decision.confidence < 0.8
        ):
            stability_reward -= 0.2  # Unnecessary intervention penalty

        return float(np.clip(stability_reward, -1.0, 1.0))

    def _compute_safety_reward(
        self, decision: AdaptationDecision, graph_state: ModelGraphState
    ) -> float:
        """Compute safety-based reward component."""
        # Start with high safety score
        safety_score = 0.9

        # Check decision safety characteristics
        if decision.confidence < 0.5:
            safety_score -= 0.2  # Low confidence decisions are riskier

        if decision.urgency > 0.8 and decision.confidence < 0.7:
            safety_score -= 0.3  # Urgent but uncertain decisions are dangerous

        # Check for safety metadata
        if "safety_score" in decision.metadata:
            metadata_safety = decision.metadata["safety_score"]
            safety_score = 0.7 * safety_score + 0.3 * metadata_safety

        # Penalty for acting on stable systems
        if len(graph_state.problematic_layers) == 0:
            safety_score -= 0.1  # Small penalty for unnecessary intervention

        return float(np.clip(safety_score, 0.0, 1.0))

    def _compute_innovation_reward(self, decision: AdaptationDecision) -> float:
        """Compute innovation/novelty reward component."""
        # Check if this is a novel type of decision
        recent_decisions = list(self.decision_history)[-20:]  # Last 20 decisions

        if not recent_decisions:
            return 0.1  # Small reward for first decisions

        # Count similar recent decisions
        similar_count = 0
        for past_decision_data in recent_decisions:
            past_decision = past_decision_data["decision"]

            # Simple similarity check
            if past_decision.layer_name == decision.layer_name or str(
                past_decision.blueprint_request
            ) == str(decision.blueprint_request):
                similar_count += 1

        # Reward novel decisions, but not too much
        novelty_ratio = 1.0 - (similar_count / len(recent_decisions))
        innovation_reward = novelty_ratio * 0.3  # Cap innovation reward

        return float(innovation_reward)

    def _compute_consistency_reward(self, decision: AdaptationDecision) -> float:
        """Compute consistency reward based on past successful patterns."""
        if not self.reward_history:
            return 0.0

        # Look for similar past decisions and their outcomes
        recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions

        similar_outcomes = []
        for past_data in recent_decisions:
            past_decision = past_data["decision"]

            # Check similarity
            similarity_score = 0.0
            if past_decision.layer_name == decision.layer_name:
                similarity_score += 0.3
            if past_decision.confidence >= decision.confidence - 0.1:
                similarity_score += 0.2
            if past_decision.urgency >= decision.urgency - 0.1:
                similarity_score += 0.2
            if str(past_decision.blueprint_request) == str(decision.blueprint_request):
                similarity_score += 0.3

            # If sufficiently similar, record the outcome
            if similarity_score >= 0.4:
                similar_outcomes.append(past_data["reward"])

        if not similar_outcomes:
            return 0.0

        # Reward consistency with past successful patterns
        avg_outcome = np.mean(similar_outcomes)
        return float(np.clip(avg_outcome * 0.5, -0.5, 0.5))  # Scaled consistency reward

    def _estimate_temporal_impacts(
        self, decision: AdaptationDecision, _current_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Estimate temporal impacts of the decision."""
        base_impact = (
            decision.confidence * 0.5
        )  # Base impact proportional to confidence

        # Immediate impact (usually highest)
        immediate = base_impact * (1.0 + decision.urgency * 0.3)

        # Short-term impact (depends on adaptation complexity)
        complexity_factor = min(len(str(decision.blueprint_request)), 200) / 200.0
        short_term = base_impact * (0.8 + complexity_factor * 0.4)

        # Medium-term impact (benefits from learning and optimization)
        medium_term = base_impact * (0.6 + decision.confidence * 0.3)

        # Long-term impact (architectural improvements have lasting effects)
        long_term = base_impact * (0.4 + complexity_factor * 0.4)

        return {
            "immediate": float(np.clip(immediate, -1.0, 1.0)),
            "short_term": float(np.clip(short_term, -1.0, 1.0)),
            "medium_term": float(np.clip(medium_term, -1.0, 1.0)),
            "long_term": float(np.clip(long_term, -1.0, 1.0)),
        }

    def _assess_risk(
        self, decision: AdaptationDecision, graph_state: ModelGraphState
    ) -> float:
        """Assess risk level of the adaptation decision."""
        risk = 0.0

        # Low confidence decisions are risky
        if decision.confidence < 0.5:
            risk += 0.3
        elif decision.confidence < 0.7:
            risk += 0.1

        # High urgency with low confidence is very risky
        if decision.urgency > 0.7 and decision.confidence < 0.6:
            risk += 0.4

        # Acting on non-problematic layers is risky
        if (
            len(graph_state.problematic_layers) > 0
            and decision.layer_name not in graph_state.problematic_layers
        ):
            risk += 0.2

        # Complex adaptations are inherently riskier
        complexity = min(len(str(decision.blueprint_request)), 200) / 200.0
        risk += complexity * 0.3

        return float(np.clip(risk, 0.0, 1.0))

    def _combine_reward_components(self, metrics: RewardMetrics) -> float:
        """Combine individual reward components into final reward."""
        # Get current weights (may be adaptive)
        weights = self._get_current_weights()

        # Primary components
        primary_reward = (
            weights["accuracy"] * metrics.accuracy_improvement
            + weights["speed"] * metrics.speed_improvement
            + weights["memory"] * metrics.memory_efficiency
            + weights["stability"] * metrics.stability_score
            + weights["safety"] * (metrics.safety_score - 0.5) * 2.0  # Center around 0
            + weights["innovation"] * metrics.innovation_score
        )

        # Temporal discounting
        temporal_reward = (
            self.config.immediate_discount * metrics.immediate_impact
            + self.config.short_term_discount * metrics.short_term_impact
            + self.config.medium_term_discount * metrics.medium_term_impact
            + self.config.long_term_discount * metrics.long_term_impact
        )

        # Combine primary and temporal components
        total_reward = 0.7 * primary_reward + 0.3 * temporal_reward

        # Apply consistency bonus
        total_reward += 0.1 * metrics.consistency_score

        return float(total_reward)

    def _apply_safety_penalties(self, reward: float, metrics: RewardMetrics) -> float:
        """Apply safety and stability penalties to the reward."""
        penalized_reward = reward

        # Safety violation penalty
        if metrics.safety_score < 0.5:
            safety_penalty = self.config.safety_failure_penalty * (
                0.5 - metrics.safety_score
            )
            penalized_reward += safety_penalty
            logger.warning("Applied safety penalty: %.3f", safety_penalty)

        # Stability penalty
        if metrics.stability_score < 0.3:
            stability_penalty = self.config.stability_failure_penalty * (
                0.3 - metrics.stability_score
            )
            penalized_reward += stability_penalty
            logger.warning("Applied stability penalty: %.3f", stability_penalty)

        return float(np.clip(penalized_reward, -5.0, 5.0))  # Reasonable bounds

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current component weights (adaptive or static)."""
        if self.weight_optimizer:
            return self.weight_optimizer.get_current_weights()
        else:
            return {
                "accuracy": self.config.accuracy_weight,
                "speed": self.config.speed_weight,
                "memory": self.config.memory_weight,
                "stability": self.config.stability_weight,
                "safety": self.config.safety_weight,
                "innovation": self.config.innovation_weight,
            }

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics."""
        if not self.reward_history:
            return {}

        rewards = list(self.reward_history)

        return {
            "total_rewards": len(rewards),
            "average_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "positive_reward_rate": self.stats["positive_rewards"]
            / max(self.stats["total_rewards_computed"], 1),
            "safety_violation_rate": self.stats["safety_violations"]
            / max(self.stats["total_rewards_computed"], 1),
            "recent_trend": (
                self.performance_tracker.trend_analyzer.compute_trend(rewards[-20:])
                if len(rewards) >= 20
                else 0.0
            ),
            "correlation_samples": len(
                self.correlation_analyzer.decision_outcome_pairs
            ),
            "current_weights": self._get_current_weights(),
        }

    def get_correlations(self) -> Dict[str, Any]:
        """Get current correlation analysis results."""
        return {
            "decision_outcome_correlations": self.correlation_analyzer.compute_correlations(),
            "mutual_information": self.correlation_analyzer.get_mutual_information(),
            "samples_analyzed": len(self.correlation_analyzer.decision_outcome_pairs),
        }


class AdaptiveWeightOptimizer:
    """Optimizes reward component weights based on learning outcomes."""

    def __init__(self, config: RewardConfig):
        self.config = config

        # Current weights (start with config defaults)
        self.weights = {
            "accuracy": config.accuracy_weight,
            "speed": config.speed_weight,
            "memory": config.memory_weight,
            "stability": config.stability_weight,
            "safety": config.safety_weight,
            "innovation": config.innovation_weight,
        }

        # Weight momentum for stability
        self.weight_momentum = {key: 0.0 for key in self.weights}

        # Performance tracking
        self.weight_performance = defaultdict(list)

    async def update_weights(
        self, _decision: AdaptationDecision, metrics: RewardMetrics, final_reward: float
    ):
        """Update component weights based on learning outcomes."""
        # This is a simplified adaptive weight system
        # In production, this could use more sophisticated optimization

        # Record performance for each component
        component_contributions = {
            "accuracy": metrics.accuracy_improvement,
            "speed": metrics.speed_improvement,
            "memory": metrics.memory_efficiency,
            "stability": metrics.stability_score,
            "safety": metrics.safety_score,
            "innovation": metrics.innovation_score,
        }

        # Update weights based on component success
        for component, contribution in component_contributions.items():
            if final_reward > 0 and contribution > 0:
                # Increase weight for successful components
                weight_delta = self.config.weight_adaptation_rate * contribution
            elif final_reward < 0 and contribution < 0:
                # Decrease weight for failing components
                weight_delta = -self.config.weight_adaptation_rate * abs(contribution)
            else:
                weight_delta = 0.0

            # Apply momentum
            momentum = self.weight_momentum[component] * self.config.weight_momentum
            self.weight_momentum[component] = momentum + weight_delta

            # Update weight
            self.weights[component] += self.weight_momentum[component]

        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum

        # Record performance
        self.weight_performance[final_reward].append(self.weights.copy())

    def get_current_weights(self) -> Dict[str, float]:
        """Get current optimized weights."""
        return self.weights.copy()

"""
Analysis components for anomaly detection and performance insights.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List

import numpy as np

from ...utils.logging import get_logger
from .collectors import MorphogeneticMetrics

logger = get_logger(__name__)


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""

    timestamp: datetime
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_score: float
    severity: str  # low, medium, high
    description: str


@dataclass
class PerformanceIssue:
    """Detected performance issue."""

    timestamp: datetime
    issue_type: str
    affected_component: str
    impact_score: float
    details: Dict[str, Any]
    recommendation: str


class AnomalyDetector:
    """
    Detects anomalies in morphogenetic metrics using statistical methods.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._metric_windows: Dict[str, deque] = {}
        self._anomalies: List[Anomaly] = []

        # Z-score threshold for anomaly detection
        self.z_threshold = 3.0

        # Metric-specific thresholds
        self.thresholds = {
            "training_loss": {"min": 0.0, "max": 10.0},
            "training_accuracy": {"min": 0.0, "max": 1.0},
            "gpu_utilization_percent": {"min": 0.0, "max": 100.0},
            "adaptation_rollbacks": {"rate_threshold": 0.1},  # 10% rollback rate
            "kernel_compilation_failures": {"rate_threshold": 0.05},  # 5% failure rate
        }

    def detect(self, metrics_history: List[MorphogeneticMetrics]) -> List[Anomaly]:
        """Detect anomalies in recent metrics."""
        if not metrics_history:
            return []

        anomalies = []
        latest_metrics = metrics_history[-1]

        # Check training metrics
        anomalies.extend(self._check_training_anomalies(
            latest_metrics, metrics_history
        ))

        # Check adaptation metrics
        anomalies.extend(self._check_adaptation_anomalies(
            latest_metrics, metrics_history
        ))

        # Check resource metrics
        anomalies.extend(self._check_resource_anomalies(
            latest_metrics, metrics_history
        ))

        # Check compilation metrics
        anomalies.extend(self._check_compilation_anomalies(
            latest_metrics, metrics_history
        ))

        # Store detected anomalies
        self._anomalies.extend(anomalies)

        # Keep only recent anomalies
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self._anomalies = [
            a for a in self._anomalies
            if a.timestamp > cutoff_time
        ]

        return anomalies

    def _check_training_anomalies(
        self,
        latest: MorphogeneticMetrics,
        history: List[MorphogeneticMetrics]
    ) -> List[Anomaly]:
        """Check for anomalies in training metrics."""
        anomalies = []

        # Extract training loss history
        losses = [m.training_loss for m in history[-self.window_size:]]

        if len(losses) >= 3:
            # Check for sudden loss spike
            if latest.training_loss > 0:
                mean_loss = np.mean(losses[:-1])
                std_loss = np.std(losses[:-1])

                if std_loss > 0:
                    z_score = (latest.training_loss - mean_loss) / std_loss

                    if abs(z_score) > self.z_threshold:
                        anomalies.append(Anomaly(
                            timestamp=latest.collected_at,
                            metric_name="training_loss",
                            expected_value=mean_loss,
                            actual_value=latest.training_loss,
                            deviation_score=abs(z_score),
                            severity="high" if abs(z_score) > 4 else "medium",
                            description=f"Training loss deviation: z-score={z_score:.2f}"
                        ))

            # Check for training stall (loss not decreasing)
            recent_losses = losses[-5:]
            if len(recent_losses) == 5 and all(recent_losses):
                if np.std(recent_losses) < 0.001:  # Very low variance
                    anomalies.append(Anomaly(
                        timestamp=latest.collected_at,
                        metric_name="training_loss",
                        expected_value=recent_losses[0] * 0.95,  # Expect 5% improvement
                        actual_value=recent_losses[-1],
                        deviation_score=1.0,
                        severity="medium",
                        description="Training appears to be stalled"
                    ))

        return anomalies

    def _check_adaptation_anomalies(
        self,
        latest: MorphogeneticMetrics,
        history: List[MorphogeneticMetrics]
    ) -> List[Anomaly]:
        """Check for anomalies in adaptation behavior."""
        anomalies = []

        # Check rollback rate
        if latest.adaptation_attempts > 0:
            rollback_rate = latest.adaptation_rollbacks / latest.adaptation_attempts

            if rollback_rate > self.thresholds["adaptation_rollbacks"]["rate_threshold"]:
                anomalies.append(Anomaly(
                    timestamp=latest.collected_at,
                    metric_name="adaptation_rollback_rate",
                    expected_value=self.thresholds["adaptation_rollbacks"]["rate_threshold"],
                    actual_value=rollback_rate,
                    deviation_score=rollback_rate / self.thresholds["adaptation_rollbacks"]["rate_threshold"],
                    severity="high",
                    description=f"High rollback rate: {rollback_rate:.1%}"
                ))

        # Check adaptation success trend
        if len(history) >= 5:
            recent_success_rates = [
                m.adaptation_successes / max(1, m.adaptation_attempts)
                for m in history[-5:]
            ]

            if all(rate < 0.5 for rate in recent_success_rates):
                anomalies.append(Anomaly(
                    timestamp=latest.collected_at,
                    metric_name="adaptation_success_rate",
                    expected_value=0.7,
                    actual_value=recent_success_rates[-1],
                    deviation_score=2.0,
                    severity="high",
                    description="Consistently low adaptation success rate"
                ))

        return anomalies

    def _check_resource_anomalies(
        self,
        latest: MorphogeneticMetrics,
        history: List[MorphogeneticMetrics]
    ) -> List[Anomaly]:
        """Check for resource usage anomalies."""
        anomalies = []

        # GPU memory spike detection
        if len(history) >= 3:
            recent_gpu_mem = [m.gpu_memory_used_mb for m in history[-3:]]
            if recent_gpu_mem[-1] > 0 and recent_gpu_mem[0] > 0:
                mem_increase = (recent_gpu_mem[-1] - recent_gpu_mem[0]) / recent_gpu_mem[0]

                if mem_increase > 0.5:  # 50% increase
                    anomalies.append(Anomaly(
                        timestamp=latest.collected_at,
                        metric_name="gpu_memory_used_mb",
                        expected_value=recent_gpu_mem[0],
                        actual_value=recent_gpu_mem[-1],
                        deviation_score=mem_increase,
                        severity="medium",
                        description=f"GPU memory spike: {mem_increase:.1%} increase"
                    ))

        # CPU utilization anomaly
        if latest.cpu_utilization_percent > 90:
            anomalies.append(Anomaly(
                timestamp=latest.collected_at,
                metric_name="cpu_utilization_percent",
                expected_value=70.0,
                actual_value=latest.cpu_utilization_percent,
                deviation_score=(latest.cpu_utilization_percent - 70) / 70,
                severity="medium",
                description="High CPU utilization"
            ))

        return anomalies

    def _check_compilation_anomalies(
        self,
        latest: MorphogeneticMetrics,
        history: List[MorphogeneticMetrics]
    ) -> List[Anomaly]:
        """Check for compilation-related anomalies."""
        anomalies = []

        # Check compilation failure rate
        if latest.kernel_compilations_total > 0:
            failure_rate = latest.kernel_compilation_failures / latest.kernel_compilations_total

            if failure_rate > self.thresholds["kernel_compilation_failures"]["rate_threshold"]:
                anomalies.append(Anomaly(
                    timestamp=latest.collected_at,
                    metric_name="kernel_compilation_failure_rate",
                    expected_value=self.thresholds["kernel_compilation_failures"]["rate_threshold"],
                    actual_value=failure_rate,
                    deviation_score=failure_rate / self.thresholds["kernel_compilation_failures"]["rate_threshold"],
                    severity="high",
                    description=f"High compilation failure rate: {failure_rate:.1%}"
                ))

        # Check compilation latency
        if latest.kernel_compilation_latency_ms:
            avg_latency = np.mean(latest.kernel_compilation_latency_ms)

            if avg_latency > 5000:  # 5 seconds
                anomalies.append(Anomaly(
                    timestamp=latest.collected_at,
                    metric_name="kernel_compilation_latency",
                    expected_value=1000.0,
                    actual_value=avg_latency,
                    deviation_score=avg_latency / 1000.0,
                    severity="medium",
                    description=f"High compilation latency: {avg_latency:.0f}ms"
                ))

        return anomalies

    def get_anomalies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent anomalies as dictionaries."""
        return [
            {
                "timestamp": a.timestamp.isoformat(),
                "metric": a.metric_name,
                "expected": a.expected_value,
                "actual": a.actual_value,
                "deviation_score": a.deviation_score,
                "severity": a.severity,
                "description": a.description
            }
            for a in self._anomalies[-limit:]
        ]


class PerformanceAnalyzer:
    """
    Analyzes performance trends and identifies optimization opportunities.
    """

    def __init__(self):
        self._performance_issues: List[PerformanceIssue] = []

    def analyze(
        self,
        metrics_history: List[MorphogeneticMetrics]
    ) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        if len(metrics_history) < 2:
            return {"status": "insufficient_data"}

        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "training_efficiency": self._analyze_training_efficiency(metrics_history),
            "adaptation_effectiveness": self._analyze_adaptation_effectiveness(metrics_history),
            "resource_utilization": self._analyze_resource_utilization(metrics_history),
            "cache_performance": self._analyze_cache_performance(metrics_history),
            "recommendations": []
        }

        # Generate recommendations based on analysis
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def find_issues(
        self,
        metrics_history: List[MorphogeneticMetrics]
    ) -> List[PerformanceIssue]:
        """Find specific performance issues."""
        issues = []

        if len(metrics_history) < 2:
            return issues

        # Check for training overhead
        latest = metrics_history[-1]
        if latest.training_overhead_percent > 5.0:
            issues.append(PerformanceIssue(
                timestamp=latest.collected_at,
                issue_type="high_training_overhead",
                affected_component="training",
                impact_score=latest.training_overhead_percent / 5.0,
                details={
                    "overhead_percent": latest.training_overhead_percent,
                    "threshold": 5.0
                },
                recommendation="Consider reducing adaptation frequency or optimizing compilation pipeline"
            ))

        # Check for poor cache performance
        if latest.kernel_cache_hits + latest.kernel_cache_misses > 0:
            cache_hit_rate = latest.kernel_cache_hits / (
                latest.kernel_cache_hits + latest.kernel_cache_misses
            )

            if cache_hit_rate < 0.8:
                issues.append(PerformanceIssue(
                    timestamp=latest.collected_at,
                    issue_type="low_cache_hit_rate",
                    affected_component="cache",
                    impact_score=(0.8 - cache_hit_rate) * 2,
                    details={
                        "hit_rate": cache_hit_rate,
                        "expected": 0.8
                    },
                    recommendation="Increase cache size or improve cache warming strategy"
                ))

        # Check for inefficient seed usage
        if latest.seed_performance_scores:
            low_performing_seeds = [
                (seed, score)
                for seed, score in latest.seed_performance_scores.items()
                if score < 0.3
            ]

            if len(low_performing_seeds) > len(latest.seed_performance_scores) * 0.3:
                issues.append(PerformanceIssue(
                    timestamp=latest.collected_at,
                    issue_type="inefficient_seed_usage",
                    affected_component="seeds",
                    impact_score=len(low_performing_seeds) / len(latest.seed_performance_scores),
                    details={
                        "low_performing_count": len(low_performing_seeds),
                        "total_seeds": len(latest.seed_performance_scores)
                    },
                    recommendation="Review seed selection strategy or increase exploration"
                ))

        # Store issues
        self._performance_issues.extend(issues)

        # Keep only recent issues
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self._performance_issues = [
            i for i in self._performance_issues
            if i.timestamp > cutoff_time
        ]

        return issues

    def _analyze_training_efficiency(
        self,
        history: List[MorphogeneticMetrics]
    ) -> Dict[str, Any]:
        """Analyze training efficiency metrics."""
        recent = history[-10:]

        # Calculate loss reduction rate
        if len(recent) >= 2 and recent[0].training_loss > 0:
            loss_reduction = (
                (recent[0].training_loss - recent[-1].training_loss) /
                recent[0].training_loss
            )
        else:
            loss_reduction = 0.0

        # Calculate average overhead
        avg_overhead = np.mean([m.training_overhead_percent for m in recent])

        return {
            "loss_reduction_rate": loss_reduction,
            "avg_overhead_percent": avg_overhead,
            "efficiency_score": max(0, (1 - avg_overhead/10) * (1 + loss_reduction))
        }

    def _analyze_adaptation_effectiveness(
        self,
        history: List[MorphogeneticMetrics]
    ) -> Dict[str, Any]:
        """Analyze adaptation effectiveness."""
        if not history:
            return {}

        latest = history[-1]

        success_rate = (
            latest.adaptation_successes / max(1, latest.adaptation_attempts)
        )

        rollback_rate = (
            latest.adaptation_rollbacks / max(1, latest.adaptation_attempts)
        )

        # Calculate adaptation velocity (adaptations per epoch)
        if latest.training_epochs_completed > 0:
            adaptation_velocity = (
                latest.adaptation_attempts / latest.training_epochs_completed
            )
        else:
            adaptation_velocity = 0.0

        return {
            "success_rate": success_rate,
            "rollback_rate": rollback_rate,
            "adaptation_velocity": adaptation_velocity,
            "effectiveness_score": success_rate * (1 - rollback_rate)
        }

    def _analyze_resource_utilization(
        self,
        history: List[MorphogeneticMetrics]
    ) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        recent = history[-10:]

        return {
            "avg_gpu_utilization": np.mean([m.gpu_utilization_percent for m in recent]),
            "max_gpu_memory_mb": max(m.gpu_memory_used_mb for m in recent),
            "avg_cpu_utilization": np.mean([m.cpu_utilization_percent for m in recent]),
            "resource_efficiency": self._calculate_resource_efficiency(recent)
        }

    def _analyze_cache_performance(
        self,
        history: List[MorphogeneticMetrics]
    ) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        total_hits = sum(m.kernel_cache_hits for m in history)
        total_misses = sum(m.kernel_cache_misses for m in history)

        if total_hits + total_misses > 0:
            overall_hit_rate = total_hits / (total_hits + total_misses)
        else:
            overall_hit_rate = 0.0

        return {
            "overall_hit_rate": overall_hit_rate,
            "total_operations": total_hits + total_misses,
            "cache_effectiveness": overall_hit_rate * 100
        }

    def _calculate_resource_efficiency(
        self,
        metrics: List[MorphogeneticMetrics]
    ) -> float:
        """Calculate resource efficiency score."""
        if not metrics:
            return 0.0

        # Efficiency = work done / resources used
        # Simplified: adaptations per resource unit

        total_adaptations = metrics[-1].adaptation_attempts
        avg_gpu_util = np.mean([m.gpu_utilization_percent for m in metrics])
        avg_cpu_util = np.mean([m.cpu_utilization_percent for m in metrics])

        if avg_gpu_util + avg_cpu_util > 0:
            efficiency = total_adaptations / (avg_gpu_util + avg_cpu_util)
        else:
            efficiency = 0.0

        return min(100, efficiency * 10)  # Scale to 0-100

    def _generate_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Training efficiency recommendations
        if analysis["training_efficiency"]["avg_overhead_percent"] > 5:
            recommendations.append(
                "High training overhead detected. Consider batching adaptations "
                "or increasing the adaptation interval."
            )

        # Adaptation recommendations
        adapt_analysis = analysis["adaptation_effectiveness"]
        if adapt_analysis.get("success_rate", 1) < 0.7:
            recommendations.append(
                "Low adaptation success rate. Review adaptation criteria and "
                "validation thresholds."
            )

        if adapt_analysis.get("rollback_rate", 0) > 0.1:
            recommendations.append(
                "High rollback rate indicates unstable adaptations. Consider "
                "more conservative adaptation strategies."
            )

        # Resource recommendations
        resource_analysis = analysis["resource_utilization"]
        if resource_analysis.get("avg_gpu_utilization", 0) < 50:
            recommendations.append(
                "GPU underutilized. Consider increasing batch size or "
                "enabling more parallel operations."
            )

        # Cache recommendations
        cache_analysis = analysis["cache_performance"]
        if cache_analysis.get("overall_hit_rate", 1) < 0.8:
            recommendations.append(
                "Cache hit rate below optimal. Consider increasing cache size "
                "or implementing predictive cache warming."
            )

        return recommendations

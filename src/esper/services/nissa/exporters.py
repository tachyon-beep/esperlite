"""
Metric exporters for various monitoring systems.
"""

import json
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest

from ...utils.logging import get_logger
from .collectors import MorphogeneticMetrics
from .collectors import SystemMetrics

logger = get_logger(__name__)


class PrometheusExporter:
    """
    Exports metrics in Prometheus format.
    
    Provides both pull-based HTTP endpoint and push gateway support.
    """

    def __init__(self, namespace: str = "esper"):
        self.namespace = namespace
        self.registry = CollectorRegistry()

        # Morphogenetic metrics
        self.seed_activations = Counter(
            f"{namespace}_seed_activations_total",
            "Total number of seed activations",
            ["layer", "seed_idx"],
            registry=self.registry
        )

        self.seed_performance = Gauge(
            f"{namespace}_seed_performance_score",
            "Current seed performance score",
            ["layer", "seed_idx"],
            registry=self.registry
        )

        self.kernel_compilations = Counter(
            f"{namespace}_kernel_compilations_total",
            "Total kernel compilations",
            ["status"],
            registry=self.registry
        )

        self.kernel_compilation_latency = Histogram(
            f"{namespace}_kernel_compilation_latency_milliseconds",
            "Kernel compilation latency",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
            registry=self.registry
        )

        self.kernel_cache_operations = Counter(
            f"{namespace}_kernel_cache_operations_total",
            "Kernel cache operations",
            ["operation", "result"],
            registry=self.registry
        )

        self.adaptations = Counter(
            f"{namespace}_adaptations_total",
            "Total adaptation attempts",
            ["result"],
            registry=self.registry
        )

        self.adaptation_latency = Histogram(
            f"{namespace}_adaptation_latency_milliseconds",
            "Adaptation latency",
            buckets=[100, 250, 500, 1000, 2500, 5000, 10000],
            registry=self.registry
        )

        self.rollbacks = Counter(
            f"{namespace}_rollbacks_total",
            "Total adaptation rollbacks",
            registry=self.registry
        )

        # Training metrics
        self.training_epochs = Gauge(
            f"{namespace}_training_epochs_completed",
            "Number of training epochs completed",
            registry=self.registry
        )

        self.training_steps = Counter(
            f"{namespace}_training_steps_total",
            "Total training steps",
            registry=self.registry
        )

        self.training_loss = Gauge(
            f"{namespace}_training_loss",
            "Current training loss",
            registry=self.registry
        )

        self.training_accuracy = Gauge(
            f"{namespace}_training_accuracy",
            "Current training accuracy",
            registry=self.registry
        )

        self.training_overhead = Gauge(
            f"{namespace}_training_overhead_percent",
            "Training overhead percentage",
            registry=self.registry
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            f"{namespace}_gpu_utilization_percent",
            "GPU utilization percentage",
            registry=self.registry
        )

        self.gpu_memory_used = Gauge(
            f"{namespace}_gpu_memory_used_megabytes",
            "GPU memory used in MB",
            registry=self.registry
        )

        self.cpu_utilization = Gauge(
            f"{namespace}_cpu_utilization_percent",
            "CPU utilization percentage",
            registry=self.registry
        )

        self.memory_used = Gauge(
            f"{namespace}_memory_used_gigabytes",
            "Memory used in GB",
            registry=self.registry
        )

        # Cache tier metrics
        self.cache_tier_hits = Counter(
            f"{namespace}_cache_tier_hits_total",
            "Cache hits by tier",
            ["tier"],
            registry=self.registry
        )

        self.cache_tier_size = Gauge(
            f"{namespace}_cache_tier_size_bytes",
            "Cache size by tier",
            ["tier"],
            registry=self.registry
        )

        # Asset metrics
        self.asset_count = Gauge(
            f"{namespace}_asset_count",
            "Number of assets by type",
            ["type", "status"],
            registry=self.registry
        )

        # Custom morphogenetic metrics
        self.morphogenetic_score = Gauge(
            f"{namespace}_morphogenetic_score",
            "Overall morphogenetic adaptation score",
            registry=self.registry
        )

        self.architecture_diversity = Gauge(
            f"{namespace}_architecture_diversity",
            "Architectural diversity score",
            registry=self.registry
        )

    def update_metrics(
        self,
        morpho_metrics: MorphogeneticMetrics,
        system_metrics: Optional[SystemMetrics] = None
    ):
        """Update Prometheus metrics from collected data."""

        # Update seed metrics
        for seed_key, count in morpho_metrics.seed_activations.items():
            layer, seed_idx = seed_key.rsplit('_', 1)
            self.seed_activations.labels(
                layer=layer, seed_idx=seed_idx
            )._value._value = count

        for seed_key, score in morpho_metrics.seed_performance_scores.items():
            layer, seed_idx = seed_key.rsplit('_', 1)
            self.seed_performance.labels(
                layer=layer, seed_idx=seed_idx
            ).set(score)

        # Update kernel metrics
        successful_compilations = (
            morpho_metrics.kernel_compilations_total -
            morpho_metrics.kernel_compilation_failures
        )

        self.kernel_compilations.labels(status="success")._value._value = (
            successful_compilations
        )
        self.kernel_compilations.labels(status="failure")._value._value = (
            morpho_metrics.kernel_compilation_failures
        )

        # Update compilation latencies
        for latency in morpho_metrics.kernel_compilation_latency_ms:
            self.kernel_compilation_latency.observe(latency)

        # Cache operations
        self.kernel_cache_operations.labels(
            operation="get", result="hit"
        )._value._value = morpho_metrics.kernel_cache_hits

        self.kernel_cache_operations.labels(
            operation="get", result="miss"
        )._value._value = morpho_metrics.kernel_cache_misses

        # Adaptation metrics
        self.adaptations.labels(result="success")._value._value = (
            morpho_metrics.adaptation_successes
        )
        self.adaptations.labels(result="failure")._value._value = (
            morpho_metrics.adaptation_attempts - morpho_metrics.adaptation_successes
        )

        self.rollbacks._value._value = morpho_metrics.adaptation_rollbacks

        # Update adaptation latencies
        for latency in morpho_metrics.adaptation_latency_ms:
            self.adaptation_latency.observe(latency)

        # Training metrics
        self.training_epochs.set(morpho_metrics.training_epochs_completed)
        self.training_steps._value._value = morpho_metrics.training_steps_total
        self.training_loss.set(morpho_metrics.training_loss)
        self.training_accuracy.set(morpho_metrics.training_accuracy)
        self.training_overhead.set(morpho_metrics.training_overhead_percent)

        # Resource metrics
        self.gpu_utilization.set(morpho_metrics.gpu_utilization_percent)
        self.gpu_memory_used.set(morpho_metrics.gpu_memory_used_mb)

        if system_metrics:
            self.cpu_utilization.set(system_metrics.cpu_percent)
            self.memory_used.set(system_metrics.memory_used_gb)

        # Calculate and set morphogenetic score
        morpho_score = self._calculate_morphogenetic_score(morpho_metrics)
        self.morphogenetic_score.set(morpho_score)

        # Calculate architectural diversity
        diversity_score = self._calculate_diversity_score(morpho_metrics)
        self.architecture_diversity.set(diversity_score)

    def update_cache_metrics(self, cache_stats: Dict[str, Any]):
        """Update cache-specific metrics."""
        # L1 cache
        if "l1" in cache_stats:
            self.cache_tier_size.labels(tier="L1").set(
                cache_stats["l1"]["size_bytes"]
            )
            self.cache_tier_hits.labels(tier="L1")._value._value = (
                cache_stats.get("l1_hits", 0)
            )

        # L2 cache
        if "l2" in cache_stats:
            self.cache_tier_size.labels(tier="L2").set(
                cache_stats["l2"].get("memory_used_bytes", 0)
            )
            self.cache_tier_hits.labels(tier="L2")._value._value = (
                cache_stats.get("l2_hits", 0)
            )

        # L3 cache
        if "l3" in cache_stats:
            self.cache_tier_size.labels(tier="L3").set(
                cache_stats["l3"].get("total_bytes", 0)
            )
            self.cache_tier_hits.labels(tier="L3")._value._value = (
                cache_stats.get("l3_hits", 0)
            )

    def update_asset_metrics(self, asset_stats: Dict[str, Any]):
        """Update asset management metrics."""
        for asset_type, counts in asset_stats.items():
            for status, count in counts.items():
                self.asset_count.labels(
                    type=asset_type, status=status
                ).set(count)

    def generate_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        return generate_latest(self.registry)

    def _calculate_morphogenetic_score(
        self,
        metrics: MorphogeneticMetrics
    ) -> float:
        """Calculate overall morphogenetic adaptation score."""
        # Weighted combination of key metrics
        compilation_success_rate = 1.0 - (
            metrics.kernel_compilation_failures /
            max(1, metrics.kernel_compilations_total)
        )

        adaptation_success_rate = (
            metrics.adaptation_successes /
            max(1, metrics.adaptation_attempts)
        )

        rollback_penalty = 1.0 - (
            metrics.adaptation_rollbacks /
            max(1, metrics.adaptation_attempts)
        )

        cache_efficiency = (
            metrics.kernel_cache_hits /
            max(1, metrics.kernel_cache_hits + metrics.kernel_cache_misses)
        )

        # Weighted score (0-100)
        score = (
            compilation_success_rate * 25 +
            adaptation_success_rate * 35 +
            rollback_penalty * 20 +
            cache_efficiency * 20
        )

        return round(score, 2)

    def _calculate_diversity_score(
        self,
        metrics: MorphogeneticMetrics
    ) -> float:
        """Calculate architectural diversity score."""
        # Based on number of unique active seeds
        active_seeds = sum(
            1 for count in metrics.seed_activations.values()
            if count > 0
        )

        total_seeds = len(metrics.seed_activations)

        if total_seeds == 0:
            return 0.0

        # Diversity ratio * 100
        diversity = (active_seeds / total_seeds) * 100

        return round(diversity, 2)


class JsonExporter:
    """Exports metrics in JSON format for custom consumers."""

    def __init__(self):
        self.last_export = None

    def export(
        self,
        morpho_metrics: MorphogeneticMetrics,
        system_metrics: Optional[SystemMetrics] = None,
        include_history: bool = False,
        history: Optional[List[MorphogeneticMetrics]] = None
    ) -> str:
        """Export metrics as JSON."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "morphogenetic": {
                "seeds": {
                    "activations": morpho_metrics.seed_activations,
                    "performance_scores": morpho_metrics.seed_performance_scores,
                    "lifecycle_transitions": morpho_metrics.seed_lifecycle_transitions,
                },
                "kernels": {
                    "compilations_total": morpho_metrics.kernel_compilations_total,
                    "compilation_failures": morpho_metrics.kernel_compilation_failures,
                    "avg_compilation_latency_ms": (
                        sum(morpho_metrics.kernel_compilation_latency_ms) /
                        len(morpho_metrics.kernel_compilation_latency_ms)
                    ) if morpho_metrics.kernel_compilation_latency_ms else 0,
                    "cache_hits": morpho_metrics.kernel_cache_hits,
                    "cache_misses": morpho_metrics.kernel_cache_misses,
                },
                "adaptations": {
                    "attempts": morpho_metrics.adaptation_attempts,
                    "successes": morpho_metrics.adaptation_successes,
                    "rollbacks": morpho_metrics.adaptation_rollbacks,
                    "avg_latency_ms": (
                        sum(morpho_metrics.adaptation_latency_ms) /
                        len(morpho_metrics.adaptation_latency_ms)
                    ) if morpho_metrics.adaptation_latency_ms else 0,
                },
                "training": {
                    "epochs_completed": morpho_metrics.training_epochs_completed,
                    "steps_total": morpho_metrics.training_steps_total,
                    "loss": morpho_metrics.training_loss,
                    "accuracy": morpho_metrics.training_accuracy,
                    "overhead_percent": morpho_metrics.training_overhead_percent,
                },
                "resources": {
                    "gpu_utilization_percent": morpho_metrics.gpu_utilization_percent,
                    "gpu_memory_used_mb": morpho_metrics.gpu_memory_used_mb,
                },
            }
        }

        if system_metrics:
            export_data["system"] = {
                "cpu_count": system_metrics.cpu_count,
                "cpu_percent": system_metrics.cpu_percent,
                "memory_total_gb": system_metrics.memory_total_gb,
                "memory_used_gb": system_metrics.memory_used_gb,
                "memory_percent": system_metrics.memory_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "network_bytes_sent": system_metrics.network_bytes_sent,
                "network_bytes_recv": system_metrics.network_bytes_recv,
                "process_count": system_metrics.process_count,
            }

        if include_history and history:
            export_data["history"] = [
                {
                    "timestamp": m.collected_at.isoformat(),
                    "training_loss": m.training_loss,
                    "training_accuracy": m.training_accuracy,
                    "adaptation_success_rate": (
                        m.adaptation_successes / max(1, m.adaptation_attempts)
                    ),
                }
                for m in history[-10:]  # Last 10 samples
            ]

        self.last_export = datetime.utcnow()
        return json.dumps(export_data, indent=2)

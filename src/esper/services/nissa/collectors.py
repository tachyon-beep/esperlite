"""
Metrics collectors for various subsystems.
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import psutil

from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MorphogeneticMetrics:
    """Core morphogenetic training metrics."""

    # Seed metrics
    seed_activations: Dict[str, int] = field(default_factory=dict)
    seed_performance_scores: Dict[str, float] = field(default_factory=dict)
    seed_lifecycle_transitions: Dict[str, int] = field(default_factory=dict)

    # Kernel metrics
    kernel_compilations_total: int = 0
    kernel_compilation_failures: int = 0
    kernel_compilation_latency_ms: List[float] = field(default_factory=list)
    kernel_cache_hits: int = 0
    kernel_cache_misses: int = 0

    # Adaptation metrics
    adaptation_attempts: int = 0
    adaptation_successes: int = 0
    adaptation_rollbacks: int = 0
    adaptation_latency_ms: List[float] = field(default_factory=list)

    # Training metrics
    training_epochs_completed: int = 0
    training_steps_total: int = 0
    training_loss: float = 0.0
    training_accuracy: float = 0.0
    training_overhead_percent: float = 0.0

    # Resource metrics
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    memory_used_gb: float = 0.0

    # Timestamp
    collected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """System-level metrics."""

    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    collected_at: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """
    Collects metrics from all subsystems.
    
    Provides both pull-based (polling) and push-based (callback)
    metric collection mechanisms.
    """

    def __init__(self):
        self.metrics = MorphogeneticMetrics()
        self.system_metrics = SystemMetrics()
        self._collectors: Dict[str, Callable] = {}
        self._collection_interval = 15  # seconds
        self._collection_task: Optional[asyncio.Task] = None

        # Metric history for trend analysis
        self._metric_history: List[MorphogeneticMetrics] = []
        self._history_size = 100

        # Register default system collectors
        self._register_system_collectors()

    def register_collector(self, name: str, collector: Callable):
        """Register a custom metric collector."""
        self._collectors[name] = collector
        logger.info(f"Registered collector: {name}")

    async def start_collection(self):
        """Start automatic metric collection."""
        if self._collection_task:
            return

        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metric collection")

    async def stop_collection(self):
        """Stop automatic metric collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        logger.info("Stopped metric collection")

    async def collect_once(self) -> MorphogeneticMetrics:
        """Collect metrics once and return."""
        # Collect system metrics
        await self._collect_system_metrics()

        # Run custom collectors
        for name, collector in self._collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector):
                    metrics = await collector()
                else:
                    metrics = collector()

                # Merge collected metrics
                self._merge_metrics(metrics)

            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")

        # Update timestamp
        self.metrics.collected_at = datetime.utcnow()

        # Add to history
        self._add_to_history(self.metrics)

        return self.metrics

    def update_seed_metrics(
        self,
        layer_name: str,
        seed_idx: int,
        performance_score: float,
        activated: bool = True
    ):
        """Update seed-related metrics."""
        seed_key = f"{layer_name}_{seed_idx}"

        if activated:
            self.metrics.seed_activations[seed_key] = (
                self.metrics.seed_activations.get(seed_key, 0) + 1
            )

        self.metrics.seed_performance_scores[seed_key] = performance_score

    def record_kernel_compilation(
        self,
        success: bool,
        latency_ms: float,
        kernel_id: Optional[str] = None
    ):
        """Record kernel compilation event."""
        self.metrics.kernel_compilations_total += 1

        if not success:
            self.metrics.kernel_compilation_failures += 1

        self.metrics.kernel_compilation_latency_ms.append(latency_ms)

        # Keep only recent latencies
        if len(self.metrics.kernel_compilation_latency_ms) > 1000:
            self.metrics.kernel_compilation_latency_ms = (
                self.metrics.kernel_compilation_latency_ms[-1000:]
            )

    def record_adaptation(
        self,
        success: bool,
        latency_ms: float,
        rollback: bool = False
    ):
        """Record adaptation event."""
        self.metrics.adaptation_attempts += 1

        if success:
            self.metrics.adaptation_successes += 1

        if rollback:
            self.metrics.adaptation_rollbacks += 1

        self.metrics.adaptation_latency_ms.append(latency_ms)

        # Keep only recent latencies
        if len(self.metrics.adaptation_latency_ms) > 1000:
            self.metrics.adaptation_latency_ms = (
                self.metrics.adaptation_latency_ms[-1000:]
            )

    def update_training_metrics(
        self,
        epoch: Optional[int] = None,
        steps: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        overhead_percent: Optional[float] = None
    ):
        """Update training-related metrics."""
        if epoch is not None:
            self.metrics.training_epochs_completed = epoch

        if steps is not None:
            self.metrics.training_steps_total = steps

        if loss is not None:
            self.metrics.training_loss = loss

        if accuracy is not None:
            self.metrics.training_accuracy = accuracy

        if overhead_percent is not None:
            self.metrics.training_overhead_percent = overhead_percent

    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        return {
            "morphogenetic": {
                "seed_count": len(self.metrics.seed_activations),
                "active_seeds": sum(
                    1 for count in self.metrics.seed_activations.values()
                    if count > 0
                ),
                "avg_seed_performance": np.mean(
                    list(self.metrics.seed_performance_scores.values())
                ) if self.metrics.seed_performance_scores else 0.0,
                "kernel_compilation_success_rate": (
                    1.0 - (self.metrics.kernel_compilation_failures /
                          max(1, self.metrics.kernel_compilations_total))
                ),
                "avg_compilation_latency_ms": np.mean(
                    self.metrics.kernel_compilation_latency_ms
                ) if self.metrics.kernel_compilation_latency_ms else 0.0,
                "adaptation_success_rate": (
                    self.metrics.adaptation_successes /
                    max(1, self.metrics.adaptation_attempts)
                ),
                "rollback_rate": (
                    self.metrics.adaptation_rollbacks /
                    max(1, self.metrics.adaptation_attempts)
                ),
                "cache_hit_rate": (
                    self.metrics.kernel_cache_hits /
                    max(1, self.metrics.kernel_cache_hits +
                        self.metrics.kernel_cache_misses)
                ),
            },
            "training": {
                "epochs": self.metrics.training_epochs_completed,
                "steps": self.metrics.training_steps_total,
                "loss": self.metrics.training_loss,
                "accuracy": self.metrics.training_accuracy,
                "overhead_percent": self.metrics.training_overhead_percent,
            },
            "resources": {
                "gpu_util": self.metrics.gpu_utilization_percent,
                "gpu_memory_mb": self.metrics.gpu_memory_used_mb,
                "cpu_util": self.system_metrics.cpu_percent,
                "memory_gb": self.system_metrics.memory_used_gb,
            },
            "timestamp": self.metrics.collected_at.isoformat()
        }

    def get_trends(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze metric trends over time window."""
        if not self._metric_history:
            return {}

        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self._metric_history
            if m.collected_at > cutoff_time
        ]

        if len(recent_metrics) < 2:
            return {}

        # Calculate trends
        first = recent_metrics[0]
        last = recent_metrics[-1]

        return {
            "seed_activation_trend": (
                sum(last.seed_activations.values()) -
                sum(first.seed_activations.values())
            ),
            "compilation_latency_trend": (
                np.mean(last.kernel_compilation_latency_ms or [0]) -
                np.mean(first.kernel_compilation_latency_ms or [0])
            ),
            "training_loss_trend": last.training_loss - first.training_loss,
            "resource_usage_trend": {
                "gpu_memory_change_mb": (
                    last.gpu_memory_used_mb - first.gpu_memory_used_mb
                ),
                "cpu_change_percent": (
                    last.cpu_utilization_percent - first.cpu_utilization_percent
                ),
            },
            "window_minutes": window_minutes,
            "samples": len(recent_metrics)
        }

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            self.system_metrics.cpu_count = psutil.cpu_count()
            self.system_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            mem = psutil.virtual_memory()
            self.system_metrics.memory_total_gb = mem.total / (1024**3)
            self.system_metrics.memory_used_gb = mem.used / (1024**3)
            self.system_metrics.memory_percent = mem.percent

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_usage_percent = disk.percent

            # Network metrics
            net = psutil.net_io_counters()
            self.system_metrics.network_bytes_sent = net.bytes_sent
            self.system_metrics.network_bytes_recv = net.bytes_recv

            # Process metrics
            self.system_metrics.process_count = len(psutil.pids())

            # GPU metrics (if available)
            gpu_metrics = await self._collect_gpu_metrics()
            if gpu_metrics:
                self.metrics.gpu_utilization_percent = gpu_metrics["utilization"]
                self.metrics.gpu_memory_used_mb = gpu_metrics["memory_used_mb"]

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _collect_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Collect GPU metrics using nvidia-smi."""
        try:
            # This is a simplified version - in production would use py3nvml
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "utilization": float(parts[0]),
                    "memory_used_mb": float(parts[1])
                }
        except:
            pass

        return None

    def _register_system_collectors(self):
        """Register default system metric collectors."""
        # These would be actual implementations in production
        pass

    def _merge_metrics(self, new_metrics: Dict[str, Any]):
        """Merge new metrics into current metrics."""
        # Simple merge - in production would be more sophisticated
        for key, value in new_metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def _add_to_history(self, metrics: MorphogeneticMetrics):
        """Add metrics to history, maintaining size limit."""
        # Create a copy to avoid reference issues
        import copy
        metrics_copy = copy.deepcopy(metrics)

        self._metric_history.append(metrics_copy)

        # Maintain history size
        if len(self._metric_history) > self._history_size:
            self._metric_history = self._metric_history[-self._history_size:]

    async def _collection_loop(self):
        """Background loop for metric collection."""
        while True:
            try:
                await self.collect_once()
                await asyncio.sleep(self._collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(self._collection_interval)

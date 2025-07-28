"""
Monitoring and metrics export for the morphogenetic message bus.

This module provides:
- Real-time metrics collection and aggregation
- Export to various monitoring systems (Prometheus, CloudWatch, etc.)
- Performance profiling and tracing
- Health checks and alerting
"""

import asyncio
import json
import logging
import time
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Statistical summary


@dataclass
class Metric:
    """Single metric definition."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """Metric value with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsExporter(ABC):
    """Abstract base for metrics exporters."""

    @abstractmethod
    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics batch."""
        pass

    @abstractmethod
    async def close(self):
        """Close exporter connections."""
        pass


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus format."""

    def __init__(self, pushgateway_url: str, job_name: str = "morphogenetic"):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.metric_definitions: Dict[str, Metric] = {}

    def register_metric(self, metric: Metric):
        """Register a metric definition."""
        self.metric_definitions[metric.name] = metric

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics to Prometheus pushgateway."""
        try:
            # Format metrics in Prometheus exposition format
            exposition_data = self._format_metrics(metrics)

            # Would send to pushgateway here
            # For now, just log
            logger.debug("Prometheus export: %d metrics", len(metrics))

            return True

        except Exception as e:
            logger.error("Prometheus export error: %s", e)
            return False

    def _format_metrics(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []

        for metric_name, values in metrics.items():
            metric_def = self.metric_definitions.get(metric_name)
            if not metric_def:
                continue

            # Add HELP and TYPE
            lines.append(f"# HELP {metric_name} {metric_def.description}")
            lines.append(f"# TYPE {metric_name} {metric_def.type.value}")

            # Add values
            for value in values:
                label_str = self._format_labels(value.labels)
                lines.append(f"{metric_name}{label_str} {value.value}")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""

        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(pairs) + "}"

    async def close(self):
        """Close exporter."""
        pass


class CloudWatchExporter(MetricsExporter):
    """Export metrics to AWS CloudWatch."""

    def __init__(self, namespace: str = "Morphogenetic", region: str = "us-east-1"):
        self.namespace = namespace
        self.region = region
        # Would initialize AWS client here

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics to CloudWatch."""
        try:
            # Format for CloudWatch
            metric_data = []

            for metric_name, values in metrics.items():
                for value in values:
                    metric_datum = {
                        "MetricName": metric_name,
                        "Value": value.value,
                        "Timestamp": value.timestamp,
                        "Dimensions": [
                            {"Name": k, "Value": v}
                            for k, v in value.labels.items()
                        ]
                    }
                    metric_data.append(metric_datum)

            # Would send to CloudWatch here
            logger.debug("CloudWatch export: %d metrics", len(metric_data))

            return True

        except Exception as e:
            logger.error("CloudWatch export error: %s", e)
            return False

    async def close(self):
        """Close exporter."""
        pass


class OpenTelemetryExporter(MetricsExporter):
    """Export metrics using OpenTelemetry."""

    def __init__(self, endpoint: str, service_name: str = "morphogenetic"):
        self.endpoint = endpoint
        self.service_name = service_name
        # Would initialize OTLP client here

    async def export(self, metrics: Dict[str, List[MetricValue]]) -> bool:
        """Export metrics via OTLP."""
        try:
            # Convert to OTLP format
            # This would use the OpenTelemetry SDK

            logger.debug("OpenTelemetry export: %d metrics", len(metrics))
            return True

        except Exception as e:
            logger.error("OpenTelemetry export error: %s", e)
            return False

    async def close(self):
        """Close exporter."""
        pass


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    # Collection
    collection_interval_ms: int = 10000
    aggregation_window_ms: int = 60000

    # Export
    export_interval_ms: int = 30000
    export_batch_size: int = 1000

    # Retention
    metric_retention_seconds: int = 3600

    # Health checks
    enable_health_checks: bool = True
    health_check_interval_ms: int = 5000

    # Tracing
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1


class MessageBusMonitor:
    """
    Comprehensive monitoring for the message bus system.
    
    Features:
    - Multi-backend export support
    - Automatic metric aggregation
    - Health check monitoring
    - Performance profiling
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        exporters: Optional[List[MetricsExporter]] = None
    ):
        self.config = config or MonitoringConfig()
        self.exporters = exporters or []

        # Metric definitions
        self._define_standard_metrics()

        # Metric storage
        self.metric_values: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Aggregated metrics
        self.aggregated_metrics: Dict[str, List[MetricValue]] = {}

        # Health checks
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, bool] = {}

        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._export_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _define_standard_metrics(self):
        """Define standard message bus metrics."""
        self.metrics = {
            # Message metrics
            "messages_published": Metric(
                "messages_published",
                MetricType.COUNTER,
                "Total messages published",
                labels=["topic", "type"]
            ),
            "messages_received": Metric(
                "messages_received",
                MetricType.COUNTER,
                "Total messages received",
                labels=["topic", "type"]
            ),
            "message_processing_duration": Metric(
                "message_processing_duration",
                MetricType.HISTOGRAM,
                "Message processing duration",
                labels=["topic", "handler"],
                unit="seconds",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            ),

            # Queue metrics
            "queue_size": Metric(
                "queue_size",
                MetricType.GAUGE,
                "Current queue size",
                labels=["queue_name"]
            ),
            "queue_lag": Metric(
                "queue_lag",
                MetricType.GAUGE,
                "Queue processing lag",
                labels=["queue_name"],
                unit="seconds"
            ),

            # Command metrics
            "commands_executed": Metric(
                "commands_executed",
                MetricType.COUNTER,
                "Total commands executed",
                labels=["command_type", "status"]
            ),
            "command_duration": Metric(
                "command_duration",
                MetricType.HISTOGRAM,
                "Command execution duration",
                labels=["command_type"],
                unit="seconds",
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            ),

            # Connection metrics
            "active_connections": Metric(
                "active_connections",
                MetricType.GAUGE,
                "Active message bus connections",
                labels=["client_type"]
            ),
            "connection_errors": Metric(
                "connection_errors",
                MetricType.COUNTER,
                "Connection error count",
                labels=["error_type"]
            ),

            # System metrics
            "memory_usage": Metric(
                "memory_usage",
                MetricType.GAUGE,
                "Memory usage",
                unit="bytes"
            ),
            "cpu_usage": Metric(
                "cpu_usage",
                MetricType.GAUGE,
                "CPU usage percentage",
                unit="percent"
            )
        }

        # Register with exporters
        for exporter in self.exporters:
            if hasattr(exporter, 'register_metric'):
                for metric in self.metrics.values():
                    exporter.register_metric(metric)

    async def start(self):
        """Start monitoring system."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._export_task = asyncio.create_task(self._export_loop())

        if self.config.enable_health_checks:
            self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info("Message bus monitor started with %d exporters",
                   len(self.exporters))

    async def stop(self):
        """Stop monitoring system."""
        self._running = False

        # Cancel tasks
        for task in [self._collection_task, self._export_task, self._health_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close exporters
        for exporter in self.exporters:
            await exporter.close()

        logger.info("Message bus monitor stopped")

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        if metric_name not in self.metrics:
            logger.warning("Unknown metric: %s", metric_name)
            return

        async with self._lock:
            metric_value = MetricValue(
                value=value,
                labels=labels or {},
                timestamp=time.time()
            )

            self.metric_values[metric_name].append(metric_value)

    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.health_status[name] = True  # Assume healthy initially

    async def _collection_loop(self):
        """Background task for metric collection."""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval_ms / 1000)

                # Aggregate metrics
                await self._aggregate_metrics()

                # Clean old metrics
                await self._cleanup_old_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Collection error: %s", e)

    async def _aggregate_metrics(self):
        """Aggregate raw metrics."""
        current_time = time.time()
        window_start = current_time - (self.config.aggregation_window_ms / 1000)

        async with self._lock:
            aggregated = {}

            for metric_name, values in self.metric_values.items():
                metric_def = self.metrics[metric_name]

                # Filter values in window
                window_values = [
                    v for v in values
                    if v.timestamp >= window_start
                ]

                if not window_values:
                    continue

                # Aggregate based on metric type
                if metric_def.type == MetricType.COUNTER:
                    # Sum for counters
                    aggregated[metric_name] = self._aggregate_counter(window_values)

                elif metric_def.type == MetricType.GAUGE:
                    # Latest value for gauges
                    aggregated[metric_name] = [window_values[-1]]

                elif metric_def.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                    # Calculate percentiles
                    aggregated[metric_name] = self._aggregate_histogram(
                        window_values, metric_def
                    )

            self.aggregated_metrics = aggregated

    def _aggregate_counter(self, values: List[MetricValue]) -> List[MetricValue]:
        """Aggregate counter values by labels."""
        by_labels = defaultdict(float)

        for value in values:
            label_key = json.dumps(value.labels, sort_keys=True)
            by_labels[label_key] += value.value

        return [
            MetricValue(
                value=total,
                labels=json.loads(label_key),
                timestamp=time.time()
            )
            for label_key, total in by_labels.items()
        ]

    def _aggregate_histogram(
        self,
        values: List[MetricValue],
        metric_def: Metric
    ) -> List[MetricValue]:
        """Aggregate histogram values."""
        # Group by labels
        by_labels = defaultdict(list)

        for value in values:
            label_key = json.dumps(value.labels, sort_keys=True)
            by_labels[label_key].append(value.value)

        results = []

        for label_key, label_values in by_labels.items():
            labels = json.loads(label_key)

            # Calculate percentiles
            label_values.sort()
            count = len(label_values)

            # Standard percentiles
            percentiles = {
                "p50": label_values[int(count * 0.5)],
                "p90": label_values[int(count * 0.9)],
                "p95": label_values[int(count * 0.95)],
                "p99": label_values[int(count * 0.99)],
                "min": label_values[0],
                "max": label_values[-1],
                "avg": sum(label_values) / count,
                "count": count
            }

            # Create metric for each percentile
            for p_name, p_value in percentiles.items():
                p_labels = labels.copy()
                p_labels["quantile"] = p_name

                results.append(MetricValue(
                    value=p_value,
                    labels=p_labels,
                    timestamp=time.time()
                ))

        return results

    async def _cleanup_old_metrics(self):
        """Remove old metric values."""
        cutoff_time = time.time() - self.config.metric_retention_seconds

        async with self._lock:
            for metric_name, values in self.metric_values.items():
                # Remove old values
                while values and values[0].timestamp < cutoff_time:
                    values.popleft()

    async def _export_loop(self):
        """Background task for metric export."""
        while self._running:
            try:
                await asyncio.sleep(self.config.export_interval_ms / 1000)

                # Export aggregated metrics
                if self.aggregated_metrics:
                    await self._export_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Export error: %s", e)

    async def _export_metrics(self):
        """Export metrics to all configured exporters."""
        # Get current aggregated metrics
        async with self._lock:
            metrics_to_export = self.aggregated_metrics.copy()

        # Export to each backend
        export_tasks = []
        for exporter in self.exporters:
            task = asyncio.create_task(exporter.export(metrics_to_export))
            export_tasks.append(task)

        # Wait for all exports
        results = await asyncio.gather(*export_tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Export failed for %s: %s",
                           type(self.exporters[i]).__name__, result)

    async def _health_check_loop(self):
        """Background task for health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_ms / 1000)

                # Run all health checks
                await self._run_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", e)

    async def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                # Run check
                if asyncio.iscoroutinefunction(check_func):
                    healthy = await check_func()
                else:
                    healthy = check_func()

                # Update status
                old_status = self.health_status.get(name, True)
                self.health_status[name] = healthy

                # Log status changes
                if old_status != healthy:
                    if healthy:
                        logger.info("Health check %s recovered", name)
                    else:
                        logger.warning("Health check %s failed", name)

                # Record metric
                await self.record_metric(
                    "health_check_status",
                    1.0 if healthy else 0.0,
                    {"check_name": name}
                )

            except Exception as e:
                logger.error("Health check %s error: %s", name, e)
                self.health_status[name] = False

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        overall_healthy = all(self.health_status.values())

        return {
            "healthy": overall_healthy,
            "checks": self.health_status.copy(),
            "timestamp": time.time()
        }

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        async with self._lock:
            summary = {
                "total_metrics": len(self.metrics),
                "active_metrics": len(self.metric_values),
                "total_values": sum(len(v) for v in self.metric_values.values()),
                "exporters": [type(e).__name__ for e in self.exporters],
                "aggregated_metrics": len(self.aggregated_metrics)
            }

        return summary

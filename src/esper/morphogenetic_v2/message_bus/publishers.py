"""Publishers for the morphogenetic message bus system.

This module provides specialized publishers for telemetry data and events,
with features like batching, compression, and claim-check pattern support.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from .clients import MessageBusClient
from .schemas import AlertSeverity
from .schemas import AlertType
from .schemas import BaseMessage
from .schemas import LayerHealthReport
from .schemas import PerformanceAlert
from .schemas import SeedMetricsSnapshot
from .schemas import StateTransitionEvent
from .schemas import TelemetryBatch
from .schemas import create_topic_name

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry publishing."""
    batch_size: int = 100
    batch_window_ms: int = 100
    compression: Optional[str] = "zstd"  # None, "zstd", "gzip"
    compression_level: int = 3
    enable_aggregation: bool = True
    aggregation_window_s: float = 10.0
    claim_check_threshold: int = 100 * 1024  # 100KB
    claim_check_storage: Optional[str] = None  # Path to claim check storage
    telemetry_retention_s: float = 3600.0  # 1 hour
    anomaly_detection: bool = True
    anomaly_threshold_stddev: float = 3.0
    performance_tracking: bool = True

    def validate(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_window_ms <= 0:
            raise ValueError("batch_window_ms must be positive")
        if self.compression not in [None, "zstd", "gzip"]:
            raise ValueError("compression must be None, 'zstd', or 'gzip'")
        if self.compression_level < 1 or self.compression_level > 9:
            raise ValueError("compression_level must be between 1 and 9")


class TelemetryPublisher:
    """High-performance telemetry publisher with batching and compression."""

    def __init__(self, client: MessageBusClient, config: TelemetryConfig):
        config.validate()
        self.client = client
        self.config = config

        # Batching
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.current_batch: List[BaseMessage] = []
        self.batch_start_time: float = time.time()

        # Aggregation
        self.aggregation_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.aggregation_window_start: float = time.time()

        # Claim check storage
        if config.claim_check_storage:
            self.claim_check_path = Path(config.claim_check_storage)
            self.claim_check_path.mkdir(parents=True, exist_ok=True)
        else:
            self.claim_check_path = None

        # Anomaly detection
        if config.anomaly_detection:
            self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
            self.anomaly_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Performance tracking
        self.stats = {
            "messages_published": 0,
            "batches_sent": 0,
            "bytes_sent": 0,
            "compression_ratio": 1.0,
            "claim_checks_created": 0,
            "anomalies_detected": 0,
            "publish_latency_ms": deque(maxlen=100)
        }

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the telemetry publisher."""
        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._batch_publisher())
        )

        if self.config.enable_aggregation:
            self._tasks.append(
                asyncio.create_task(self._aggregation_publisher())
            )

        if self.config.anomaly_detection:
            self._tasks.append(
                asyncio.create_task(self._anomaly_detector())
            )

        logger.info("TelemetryPublisher started")

    async def stop(self):
        """Stop the telemetry publisher."""
        self._running = False

        # Flush any pending data
        await self._flush_batch()
        await self._flush_aggregation()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("TelemetryPublisher stopped")

    async def publish_layer_health(self, layer_id: str, health_data: torch.Tensor):
        """Publish layer health report with automatic batching.
        
        Args:
            layer_id: Layer identifier
            health_data: Tensor of shape (num_seeds, num_metrics)
        """
        start_time = time.time()

        # Convert GPU tensor to CPU efficiently
        health_dict = self._tensor_to_dict(health_data)

        # Calculate summary statistics
        active_seeds = sum(1 for metrics in health_dict.values()
                          if metrics.get('lifecycle_state', 0) > 0)

        performance_summary = self._calculate_summary(health_dict)

        # Detect anomalies in real-time
        anomalies = []
        if self.config.anomaly_detection:
            anomalies = await self._detect_anomalies(layer_id, performance_summary)

        # Create report
        report = LayerHealthReport(
            source=f"kasmina_layer_{layer_id}",
            layer_id=layer_id,
            total_seeds=len(health_dict),
            active_seeds=active_seeds,
            health_metrics=health_dict,
            performance_summary=performance_summary,
            telemetry_window=(self.aggregation_window_start, time.time()),
            anomalies=anomalies
        )

        # Add to batch or aggregation
        if self.config.enable_aggregation:
            self.aggregation_buffer[layer_id].append(report.to_dict())
        else:
            await self.batch_queue.put(report)

        # Track performance
        publish_latency = (time.time() - start_time) * 1000
        self.stats["publish_latency_ms"].append(publish_latency)

    async def publish_seed_metrics(self, layer_id: str, seed_id: int,
                                  metrics: Dict[str, float],
                                  lifecycle_state: str,
                                  blueprint_id: Optional[int] = None):
        """Publish detailed metrics for a single seed.
        
        Args:
            layer_id: Layer identifier
            seed_id: Seed identifier
            metrics: Dictionary of metric name to value
            lifecycle_state: Current lifecycle state
            blueprint_id: Optional blueprint identifier
        """
        snapshot = SeedMetricsSnapshot(
            source=f"kasmina_layer_{layer_id}_seed_{seed_id}",
            layer_id=layer_id,
            seed_id=seed_id,
            lifecycle_state=lifecycle_state,
            blueprint_id=blueprint_id,
            metrics=metrics,
            error_count=int(metrics.get('error_count', 0)),
            warning_count=int(metrics.get('warning_count', 0))
        )

        # Detect seed-level anomalies
        if self.config.anomaly_detection:
            for metric_name, value in metrics.items():
                key = f"{layer_id}_{seed_id}_{metric_name}"
                self.metric_history[key].append(value)

                if len(self.metric_history[key]) > 10:
                    if self._is_anomaly(self.metric_history[key], value):
                        alert = PerformanceAlert(
                            source=snapshot.source,
                            layer_id=layer_id,
                            seed_id=seed_id,
                            alert_type=AlertType.ANOMALY,
                            severity=AlertSeverity.WARNING,
                            metric_name=metric_name,
                            metric_value=value,
                            details={"history": list(self.metric_history[key])}
                        )
                        await self._publish_alert(alert)

        await self.batch_queue.put(snapshot)

    async def publish_event(self, event: Union[StateTransitionEvent, Any]):
        """Publish an event immediately without batching.
        
        Args:
            event: Event to publish
        """
        # Events are published immediately for low latency
        topic = self._get_event_topic(event)
        await self.client.publish(topic, event)
        self.stats["messages_published"] += 1

    def add_anomaly_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add a callback for anomaly alerts.
        
        Args:
            callback: Async function to call when anomaly detected
        """
        self.anomaly_callbacks.append(callback)

    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """Convert tensor to dictionary efficiently."""
        # Move to CPU if on GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert to numpy for efficient processing
        array = tensor.numpy()

        # Create dictionary
        result = {}
        for seed_id in range(array.shape[0]):
            metrics = {
                'lifecycle_state': float(array[seed_id, 0]),
                'loss': float(array[seed_id, 1]) if array.shape[1] > 1 else 0.0,
                'accuracy': float(array[seed_id, 2]) if array.shape[1] > 2 else 0.0,
                'compute_time_ms': float(array[seed_id, 3]) if array.shape[1] > 3 else 0.0
            }

            # Add additional metrics if present
            for i in range(4, min(array.shape[1], 10)):
                metrics[f'metric_{i}'] = float(array[seed_id, i])

            result[seed_id] = metrics

        return result

    def _calculate_summary(self, health_dict: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Calculate summary statistics from health metrics."""
        if not health_dict:
            return {}

        # Aggregate metrics
        all_metrics = defaultdict(list)
        for metrics in health_dict.values():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

        # Calculate statistics
        summary = {}
        for key, values in all_metrics.items():
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))
                summary[f"{key}_p50"] = float(np.percentile(values, 50))
                summary[f"{key}_p95"] = float(np.percentile(values, 95))
                summary[f"{key}_p99"] = float(np.percentile(values, 99))

        return summary

    def _is_anomaly(self, history: deque, value: float) -> bool:
        """Check if value is anomalous based on history."""
        if len(history) < 10:
            return False

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return value != mean

        z_score = abs(value - mean) / std
        return z_score > self.config.anomaly_threshold_stddev

    async def _detect_anomalies(self, layer_id: str,
                               summary: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in layer metrics."""
        anomalies = []

        for metric_name, value in summary.items():
            if metric_name.endswith('_mean'):
                base_metric = metric_name[:-5]
                key = f"{layer_id}_{base_metric}"

                self.metric_history[key].append(value)

                if len(self.metric_history[key]) > 10 and self._is_anomaly(self.metric_history[key], value):
                    anomaly = {
                        "metric": base_metric,
                        "value": value,
                        "expected_range": (
                            float(np.mean(self.metric_history[key]) -
                                  self.config.anomaly_threshold_stddev * np.std(self.metric_history[key])),
                            float(np.mean(self.metric_history[key]) +
                                  self.config.anomaly_threshold_stddev * np.std(self.metric_history[key]))
                        ),
                        "severity": "high" if abs(value - np.mean(self.metric_history[key])) >
                                   5 * np.std(self.metric_history[key]) else "medium"
                    }
                    anomalies.append(anomaly)
                    self.stats["anomalies_detected"] += 1

        return anomalies

    async def _publish_alert(self, alert: PerformanceAlert):
        """Publish an alert and notify callbacks."""
        # Publish to message bus
        topic = create_topic_name("alert", alert.layer_id, alert.seed_id)
        await self.client.publish(topic, alert)

        # Notify callbacks
        for callback in self.anomaly_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error("Anomaly callback error: %s", e)

    def _get_event_topic(self, event: BaseMessage) -> str:
        """Determine topic for an event."""
        if isinstance(event, StateTransitionEvent):
            return create_topic_name("event.transition", event.layer_id, event.seed_id)
        else:
            # Generic event topic
            return create_topic_name("event", getattr(event, 'layer_id', None))

    async def _batch_publisher(self):
        """Background task for batched publishing."""
        while self._running:
            try:
                # Collect messages for batch
                deadline = time.time() + (self.config.batch_window_ms / 1000)

                while len(self.current_batch) < self.config.batch_size and time.time() < deadline:
                    try:
                        timeout = max(0.001, deadline - time.time())
                        message = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=timeout
                        )
                        self.current_batch.append(message)

                    except asyncio.TimeoutError:
                        break

                # Publish batch if we have messages
                if self.current_batch:
                    await self._flush_batch()

            except Exception as e:
                logger.error("Batch publisher error: %s", e)
                await asyncio.sleep(0.1)

    async def _flush_batch(self):
        """Flush current batch."""
        if not self.current_batch:
            return

        start_time = time.time()

        # Create batch message
        batch = TelemetryBatch(
            source="telemetry_publisher",
            messages=self.current_batch.copy(),
            compression=self.config.compression
        )

        # Check if we need claim check
        batch_size = len(json.dumps(batch.to_dict()))

        if self.claim_check_path and batch_size > self.config.claim_check_threshold:
            # Store payload and send reference
            claim_check_id = await self._store_claim_check(batch)
            batch.messages = []  # Clear messages
            batch.metadata["claim_check_id"] = claim_check_id
            batch.metadata["claim_check_size"] = batch_size

        # Publish batch
        topic = create_topic_name("telemetry.batch")
        await self.client.publish(topic, batch)

        # Update stats
        self.stats["messages_published"] += len(self.current_batch)
        self.stats["batches_sent"] += 1
        self.stats["bytes_sent"] += batch_size

        # Clear batch
        self.current_batch.clear()
        self.batch_start_time = time.time()

        # Log performance
        latency = (time.time() - start_time) * 1000
        if latency > 100:
            logger.warning("Slow batch publish: %.1fms", latency)

    async def _store_claim_check(self, batch: TelemetryBatch) -> str:
        """Store large payload and return claim check ID."""
        claim_check_id = str(uuid.uuid4())

        # Serialize and compress
        data = json.dumps(batch.to_dict())

        if self.config.compression == "zstd":
            import zstandard
            cctx = zstandard.ZstdCompressor(level=self.config.compression_level)
            compressed = cctx.compress(data.encode())
        elif self.config.compression == "gzip":
            import gzip
            compressed = gzip.compress(data.encode(), compresslevel=self.config.compression_level)
        else:
            compressed = data.encode()

        # Store to file
        file_path = self.claim_check_path / f"{claim_check_id}.bin"
        file_path.write_bytes(compressed)

        # Track compression ratio
        self.stats["compression_ratio"] = len(data) / len(compressed)
        self.stats["claim_checks_created"] += 1

        logger.debug("Created claim check %s (%s bytes)", claim_check_id, len(compressed))

        return claim_check_id

    async def _aggregation_publisher(self):
        """Background task for aggregation window publishing."""
        while self._running:
            try:
                # Wait for aggregation window
                await asyncio.sleep(self.config.aggregation_window_s)

                # Flush aggregation buffer
                await self._flush_aggregation()

            except Exception as e:
                logger.error("Aggregation publisher error: %s", e)

    async def _flush_aggregation(self):
        """Flush aggregation buffer."""
        if not self.aggregation_buffer:
            return

        # Process each layer's aggregated data
        for layer_id, reports in self.aggregation_buffer.items():
            if not reports:
                continue

            # Aggregate multiple reports into one
            aggregated = self._aggregate_reports(reports)

            # Publish aggregated report
            topic = create_topic_name("telemetry.aggregated", layer_id)
            await self.client.publish(topic, aggregated)

        # Clear buffer and reset window
        self.aggregation_buffer.clear()
        self.aggregation_window_start = time.time()

    def _aggregate_reports(self, reports: List[Dict[str, Any]]) -> LayerHealthReport:
        """Aggregate multiple reports into one."""
        # Combine all metrics
        all_metrics = {}
        total_seeds = 0
        active_seeds = 0

        for report in reports:
            for seed_id, metrics in report.get('health_metrics', {}).items():
                if seed_id not in all_metrics:
                    all_metrics[seed_id] = []
                all_metrics[seed_id].append(metrics)

            total_seeds = max(total_seeds, report.get('total_seeds', 0))
            active_seeds = max(active_seeds, report.get('active_seeds', 0))

        # Average metrics over time window
        aggregated_metrics = {}
        for seed_id, metrics_list in all_metrics.items():
            aggregated = {}

            # Get all metric keys
            all_keys = set()
            for m in metrics_list:
                all_keys.update(m.keys())

            # Average each metric
            for key in all_keys:
                values = [m.get(key, 0) for m in metrics_list if key in m]
                if values:
                    aggregated[key] = float(np.mean(values))

            aggregated_metrics[int(seed_id)] = aggregated

        # Create aggregated report
        return LayerHealthReport(
            source=reports[0].get('source', ''),
            layer_id=reports[0].get('layer_id', ''),
            total_seeds=total_seeds,
            active_seeds=active_seeds,
            health_metrics=aggregated_metrics,
            performance_summary=self._calculate_summary(aggregated_metrics),
            telemetry_window=(self.aggregation_window_start, time.time())
        )

    async def _anomaly_detector(self):
        """Background task for anomaly detection."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                # Analyze metric trends
                for key, history in self.metric_history.items():
                    if len(history) < 20:
                        continue

                    # Detect trend changes
                    recent = list(history)[-10:]
                    older = list(history)[-20:-10]

                    recent_mean = np.mean(recent)
                    older_mean = np.mean(older)

                    # Significant change detection
                    if abs(recent_mean - older_mean) > 2 * np.std(history):
                        parts = key.split('_')
                        layer_id = parts[0] if parts else ""

                        alert = PerformanceAlert(
                            source="anomaly_detector",
                            layer_id=layer_id,
                            alert_type=AlertType.DEGRADATION if recent_mean < older_mean else AlertType.IMPROVEMENT,
                            severity=AlertSeverity.INFO,
                            metric_name=key,
                            metric_value=recent_mean,
                            details={
                                "trend": "decreasing" if recent_mean < older_mean else "increasing",
                                "change_percent": abs(recent_mean - older_mean) / older_mean * 100
                            }
                        )

                        await self._publish_alert(alert)

            except Exception as e:
                logger.error("Anomaly detector error: %s", e)

    async def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        stats = self.stats.copy()

        # Calculate latency statistics
        if self.stats["publish_latency_ms"]:
            latencies = list(self.stats["publish_latency_ms"])
            stats["latency_p50"] = float(np.percentile(latencies, 50))
            stats["latency_p95"] = float(np.percentile(latencies, 95))
            stats["latency_p99"] = float(np.percentile(latencies, 99))

        # Add current state
        stats["batch_queue_size"] = self.batch_queue.qsize()
        stats["current_batch_size"] = len(self.current_batch)
        stats["aggregation_buffer_size"] = sum(len(v) for v in self.aggregation_buffer.values())

        return stats


class EventPublisher:
    """Specialized publisher for events with guaranteed delivery."""

    def __init__(self, client: MessageBusClient):
        self.client = client
        self.stats = {
            "events_published": 0,
            "events_failed": 0,
            "retry_count": 0
        }

    async def publish_state_transition(self, layer_id: str, seed_id: int,
                                     from_state: str, to_state: str,
                                     reason: str = "",
                                     metrics: Optional[Dict[str, float]] = None):
        """Publish a state transition event.
        
        Args:
            layer_id: Layer identifier
            seed_id: Seed identifier  
            from_state: Previous state
            to_state: New state
            reason: Reason for transition
            metrics: Optional metrics snapshot
        """
        event = StateTransitionEvent(
            source=f"kasmina_layer_{layer_id}",
            layer_id=layer_id,
            seed_id=seed_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metrics_snapshot=metrics or {},
            transition_duration_ms=0.0,  # Will be set by layer
            triggered_by="system"
        )

        await self._publish_with_retry(event)

    async def _publish_with_retry(self, event: BaseMessage, max_retries: int = 3):
        """Publish event with retry logic."""
        topic = self._get_event_topic(event)

        for attempt in range(max_retries):
            try:
                await self.client.publish(topic, event)
                self.stats["events_published"] += 1
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    self.stats["retry_count"] += 1
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats["events_failed"] += 1
                    logger.error("Failed to publish event after %s attempts: %s", max_retries, e)
                    raise

    def _get_event_topic(self, event: BaseMessage) -> str:
        """Determine topic for an event."""
        event_type = event.__class__.__name__.lower()

        if hasattr(event, 'layer_id'):
            return create_topic_name(f"event.{event_type}", event.layer_id)
        else:
            return create_topic_name(f"event.{event_type}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        return self.stats.copy()

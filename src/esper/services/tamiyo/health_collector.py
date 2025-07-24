"""
Production health signal collection with intelligent filtering.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from esper.contracts.operational import HealthSignal
from esper.services.oona_client import OonaClient

logger = logging.getLogger(__name__)


class HealthSignalBuffer:
    """High-performance circular buffer for health signals."""

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.layer_index: Dict[str, List[int]] = {}

    def add(self, signal: HealthSignal):
        """Add signal to buffer."""
        # Add to main buffer
        position = len(self.buffer)
        self.buffer.append(signal)

        # Update layer index
        layer_key = f"{signal.layer_id}:{signal.seed_id}"
        if layer_key not in self.layer_index:
            self.layer_index[layer_key] = []
        self.layer_index[layer_key].append(position)

    def get_recent(self, window_size: int = 1000) -> List[HealthSignal]:
        """Get recent signals."""
        return list(self.buffer)[-window_size:]

    def get_by_layer(self, layer_id: str, limit: int = 100) -> List[HealthSignal]:
        """Get signals for specific layer."""
        results = []
        for key, positions in self.layer_index.items():
            if key.startswith(f"{layer_id}:"):
                for pos in positions[-limit:]:
                    if pos < len(self.buffer):
                        results.append(self.buffer[pos])
        return results


class SignalFilterEngine:
    """Intelligent filtering for health signals."""

    def __init__(self, anomaly_threshold: float = 2.0):
        self.priority_thresholds = {
            "health_score": 0.3,      # Alert if below
            "error_rate": 0.1,        # Alert if above
            "latency_spike": 2.0,     # Alert if 2x normal
        }
        self.baseline_latencies: Dict[str, float] = {}
        self.error_layers = set()
        self.anomaly_threshold = anomaly_threshold

        # Anomaly detection state
        self.health_history: Dict[int, deque] = {}  # layer_id -> deque of health scores
        self.history_window = 20  # Keep last 20 signals per layer

    def should_process(self, signal: HealthSignal) -> bool:
        """Determine if signal should be processed."""
        # Track health history for anomaly detection
        if signal.layer_id not in self.health_history:
            self.health_history[signal.layer_id] = deque(maxlen=self.history_window)
        self.health_history[signal.layer_id].append(signal.health_score)

        # Always process unhealthy signals
        if signal.health_score < self.priority_thresholds["health_score"]:
            return True

        # Process anomalous signals
        if self._is_anomalous(signal):
            return True

        # Process unstable gradients - critical for training stability
        if signal.gradient_norm > 5.0 or signal.gradient_variance > 1.0:
            return True

        # Process low gradient sign stability - indicates oscillating training
        if signal.gradient_sign_stability < 0.5:
            return True

        # Process abnormal parameter norm ratios
        if signal.param_norm_ratio < 0.5 or signal.param_norm_ratio > 2.0:
            return True

        # Process high error rates
        if signal.error_count > 0:
            error_rate = signal.error_count / max(signal.total_executions, 1)
            if error_rate > self.priority_thresholds["error_rate"]:
                self.error_layers.add(signal.layer_id)
                return True

        # Process latency spikes
        baseline = self.baseline_latencies.get(signal.layer_id, signal.execution_latency)
        if signal.execution_latency > baseline * self.priority_thresholds["latency_spike"]:
            return True

        # Update baseline
        self.baseline_latencies[signal.layer_id] = (
            0.9 * baseline + 0.1 * signal.execution_latency
        )

        # Sample some healthy signals (reduced rate for throttling)
        return hash(signal.timestamp) % 8 == 0  # 12.5% sampling rate

    def calculate_priority(self, signal: HealthSignal) -> float:
        """Calculate processing priority."""
        priority = 0.0

        # Health score component (inverted - lower health = higher priority)
        # Scale to keep healthy signals (0.8-1.0) in the 0.3-0.7 range
        health_factor = (1.0 - signal.health_score) * 3.5  # Reduced from 10.0
        priority += health_factor

        # Gradient instability component - critical for training
        gradient_priority = 0.0
        if signal.gradient_norm > 10.0:
            gradient_priority += 5.0  # Exploding gradients - very high priority
        elif signal.gradient_norm > 5.0:
            gradient_priority += 2.0

        if signal.gradient_variance > 2.0:
            gradient_priority += 3.0  # High variance indicates instability

        if signal.gradient_sign_stability < 0.3:
            gradient_priority += 4.0  # Oscillating gradients - needs attention

        priority += gradient_priority

        # Parameter health component
        if signal.param_norm_ratio < 0.5 or signal.param_norm_ratio > 2.0:
            priority += 2.0  # Unhealthy parameter ratios

        # Error component
        if signal.error_count > 0:
            priority += min(signal.error_count, 10) * 2.0

        # Performance component
        if signal.cache_hit_rate < 0.5 and signal.total_executions > 100:
            priority += 1.0  # Poor cache performance

        # Latency component
        if signal.execution_latency > 10.0:  # >10ms is concerning
            priority += signal.execution_latency / 10.0

        # Base priority for all signals
        priority += 0.3  # Ensure minimum priority

        return priority

    def _is_anomalous(self, signal: HealthSignal) -> bool:
        """Detect anomalous signals using statistical methods."""
        history = self.health_history.get(signal.layer_id, deque())

        # Need at least 10 samples for meaningful statistics
        if len(history) < 10:
            return False

        # Calculate mean and std of recent history (excluding current)
        recent_scores = list(history)[:-1]  # Exclude the current signal we just added
        if not recent_scores:
            return False

        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)

        # If std is too small, use a minimum threshold
        if std_score < 0.01:
            std_score = 0.01

        # Check if current score is outside N-sigma bounds
        z_score = abs(signal.health_score - mean_score) / std_score
        return z_score > self.anomaly_threshold


class CollectionStatistics:
    """Track collection performance."""

    def __init__(self):
        self.signals_processed = 0
        self.signals_filtered = 0
        self.start_time = time.time()

    def record_signal_processed(self):
        """Record processed signal."""
        self.signals_processed += 1

    def record_signal_filtered(self):
        """Record filtered signal."""
        self.signals_filtered += 1

    def get_stats(self) -> Dict[str, float]:
        """Get collection statistics."""
        elapsed = time.time() - self.start_time
        return {
            "signals_processed": self.signals_processed,
            "signals_filtered": self.signals_filtered,
            "signals_per_second": self.signals_processed / elapsed if elapsed > 0 else 0,
            "filter_rate": self.signals_filtered / max(self.signals_processed, 1),
        }


class ErrorRecoveryIntegration:
    """Integration with Phase 1 error recovery system."""

    def convert_to_health_signal(self, error_event: Dict[str, Any]) -> Optional[HealthSignal]:
        """Convert error recovery event to health signal."""
        try:
            return HealthSignal(
                layer_id=error_event["layer_id"],
                seed_id=error_event.get("seed_id", 0),
                chunk_id=error_event.get("chunk_id", 0),
                epoch=error_event.get("epoch", 0),
                health_score=0.7 if error_event.get("recovery_success", False) else 0.1,  # Better health if recovered
                activation_variance=0.0,
                dead_neuron_ratio=0.0,
                avg_correlation=0.0,
                is_ready_for_transition=False,
                execution_latency=error_event.get("execution_latency", error_event.get("recovery_time_ms", 0)),
                error_count=0 if error_event.get("recovery_success", False) else 1,
                active_seeds=error_event.get("active_seeds", 1),
                total_seeds=error_event.get("total_seeds", 1),
                timestamp=error_event.get("timestamp", time.time()),
                # Training dynamics - errors indicate instability
                gradient_norm=error_event.get("gradient_norm", 10.0),  # High for errors
                gradient_variance=error_event.get("gradient_variance", 5.0),  # High variance
                gradient_sign_stability=error_event.get("gradient_sign_stability", 0.2),  # Low stability
                param_norm_ratio=error_event.get("param_norm_ratio", 0.5),  # Degraded ratio
                # Performance metrics
                total_executions=error_event.get("total_executions", 1),
                cache_hit_rate=error_event.get("cache_hit_rate", 0.0)  # No cache hits on error
            )
        except Exception as e:
            logger.warning(f"Failed to convert error event: {e}")
            return None


class HealthAggregator:
    """Aggregate health signals for analysis."""

    def aggregate_by_layer(
        self,
        signals: List[HealthSignal]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate signals by layer."""
        layer_aggregates = {}

        for signal in signals:
            if signal.layer_id not in layer_aggregates:
                layer_aggregates[signal.layer_id] = {
                    "count": 0,
                    "avg_health": 0.0,
                    "min_health": 1.0,
                    "max_health": 0.0,
                    "avg_latency": 0.0,
                    "error_rate": 0.0,
                }

            agg = layer_aggregates[signal.layer_id]
            agg["count"] += 1

            # Update aggregates
            agg["avg_health"] += signal.health_score
            agg["min_health"] = min(agg["min_health"], signal.health_score)
            agg["max_health"] = max(agg["max_health"], signal.health_score)
            agg["avg_latency"] += signal.execution_latency

            if signal.error_count > 0:
                agg["error_rate"] += 1

        # Finalize averages
        for layer_id, agg in layer_aggregates.items():
            if agg["count"] > 0:
                agg["avg_health"] /= agg["count"]
                agg["avg_latency"] /= agg["count"]
                agg["error_rate"] /= agg["count"]

        return layer_aggregates


class BatchProcessor:
    """Batch processing for efficiency."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.pending_batch = []

    def add(self, item: Any) -> Optional[List[Any]]:
        """Add item to batch, return full batch if ready."""
        self.pending_batch.append(item)

        if len(self.pending_batch) >= self.batch_size:
            batch = self.pending_batch
            self.pending_batch = []
            return batch

        return None

    def flush(self) -> List[Any]:
        """Flush pending batch."""
        batch = self.pending_batch
        self.pending_batch = []
        return batch


class ProductionHealthCollector:
    """
    Production-grade health signal collection with intelligent filtering.
    
    Integrates with Phase 1 error recovery system to collect and process
    health signals in real-time for policy training.
    """

    def __init__(
        self,
        oona_client: OonaClient,
        buffer_size: int = 50000,
        processing_batch_size: int = 1000
    ):
        self.oona_client = oona_client
        self.signal_buffer = HealthSignalBuffer(max_size=buffer_size)
        self.aggregator = HealthAggregator()
        self.filter_engine = SignalFilterEngine()

        # Performance optimization
        self.batch_processor = BatchProcessor(batch_size=processing_batch_size)
        self.statistics = CollectionStatistics()

        # Integration with Phase 1
        self.error_recovery_integration = ErrorRecoveryIntegration()

        # State management
        self._running = False
        self._collection_task = None

    async def start_intelligent_collection(self):
        """Start health signal collection with intelligent filtering."""

        # Subscribe to all telemetry streams

        # Store collection task for proper shutdown
        self._collection_task = None
        self._running = True

        # For production, would use real subscription
        # For now, simulate collection
        await self._simulated_collection_loop()

    async def stop_collection(self):
        """Stop health signal collection gracefully."""
        self._running = False
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

    async def get_recent_signals(self, count: int) -> List[HealthSignal]:
        """Get recent health signals from buffer."""
        return self.signal_buffer.get_recent(count)

    async def _simulated_collection_loop(self):
        """Simulated collection for testing."""
        epoch_counter = 0
        while self._running:
            try:
                # Simulate health signals with realistic gradient dynamics
                for i in range(10):
                    # Simulate different layer behaviors
                    layer_id = i % 4
                    health_base = 0.7 + 0.3 * (i % 3) / 3

                    # Simulate gradient dynamics based on layer health
                    if health_base > 0.8:  # Healthy layer
                        gradient_norm = np.random.lognormal(-2, 0.5)  # Low, stable gradients
                        gradient_variance = np.random.exponential(0.1)
                        gradient_sign_stability = 0.8 + np.random.normal(0, 0.1)
                        param_norm_ratio = 1.0 + np.random.normal(0, 0.05)
                    else:  # Struggling layer
                        gradient_norm = np.random.lognormal(0, 1.0)  # Higher, unstable gradients
                        gradient_variance = np.random.exponential(0.5)
                        gradient_sign_stability = 0.4 + np.random.normal(0, 0.2)
                        param_norm_ratio = 0.8 + np.random.normal(0, 0.1)

                    # Ensure bounds
                    gradient_sign_stability = max(0.0, min(1.0, gradient_sign_stability))
                    param_norm_ratio = max(0.1, param_norm_ratio)

                    # Simulate performance metrics
                    total_executions = 1000 + epoch_counter * 100 + i * 10
                    cache_hit_rate = 0.7 + 0.2 * health_base if total_executions > 100 else 0.3

                    signal = HealthSignal(
                        layer_id=layer_id,
                        seed_id=i % 4,
                        chunk_id=0,
                        epoch=int(time.time() / 100),
                        health_score=health_base,
                        activation_variance=0.1,
                        dead_neuron_ratio=0.02,
                        avg_correlation=0.85,
                        is_ready_for_transition=False,
                        execution_latency=1.0 + 0.5 * (i % 5),
                        error_count=1 if i % 10 == 0 else 0,
                        active_seeds=5,
                        total_seeds=10,
                        timestamp=time.time(),
                        # Training dynamics
                        gradient_norm=gradient_norm,
                        gradient_variance=gradient_variance,
                        gradient_sign_stability=gradient_sign_stability,
                        param_norm_ratio=param_norm_ratio,
                        # Performance metrics
                        total_executions=total_executions,
                        cache_hit_rate=cache_hit_rate
                    )

                    if self.filter_engine.should_process(signal):
                        self.filter_engine.calculate_priority(signal)
                        self.signal_buffer.add(signal)
                        self.statistics.record_signal_processed()
                    else:
                        self.statistics.record_signal_filtered()

                epoch_counter += 1
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(1.0)

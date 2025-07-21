"""
Production Health Signal Collection System for Tamiyo Intelligence

This module implements real-time health signal collection and processing,
integrating with Phase 1 execution system to provide training data for
the GNN policy network.
"""

import asyncio
import logging
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from esper.contracts.operational import HealthSignal
from esper.contracts.messages import OonaMessage, TopicNames
from esper.services.oona_client import OonaClient

logger = logging.getLogger(__name__)


@dataclass
class CollectionStatistics:
    """Statistics for health signal collection performance."""
    
    signals_processed: int = 0
    signals_dropped: int = 0
    signals_filtered: int = 0
    processing_time_total: float = 0.0
    last_processed_timestamp: float = 0.0
    error_count: int = 0
    
    def record_signal_processed(self, processing_time: float = 0.0):
        """Record successful signal processing."""
        self.signals_processed += 1
        self.processing_time_total += processing_time
        self.last_processed_timestamp = time.time()
    
    def record_signal_dropped(self):
        """Record dropped signal."""
        self.signals_dropped += 1
    
    def record_signal_filtered(self):
        """Record filtered signal."""
        self.signals_filtered += 1
    
    def record_error(self):
        """Record processing error."""
        self.error_count += 1
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per signal."""
        if self.signals_processed == 0:
            return 0.0
        return self.processing_time_total / self.signals_processed
    
    @property
    def throughput_per_second(self) -> float:
        """Signals processed per second."""
        elapsed = time.time() - self.last_processed_timestamp
        if elapsed == 0:
            return 0.0
        return self.signals_processed / elapsed
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "signals_processed": self.signals_processed,
            "signals_dropped": self.signals_dropped,
            "signals_filtered": self.signals_filtered,
            "error_count": self.error_count,
            "avg_processing_time_ms": self.avg_processing_time * 1000,
            "throughput_per_second": self.throughput_per_second,
            "total_processing_time": self.processing_time_total
        }


class SignalFilterEngine:
    """Intelligent filtering engine for health signals."""
    
    def __init__(self, anomaly_threshold: float = 2.0):
        self.anomaly_threshold = anomaly_threshold
        self.signal_history = defaultdict(list)  # layer_id -> list of recent signals
        self.history_size = 100
    
    def should_process(self, signal: HealthSignal) -> bool:
        """Determine if signal should be processed based on intelligent filtering."""
        # Always process signals with errors
        if signal.error_count > 0:
            return True
        
        # Always process signals indicating readiness for transition
        if signal.is_ready_for_transition:
            return True
        
        # Process signals with anomalous health scores
        if self._is_anomalous(signal):
            return True
        
        # Throttle normal signals (process every 10th)
        layer_key = f"{signal.layer_id}_{signal.seed_id}"
        history = self.signal_history[layer_key]
        
        if len(history) == 0 or len(history) % 10 == 0:
            return True
        
        return False
    
    def calculate_priority(self, signal: HealthSignal) -> float:
        """Calculate priority score for signal processing."""
        priority = 0.5  # Base priority
        
        # Higher priority for unhealthy signals
        if signal.health_score < 0.5:
            priority += 0.3
        
        # Higher priority for signals with errors
        if signal.error_count > 0:
            priority += 0.4 * min(signal.error_count / 5.0, 1.0)
        
        # Higher priority for signals ready for transition
        if signal.is_ready_for_transition:
            priority += 0.2
        
        # Higher priority for anomalous signals
        if self._is_anomalous(signal):
            priority += 0.3
        
        return min(priority, 1.0)
    
    def _is_anomalous(self, signal: HealthSignal) -> bool:
        """Check if signal is anomalous compared to recent history."""
        layer_key = f"{signal.layer_id}_{signal.seed_id}"
        history = self.signal_history[layer_key]
        
        if len(history) < 10:  # Need minimum history
            self._add_to_history(signal, layer_key)
            return False
        
        # Check if health score is anomalous
        recent_scores = [s.health_score for s in history[-10:]]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if std_score > 0:
            z_score = abs(signal.health_score - mean_score) / std_score
            is_anomalous = z_score > self.anomaly_threshold
        else:
            is_anomalous = False
        
        self._add_to_history(signal, layer_key)
        return is_anomalous
    
    def _add_to_history(self, signal: HealthSignal, layer_key: str):
        """Add signal to history and maintain size."""
        history = self.signal_history[layer_key]
        history.append(signal)
        
        if len(history) > self.history_size:
            history.pop(0)


class HealthSignalBuffer:
    """High-performance buffer for health signals with priority queueing."""
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.signals = deque(maxlen=max_size)
        self.priority_signals = deque()  # High priority signals
        self.signal_index = {}  # layer_id -> signal for fast lookup
        self._lock = asyncio.Lock()
    
    async def add_with_priority(self, signal: HealthSignal, priority: float):
        """Add signal with priority level."""
        async with self._lock:
            if priority > 0.7:  # High priority
                self.priority_signals.append(signal)
                if len(self.priority_signals) > self.max_size // 10:  # 10% for high priority
                    self.priority_signals.popleft()
            else:
                self.signals.append(signal)
            
            # Update index for fast lookup
            layer_key = f"{signal.layer_id}_{signal.seed_id}"
            self.signal_index[layer_key] = signal
    
    async def get_recent_signals(self, count: int = 1000) -> List[HealthSignal]:
        """Get recent signals for analysis."""
        async with self._lock:
            # Combine priority and normal signals
            recent = list(self.priority_signals) + list(self.signals)[-count:]
            return recent[-count:]  # Return most recent
    
    async def get_signals_for_layer(self, layer_id: int, count: int = 100) -> List[HealthSignal]:
        """Get recent signals for specific layer."""
        async with self._lock:
            layer_signals = [
                s for s in list(self.signals)[-count*2:] 
                if s.layer_id == layer_id
            ]
            return layer_signals[-count:]
    
    def __len__(self) -> int:
        return len(self.signals) + len(self.priority_signals)


class ErrorRecoveryIntegration:
    """Integration with Phase 1 error recovery system."""
    
    def convert_to_health_signal(self, error_payload: Dict[str, Any]) -> Optional[HealthSignal]:
        """Convert error recovery event to health signal."""
        try:
            # Extract error information
            error_type = error_payload.get("error_type", "unknown")
            layer_name = error_payload.get("layer_name", "unknown")
            seed_idx = error_payload.get("seed_idx", 0)
            recovery_success = error_payload.get("recovery_success", False)
            
            # Map to health signal components
            health_score = 0.8 if recovery_success else 0.2
            error_count = 1 if not recovery_success else 0
            
            return HealthSignal(
                layer_id=hash(layer_name) % 1000,  # Convert string to int
                seed_id=seed_idx,
                chunk_id=0,
                epoch=error_payload.get("epoch", 0),
                activation_variance=0.0,
                dead_neuron_ratio=0.0,
                avg_correlation=0.5,
                is_ready_for_transition=False,
                
                # Phase 1 integration fields
                health_score=health_score,
                execution_latency=error_payload.get("execution_latency", 0.0),
                error_count=error_count,
                active_seeds=1,
                total_seeds=1,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert error event to health signal: {e}")
            return None


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
        self.filter_engine = SignalFilterEngine()
        self.statistics = CollectionStatistics()
        
        # Processing configuration
        self.processing_batch_size = processing_batch_size
        self.running = False
        
        # Integration components
        self.error_recovery_integration = ErrorRecoveryIntegration()
        
        # Consumer configuration
        self.consumer_group = "tamiyo_health_collectors"
        self.consumer_name = f"collector_{socket.gethostname()}_{int(time.time())}"
        
        logger.info(f"Initialized ProductionHealthCollector with buffer_size={buffer_size}")

    async def start_intelligent_collection(self):
        """Start health signal collection with intelligent filtering."""
        
        if self.running:
            logger.warning("Health collector already running")
            return
        
        logger.info("Starting intelligent health signal collection")
        
        # Subscribe to all telemetry streams
        topics = [
            "telemetry.execution.kernel_performance",
            "telemetry.cache.hit_rates", 
            "telemetry.error_recovery.events",
            "telemetry.layer.health_signals",
            "telemetry.model.performance_metrics"
        ]
        
        try:
            # Start processing pipeline
            self.running = True
            await asyncio.gather(
                self._message_ingestion_loop(),
                self._statistics_reporting_loop(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Health collection error: {e}")
            self.running = False
            raise

    async def stop_collection(self):
        """Stop health signal collection gracefully."""
        logger.info("Stopping health signal collection")
        self.running = False

    async def _message_ingestion_loop(self):
        """High-performance message ingestion with intelligent filtering."""
        logger.info("Starting message ingestion loop")
        
        while self.running:
            try:
                # Consume messages in batches for efficiency
                messages = await self.oona_client.consume(
                    streams=[
                        "telemetry.execution.kernel_performance",
                        "telemetry.error_recovery.events",
                        "telemetry.layer.health_signals"
                    ],
                    consumer_group=self.consumer_group,
                    consumer_name=self.consumer_name,
                    count=min(self.processing_batch_size, 100),
                    timeout=50  # 50ms timeout for responsive processing
                )
                
                if not messages:
                    await asyncio.sleep(0.01)  # Brief pause if no messages
                    continue
                
                # Process messages
                for message in messages:
                    await self._process_message(message)
                    
            except Exception as e:
                logger.error(f"Message ingestion error: {e}")
                self.statistics.record_error()
                await asyncio.sleep(0.1)  # Brief backoff on error

    async def _process_message(self, message: OonaMessage):
        """Process individual message into health signal."""
        start_time = time.perf_counter()
        
        try:
            # Convert to health signal
            health_signal = self._parse_health_signal(message)
            
            if health_signal and self.filter_engine.should_process(health_signal):
                # Add to buffer with priority scoring
                priority = self.filter_engine.calculate_priority(health_signal)
                await self.signal_buffer.add_with_priority(health_signal, priority)
                
                processing_time = time.perf_counter() - start_time
                self.statistics.record_signal_processed(processing_time)
            elif health_signal:
                self.statistics.record_signal_filtered()
            else:
                self.statistics.record_signal_dropped()
                
        except Exception as e:
            logger.warning(f"Failed to process message: {e}")
            self.statistics.record_error()

    def _parse_health_signal(self, message: OonaMessage) -> Optional[HealthSignal]:
        """Parse Oona message into structured health signal."""
        try:
            payload = message.payload
            
            # Handle different message types
            if message.topic.value == "telemetry.execution.kernel_performance":
                return self._parse_execution_signal(payload)
                
            elif message.topic.value == "telemetry.error_recovery.events":
                return self.error_recovery_integration.convert_to_health_signal(payload)
                
            elif message.topic.value == "telemetry.layer.health_signals":
                return self._parse_layer_signal(payload)
                
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse health signal: {e}")
            return None

    def _parse_execution_signal(self, payload: Dict[str, Any]) -> Optional[HealthSignal]:
        """Parse execution performance signal."""
        try:
            return HealthSignal(
                layer_id=payload["layer_id"],
                seed_id=payload["seed_id"],
                chunk_id=payload.get("chunk_id", 0),
                epoch=payload["epoch"],
                
                # Phase 1 integration: use actual execution metrics
                execution_latency=payload["execution_latency_ms"],
                health_score=self._compute_health_score(payload),
                error_count=payload.get("error_count", 0),
                
                # Default values for missing fields
                activation_variance=payload.get("activation_variance", 0.0),
                dead_neuron_ratio=payload.get("dead_neuron_ratio", 0.0),
                avg_correlation=payload.get("avg_correlation", 0.5),
                is_ready_for_transition=payload.get("ready_for_transition", False),
                
                # Additional metadata
                active_seeds=payload.get("active_seeds", 1),
                total_seeds=payload.get("total_seeds", 1),
                timestamp=time.time()
            )
            
        except KeyError as e:
            logger.warning(f"Missing required field in execution signal: {e}")
            return None

    def _parse_layer_signal(self, payload: Dict[str, Any]) -> Optional[HealthSignal]:
        """Parse layer health signal."""
        try:
            return HealthSignal(
                layer_id=payload["layer_id"],
                seed_id=payload.get("seed_id", 0),
                chunk_id=payload.get("chunk_id", 0),
                epoch=payload["epoch"],
                activation_variance=payload["activation_variance"],
                dead_neuron_ratio=payload["dead_neuron_ratio"],
                avg_correlation=payload["avg_correlation"],
                is_ready_for_transition=payload.get("is_ready_for_transition", False),
                health_score=payload.get("health_score", 0.5),
                execution_latency=payload.get("execution_latency", 0.0),
                error_count=payload.get("error_count", 0),
                active_seeds=payload.get("active_seeds", 1),
                total_seeds=payload.get("total_seeds", 1),
                timestamp=time.time()
            )
            
        except KeyError as e:
            logger.warning(f"Missing required field in layer signal: {e}")
            return None

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """Compute health score from execution metrics."""
        # Base score from execution performance
        execution_score = 1.0 - min(metrics.get("execution_latency_ms", 0) / 10.0, 1.0)
        
        # Error rate impact
        error_rate = metrics.get("error_count", 0) / max(metrics.get("total_executions", 1), 1)
        error_score = 1.0 - min(error_rate * 5, 1.0)
        
        # Cache performance impact
        cache_score = metrics.get("cache_hit_rate", 1.0)
        
        # Weighted combination
        return 0.4 * execution_score + 0.4 * error_score + 0.2 * cache_score

    async def _statistics_reporting_loop(self):
        """Periodic statistics reporting."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                stats = self.statistics.get_stats_dict()
                buffer_size = len(self.signal_buffer)
                
                logger.info(
                    f"Health collector stats: "
                    f"processed={stats['signals_processed']}, "
                    f"dropped={stats['signals_dropped']}, "
                    f"filtered={stats['signals_filtered']}, "
                    f"errors={stats['error_count']}, "
                    f"buffer_size={buffer_size}, "
                    f"avg_time={stats['avg_processing_time_ms']:.2f}ms"
                )
                
            except Exception as e:
                logger.error(f"Statistics reporting error: {e}")

    async def get_recent_signals(self, count: int = 1000) -> List[HealthSignal]:
        """Get recent health signals for analysis."""
        return await self.signal_buffer.get_recent_signals(count)

    async def get_signals_for_layer(self, layer_id: int, count: int = 100) -> List[HealthSignal]:
        """Get recent signals for specific layer."""
        return await self.signal_buffer.get_signals_for_layer(layer_id, count)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        stats = self.statistics.get_stats_dict()
        stats.update({
            "running": self.running,
            "buffer_size": len(self.signal_buffer),
            "buffer_max_size": self.signal_buffer.max_size,
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name
        })
        return stats
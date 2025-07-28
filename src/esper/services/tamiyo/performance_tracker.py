"""
Performance tracking system for intelligent seed selection.

This module tracks the performance history of seeds across layers,
enabling data-driven selection strategies based on actual execution results.
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional

import numpy as np


@dataclass
class PerformanceDelta:
    """Performance change from a single execution."""

    accuracy_delta: float  # Change in accuracy
    loss_delta: float  # Change in loss (negative is better)
    latency_ms: float  # Execution latency in milliseconds
    memory_mb: float  # Memory usage in MB
    timestamp: float = field(default_factory=time.time)

    @property
    def is_improvement(self) -> bool:
        """Check if this represents an improvement."""
        return self.accuracy_delta > 0 or self.loss_delta < 0


@dataclass
class SeedPerformanceMetrics:
    """Comprehensive performance history for a single seed."""

    seed_idx: int
    layer_name: str
    kernel_id: Optional[str] = None

    # Performance arrays (sliding window)
    accuracy_improvements: List[float] = field(default_factory=list)
    loss_reductions: List[float] = field(default_factory=list)
    execution_latencies: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)

    # Counters
    total_activations: int = 0
    successful_grafts: int = 0
    failed_attempts: int = 0
    last_activation_epoch: int = -1

    # Computed scores
    ucb_score: float = 0.0
    thompson_score: float = 0.0
    performance_score: float = 0.0

    def get_mean_performance(self) -> float:
        """Calculate mean performance metric."""
        if not self.accuracy_improvements:
            return 0.0

        # Combine accuracy and loss improvements
        acc_mean = np.mean(self.accuracy_improvements[-20:])  # Last 20 samples
        loss_mean = -np.mean(
            self.loss_reductions[-20:]
        )  # Negative because lower is better

        # Weighted combination (accuracy more important)
        return 0.7 * acc_mean + 0.3 * loss_mean

    def get_variance(self) -> float:
        """Calculate performance variance."""
        if len(self.accuracy_improvements) < 2:
            return 1.0  # High variance for new seeds

        return np.var(self.accuracy_improvements[-20:])

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_grafts + self.failed_attempts
        if total == 0:
            return 0.5  # Neutral prior

        return self.successful_grafts / total


class PerformanceTracker:
    """
    Centralized performance tracking for all seeds.

    Tracks execution history, computes selection scores,
    and provides data for intelligent seed selection.
    """

    def __init__(
        self,
        window_size: int = 50,
        decay_factor: float = 0.95,
        redis_client: Optional[object] = None,
    ):
        """
        Initialize performance tracker.

        Args:
            window_size: Maximum history per metric
            decay_factor: Exponential decay for old measurements
            redis_client: Optional Redis for persistence
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.redis_client = redis_client

        # In-memory metrics storage
        self.metrics: Dict[str, SeedPerformanceMetrics] = {}

        # Global counters for UCB
        self.total_selections: Dict[str, int] = defaultdict(int)

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def update_metrics(
        self,
        layer_name: str,
        seed_idx: int,
        performance_delta: PerformanceDelta,
        kernel_id: Optional[str] = None,
    ) -> None:
        """
        Update seed performance based on execution results.

        Args:
            layer_name: Layer containing the seed
            seed_idx: Seed index
            performance_delta: Performance change
            kernel_id: Optional kernel identifier
        """
        async with self._lock:
            key = f"{layer_name}:{seed_idx}"

            # Initialize if needed
            if key not in self.metrics:
                self.metrics[key] = SeedPerformanceMetrics(
                    seed_idx=seed_idx, layer_name=layer_name, kernel_id=kernel_id
                )

            metric = self.metrics[key]

            # Update kernel ID if provided
            if kernel_id:
                metric.kernel_id = kernel_id

            # Update performance arrays
            metric.accuracy_improvements.append(performance_delta.accuracy_delta)
            metric.loss_reductions.append(performance_delta.loss_delta)
            metric.execution_latencies.append(performance_delta.latency_ms)
            metric.memory_usage.append(performance_delta.memory_mb)

            # Maintain window size
            if len(metric.accuracy_improvements) > self.window_size:
                metric.accuracy_improvements.pop(0)
                metric.loss_reductions.pop(0)
                metric.execution_latencies.pop(0)
                metric.memory_usage.pop(0)

            # Update counters
            metric.total_activations += 1
            if performance_delta.is_improvement:
                metric.successful_grafts += 1
            else:
                metric.failed_attempts += 1

            # Recompute scores
            self._update_scores(metric, layer_name)

            # Persist if Redis available
            if self.redis_client:
                await self._persist_metrics(key, metric)

    def _update_scores(self, metric: SeedPerformanceMetrics, layer_name: str) -> None:
        """Recompute selection scores for a seed."""
        # UCB score
        total_trials = sum(
            m.total_activations
            for k, m in self.metrics.items()
            if k.startswith(f"{layer_name}:")
        )

        if total_trials > 0 and metric.total_activations > 0:
            mean_perf = metric.get_mean_performance()
            exploration_term = np.sqrt(
                2 * np.log(total_trials) / metric.total_activations
            )
            metric.ucb_score = mean_perf + exploration_term
        else:
            metric.ucb_score = float("inf")  # Unexplored seeds get high score

        # Thompson score (Beta distribution)
        alpha = metric.successful_grafts + 1
        beta = metric.failed_attempts + 1
        metric.thompson_score = np.random.beta(alpha, beta)

        # Direct performance score with decay
        if metric.accuracy_improvements:
            # Apply exponential decay to historical values
            weights = np.power(
                self.decay_factor,
                np.arange(len(metric.accuracy_improvements) - 1, -1, -1),
            )
            weighted_acc = np.average(metric.accuracy_improvements, weights=weights)
            weighted_loss = np.average(metric.loss_reductions, weights=weights)

            metric.performance_score = 0.7 * weighted_acc - 0.3 * weighted_loss
        else:
            metric.performance_score = 0.0

    async def get_layer_metrics(
        self, layer_name: str
    ) -> Dict[int, SeedPerformanceMetrics]:
        """
        Get all seed metrics for a layer.

        Args:
            layer_name: Target layer

        Returns:
            Dict mapping seed_idx to metrics
        """
        async with self._lock:
            result = {}

            for key, metric in self.metrics.items():
                if key.startswith(f"{layer_name}:"):
                    result[metric.seed_idx] = metric

            return result

    async def record_selection(
        self, layer_name: str, seed_idx: int, epoch: int
    ) -> None:
        """
        Record that a seed was selected.

        Args:
            layer_name: Layer name
            seed_idx: Selected seed
            epoch: Current epoch
        """
        async with self._lock:
            key = f"{layer_name}:{seed_idx}"
            self.total_selections[layer_name] += 1

            if key in self.metrics:
                self.metrics[key].last_activation_epoch = epoch

    async def get_seed_history(
        self, layer_name: str, seed_idx: int
    ) -> Optional[SeedPerformanceMetrics]:
        """
        Get performance history for a specific seed.

        Args:
            layer_name: Layer name
            seed_idx: Seed index

        Returns:
            Performance metrics or None
        """
        async with self._lock:
            key = f"{layer_name}:{seed_idx}"
            return self.metrics.get(key)

    async def _persist_metrics(self, key: str, metric: SeedPerformanceMetrics) -> None:
        """Persist metrics to Redis if available."""
        if not self.redis_client:
            return

        try:
            # Convert to JSON-serializable format
            data = {
                "seed_idx": metric.seed_idx,
                "layer_name": metric.layer_name,
                "kernel_id": metric.kernel_id,
                "accuracy_improvements": metric.accuracy_improvements[
                    -20:
                ],  # Keep recent
                "loss_reductions": metric.loss_reductions[-20:],
                "total_activations": metric.total_activations,
                "successful_grafts": metric.successful_grafts,
                "failed_attempts": metric.failed_attempts,
                "last_activation_epoch": metric.last_activation_epoch,
                "scores": {
                    "ucb": metric.ucb_score,
                    "thompson": metric.thompson_score,
                    "performance": metric.performance_score,
                },
            }

            await self.redis_client.set(
                f"seed_metrics:{key}",
                json.dumps(data),
                ex=86400,  # 24 hour TTL
            )
        except Exception as e:
            # Log but don't fail
            import logging

            logging.warning(f"Failed to persist metrics: {e}")

    async def load_from_redis(self) -> None:
        """Load persisted metrics from Redis."""
        if not self.redis_client:
            return

        try:
            # Scan for all seed metrics keys
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match="seed_metrics:*", count=100
                )

                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        metric_data = json.loads(data)

                        # Reconstruct metric object
                        metric = SeedPerformanceMetrics(
                            seed_idx=metric_data["seed_idx"],
                            layer_name=metric_data["layer_name"],
                            kernel_id=metric_data.get("kernel_id"),
                            accuracy_improvements=metric_data.get(
                                "accuracy_improvements", []
                            ),
                            loss_reductions=metric_data.get("loss_reductions", []),
                            total_activations=metric_data.get("total_activations", 0),
                            successful_grafts=metric_data.get("successful_grafts", 0),
                            failed_attempts=metric_data.get("failed_attempts", 0),
                            last_activation_epoch=metric_data.get(
                                "last_activation_epoch", -1
                            ),
                        )

                        # Restore scores
                        scores = metric_data.get("scores", {})
                        metric.ucb_score = scores.get("ucb", 0.0)
                        metric.thompson_score = scores.get("thompson", 0.0)
                        metric.performance_score = scores.get("performance", 0.0)

                        # Store in memory
                        metric_key = key.decode().replace("seed_metrics:", "")
                        self.metrics[metric_key] = metric

                if cursor == 0:
                    break

        except Exception as e:
            # Log but don't fail
            import logging

            logging.warning(f"Failed to load metrics from Redis: {e}")

    def get_summary_stats(self) -> Dict[str, Dict]:
        """Get summary statistics for all tracked seeds."""
        summary = {}

        for key, metric in self.metrics.items():
            summary[key] = {
                "total_activations": metric.total_activations,
                "success_rate": metric.get_success_rate(),
                "mean_performance": metric.get_mean_performance(),
                "variance": metric.get_variance(),
                "last_activation": metric.last_activation_epoch,
                "scores": {
                    "ucb": metric.ucb_score,
                    "thompson": metric.thompson_score,
                    "performance": metric.performance_score,
                },
            }

        return summary

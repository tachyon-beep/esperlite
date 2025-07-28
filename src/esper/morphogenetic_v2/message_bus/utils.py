"""Utilities for the morphogenetic message bus system.

This module provides utility classes and functions for resilience patterns,
monitoring, and performance optimization.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    excluded_exceptions: Set[type] = field(default_factory=set)

    def should_count_exception(self, exception: Exception) -> bool:
        """Check if exception should count as failure."""
        return type(exception) not in self.excluded_exceptions


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures.
    
    The circuit breaker pattern prevents repeated calls to a failing service,
    giving it time to recover while failing fast for clients.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self._lock = asyncio.Lock()

        # Metrics
        self.stats = {
            "calls_allowed": 0,
            "calls_rejected": 0,
            "state_changes": 0,
            "total_failures": 0,
            "total_successes": 0
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        async with self._lock:
            # Check state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats["calls_rejected"] += 1
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is open"
                    )

            self.stats["calls_allowed"] += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except Exception as e:
            if self.config.should_count_exception(e):
                await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.stats["total_successes"] += 1

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1

                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                # Reset failure count on success in closed state
                self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.stats["total_failures"] += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1

                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

            elif self.state == CircuitState.HALF_OPEN:
                # Single failure in half-open returns to open
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from open state."""
        if not self.last_failure_time:
            return True

        return time.time() - self.last_failure_time >= self.config.timeout_seconds

    def _transition_to_closed(self):
        """Transition to closed state."""
        logger.info("Circuit breaker '%s' transitioning to CLOSED", self.name)
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()
        self.stats["state_changes"] += 1

    def _transition_to_open(self):
        """Transition to open state."""
        logger.warning("Circuit breaker '%s' transitioning to OPEN", self.name)
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.stats["state_changes"] += 1

    def _transition_to_half_open(self):
        """Transition to half-open state."""
        logger.info("Circuit breaker '%s' transitioning to HALF_OPEN", self.name)
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self.last_state_change = time.time()
        self.stats["state_changes"] += 1

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
            "stats": self.stats.copy()
        }

    async def reset(self):
        """Manually reset circuit breaker."""
        async with self._lock:
            self._transition_to_closed()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class MessageDeduplicator:
    """Prevents duplicate message processing using bloom filter.
    
    Uses a combination of bloom filter for fast checks and
    LRU cache for accurate deduplication.
    """

    def __init__(self, window_size: int = 10000, error_rate: float = 0.001):
        self.window_size = window_size
        self.seen_messages: Deque[str] = deque(maxlen=window_size)
        self.seen_set: Set[str] = set()
        self._lock = asyncio.Lock()

        # Stats
        self.stats = {
            "messages_checked": 0,
            "duplicates_found": 0,
            "false_positives": 0
        }

        # Simple bloom filter implementation
        self.bloom_size = window_size * 10
        self.bloom_array = bytearray(self.bloom_size // 8 + 1)
        self.hash_count = max(1, int(-window_size * error_rate / 0.693))

    async def is_duplicate(self, message_id: str) -> bool:
        """Check if message has been seen recently.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            True if message is duplicate
        """
        async with self._lock:
            self.stats["messages_checked"] += 1

            # Quick bloom filter check
            if not self._bloom_contains(message_id):
                # Definitely not seen
                return False

            # Accurate check
            is_dup = message_id in self.seen_set

            if is_dup:
                self.stats["duplicates_found"] += 1
            elif self._bloom_contains(message_id):
                # Bloom filter false positive
                self.stats["false_positives"] += 1

            return is_dup

    async def mark_processed(self, message_id: str):
        """Mark message as processed.
        
        Args:
            message_id: Unique message identifier
        """
        async with self._lock:
            # Add to bloom filter
            self._bloom_add(message_id)

            # Add to accurate tracking
            if len(self.seen_messages) >= self.window_size:
                # Remove oldest
                oldest = self.seen_messages[0]
                self.seen_set.discard(oldest)

            self.seen_messages.append(message_id)
            self.seen_set.add(message_id)

    def _bloom_add(self, item: str):
        """Add item to bloom filter."""
        for i in range(self.hash_count):
            pos = self._hash(item, i) % self.bloom_size
            byte_pos = pos // 8
            bit_pos = pos % 8
            self.bloom_array[byte_pos] |= (1 << bit_pos)

    def _bloom_contains(self, item: str) -> bool:
        """Check if item might be in bloom filter."""
        for i in range(self.hash_count):
            pos = self._hash(item, i) % self.bloom_size
            byte_pos = pos // 8
            bit_pos = pos % 8

            if not (self.bloom_array[byte_pos] & (1 << bit_pos)):
                return False

        return True

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash for bloom filter."""
        h = hashlib.sha256(f"{item}:{seed}".encode()).digest()
        return int.from_bytes(h[:4], byteorder='big')

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        stats = self.stats.copy()
        stats["current_window_size"] = len(self.seen_messages)
        stats["bloom_filter_size_bytes"] = len(self.bloom_array)

        if stats["messages_checked"] > 0:
            stats["duplicate_rate"] = stats["duplicates_found"] / stats["messages_checked"]
            stats["false_positive_rate"] = stats["false_positives"] / stats["messages_checked"]

        return stats

    async def clear(self):
        """Clear all tracking data."""
        async with self._lock:
            self.seen_messages.clear()
            self.seen_set.clear()
            self.bloom_array = bytearray(self.bloom_size // 8 + 1)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""
    max_calls: int = 100
    time_window_seconds: float = 1.0
    burst_allowance: float = 1.5  # Allow burst up to 150% of limit


class RateLimiter:
    """Token bucket rate limiter for controlling message flow."""

    def __init__(self, name: str, config: Optional[RateLimiterConfig] = None):
        self.name = name
        self.config = config or RateLimiterConfig()

        # Token bucket
        self.tokens = float(config.max_calls)
        self.max_tokens = config.max_calls * config.burst_allowance
        self.refill_rate = config.max_calls / config.time_window_seconds
        self.last_refill = time.time()

        self._lock = asyncio.Lock()

        # Stats
        self.stats = {
            "calls_allowed": 0,
            "calls_rejected": 0,
            "total_wait_time_ms": 0.0
        }

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False if rate limited
        """
        async with self._lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.stats["calls_allowed"] += 1
                return True
            else:
                self.stats["calls_rejected"] += 1
                return False

    async def acquire_with_wait(self, tokens: int = 1,
                               max_wait_seconds: float = 10.0) -> bool:
        """Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if acquired within timeout
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            if await self.acquire(tokens):
                wait_time = (time.time() - start_time) * 1000
                self.stats["total_wait_time_ms"] += wait_time
                return True

            # Calculate wait time
            async with self._lock:
                needed = tokens - self.tokens
                wait_time = needed / self.refill_rate

            # Wait for tokens
            wait_time = min(wait_time, max_wait_seconds - (time.time() - start_time))
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                break

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        stats = self.stats.copy()
        stats["current_tokens"] = self.tokens
        stats["max_tokens"] = self.max_tokens
        stats["refill_rate"] = self.refill_rate

        if stats["calls_allowed"] > 0:
            stats["average_wait_time_ms"] = (
                stats["total_wait_time_ms"] / stats["calls_allowed"]
            )

        return stats


class MessageBatcher:
    """Utility for batching messages efficiently."""

    def __init__(self, batch_size: int = 100,
                 window_ms: int = 100,
                 max_bytes: int = 1024 * 1024):
        self.batch_size = batch_size
        self.window_ms = window_ms
        self.max_bytes = max_bytes

        self.current_batch: List[Any] = []
        self.current_size = 0
        self.batch_start_time = time.time()
        self._lock = asyncio.Lock()

        # Stats
        self.stats = {
            "batches_created": 0,
            "messages_batched": 0,
            "average_batch_size": 0.0,
            "bytes_batched": 0
        }

    async def add(self, message: Any) -> Optional[List[Any]]:
        """Add message to batch.
        
        Args:
            message: Message to add
            
        Returns:
            Completed batch if ready, None otherwise
        """
        async with self._lock:
            # Estimate message size
            msg_size = len(json.dumps(message)) if not isinstance(message, bytes) else len(message)

            # Check if batch is full
            if (len(self.current_batch) >= self.batch_size or
                self.current_size + msg_size > self.max_bytes or
                (time.time() - self.batch_start_time) * 1000 > self.window_ms):

                # Return current batch and start new one
                batch = self.current_batch.copy()
                self._reset_batch()

                # Add message to new batch
                self.current_batch.append(message)
                self.current_size = msg_size

                if batch:
                    self._update_stats(batch, self.current_size)
                    return batch
            else:
                # Add to current batch
                self.current_batch.append(message)
                self.current_size += msg_size

            return None

    async def flush(self) -> Optional[List[Any]]:
        """Flush current batch.
        
        Returns:
            Current batch if not empty
        """
        async with self._lock:
            if self.current_batch:
                batch = self.current_batch.copy()
                self._reset_batch()
                self._update_stats(batch, self.current_size)
                return batch

            return None

    def _reset_batch(self):
        """Reset batch state."""
        self.current_batch = []
        self.current_size = 0
        self.batch_start_time = time.time()

    def _update_stats(self, batch: List[Any], size: int):
        """Update statistics."""
        self.stats["batches_created"] += 1
        self.stats["messages_batched"] += len(batch)
        self.stats["bytes_batched"] += size

        # Update average
        count = self.stats["batches_created"]
        avg = self.stats["average_batch_size"]
        self.stats["average_batch_size"] = (
            (avg * (count - 1) + len(batch)) / count
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        stats = self.stats.copy()
        stats["current_batch_size"] = len(self.current_batch)
        stats["current_batch_bytes"] = self.current_size
        return stats


class RetryPolicy:
    """Configurable retry policy with exponential backoff."""

    def __init__(self, max_retries: int = 3,
                 base_delay_ms: int = 100,
                 max_delay_ms: int = 60000,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for given attempt.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in milliseconds
        """
        if attempt >= self.max_retries:
            return 0  # No more retries

        # Exponential backoff
        delay = self.base_delay_ms * (self.exponential_base ** attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay_ms)

        # Add jitter
        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay

    def should_retry(self, attempt: int, exception: Optional[Exception] = None) -> bool:
        """Check if should retry.
        
        Args:
            attempt: Current attempt number
            exception: Optional exception to check
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False

        # Could add exception-based logic here
        if exception:
            # Don't retry on certain exceptions
            non_retryable = (ValueError, TypeError, KeyError)
            if isinstance(exception, non_retryable):
                return False

        return True

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry policy.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(attempt, e):
                    raise

                if attempt < self.max_retries:
                    delay_ms = self.get_delay_ms(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} "
                        f"after {delay_ms:.0f}ms: {e}"
                    )
                    await asyncio.sleep(delay_ms / 1000)

        raise last_exception


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""

    def __init__(self, window_size_seconds: float = 60.0):
        self.window_size = window_size_seconds
        self.metrics: Dict[str, Deque[tuple]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._lock = asyncio.Lock()

    async def record(self, metric_name: str, value: float,
                    tags: Optional[Dict[str, str]] = None):
        """Record a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags
        """
        async with self._lock:
            timestamp = time.time()
            self.metrics[metric_name].append((timestamp, value, tags or {}))

            # Clean old entries
            cutoff = timestamp - self.window_size
            while (self.metrics[metric_name] and
                   self.metrics[metric_name][0][0] < cutoff):
                self.metrics[metric_name].popleft()

    async def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Statistics dictionary
        """
        async with self._lock:
            if metric_name not in self.metrics:
                return {}

            values = [v for _, v, _ in self.metrics[metric_name]]

            if not values:
                return {}

            import numpy as np

            return {
                "count": len(values),
                "sum": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }

    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        result = {}

        for metric_name in list(self.metrics.keys()):
            stats = await self.get_stats(metric_name)
            if stats:
                result[metric_name] = stats

        return result

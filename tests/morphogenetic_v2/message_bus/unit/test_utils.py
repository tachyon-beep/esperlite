"""Unit tests for message bus utilities."""

import time
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from src.esper.morphogenetic_v2.message_bus.utils import CircuitBreaker
from src.esper.morphogenetic_v2.message_bus.utils import CircuitBreakerConfig
from src.esper.morphogenetic_v2.message_bus.utils import CircuitOpenError
from src.esper.morphogenetic_v2.message_bus.utils import CircuitState
from src.esper.morphogenetic_v2.message_bus.utils import MessageBatcher
from src.esper.morphogenetic_v2.message_bus.utils import MessageDeduplicator
from src.esper.morphogenetic_v2.message_bus.utils import MetricsCollector
from src.esper.morphogenetic_v2.message_bus.utils import RateLimiter
from src.esper.morphogenetic_v2.message_bus.utils import RateLimiterConfig
from src.esper.morphogenetic_v2.message_bus.utils import RetryPolicy


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0
        assert len(config.excluded_exceptions) == 0

    def test_should_count_exception(self):
        """Test exception counting logic."""
        config = CircuitBreakerConfig(
            excluded_exceptions={ValueError, KeyError}
        )

        # Should count
        assert config.should_count_exception(RuntimeError("test"))
        assert config.should_count_exception(TypeError("test"))

        # Should not count
        assert not config.should_count_exception(ValueError("test"))
        assert not config.should_count_exception(KeyError("test"))


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.fixture
    def breaker(self):
        """Create test circuit breaker."""
        return CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1
        ))

    @pytest.mark.asyncio
    async def test_initial_state(self, breaker):
        """Test initial circuit breaker state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

        state = breaker.get_state()
        assert state["name"] == "test"
        assert state["state"] == "closed"

    @pytest.mark.asyncio
    async def test_successful_calls(self, breaker):
        """Test successful calls through breaker."""
        async def success_func(value):
            return value * 2

        result = await breaker.call(success_func, 5)
        assert result == 10
        assert breaker.stats["calls_allowed"] == 1
        assert breaker.stats["total_successes"] == 1

    @pytest.mark.asyncio
    async def test_failure_threshold(self, breaker):
        """Test circuit opens after failure threshold."""
        async def failing_func():
            raise RuntimeError("Test error")

        # First failure
        with pytest.raises(RuntimeError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats["state_changes"] == 1

    @pytest.mark.asyncio
    async def test_open_circuit_rejection(self, breaker):
        """Test calls rejected when circuit is open."""
        # Open the circuit
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time()

        async def test_func():
            return "should not execute"

        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.call(test_func)

        assert "Circuit breaker 'test' is open" in str(exc_info.value)
        assert breaker.stats["calls_rejected"] == 1

    @pytest.mark.asyncio
    async def test_half_open_recovery(self, breaker):
        """Test recovery through half-open state."""
        # Open the circuit
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time() - 0.2  # Past timeout

        async def success_func():
            return "success"

        # Should transition to half-open and allow call
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.success_count == 1

        # Second success should close circuit
        await breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats["state_changes"] == 2

    @pytest.mark.asyncio
    async def test_half_open_failure(self, breaker):
        """Test failure in half-open returns to open."""
        breaker.state = CircuitState.HALF_OPEN

        async def failing_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self, breaker):
        """Test excluded exceptions don't count as failures."""
        breaker.config.excluded_exceptions.add(ValueError)

        async def func():
            raise ValueError("Excluded error")

        # Should not count as failure
        with pytest.raises(ValueError):
            await breaker.call(func)

        assert breaker.failure_count == 0
        assert breaker.stats["total_failures"] == 0

    @pytest.mark.asyncio
    async def test_manual_reset(self, breaker):
        """Test manual circuit reset."""
        breaker.state = CircuitState.OPEN
        breaker.failure_count = 5

        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0


class TestMessageDeduplicator:
    """Test MessageDeduplicator functionality."""

    @pytest.fixture
    def dedup(self):
        """Create test deduplicator."""
        return MessageDeduplicator(window_size=100, error_rate=0.01)

    @pytest.mark.asyncio
    async def test_not_duplicate(self, dedup):
        """Test non-duplicate messages."""
        assert not await dedup.is_duplicate("msg1")
        assert not await dedup.is_duplicate("msg2")
        assert dedup.stats["duplicates_found"] == 0

    @pytest.mark.asyncio
    async def test_mark_and_check_duplicate(self, dedup):
        """Test marking and checking duplicates."""
        # Mark as processed
        await dedup.mark_processed("msg1")

        # Check - should be duplicate
        assert await dedup.is_duplicate("msg1")
        assert dedup.stats["duplicates_found"] == 1

        # Different message - not duplicate
        assert not await dedup.is_duplicate("msg2")

    @pytest.mark.asyncio
    async def test_window_size_limit(self, dedup):
        """Test window size enforcement."""
        # Fill window
        for i in range(100):
            await dedup.mark_processed(f"msg{i}")

        # Old messages should be forgotten
        assert len(dedup.seen_messages) == 100
        assert len(dedup.seen_set) == 100

        # Add one more
        await dedup.mark_processed("msg100")

        # Oldest should be removed
        assert len(dedup.seen_messages) == 100
        assert "msg0" not in dedup.seen_set
        assert "msg100" in dedup.seen_set

    @pytest.mark.asyncio
    async def test_bloom_filter(self, dedup):
        """Test bloom filter functionality."""
        # Add to bloom filter
        await dedup.mark_processed("test_msg")

        # Should be in bloom filter
        assert dedup._bloom_contains("test_msg")

        # Random message unlikely to be in bloom
        # (but could have false positive)
        random_msg = "unlikely_msg_12345"
        if dedup._bloom_contains(random_msg):
            # False positive
            assert not await dedup.is_duplicate(random_msg)
            assert dedup.stats["false_positives"] > 0

    @pytest.mark.asyncio
    async def test_get_stats(self, dedup):
        """Test statistics retrieval."""
        # Generate some activity
        await dedup.mark_processed("msg1")
        await dedup.is_duplicate("msg1")  # Duplicate
        await dedup.is_duplicate("msg2")  # Not duplicate

        stats = dedup.get_stats()

        assert stats["messages_checked"] == 2
        assert stats["duplicates_found"] == 1
        assert stats["current_window_size"] == 1
        assert "duplicate_rate" in stats
        assert stats["duplicate_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_clear(self, dedup):
        """Test clearing deduplicator."""
        # Add some messages
        for i in range(10):
            await dedup.mark_processed(f"msg{i}")

        assert len(dedup.seen_set) == 10

        # Clear
        await dedup.clear()

        assert len(dedup.seen_messages) == 0
        assert len(dedup.seen_set) == 0

        # Should not be duplicates anymore
        assert not await dedup.is_duplicate("msg0")


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.fixture
    def limiter(self):
        """Create test rate limiter."""
        config = RateLimiterConfig(
            max_calls=10,
            time_window_seconds=1.0,
            burst_allowance=1.5
        )
        return RateLimiter("test", config)

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self, limiter):
        """Test acquiring tokens within limit."""
        # Should succeed
        for i in range(10):
            assert await limiter.acquire()

        assert limiter.stats["calls_allowed"] == 10
        assert limiter.stats["calls_rejected"] == 0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self, limiter):
        """Test rate limiting when exceeding limit."""
        # Use all tokens
        limiter.tokens = 0

        # Should fail
        assert not await limiter.acquire()
        assert limiter.stats["calls_rejected"] == 1

    @pytest.mark.asyncio
    async def test_burst_allowance(self, limiter):
        """Test burst allowance."""
        # Start with max_tokens to test burst
        limiter.tokens = limiter.max_tokens  # 15 tokens
        
        # Can burst up to 150% (15 calls)
        success_count = 0
        for i in range(20):
            if await limiter.acquire():
                success_count += 1

        assert success_count == 15  # 10 * 1.5

    @pytest.mark.asyncio
    async def test_token_refill(self, limiter):
        """Test token refill over time."""
        # Use all tokens
        limiter.tokens = 0
        limiter.last_refill = time.time() - 0.1  # 100ms ago

        # Should have refilled ~1 token (10 tokens/sec * 0.1s)
        assert await limiter.acquire()
        assert limiter.tokens < 1  # Some tokens used

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self, limiter):
        """Test acquiring with wait."""
        # Use all tokens
        limiter.tokens = 0

        # Should wait and succeed
        start_time = time.time()
        result = await limiter.acquire_with_wait(1, max_wait_seconds=0.2)
        wait_time = time.time() - start_time

        assert result
        assert wait_time >= 0.09  # Should wait ~100ms for 1 token
        assert wait_time < 0.2
        assert limiter.stats["total_wait_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_acquire_with_wait_timeout(self, limiter):
        """Test wait timeout."""
        # Need many tokens, short timeout
        limiter.tokens = 0

        result = await limiter.acquire_with_wait(50, max_wait_seconds=0.1)
        assert not result  # Should timeout

    def test_get_stats(self, limiter):
        """Test statistics retrieval."""
        stats = limiter.get_stats()

        assert "current_tokens" in stats
        assert stats["max_tokens"] == 15  # 10 * 1.5
        assert stats["refill_rate"] == 10.0


class TestMessageBatcher:
    """Test MessageBatcher functionality."""

    @pytest.fixture
    def batcher(self):
        """Create test batcher."""
        return MessageBatcher(
            batch_size=3,
            window_ms=100,
            max_bytes=1000
        )

    @pytest.mark.asyncio
    async def test_add_below_threshold(self, batcher):
        """Test adding messages below batch threshold."""
        msg1 = {"id": 1, "data": "test1"}
        msg2 = {"id": 2, "data": "test2"}

        result1 = await batcher.add(msg1)
        assert result1 is None  # Not ready

        result2 = await batcher.add(msg2)
        assert result2 is None  # Still not ready

        assert len(batcher.current_batch) == 2

    @pytest.mark.asyncio
    async def test_batch_size_trigger(self, batcher):
        """Test batch completion by size."""
        messages = [{"id": i} for i in range(4)]

        # First 3 messages
        for i in range(3):
            result = await batcher.add(messages[i])
            if i < 2:
                assert result is None

        # Third message should trigger batch
        result = await batcher.add(messages[3])
        assert result is not None
        assert len(result) == 3
        assert result[0]["id"] == 0

        # Fourth message in new batch
        assert len(batcher.current_batch) == 1

    @pytest.mark.asyncio
    async def test_batch_bytes_trigger(self, batcher):
        """Test batch completion by size limit."""
        # Large message
        large_msg = {"data": "x" * 900}  # ~908 bytes
        small_msg = {"id": 1, "data": "x" * 100}  # ~110 bytes

        # Add large message
        result = await batcher.add(large_msg)
        assert result is None

        # Small message should trigger due to size
        result = await batcher.add(small_msg)
        assert result is not None
        assert len(result) == 1  # Only large message in batch

    @pytest.mark.asyncio
    async def test_batch_time_trigger(self, batcher):
        """Test batch completion by time window."""
        # Add a message first
        await batcher.add({"id": 0})
        
        # Set old start time
        batcher.batch_start_time = time.time() - 0.2  # 200ms ago

        msg = {"id": 1}
        result = await batcher.add(msg)

        # Should trigger due to time window
        assert result is not None
        assert len(result) == 1  # Previous batch had one message
        assert result[0]["id"] == 0

    @pytest.mark.asyncio
    async def test_flush(self, batcher):
        """Test manual flush."""
        # Add some messages
        await batcher.add({"id": 1})
        await batcher.add({"id": 2})

        # Flush
        batch = await batcher.flush()
        assert batch is not None
        assert len(batch) == 2

        # Batch should be reset
        assert len(batcher.current_batch) == 0

        # Flush empty batch
        batch = await batcher.flush()
        assert batch is None

    def test_get_stats(self, batcher):
        """Test statistics retrieval."""
        stats = batcher.get_stats()

        assert stats["batches_created"] == 0
        assert stats["messages_batched"] == 0
        assert "current_batch_size" in stats


class TestRetryPolicy:
    """Test RetryPolicy functionality."""

    @pytest.fixture
    def policy(self):
        """Create test retry policy."""
        return RetryPolicy(
            max_retries=3,
            base_delay_ms=100,
            max_delay_ms=1000,
            exponential_base=2.0,
            jitter=False
        )

    def test_get_delay_ms(self, policy):
        """Test delay calculation."""
        # First attempt - base delay
        assert policy.get_delay_ms(0) == 100

        # Exponential backoff
        assert policy.get_delay_ms(1) == 200
        assert policy.get_delay_ms(2) == 400

        # Beyond max_retries returns 0
        assert policy.get_delay_ms(3) == 0
        assert policy.get_delay_ms(5) == 0

        # No delay after max retries
        assert policy.get_delay_ms(3) == 0

    def test_get_delay_with_jitter(self):
        """Test delay with jitter."""
        policy = RetryPolicy(base_delay_ms=100, jitter=True)

        delays = [policy.get_delay_ms(0) for _ in range(10)]

        # Should have variation
        assert min(delays) >= 50  # 0.5 * base
        assert max(delays) <= 150  # 1.5 * base
        assert len(set(delays)) > 1  # Not all same

    def test_should_retry(self, policy):
        """Test retry decision logic."""
        # Should retry within limit
        assert policy.should_retry(0)
        assert policy.should_retry(2)

        # Should not retry after max
        assert not policy.should_retry(3)
        assert not policy.should_retry(10)

    def test_should_retry_with_exceptions(self, policy):
        """Test exception-based retry logic."""
        # Retryable exceptions
        assert policy.should_retry(0, RuntimeError("test"))
        assert policy.should_retry(0, ConnectionError("test"))

        # Non-retryable exceptions
        assert not policy.should_retry(0, ValueError("test"))
        assert not policy.should_retry(0, TypeError("test"))
        assert not policy.should_retry(0, KeyError("test"))

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, policy):
        """Test successful retry execution."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary error")
            return "success"

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await policy.execute_with_retry(flaky_func)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self, policy):
        """Test retry exhaustion."""
        async def always_fail():
            raise RuntimeError("Persistent error")

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as exc_info:
                await policy.execute_with_retry(always_fail)

        assert "Persistent error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_non_retryable(self, policy):
        """Test non-retryable exception."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")

        with pytest.raises(ValueError):
            await policy.execute_with_retry(func)

        assert call_count == 1  # No retries


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.fixture
    def collector(self):
        """Create test metrics collector."""
        return MetricsCollector(window_size_seconds=60.0)

    @pytest.mark.asyncio
    async def test_record_metric(self, collector):
        """Test recording metrics."""
        await collector.record("latency", 10.5)
        await collector.record("latency", 15.3)
        await collector.record("latency", 12.7, {"endpoint": "/api"})

        # Check stored metrics
        assert "latency" in collector.metrics
        assert len(collector.metrics["latency"]) == 3

    @pytest.mark.asyncio
    async def test_window_cleanup(self, collector):
        """Test old metrics cleanup."""
        # Add old metric
        old_time = time.time() - 70  # Outside window
        collector.metrics["test"].append((old_time, 1.0, {}))

        # Add new metric
        await collector.record("test", 2.0)

        # Old metric should be removed
        assert len(collector.metrics["test"]) == 1
        assert collector.metrics["test"][0][1] == 2.0

    @pytest.mark.asyncio
    async def test_get_stats(self, collector):
        """Test statistics calculation."""
        # Record some metrics
        values = [10, 20, 30, 40, 50]
        for v in values:
            await collector.record("response_time", v)

        stats = await collector.get_stats("response_time")

        assert stats["count"] == 5
        assert stats["sum"] == 150
        assert stats["mean"] == 30
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["p50"] == 30
        assert "std" in stats
        assert "p95" in stats
        assert "p99" in stats

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, collector):
        """Test stats for non-existent metric."""
        stats = await collector.get_stats("nonexistent")
        assert stats == {}

    @pytest.mark.asyncio
    async def test_get_all_metrics(self, collector):
        """Test retrieving all metrics."""
        # Record different metrics
        await collector.record("cpu", 45.5)
        await collector.record("memory", 1024)
        await collector.record("requests", 100)

        all_stats = await collector.get_all_metrics()

        assert len(all_stats) == 3
        assert "cpu" in all_stats
        assert "memory" in all_stats
        assert "requests" in all_stats

        assert all_stats["cpu"]["mean"] == 45.5
        assert all_stats["memory"]["mean"] == 1024

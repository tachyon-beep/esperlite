"""
Tests for circuit breaker pattern implementation.
"""

import asyncio
from unittest.mock import Mock

import pytest

from esper.utils.circuit_breaker import CircuitBreaker
from esper.utils.circuit_breaker import CircuitBreakerConfig
from esper.utils.circuit_breaker import CircuitBreakerOpenError
from esper.utils.circuit_breaker import CircuitBreakerState


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with test configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for fast tests
            success_threshold=2,
            timeout=1,
        )
        return CircuitBreaker(name="test_breaker", config=config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_success(self, circuit_breaker):
        """Test successful calls in CLOSED state."""

        async def successful_function():
            return "success"

        # Should start in CLOSED state
        assert circuit_breaker.is_closed

        # Successful call should work
        result = await circuit_breaker.call(successful_function)
        assert result == "success"

        # Stats should reflect success
        stats = circuit_breaker.get_stats()
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 0
        assert stats["state"] == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""

        async def failing_function():
            raise Exception("Test failure")

        # Circuit should start closed
        assert circuit_breaker.is_closed

        # First two failures should keep circuit closed
        for i in range(2):
            with pytest.raises(Exception, match="Test failure"):
                await circuit_breaker.call(failing_function)
            assert circuit_breaker.is_closed

        # Third failure should open the circuit
        with pytest.raises(Exception, match="Test failure"):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.is_open

        # Further calls should fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_function)

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_failure(self, circuit_breaker):
        """Test circuit breaker handles timeout failures."""

        async def slow_function():
            await asyncio.sleep(2)  # Longer than timeout
            return "too_slow"

        # Call should timeout and count as failure
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_function)

        stats = circuit_breaker.get_stats()
        assert stats["total_timeouts"] == 1
        assert stats["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_cycle(self, circuit_breaker):
        """Test complete recovery cycle from OPEN to CLOSED."""

        async def failing_function():
            raise Exception("Test failure")

        async def successful_function():
            return "success"

        # Open the circuit with failures
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)

        assert circuit_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Circuit should transition to HALF_OPEN on next call
        result = await circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.is_half_open

        # Another success should close the circuit
        result = await circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self, circuit_breaker):
        """Test circuit breaker returns to OPEN from HALF_OPEN on failure."""

        async def failing_function():
            raise Exception("Test failure")

        async def successful_function():
            return "success"

        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)

        assert circuit_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # First call should transition to HALF_OPEN
        result = await circuit_breaker.call(successful_function)
        assert circuit_breaker.is_half_open

        # Failure should return to OPEN
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)

        assert circuit_breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_statistics(self, circuit_breaker):
        """Test circuit breaker statistics tracking."""

        async def mixed_function(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # Mix of successes and failures
        await circuit_breaker.call(mixed_function, should_fail=False)
        await circuit_breaker.call(mixed_function, should_fail=False)

        with pytest.raises(Exception):
            await circuit_breaker.call(mixed_function, should_fail=True)

        stats = circuit_breaker.get_stats()
        assert stats["total_requests"] == 3
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1
        assert stats["name"] == "test_breaker"
        assert "config" in stats
        assert stats["config"]["failure_threshold"] == 3

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality."""

        # Manually set some state
        circuit_breaker.failure_count = 5
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.total_requests = 10

        # Reset should restore initial state
        circuit_breaker.reset()

        assert circuit_breaker.is_closed
        assert circuit_breaker.failure_count == 0
        # Note: reset doesn't clear total statistics, only state

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_calls(self, circuit_breaker):
        """Test circuit breaker with concurrent calls."""

        call_count = 0

        async def counting_function():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Small delay
            return f"call_{call_count}"

        # Make concurrent calls
        tasks = [circuit_breaker.call(counting_function) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All calls should succeed
        assert len(results) == 5
        assert all("call_" in result for result in results)

        stats = circuit_breaker.get_stats()
        assert stats["total_successes"] == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_config_validation(self):
        """Test circuit breaker with various configurations."""

        # Test with custom config
        config = CircuitBreakerConfig(
            failure_threshold=1,  # Very sensitive
            recovery_timeout=0.5,
            success_threshold=1,
            timeout=5,
        )

        breaker = CircuitBreaker(name="sensitive_breaker", config=config)

        async def failing_function():
            raise Exception("Immediate failure")

        # Should open after just one failure
        with pytest.raises(Exception):
            await breaker.call(failing_function)

        assert breaker.is_open

        # Quick recovery
        await asyncio.sleep(0.6)

        async def success_function():
            return "recovered"

        # Should recover with just one success
        result = await breaker.call(success_function)
        assert result == "recovered"
        assert breaker.is_closed


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with real components."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_http_client_mock(self):
        """Test circuit breaker protecting HTTP client calls."""

        # Mock HTTP client that fails then succeeds
        mock_client = Mock()
        call_count = 0

        async def mock_http_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Network error")
            return {"status": "ok"}

        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5,
            success_threshold=1,
            timeout=1,
        )

        breaker = CircuitBreaker(name="http_breaker", config=config)

        # First 3 calls should fail and open circuit
        for i in range(3):
            with pytest.raises(Exception, match="Network error"):
                await breaker.call(mock_http_call)

        assert breaker.is_open

        # Circuit should be open
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(mock_http_call)

        # Wait for recovery
        await asyncio.sleep(0.6)

        # Next call should succeed and close circuit
        result = await breaker.call(mock_http_call)
        assert result == {"status": "ok"}
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self):
        """Test circuit breaker metrics for monitoring."""

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            success_threshold=1,
            timeout=1,
        )

        breaker = CircuitBreaker(name="metrics_breaker", config=config)

        async def variable_function(behavior):
            if behavior == "success":
                return "ok"
            elif behavior == "timeout":
                await asyncio.sleep(2)  # Will timeout
                return "too_slow"
            else:
                raise Exception("Failure")

        # Collect various metrics
        await breaker.call(variable_function, "success")

        with pytest.raises(asyncio.TimeoutError):
            await breaker.call(variable_function, "timeout")

        with pytest.raises(Exception):
            await breaker.call(variable_function, "fail")

        # Open circuit
        with pytest.raises(Exception):
            await breaker.call(variable_function, "fail")

        assert breaker.is_open

        # Get comprehensive stats
        stats = breaker.get_stats()

        assert stats["total_requests"] == 4
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 2
        assert stats["total_timeouts"] == 1
        assert stats["total_circuit_opens"] == 1
        assert stats["state"] == "open"

        # Verify config is included
        assert stats["config"]["failure_threshold"] == 2
        assert stats["config"]["timeout"] == 1

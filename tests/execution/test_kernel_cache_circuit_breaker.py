"""
Tests for KernelCache circuit breaker integration.
"""

import asyncio

import pytest
import torch

from esper.execution.kernel_cache import KernelCache
from esper.utils.circuit_breaker import CircuitBreakerOpenError
from esper.utils.config import ServiceConfig


class TestKernelCacheCircuitBreaker:
    """Test KernelCache with circuit breaker functionality."""

    @pytest.fixture
    def config(self):
        """Test configuration with short timeouts."""
        return ServiceConfig(
            urza_url="http://test-urza:8000",
            http_timeout=1,
            retry_attempts=1,
            cache_size_mb=16,
        )

    @pytest.fixture
    def kernel_cache(self, config):
        """Create KernelCache with test configuration."""
        return KernelCache(config=config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection_on_urza_failures(self, kernel_cache):
        """Test circuit breaker opens after Urza failures."""

        # Mock the implementation to fail consistently
        async def failing_fetch(artifact_id):
            raise Exception("Urza service error")

        kernel_cache._fetch_from_urza_impl = failing_fetch

        # First 5 failures should cause circuit to open
        for i in range(5):
            result = await kernel_cache.load_kernel(f"test-kernel-{i}")
            assert result is None

        # Circuit should now be open, and next call should trigger circuit breaker protection
        result = await kernel_cache.load_kernel("test-kernel-protected")
        assert result is None

        # Check statistics
        stats = kernel_cache.get_cache_stats()
        assert "circuit_breaker_stats" in stats
        cb_stats = stats["circuit_breaker_stats"]
        assert cb_stats["state"] == "open"
        assert cb_stats["total_failures"] >= 5

        # The circuit breaker failure counter should show circuit breaker protection was triggered
        assert stats["circuit_breaker_failures"] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_after_urza_recovery(self, kernel_cache):
        """Test circuit breaker recovery when Urza service recovers."""

        call_count = 0

        async def variable_fetch(artifact_id):
            nonlocal call_count
            call_count += 1

            # Fail first 5 calls, then succeed
            if call_count <= 5:
                raise Exception("Urza temporarily down")

            # Return mock tensor for success
            return torch.randn(1024, dtype=torch.float32)

        kernel_cache._fetch_from_urza_impl = variable_fetch

        # Cause failures to open circuit
        for i in range(5):
            result = await kernel_cache.load_kernel(f"test-kernel-{i}")
            assert result is None

        # Wait for recovery timeout (circuit breaker uses 30s, but we can manipulate it for testing)
        kernel_cache._circuit_breaker.config.recovery_timeout = 0.1
        await asyncio.sleep(0.2)

        # Next call should succeed and recover circuit
        result = await kernel_cache.load_kernel("recovery-kernel")
        assert result is not None
        assert isinstance(result, torch.Tensor)

        # Circuit should eventually close after enough successes
        for i in range(3):
            result = await kernel_cache.load_kernel(f"success-kernel-{i}")
            assert result is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_handling(self, kernel_cache):
        """Test circuit breaker handles timeout failures."""

        async def slow_fetch(artifact_id):
            await asyncio.sleep(2)  # Longer than timeout
            return torch.randn(1024, dtype=torch.float32)

        kernel_cache._fetch_from_urza_impl = slow_fetch

        # Timeout should be treated as failure
        result = await kernel_cache.load_kernel("slow-kernel")
        assert result is None

        # Check that timeout was recorded
        stats = kernel_cache.get_cache_stats()
        cb_stats = stats["circuit_breaker_stats"]
        assert cb_stats["total_timeouts"] > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_statistics_tracking(self, kernel_cache):
        """Test circuit breaker statistics are properly tracked."""

        success_count = 0

        async def mixed_fetch(artifact_id):
            nonlocal success_count
            success_count += 1

            # Alternate between success and failure
            if success_count % 2 == 0:
                raise Exception("Intermittent failure")

            return torch.randn(1024, dtype=torch.float32)

        kernel_cache._fetch_from_urza_impl = mixed_fetch

        # Mix of successes and failures
        for i in range(6):
            await kernel_cache.load_kernel(f"mixed-kernel-{i}")

        # Check comprehensive stats
        stats = kernel_cache.get_cache_stats()

        assert "circuit_breaker_stats" in stats
        cb_stats = stats["circuit_breaker_stats"]

        assert cb_stats["total_requests"] == 6
        assert cb_stats["total_successes"] == 3
        assert cb_stats["total_failures"] == 3
        assert cb_stats["name"] == "kernel_cache_urza"

    @pytest.mark.asyncio
    async def test_cache_hit_bypasses_circuit_breaker(self, kernel_cache):
        """Test cache hits don't trigger circuit breaker."""

        # First, populate cache with a successful fetch
        test_tensor = torch.randn(1024, dtype=torch.float32)

        async def successful_fetch(artifact_id):
            return test_tensor

        kernel_cache._fetch_from_urza_impl = successful_fetch

        # Load kernel into cache
        result = await kernel_cache.load_kernel("cached-kernel")
        assert result is not None

        # Now make the fetch fail
        async def failing_fetch(artifact_id):
            raise Exception("Service down")

        kernel_cache._fetch_from_urza_impl = failing_fetch

        # Cache hit should still work even with failing service
        result = await kernel_cache.load_kernel("cached-kernel")
        assert result is not None
        assert torch.equal(result, test_tensor)

        # Cache stats should show hit
        stats = kernel_cache.get_cache_stats()
        assert stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_error_handling(self, kernel_cache):
        """Test specific handling of CircuitBreakerOpenError."""

        # Manually open the circuit breaker
        from esper.utils.circuit_breaker import CircuitBreakerState

        kernel_cache._circuit_breaker.failure_count = 10  # Above threshold
        kernel_cache._circuit_breaker.state = CircuitBreakerState.OPEN

        # Mock the circuit breaker to raise CircuitBreakerOpenError
        async def mock_circuit_call(*args, **kwargs):
            raise CircuitBreakerOpenError("Circuit breaker is open")

        kernel_cache._circuit_breaker.call = mock_circuit_call

        # Load should fail gracefully
        result = await kernel_cache.load_kernel("test-kernel")
        assert result is None

        # Circuit breaker failure should be tracked
        stats = kernel_cache.get_cache_stats()
        assert stats["circuit_breaker_failures"] > 0

    @pytest.mark.asyncio
    async def test_kernel_not_found_vs_service_error(self, kernel_cache):
        """Test differentiation between kernel not found and service errors."""

        call_count = 0

        async def selective_fetch(artifact_id):
            nonlocal call_count
            call_count += 1

            if artifact_id == "nonexistent-kernel":
                # Simulate 404 - kernel not found
                return None
            elif artifact_id == "service-error-kernel":
                # Simulate service error
                raise Exception("Internal server error")
            else:
                # Successful fetch
                return torch.randn(1024, dtype=torch.float32)

        kernel_cache._fetch_from_urza_impl = selective_fetch

        # Not found should return None without affecting circuit breaker much
        result = await kernel_cache.load_kernel("nonexistent-kernel")
        assert result is None

        # Service error should affect circuit breaker
        result = await kernel_cache.load_kernel("service-error-kernel")
        assert result is None

        # Successful fetch should work
        result = await kernel_cache.load_kernel("good-kernel")
        assert result is not None

        # Check that different types of failures are handled appropriately
        stats = kernel_cache.get_cache_stats()
        assert stats["misses"] == 3  # All were cache misses

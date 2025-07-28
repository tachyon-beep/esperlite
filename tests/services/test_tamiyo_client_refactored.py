"""
Tests for TamiyoClient using real service infrastructure.

Tests against actual Tamiyo service or test container instead of mocking,
following project guidelines to use real services when available.
"""

import asyncio
import os
import time

import pytest

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.services.clients.tamiyo_client import TamiyoClient
from esper.utils.circuit_breaker import CircuitBreakerOpenError
from esper.utils.config import ServiceConfig


@pytest.fixture
def tamiyo_service_config():
    """Configure Tamiyo service for testing."""
    # Use test-specific port to avoid conflicts
    return ServiceConfig(
        base_url=os.environ.get("TAMIYO_TEST_URL", "http://localhost:8102"),
        timeout=5.0,
        max_retries=3
    )


@pytest.fixture
async def tamiyo_client(tamiyo_service_config):
    """Create TamiyoClient connected to test service."""
    client = TamiyoClient(config=tamiyo_service_config)

    # Check if service is available
    try:
        # Simple health check - adjust based on actual Tamiyo API
        await client._make_request("GET", "/health")
    except Exception:
        pytest.skip("Tamiyo service not available for testing")

    yield client


class TestTamiyoClientIntegration:
    """Integration tests for TamiyoClient with real service."""

    @pytest.mark.asyncio
    async def test_analyze_healthy_model_state(self, tamiyo_client):
        """Test analyzing a healthy model state."""
        # Create healthy signals
        health_signals = [
            HealthSignal(
                layer_id=i,
                seed_id=0,
                chunk_id=0,
                epoch=100,
                activation_variance=0.1,
                dead_neuron_ratio=0.02,
                avg_correlation=0.85,
                health_score=0.95,
                execution_latency=5.0,
                error_count=0,
                active_seeds=4,
                total_seeds=4,
                timestamp=time.time(),
            )
            for i in range(3)
        ]

        decisions = await tamiyo_client.analyze_model_state(health_signals)

        # Healthy model should produce few or no adaptation decisions
        assert isinstance(decisions, list)
        assert all(isinstance(d, AdaptationDecision) for d in decisions)

        # Most layers should have no action needed
        no_action_count = sum(1 for d in decisions if d.action == "no_action")
        assert no_action_count >= len(health_signals) * 0.8

    @pytest.mark.asyncio
    async def test_analyze_unhealthy_model_state(self, tamiyo_client):
        """Test analyzing model with performance issues."""
        # Create signals with various problems
        health_signals = [
            HealthSignal(
                layer_id=0,
                seed_id=0,
                chunk_id=0,
                epoch=100,
                activation_variance=0.9,  # High variance - unstable
                dead_neuron_ratio=0.4,   # Many dead neurons
                avg_correlation=0.3,     # Low correlation
                health_score=0.4,        # Poor health
                execution_latency=50.0,  # Slow
                error_count=10,          # Errors present
                active_seeds=2,
                total_seeds=4,
                timestamp=time.time(),
            ),
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=100,
                activation_variance=0.05,  # Too low - might be stuck
                dead_neuron_ratio=0.8,     # Most neurons dead
                avg_correlation=0.1,       # Very low correlation
                health_score=0.2,          # Critical health
                execution_latency=100.0,   # Very slow
                error_count=50,            # Many errors
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        decisions = await tamiyo_client.analyze_model_state(health_signals)

        # Should produce adaptation decisions
        assert len(decisions) > 0

        # Should have actions other than no_action
        action_types = {d.action for d in decisions}
        assert action_types - {"no_action"}  # Has other actions

        # Critical layers should have high priority
        critical_decisions = [d for d in decisions if d.layer_id == 1]
        if critical_decisions:
            assert any(d.priority == "high" for d in critical_decisions)

    @pytest.mark.asyncio
    async def test_get_seed_recommendations(self, tamiyo_client):
        """Test getting seed recommendations for layers."""
        layer_states = {
            "layer_0": {
                "performance": 0.3,  # Poor performance
                "stability": 0.8,
                "active_seeds": [0, 1],
                "available_seeds": [2, 3]
            },
            "layer_1": {
                "performance": 0.9,  # Good performance
                "stability": 0.9,
                "active_seeds": [0],
                "available_seeds": [1, 2, 3]
            }
        }

        recommendations = await tamiyo_client.get_seed_recommendations(layer_states)

        assert isinstance(recommendations, dict)

        # Poor performing layer should get recommendations
        if "layer_0" in recommendations:
            assert "recommended_seeds" in recommendations["layer_0"]
            assert "confidence" in recommendations["layer_0"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, tamiyo_client):
        """Test circuit breaker opens on repeated failures."""
        # Force failures by using invalid endpoint
        tamiyo_client.base_url = "http://localhost:99999"  # Non-existent port

        # Attempt multiple requests
        for _ in range(5):
            try:
                await tamiyo_client.analyze_model_state([])
            except Exception:
                pass  # Expected to fail

            # Small delay between attempts
            await asyncio.sleep(0.1)

        # Circuit breaker should be open after failures
        with pytest.raises(CircuitBreakerOpenError):
            await tamiyo_client.analyze_model_state([])

    @pytest.mark.asyncio
    async def test_batch_analysis(self, tamiyo_client):
        """Test analyzing large batches of health signals."""
        # Create a large batch of signals
        health_signals = []
        for layer_id in range(10):
            for seed_id in range(4):
                health_signals.append(
                    HealthSignal(
                        layer_id=layer_id,
                        seed_id=seed_id,
                        chunk_id=0,
                        epoch=100,
                        activation_variance=0.1 + (layer_id * 0.05),
                        dead_neuron_ratio=0.05 + (seed_id * 0.02),
                        avg_correlation=0.8 - (layer_id * 0.05),
                        health_score=0.9 - (layer_id * 0.08),
                        execution_latency=5.0 + (layer_id * 2),
                        error_count=layer_id,
                        active_seeds=4 - seed_id,
                        total_seeds=4,
                        timestamp=time.time(),
                    )
                )

        # Service should handle large batches
        decisions = await tamiyo_client.analyze_model_state(health_signals)

        assert isinstance(decisions, list)
        # Should return decisions for problematic layers
        assert len(decisions) > 0

        # Later layers (worse health) should have more decisions
        later_layer_decisions = [d for d in decisions if d.layer_id >= 7]
        early_layer_decisions = [d for d in decisions if d.layer_id <= 2]
        assert len(later_layer_decisions) >= len(early_layer_decisions)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, tamiyo_client):
        """Test client handles timeouts gracefully."""
        # Configure very short timeout
        tamiyo_client.timeout = 0.001  # 1ms timeout

        # Create large payload that takes time to process
        health_signals = [
            HealthSignal(
                layer_id=i,
                seed_id=j,
                chunk_id=0,
                epoch=100,
                activation_variance=0.1,
                dead_neuron_ratio=0.05,
                avg_correlation=0.8,
                health_score=0.9,
                execution_latency=5.0,
                error_count=0,
                active_seeds=4,
                total_seeds=4,
                timestamp=time.time(),
            )
            for i in range(100)
            for j in range(10)
        ]

        # Should timeout but handle gracefully
        with pytest.raises(asyncio.TimeoutError):
            await tamiyo_client.analyze_model_state(health_signals)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, tamiyo_client):
        """Test handling multiple concurrent requests."""
        # Create different signal sets
        signal_sets = []
        for i in range(5):
            signals = [
                HealthSignal(
                    layer_id=i,
                    seed_id=0,
                    chunk_id=0,
                    epoch=100 + i,
                    activation_variance=0.1 + i * 0.1,
                    dead_neuron_ratio=0.05,
                    avg_correlation=0.8,
                    health_score=0.9 - i * 0.1,
                    execution_latency=5.0,
                    error_count=i,
                    active_seeds=4,
                    total_seeds=4,
                    timestamp=time.time(),
                )
            ]
            signal_sets.append(signals)

        # Send concurrent requests
        tasks = [
            tamiyo_client.analyze_model_state(signals)
            for signals in signal_sets
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete
        assert len(results) == 5

        # Should get valid responses (not exceptions)
        valid_results = [r for r in results if not isinstance(r, Exception)]
        assert len(valid_results) >= 3  # Most should succeed


class TestTamiyoClientWithMockService:
    """Test client behavior with a mock Tamiyo service for edge cases."""

    @pytest.mark.asyncio
    async def test_service_unavailable_fallback(self):
        """Test fallback behavior when service is unavailable."""
        # Create client with unavailable service
        config = ServiceConfig(
            base_url="http://localhost:99999",  # Non-existent
            timeout=1.0,
            max_retries=1
        )
        client = TamiyoClient(config=config)

        # Should handle gracefully with empty decisions or error
        health_signals = [
            HealthSignal(
                layer_id=0,
                seed_id=0,
                chunk_id=0,
                epoch=100,
                activation_variance=0.1,
                dead_neuron_ratio=0.05,
                avg_correlation=0.8,
                health_score=0.9,
                execution_latency=5.0,
                error_count=0,
                active_seeds=4,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        try:
            decisions = await client.analyze_model_state(health_signals)
            # If it returns successfully, should be empty or safe defaults
            assert isinstance(decisions, list)
        except Exception as e:
            # Should be a connection or circuit breaker error
            assert "connect" in str(e).lower() or "circuit" in str(e).lower()

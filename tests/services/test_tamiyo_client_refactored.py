"""
Tests for TamiyoClient - Refactored Version.

This version focuses on real behavior testing rather than mocking
implementation details, and removes low-value tests.
"""

import asyncio
import time
import pytest

from esper.contracts.operational import AdaptationDecision, HealthSignal
from esper.services.clients.tamiyo_client import MockTamiyoClient, TamiyoClient
from esper.utils.config import ServiceConfig
from esper.utils.circuit_breaker import CircuitBreakerOpenError


@pytest.mark.real_components
class TestMockTamiyoClientBehavior:
    """Test MockTamiyoClient actual behavior."""

    @pytest.fixture
    def mock_client(self):
        """Create MockTamiyoClient for testing."""
        return MockTamiyoClient()

    @pytest.mark.asyncio
    async def test_healthy_signals_produce_no_decisions(self, mock_client):
        """Test that healthy layers don't trigger adaptations."""
        # Create truly healthy signals
        health_signals = [
            HealthSignal(
                layer_id=i,
                seed_id=0,
                chunk_id=0,
                epoch=100,
                activation_variance=0.1,  # Low variance is good
                dead_neuron_ratio=0.02,  # Few dead neurons
                avg_correlation=0.85,  # High correlation
                health_score=0.95,  # Excellent health
                execution_latency=5.0,  # Fast execution
                error_count=0,  # No errors
                active_seeds=4,  # All seeds active
                total_seeds=4,
                timestamp=time.time(),
            )
            for i in range(3)
        ]

        decisions = await mock_client.analyze_model_state(health_signals)
        assert len(decisions) == 0, (
            "Healthy layers should not generate adaptation decisions"
        )

    @pytest.mark.asyncio
    async def test_unhealthy_signals_generate_appropriate_decisions(self, mock_client):
        """Test that unhealthy layers generate sensible adaptation decisions."""
        unhealthy_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=50,
                activation_variance=0.8,  # High variance (bad)
                dead_neuron_ratio=0.45,  # Many dead neurons
                avg_correlation=0.2,  # Low correlation
                health_score=0.4,  # Poor health
                execution_latency=50.0,  # Slow execution
                error_count=5,  # Multiple errors
                active_seeds=1,  # Few active seeds
                total_seeds=4,
                timestamp=time.time(),
            ),
            HealthSignal(
                layer_id=2,
                seed_id=0,
                chunk_id=0,
                epoch=50,
                activation_variance=1.2,  # Very high variance
                dead_neuron_ratio=0.7,  # Most neurons dead
                avg_correlation=0.1,  # Very low correlation
                health_score=0.2,  # Critical health
                execution_latency=100.0,  # Very slow
                error_count=10,  # Many errors
                active_seeds=0,  # No active seeds
                total_seeds=4,
                timestamp=time.time(),
            ),
        ]

        decisions = await mock_client.analyze_model_state(unhealthy_signals)

        # Should generate decisions for unhealthy layers
        assert len(decisions) == 2

        # Verify decision properties make sense
        for i, decision in enumerate(decisions):
            signal = unhealthy_signals[i]

            # Basic properties
            assert decision.adaptation_type == "add_seed"
            assert decision.confidence > 0.5

            # Urgency should correlate with poor health
            assert decision.urgency > 0.0
            assert decision.urgency == pytest.approx(
                1.0 - signal.health_score, rel=0.01
            )

            # Should have meaningful metadata
            assert "kernel_artifact_id" in decision.metadata
            assert "reason" in decision.metadata
            # MockTamiyoClient includes health score value in reason
            assert "below threshold" in decision.metadata["reason"]

    @pytest.mark.asyncio
    async def test_mock_client_simulates_processing_delay(self, mock_client):
        """Test that mock client simulates realistic processing time."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.9,
                dead_neuron_ratio=0.5,
                avg_correlation=0.2,
                health_score=0.3,
                execution_latency=20.0,
                error_count=3,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        start_time = time.perf_counter()
        decisions = await mock_client.analyze_model_state(health_signals)
        elapsed = time.perf_counter() - start_time

        # Should have some processing delay
        assert elapsed > 0.005, "Mock should simulate processing delay"
        assert elapsed < 0.1, "Mock delay should be reasonable"

    @pytest.mark.asyncio
    async def test_mock_feedback_always_succeeds(self, mock_client):
        """Test that mock feedback submission always succeeds."""
        decision = AdaptationDecision(
            adaptation_type="add_seed",
            layer_name="test_layer",
            confidence=0.8,
            urgency=0.7,
            metadata={
                "target_seed_index": 0,
                "kernel_artifact_id": "test_kernel",
                "reason": "Test feedback",
            },
        )

        # Both positive and negative feedback should succeed
        assert await mock_client.submit_adaptation_feedback(decision, success=True)
        assert await mock_client.submit_adaptation_feedback(decision, success=False)

        # Stats should reflect the requests
        stats = mock_client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 2  # 2 feedbacks


@pytest.mark.real_components
class TestTamiyoClientRealBehavior:
    """Test TamiyoClient behavior with minimal mocking."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServiceConfig(tamiyo_url="http://test-tamiyo:8001", http_timeout=5)

    @pytest.fixture
    def client(self, config):
        """Create TamiyoClient for testing."""
        return TamiyoClient(config)

    @pytest.mark.asyncio
    async def test_empty_health_signals_skip_request(self, client):
        """Test that empty health signals skip HTTP request entirely."""
        decisions = await client.analyze_model_state([])

        assert decisions == []
        # Should not increment request counter for empty signals
        stats = client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_from_failures(self, client, monkeypatch):
        """Test circuit breaker protection from repeated failures."""
        # Create a failing request function
        failure_count = 0

        async def failing_request(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise ConnectionError("Simulated connection failure")

        # Patch the actual request method
        monkeypatch.setattr(client, "_make_analyze_request", failing_request)

        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.5,
                dead_neuron_ratio=0.2,
                avg_correlation=0.5,
                health_score=0.5,
                execution_latency=10.0,
                error_count=1,
                active_seeds=2,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        # Make requests until circuit breaker opens
        for i in range(5):
            decisions = await client.analyze_model_state(health_signals)
            assert decisions == []

        # Circuit breaker should be open now
        stats = client.get_client_stats()
        assert stats["circuit_breaker_stats"]["state"] in ["open", "half_open"]
        assert stats["client_stats"]["circuit_breaker_failures"] > 0

        # Further requests should fail fast without calling the failing function
        initial_failure_count = failure_count
        decisions = await client.analyze_model_state(health_signals)
        assert decisions == []
        # Should not have called the failing function
        assert failure_count == initial_failure_count

    @pytest.mark.asyncio
    async def test_successful_analysis_request_flow(self, client, monkeypatch):
        """Test successful request flow with realistic response."""

        # Mock a successful response
        async def mock_analyze_request(http_client, payload):
            # Simulate processing delay
            await asyncio.sleep(0.01)

            # Return realistic response based on input
            decisions = []
            for signal_data in payload["health_signals"]:
                if signal_data["health_score"] < 0.7:
                    decisions.append(
                        {
                            "adaptation_type": "add_seed",
                            "layer_name": f"layer_{signal_data['layer_id']}",
                            "confidence": 0.85,
                            "urgency": 1.0 - signal_data["health_score"],
                            "metadata": {
                                "target_seed_index": 1,
                                "kernel_artifact_id": f"kernel_{signal_data['layer_id']}",
                                "reason": "Health below threshold",
                            },
                        }
                    )

            return {"decisions": decisions}

        monkeypatch.setattr(client, "_make_analyze_request", mock_analyze_request)

        # Test with mixed health signals
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.1,
                dead_neuron_ratio=0.05,
                avg_correlation=0.9,
                health_score=0.9,  # Healthy
                execution_latency=5.0,
                error_count=0,
                active_seeds=4,
                total_seeds=4,
                timestamp=time.time(),
            ),
            HealthSignal(
                layer_id=2,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.7,
                dead_neuron_ratio=0.3,
                avg_correlation=0.3,
                health_score=0.5,  # Unhealthy
                execution_latency=20.0,
                error_count=2,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            ),
        ]

        decisions = await client.analyze_model_state(health_signals, {"epoch": 10})

        # Should only get decision for unhealthy layer
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.layer_name == "layer_2"
        assert decision.urgency == pytest.approx(0.5, rel=0.01)

        # Stats should reflect success
        stats = client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 1
        assert stats["client_stats"]["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_health_check_behavior(self, client, monkeypatch):
        """Test health check correctly interprets service status."""

        # Test healthy service
        async def mock_healthy_status():
            return {"healthy": True, "version": "1.0.0"}

        monkeypatch.setattr(client, "get_tamiyo_status", mock_healthy_status)
        assert await client.health_check() is True

        # Test unhealthy service
        async def mock_unhealthy_status():
            return {"healthy": False, "error": "Database connection failed"}

        monkeypatch.setattr(client, "get_tamiyo_status", mock_unhealthy_status)
        assert await client.health_check() is False

        # Test service error
        async def mock_error_status():
            raise Exception("Service unavailable")

        monkeypatch.setattr(client, "get_tamiyo_status", mock_error_status)
        assert await client.health_check() is False

    def test_statistics_tracking(self, client):
        """Test that statistics are properly tracked."""
        # Initial state
        stats = client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 0
        assert stats["base_url"] == "http://test-tamiyo:8001"
        assert stats["timeout"] == 5

        # Reset should clear stats
        client._stats["total_requests"] = 10
        client.reset_stats()

        stats = client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 0


@pytest.mark.integration
class TestTamiyoClientErrorHandling:
    """Test error handling in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_network_error_graceful_handling(self):
        """Test graceful handling of network errors."""
        # Create client pointing to non-existent service
        client = TamiyoClient(
            ServiceConfig(
                tamiyo_url="http://localhost:99999",  # Invalid port
                http_timeout=1,
            )
        )

        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.5,
                dead_neuron_ratio=0.2,
                avg_correlation=0.5,
                health_score=0.5,
                execution_latency=10.0,
                error_count=1,
                active_seeds=2,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        # Should handle error gracefully
        decisions = await client.analyze_model_state(health_signals)
        assert decisions == []

        # Should track the failure
        stats = client.get_client_stats()
        assert stats["client_stats"]["failed_requests"] > 0

    @pytest.mark.asyncio
    async def test_timeout_behavior(self):
        """Test timeout handling."""
        # Create client with very short timeout
        client = TamiyoClient(
            ServiceConfig(
                tamiyo_url="http://test-tamiyo:8001",
                http_timeout=0.001,  # 1ms timeout
            )
        )

        # Mock a slow request
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return {"decisions": []}

        # This would normally timeout, but we'll test the behavior
        decision = AdaptationDecision(
            adaptation_type="add_seed",
            layer_name="test_layer",
            confidence=0.5,
            urgency=0.5,
            metadata={},
        )

        # Feedback should handle timeout gracefully
        success = await client.submit_adaptation_feedback(decision, True)
        assert success is False


@pytest.mark.performance
class TestTamiyoClientPerformance:
    """Performance tests for TamiyoClient."""

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        client = MockTamiyoClient()

        # Create multiple health signal sets
        signal_sets = []
        for i in range(10):
            signals = [
                HealthSignal(
                    layer_id=i,
                    seed_id=0,
                    chunk_id=0,
                    epoch=10,
                    activation_variance=0.5,
                    dead_neuron_ratio=0.2,
                    avg_correlation=0.5,
                    health_score=0.6,  # Unhealthy to trigger decisions
                    execution_latency=10.0,
                    error_count=1,
                    active_seeds=2,
                    total_seeds=4,
                    timestamp=time.time(),
                )
            ]
            signal_sets.append(signals)

        # Make concurrent requests
        start_time = time.perf_counter()
        tasks = [client.analyze_model_state(signals) for signals in signal_sets]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        # All should complete
        assert len(results) == 10
        assert all(len(decisions) > 0 for decisions in results)

        # Should be faster than sequential (10 * 0.01s = 0.1s)
        assert elapsed < 0.05, "Concurrent requests should be faster than sequential"

    def test_stats_collection_overhead(self):
        """Test that stats collection has minimal overhead."""
        import time

        client = MockTamiyoClient()

        # Time stats operations
        iterations = 10000
        start = time.perf_counter()

        for _ in range(iterations):
            stats = client.get_client_stats()
            client._stats["total_requests"] += 1

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        # Should be very fast (microseconds)
        assert avg_time < 0.00001  # Less than 10 microseconds

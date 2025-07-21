"""
Tests for TamiyoClient and MockTamiyoClient.

This module tests the HTTP client for Tamiyo strategic controller
with circuit breaker protection and error handling.
"""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.services.clients.tamiyo_client import MockTamiyoClient
from esper.services.clients.tamiyo_client import TamiyoClient
from esper.utils.config import ServiceConfig


class TestMockTamiyoClient:
    """Test MockTamiyoClient functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create MockTamiyoClient for testing."""
        return MockTamiyoClient()

    @pytest.mark.asyncio
    async def test_mock_analyze_model_state_empty_signals(self, mock_client):
        """Test analysis with empty health signals."""
        decisions = await mock_client.analyze_model_state([])
        assert decisions == []

    @pytest.mark.asyncio
    async def test_mock_analyze_model_state_healthy_layers(self, mock_client):
        """Test analysis with healthy layers."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.1,
                dead_neuron_ratio=0.05,
                avg_correlation=0.8,
                health_score=0.9,  # High health
                execution_latency=10.0,
                error_count=0,
                active_seeds=3,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        decisions = await mock_client.analyze_model_state(health_signals)
        assert len(decisions) == 0  # No adaptations needed for healthy layers

    @pytest.mark.asyncio
    async def test_mock_analyze_model_state_unhealthy_layers(self, mock_client):
        """Test analysis with unhealthy layers."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.8,
                dead_neuron_ratio=0.4,
                avg_correlation=0.2,
                health_score=0.5,  # Low health
                execution_latency=50.0,
                error_count=3,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            ),
            HealthSignal(
                layer_id=2,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=1.2,
                dead_neuron_ratio=0.6,
                avg_correlation=0.1,
                health_score=0.3,  # Very low health
                execution_latency=100.0,
                error_count=5,
                active_seeds=0,
                total_seeds=4,
                timestamp=time.time(),
            ),
        ]

        decisions = await mock_client.analyze_model_state(health_signals)
        assert (
            len(decisions) == 2
        )  # Should generate decisions for both unhealthy layers

        for decision in decisions:
            assert decision.adaptation_type == "add_seed"
            assert decision.confidence == 0.8
            assert decision.urgency > 0.0
            assert decision.metadata.get("kernel_artifact_id") is not None

    @pytest.mark.asyncio
    async def test_mock_submit_feedback(self, mock_client):
        """Test feedback submission."""
        decision = AdaptationDecision(
            adaptation_type="add_seed",
            layer_name="test_layer",
            confidence=0.8,
            urgency=0.7,
            metadata={
                "target_seed_index": 0,
                "estimated_improvement": 0.15,
                "kernel_artifact_id": "test_kernel_123",
                "reason": "Low health score",
            },
        )

        success = await mock_client.submit_adaptation_feedback(
            decision=decision,
            success=True,
            performance_impact={"loss_improvement": 0.05},
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_mock_status_and_health(self, mock_client):
        """Test status and health check methods."""
        status = await mock_client.get_tamiyo_status()
        assert status["healthy"] is True
        assert "version" in status

        health = await mock_client.health_check()
        assert health is True

    def test_mock_client_stats(self, mock_client):
        """Test client statistics tracking."""
        stats = mock_client.get_client_stats()

        assert "client_stats" in stats
        assert "base_url" in stats
        assert "timeout" in stats

        # Initially zero requests
        assert stats["client_stats"]["total_requests"] == 0


class TestTamiyoClientWithMocks:
    """Test TamiyoClient with mocked HTTP responses."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServiceConfig(tamiyo_url="http://test-tamiyo:8001", http_timeout=5)

    @pytest.fixture
    def client(self, config):
        """Create TamiyoClient for testing."""
        return TamiyoClient(config)

    @pytest.mark.asyncio
    async def test_analyze_model_state_success(self, client):
        """Test successful model state analysis."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.3,
                dead_neuron_ratio=0.15,
                avg_correlation=0.6,
                health_score=0.6,
                execution_latency=20.0,
                error_count=1,
                active_seeds=2,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        context = {"epoch": 10, "learning_rate": 0.001}

        # Mock the HTTP client response
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "decisions": [
                    {
                        "adaptation_type": "add_seed",
                        "layer_name": "test_layer",
                        "layer_name": "test_layer",
                        "confidence": 0.85,
                        "urgency": 0.7,
                        "metadata": {
                            "target_seed_index": 1,
                            "estimated_improvement": 0.12,
                            "kernel_artifact_id": "kernel_456",
                            "reason": "Health score below threshold",
                        },
                    }
                ]
            }
        )

        with patch.object(
            client,
            "_make_analyze_request",
            return_value=mock_response.json.return_value,
        ):
            decisions = await client.analyze_model_state(health_signals, context)

        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.adaptation_type == "add_seed"
        assert decision.layer_name == "test_layer"
        assert decision.confidence == 0.85

    @pytest.mark.asyncio
    async def test_analyze_model_state_empty_response(self, client):
        """Test analysis with empty response."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.05,
                dead_neuron_ratio=0.02,
                avg_correlation=0.9,
                health_score=0.9,
                execution_latency=5.0,
                error_count=0,
                active_seeds=4,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        with patch.object(
            client, "_make_analyze_request", return_value={"decisions": []}
        ):
            decisions = await client.analyze_model_state(health_signals)

        assert len(decisions) == 0

    @pytest.mark.asyncio
    async def test_analyze_model_state_circuit_breaker_open(self, client):
        """Test behavior when circuit breaker is open."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.7,
                dead_neuron_ratio=0.3,
                avg_correlation=0.3,
                health_score=0.5,
                execution_latency=20.0,
                error_count=2,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        # Simulate circuit breaker open
        from esper.utils.circuit_breaker import CircuitBreakerOpenError

        with patch.object(
            client._circuit_breaker,
            "call",
            side_effect=CircuitBreakerOpenError("Circuit breaker is open"),
        ):
            decisions = await client.analyze_model_state(health_signals)

        assert len(decisions) == 0
        assert client._stats["circuit_breaker_failures"] > 0

    @pytest.mark.asyncio
    async def test_submit_adaptation_feedback_success(self, client):
        """Test successful feedback submission."""
        decision = AdaptationDecision(
            adaptation_type="add_seed",
            layer_name="test_layer",
            confidence=0.8,
            urgency=0.7,
            metadata={
                "target_seed_index": 0,
                "estimated_improvement": 0.15,
                "kernel_artifact_id": "test_kernel_789",
                "reason": "Performance improvement needed",
            },
        )

        with patch.object(client, "_make_feedback_request", return_value=None):
            success = await client.submit_adaptation_feedback(
                decision=decision,
                success=True,
                performance_impact={"accuracy_gain": 0.02},
            )

        assert success is True

    @pytest.mark.asyncio
    async def test_get_tamiyo_status_success(self, client):
        """Test successful status retrieval."""
        mock_status = {
            "healthy": True,
            "version": "1.0.0",
            "uptime": 3600,
            "active_analyses": 2,
        }

        with patch.object(client, "_make_status_request", return_value=mock_status):
            status = await client.get_tamiyo_status()

        assert status == mock_status

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check with healthy service."""
        with patch.object(client, "get_tamiyo_status", return_value={"healthy": True}):
            is_healthy = await client.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Test health check with unhealthy service."""
        with patch.object(
            client, "get_tamiyo_status", side_effect=Exception("Service down")
        ):
            is_healthy = await client.health_check()

        assert is_healthy is False

    def test_client_stats_tracking(self, client):
        """Test that client statistics are tracked correctly."""
        initial_stats = client.get_client_stats()

        assert initial_stats["client_stats"]["total_requests"] == 0
        assert initial_stats["client_stats"]["successful_requests"] == 0
        assert initial_stats["client_stats"]["failed_requests"] == 0
        assert "circuit_breaker_stats" in initial_stats

    def test_reset_stats(self, client):
        """Test statistics reset functionality."""
        # Simulate some usage
        client._stats["total_requests"] = 10
        client._stats["successful_requests"] = 8
        client._stats["failed_requests"] = 2

        # Reset stats
        client.reset_stats()

        stats = client.get_client_stats()
        assert stats["client_stats"]["total_requests"] == 0
        assert stats["client_stats"]["successful_requests"] == 0
        assert stats["client_stats"]["failed_requests"] == 0


class TestTamiyoClientIntegration:
    """Integration tests for TamiyoClient error handling."""

    @pytest.fixture
    def client(self):
        """Create TamiyoClient with test configuration."""
        config = ServiceConfig(
            tamiyo_url="http://nonexistent-tamiyo:8001",
            http_timeout=1,  # Short timeout for faster tests
        )
        return TamiyoClient(config)

    @pytest.mark.asyncio
    async def test_network_error_handling(self, client):
        """Test handling of network errors."""
        health_signals = [
            HealthSignal(
                layer_id=1,
                seed_id=0,
                chunk_id=0,
                epoch=10,
                activation_variance=0.6,
                dead_neuron_ratio=0.25,
                avg_correlation=0.4,
                health_score=0.5,
                execution_latency=20.0,
                error_count=2,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time(),
            )
        ]

        # This should handle network errors gracefully
        decisions = await client.analyze_model_state(health_signals)
        assert decisions == []  # Should return empty list on error

        # Check that failure was recorded
        stats = client.get_client_stats()
        assert stats["client_stats"]["failed_requests"] > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, client):
        """Test handling of request timeouts."""
        decision = AdaptationDecision(
            adaptation_type="add_seed",
            layer_name="test_layer",
            confidence=0.8,
            urgency=0.7,
            metadata={
                "target_seed_index": 0,
                "estimated_improvement": 0.15,
                "kernel_artifact_id": "test_kernel_timeout",
                "reason": "Timeout test",
            },
        )

        # This should handle timeouts gracefully
        success = await client.submit_adaptation_feedback(decision, True)
        assert success is False

        # Check that failure was recorded
        stats = client.get_client_stats()
        assert stats["client_stats"]["failed_requests"] > 0


class TestTamiyoClientConfiguration:
    """Test TamiyoClient configuration handling."""

    def test_default_configuration(self):
        """Test client with default configuration."""
        client = TamiyoClient()

        stats = client.get_client_stats()
        assert stats["base_url"].startswith("http")
        assert stats["timeout"] > 0

    def test_custom_configuration(self):
        """Test client with custom configuration."""
        config = ServiceConfig(
            tamiyo_url="https://custom-tamiyo.example.com", http_timeout=60
        )

        client = TamiyoClient(config)
        stats = client.get_client_stats()

        assert stats["base_url"] == "https://custom-tamiyo.example.com"
        assert stats["timeout"] == 60

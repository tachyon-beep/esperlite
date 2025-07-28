"""Unit tests for message bus publishers."""

import asyncio
from collections import deque
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.esper.morphogenetic_v2.message_bus.publishers import EventPublisher
from src.esper.morphogenetic_v2.message_bus.publishers import TelemetryConfig
from src.esper.morphogenetic_v2.message_bus.publishers import TelemetryPublisher
from src.esper.morphogenetic_v2.message_bus.schemas import SeedMetricsSnapshot
from src.esper.morphogenetic_v2.message_bus.schemas import StateTransitionEvent


class TestTelemetryConfig:
    """Test TelemetryConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = TelemetryConfig(
            batch_size=100,
            batch_window_ms=100,
            compression="zstd"
        )
        config.validate()  # Should not raise

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        config = TelemetryConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_invalid_batch_window(self):
        """Test invalid batch window."""
        config = TelemetryConfig(batch_window_ms=-1)
        with pytest.raises(ValueError, match="batch_window_ms must be positive"):
            config.validate()

    def test_invalid_compression(self):
        """Test invalid compression type."""
        config = TelemetryConfig(compression="invalid")
        with pytest.raises(ValueError, match="compression must be None, 'zstd', or 'gzip'"):
            config.validate()

    def test_invalid_compression_level(self):
        """Test invalid compression level."""
        config = TelemetryConfig(compression_level=10)
        with pytest.raises(ValueError, match="compression_level must be between 1 and 9"):
            config.validate()


class TestTelemetryPublisher:
    """Test TelemetryPublisher functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock message bus client."""
        client = AsyncMock()
        client.publish = AsyncMock()
        return client

    @pytest.fixture
    def telemetry_config(self):
        """Create test telemetry config."""
        return TelemetryConfig(
            batch_size=10,
            batch_window_ms=50,
            compression=None,
            enable_aggregation=False,
            anomaly_detection=False
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_client, telemetry_config):
        """Test publisher lifecycle."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)

        assert not publisher._running
        assert len(publisher._tasks) == 0

        await publisher.start()

        assert publisher._running
        assert len(publisher._tasks) > 0

        await publisher.stop()

        assert not publisher._running
        assert len(publisher._tasks) == 0

    @pytest.mark.asyncio
    async def test_publish_layer_health(self, mock_client, telemetry_config):
        """Test publishing layer health report."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)
        await publisher.start()

        # Create health data
        health_data = torch.randn(100, 4)

        await publisher.publish_layer_health("test_layer", health_data)

        # Should be batched, not published immediately
        assert mock_client.publish.call_count == 0
        assert publisher.batch_queue.qsize() == 1

        # Move message from queue to current_batch (simulating what _batch_publisher does)
        message = await publisher.batch_queue.get()
        publisher.current_batch.append(message)

        # Force flush
        await publisher._flush_batch()

        # Now should be published
        assert mock_client.publish.call_count == 1
        call_args = mock_client.publish.call_args
        assert "telemetry.batch" in call_args[0][0]

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_seed_metrics(self, mock_client, telemetry_config):
        """Test publishing seed metrics."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)
        await publisher.start()

        await publisher.publish_seed_metrics(
            layer_id="test_layer",
            seed_id=42,
            metrics={"loss": 0.1, "accuracy": 0.95},
            lifecycle_state="TRAINING",
            blueprint_id=10
        )

        assert publisher.batch_queue.qsize() == 1

        # Get the message from queue
        msg = await publisher.batch_queue.get()
        assert isinstance(msg, SeedMetricsSnapshot)
        assert msg.layer_id == "test_layer"
        assert msg.seed_id == 42
        assert msg.metrics["loss"] == 0.1

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_event_immediate(self, mock_client, telemetry_config):
        """Test that events are published immediately."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)
        await publisher.start()

        event = StateTransitionEvent(
            layer_id="test_layer",
            seed_id=0,
            from_state="DORMANT",
            to_state="GERMINATED"
        )

        await publisher.publish_event(event)

        # Events should be published immediately
        assert mock_client.publish.call_count == 1
        assert publisher.stats["messages_published"] == 1

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_batch_window(self, mock_client):
        """Test batch window timing."""
        config = TelemetryConfig(
            batch_size=100,  # Large batch size
            batch_window_ms=100,  # 100ms window
            enable_aggregation=False  # Disable aggregation for this test
        )
        publisher = TelemetryPublisher(mock_client, config)
        await publisher.start()

        # Add one message
        health_data = torch.randn(10, 4)
        await publisher.publish_layer_health("test", health_data)

        # Wait for batch window
        await asyncio.sleep(0.15)

        # Should have published due to window timeout
        assert mock_client.publish.call_count == 1

        # Clean up
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_batch_size_trigger(self, mock_client):
        """Test batch size triggering."""
        config = TelemetryConfig(
            batch_size=2,
            batch_window_ms=1000  # Long window
        )
        publisher = TelemetryPublisher(mock_client, config)
        await publisher.start()

        # Add messages to trigger batch
        for i in range(2):
            await publisher.publish_seed_metrics(
                layer_id="test",
                seed_id=i,
                metrics={"loss": 0.1},
                lifecycle_state="TRAINING"
            )

        # Give batch publisher time to process
        await asyncio.sleep(0.1)

        # Should have published due to batch size
        assert mock_client.publish.call_count == 1

        await publisher.stop()

    def test_tensor_to_dict(self, mock_client, telemetry_config):
        """Test tensor to dictionary conversion."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)

        # CPU tensor
        tensor = torch.tensor([
            [1.0, 0.5, 0.9, 10.0],
            [2.0, 0.3, 0.95, 15.0]
        ])

        result = publisher._tensor_to_dict(tensor)

        assert len(result) == 2
        assert result[0]["lifecycle_state"] == 1.0
        assert result[0]["loss"] == 0.5
        assert abs(result[0]["accuracy"] - 0.9) < 1e-6
        assert result[0]["compute_time_ms"] == 10.0
        
        assert result[1]["lifecycle_state"] == 2.0
        assert abs(result[1]["loss"] - 0.3) < 1e-6
        assert abs(result[1]["accuracy"] - 0.95) < 1e-6
        assert result[1]["compute_time_ms"] == 15.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_to_dict_gpu(self, mock_client, telemetry_config):
        """Test GPU tensor to dictionary conversion."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)

        # GPU tensor
        tensor = torch.randn(10, 4).cuda()
        result = publisher._tensor_to_dict(tensor)

        assert len(result) == 10
        assert all("lifecycle_state" in metrics for metrics in result.values())

    def test_calculate_summary(self, mock_client, telemetry_config):
        """Test summary statistics calculation."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)

        health_dict = {
            0: {"loss": 0.1, "accuracy": 0.9},
            1: {"loss": 0.2, "accuracy": 0.8},
            2: {"loss": 0.3, "accuracy": 0.7}
        }

        summary = publisher._calculate_summary(health_dict)

        assert "loss_mean" in summary
        assert summary["loss_mean"] == pytest.approx(0.2, rel=1e-5)
        assert "loss_std" in summary
        assert "loss_min" in summary
        assert summary["loss_min"] == 0.1
        assert "loss_max" in summary
        assert summary["loss_max"] == 0.3
        assert "loss_p50" in summary
        assert "loss_p95" in summary
        assert "loss_p99" in summary

    def test_is_anomaly(self, mock_client, telemetry_config):
        """Test anomaly detection logic."""
        config = TelemetryConfig(anomaly_detection=True, anomaly_threshold_stddev=2.0)
        publisher = TelemetryPublisher(mock_client, config)

        # Build history with some variation
        history = deque([1.0, 1.1, 0.9, 1.0, 1.05, 0.95] * 3, maxlen=100)

        # Normal value (within 2 std devs)
        assert not publisher._is_anomaly(history, 1.1)

        # Anomalous value (beyond 2 std devs)
        assert publisher._is_anomaly(history, 5.0)

        # Edge case: zero std - any different value is anomaly
        history = deque([1.0] * 20, maxlen=100)
        assert publisher._is_anomaly(history, 1.1)  # Even small difference is anomaly when std=0

    @pytest.mark.asyncio
    async def test_anomaly_detection_flow(self, mock_client):
        """Test complete anomaly detection flow."""
        config = TelemetryConfig(
            batch_size=100,
            anomaly_detection=True,
            anomaly_threshold_stddev=2.0
        )
        publisher = TelemetryPublisher(mock_client, config)
        await publisher.start()

        # Track alerts
        alerts = []

        async def callback(alert):
            alerts.append(alert)

        publisher.add_anomaly_callback(callback)

        # Publish normal metrics to build history
        for i in range(15):
            await publisher.publish_seed_metrics(
                layer_id="test",
                seed_id=0,
                metrics={"loss": 0.1 + np.random.normal(0, 0.01)},
                lifecycle_state="TRAINING"
            )

        # Publish anomalous metric
        await publisher.publish_seed_metrics(
            layer_id="test",
            seed_id=0,
            metrics={"loss": 1.0},  # Anomaly
            lifecycle_state="TRAINING"
        )

        # Give time for processing
        await asyncio.sleep(0.1)

        # Should have detected anomaly
        assert len(alerts) == 1
        assert alerts[0].metric_name == "loss"
        assert alerts[0].metric_value == 1.0

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_aggregation_window(self, mock_client):
        """Test aggregation window functionality."""
        config = TelemetryConfig(
            enable_aggregation=True,
            aggregation_window_s=0.1  # 100ms window
        )
        publisher = TelemetryPublisher(mock_client, config)
        await publisher.start()

        # Publish multiple reports
        for i in range(3):
            health_data = torch.ones(10, 4) * i
            await publisher.publish_layer_health("test_layer", health_data)

        # Wait for aggregation
        await asyncio.sleep(0.15)

        # Should have published aggregated report
        assert mock_client.publish.call_count >= 1
        call_args = mock_client.publish.call_args
        assert "telemetry.aggregated" in call_args[0][0]

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_claim_check_large_payload(self, mock_client, tmp_path):
        """Test claim check pattern for large payloads."""
        config = TelemetryConfig(
            claim_check_threshold=1000,  # 1KB threshold
            claim_check_storage=str(tmp_path),
            enable_aggregation=False,  # Disable aggregation
            compression=None  # Disable compression
        )
        publisher = TelemetryPublisher(mock_client, config)
        await publisher.start()

        # Create large batch
        for i in range(20):
            health_data = torch.randn(100, 10)  # Large tensor
            await publisher.publish_layer_health(f"layer_{i}", health_data)

        # Move messages from queue to current_batch
        while not publisher.batch_queue.empty():
            msg = await publisher.batch_queue.get()
            publisher.current_batch.append(msg)

        # Force flush
        await publisher._flush_batch()

        # Should have used claim check
        call_args = mock_client.publish.call_args
        batch_msg = call_args[0][1]
        assert "claim_check_id" in batch_msg.metadata
        assert len(batch_msg.messages) == 0  # Messages moved to storage

        # Verify file exists
        claim_check_id = batch_msg.metadata["claim_check_id"]
        file_path = tmp_path / f"{claim_check_id}.bin"
        assert file_path.exists()

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_client, telemetry_config):
        """Test statistics retrieval."""
        publisher = TelemetryPublisher(mock_client, telemetry_config)
        await publisher.start()

        # Generate some activity
        for i in range(5):
            await publisher.publish_seed_metrics(
                layer_id="test",
                seed_id=i,
                metrics={"loss": 0.1},
                lifecycle_state="TRAINING"
            )

        # Move messages from queue to current_batch
        while not publisher.batch_queue.empty():
            msg = await publisher.batch_queue.get()
            publisher.current_batch.append(msg)

        await publisher._flush_batch()

        stats = await publisher.get_stats()

        assert stats["messages_published"] == 5
        assert stats["batches_sent"] == 1
        assert "batch_queue_size" in stats
        assert "current_batch_size" in stats

        await publisher.stop()


class TestEventPublisher:
    """Test EventPublisher functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock message bus client."""
        client = AsyncMock()
        client.publish = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_publish_state_transition(self, mock_client):
        """Test publishing state transition event."""
        publisher = EventPublisher(mock_client)

        await publisher.publish_state_transition(
            layer_id="test_layer",
            seed_id=42,
            from_state="DORMANT",
            to_state="GERMINATED",
            reason="Activation",
            metrics={"init_loss": 1.0}
        )

        assert mock_client.publish.call_count == 1
        call_args = mock_client.publish.call_args

        topic = call_args[0][0]
        event = call_args[0][1]

        assert "event.statetransitionevent" in topic
        assert isinstance(event, StateTransitionEvent)
        assert event.from_state == "DORMANT"
        assert event.to_state == "GERMINATED"
        assert event.metrics_snapshot["init_loss"] == 1.0
        assert publisher.stats["events_published"] == 1

    @pytest.mark.asyncio
    async def test_publish_with_retry_success(self, mock_client):
        """Test event publishing with retry on failure."""
        publisher = EventPublisher(mock_client)

        # Fail once, then succeed
        mock_client.publish.side_effect = [
            Exception("Network error"),
            None
        ]

        event = StateTransitionEvent(
            layer_id="test",
            seed_id=0,
            from_state="A",
            to_state="B"
        )

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await publisher._publish_with_retry(event, max_retries=2)

        assert mock_client.publish.call_count == 2
        assert publisher.stats["retry_count"] == 1
        assert publisher.stats["events_published"] == 1

    @pytest.mark.asyncio
    async def test_publish_with_retry_failure(self, mock_client):
        """Test event publishing failure after max retries."""
        publisher = EventPublisher(mock_client)

        # Always fail
        mock_client.publish.side_effect = Exception("Persistent error")

        event = StateTransitionEvent(
            layer_id="test",
            seed_id=0,
            from_state="A",
            to_state="B"
        )

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent error"):
                await publisher._publish_with_retry(event, max_retries=2)

        assert mock_client.publish.call_count == 2
        assert publisher.stats["events_failed"] == 1

    def test_get_event_topic(self, mock_client):
        """Test event topic generation."""
        publisher = EventPublisher(mock_client)

        # State transition event
        event = StateTransitionEvent(layer_id="test", seed_id=0)
        topic = publisher._get_event_topic(event)
        assert topic == "morphogenetic.event.statetransitionevent.layer.test"

        # Event without layer_id
        generic_event = Mock(spec=[], layer_id=None)
        generic_event.__class__.__name__ = "TestEvent"
        topic = publisher._get_event_topic(generic_event)
        assert topic == "morphogenetic.event.testevent"

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_client):
        """Test statistics retrieval."""
        publisher = EventPublisher(mock_client)

        # Generate some activity
        await publisher.publish_state_transition(
            "layer1", 0, "A", "B"
        )

        # Force a failure
        mock_client.publish.side_effect = Exception("Error")
        try:
            await publisher.publish_state_transition(
                "layer2", 0, "B", "C"
            )
        except:
            pass

        stats = await publisher.get_stats()

        assert stats["events_published"] == 1
        assert stats["events_failed"] == 1

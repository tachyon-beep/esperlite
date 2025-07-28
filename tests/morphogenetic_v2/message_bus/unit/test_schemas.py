"""Unit tests for message schemas."""

import time
import uuid

import pytest

from src.esper.morphogenetic_v2.message_bus.schemas import AlertSeverity
from src.esper.morphogenetic_v2.message_bus.schemas import AlertType
from src.esper.morphogenetic_v2.message_bus.schemas import BaseMessage
from src.esper.morphogenetic_v2.message_bus.schemas import LayerHealthReport
from src.esper.morphogenetic_v2.message_bus.schemas import LifecycleTransitionCommand
from src.esper.morphogenetic_v2.message_bus.schemas import MessageFactory
from src.esper.morphogenetic_v2.message_bus.schemas import PerformanceAlert
from src.esper.morphogenetic_v2.message_bus.schemas import SeedMetricsSnapshot
from src.esper.morphogenetic_v2.message_bus.schemas import StateTransitionEvent
from src.esper.morphogenetic_v2.message_bus.schemas import create_topic_name
from src.esper.morphogenetic_v2.message_bus.schemas import parse_topic_name


class TestBaseMessage:
    """Test BaseMessage functionality."""

    def test_base_message_creation(self):
        """Test creating a base message."""
        msg = BaseMessage(source="test_source")

        assert msg.message_id
        assert isinstance(msg.message_id, str)
        assert msg.timestamp > 0
        assert msg.source == "test_source"
        assert msg.version == "1.0"
        assert msg.correlation_id is None
        assert msg.metadata == {}

    def test_base_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = BaseMessage(
            source="test",
            correlation_id="corr_123",
            metadata={"key": "value"}
        )

        data = msg.to_dict()

        assert data["message_id"] == msg.message_id
        assert data["timestamp"] == msg.timestamp
        assert data["source"] == "test"
        assert data["version"] == "1.0"
        assert data["correlation_id"] == "corr_123"
        assert data["metadata"] == {"key": "value"}
        assert data["message_type"] == "BaseMessage"

    def test_base_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "message_id": "test_id",
            "timestamp": 123.456,
            "source": "test",
            "version": "2.0",
            "correlation_id": "corr_123",
            "metadata": {"key": "value"}
        }

        msg = BaseMessage.from_dict(data)

        assert msg.message_id == "test_id"
        assert msg.timestamp == 123.456
        assert msg.source == "test"
        assert msg.version == "2.0"
        assert msg.correlation_id == "corr_123"
        assert msg.metadata == {"key": "value"}


class TestTelemetryMessages:
    """Test telemetry message types."""

    def test_layer_health_report(self):
        """Test LayerHealthReport creation and serialization."""
        report = LayerHealthReport(
            layer_id="layer_1",
            total_seeds=100,
            active_seeds=30,
            health_metrics={
                0: {"loss": 0.1, "accuracy": 0.95},
                1: {"loss": 0.2, "accuracy": 0.90}
            },
            performance_summary={"loss_mean": 0.15},
            telemetry_window=(100.0, 110.0),
            anomalies=[{"metric": "loss", "severity": "high"}]
        )

        assert report.layer_id == "layer_1"
        assert report.inactive_seeds == 70
        assert report.window_duration == 10.0

        # Test serialization
        data = report.to_dict()
        assert data["message_type"] == "LayerHealthReport"
        assert data["layer_id"] == "layer_1"
        assert data["total_seeds"] == 100

        # Test deserialization
        restored = LayerHealthReport.from_dict(data)
        assert restored.layer_id == report.layer_id
        assert restored.total_seeds == report.total_seeds

    def test_seed_metrics_snapshot(self):
        """Test SeedMetricsSnapshot creation."""
        snapshot = SeedMetricsSnapshot(
            layer_id="layer_1",
            seed_id=42,
            lifecycle_state="TRAINING",
            blueprint_id=10,
            metrics={"loss": 0.1, "accuracy": 0.95},
            error_count=2,
            warning_count=5,
            checkpoint_id="ckpt_123"
        )

        assert snapshot.seed_id == 42
        assert snapshot.lifecycle_state == "TRAINING"
        assert snapshot.error_count == 2

        # Test serialization roundtrip
        data = snapshot.to_dict()
        restored = SeedMetricsSnapshot.from_dict(data)
        assert restored.seed_id == snapshot.seed_id
        assert restored.metrics == snapshot.metrics


class TestControlCommands:
    """Test control command messages."""

    def test_lifecycle_transition_command(self):
        """Test LifecycleTransitionCommand."""
        cmd = LifecycleTransitionCommand(
            layer_id="layer_1",
            seed_id=0,
            target_state="GRAFTING",
            parameters={"learning_rate": 0.001},
            priority="high",
            timeout_ms=10000,
            force=True,
            reason="Performance improvement"
        )

        assert cmd.target_state == "GRAFTING"
        assert cmd.parameters["learning_rate"] == 0.001
        assert cmd.priority == "high"
        assert cmd.force is True

        # Test serialization
        data = cmd.to_dict()
        assert data["message_type"] == "LifecycleTransitionCommand"
        assert data["force"] is True


class TestEventMessages:
    """Test event message types."""

    def test_state_transition_event(self):
        """Test StateTransitionEvent."""
        event = StateTransitionEvent(
            layer_id="layer_1",
            seed_id=0,
            from_state="TRAINING",
            to_state="GRAFTING",
            reason="Automatic progression",
            metrics_snapshot={"loss": 0.1},
            transition_duration_ms=150.5,
            triggered_by="system",
            success=True
        )

        assert event.from_state == "TRAINING"
        assert event.to_state == "GRAFTING"
        assert event.success is True

        # Test serialization
        data = event.to_dict()
        restored = StateTransitionEvent.from_dict(data)
        assert restored.from_state == event.from_state
        assert restored.metrics_snapshot == event.metrics_snapshot

    def test_performance_alert(self):
        """Test PerformanceAlert."""
        alert = PerformanceAlert(
            layer_id="layer_1",
            seed_id=42,
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.WARNING,
            metric_name="loss",
            metric_value=0.5,
            threshold=0.2,
            details={"stddev": 0.1},
            recommended_action="Check training data"
        )

        assert alert.alert_type == AlertType.ANOMALY
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value == 0.5

        # Test enum serialization
        data = alert.to_dict()
        assert data["alert_type"] == "anomaly"
        assert data["severity"] == "warning"


class TestMessageFactory:
    """Test MessageFactory functionality."""

    def test_create_from_dict(self):
        """Test creating messages from dictionaries."""
        # Test health report
        data = {
            "message_type": "LayerHealthReport",
            "layer_id": "test",
            "total_seeds": 100,
            "active_seeds": 50,
            "health_metrics": {},
            "performance_summary": {},
            "telemetry_window": [0, 10]
        }

        msg = MessageFactory.create(data)
        assert isinstance(msg, LayerHealthReport)
        assert msg.layer_id == "test"
        assert msg.total_seeds == 100

    def test_unknown_message_type(self):
        """Test error on unknown message type."""
        with pytest.raises(ValueError, match="Unknown message type"):
            MessageFactory.create({"message_type": "UnknownType"})

    def test_missing_message_type(self):
        """Test error on missing message type."""
        with pytest.raises(ValueError, match="Missing message_type"):
            MessageFactory.create({"some": "data"})


class TestTopicUtilities:
    """Test topic utility functions."""

    def test_create_topic_name(self):
        """Test topic name creation."""
        # Basic topic
        topic = create_topic_name("telemetry")
        assert topic == "morphogenetic.telemetry"

        # With layer
        topic = create_topic_name("telemetry", layer_id="layer_1")
        assert topic == "morphogenetic.telemetry.layer.layer_1"

        # With layer and seed
        topic = create_topic_name("control", layer_id="layer_1", seed_id=42)
        assert topic == "morphogenetic.control.layer.layer_1.seed.42"

    def test_parse_topic_name(self):
        """Test topic name parsing."""
        # Basic topic
        parsed = parse_topic_name("morphogenetic.telemetry")
        assert parsed["system"] == "morphogenetic"
        assert parsed["message_type"] == "telemetry"
        assert "layer_id" not in parsed

        # With layer
        parsed = parse_topic_name("morphogenetic.control.layer.layer_1")
        assert parsed["layer_id"] == "layer_1"
        assert "seed_id" not in parsed

        # With layer and seed
        parsed = parse_topic_name("morphogenetic.event.layer.layer_1.seed.42")
        assert parsed["layer_id"] == "layer_1"
        assert parsed["seed_id"] == 42

        # Complex topic
        parsed = parse_topic_name("morphogenetic.telemetry.aggregated.layer.test")
        assert parsed["message_type"] == "telemetry"
        assert parsed["layer_id"] == "test"


class TestMessageValidation:
    """Test message validation and edge cases."""

    def test_message_id_generation(self):
        """Test automatic message ID generation."""
        msg1 = BaseMessage()
        msg2 = BaseMessage()

        # Should have different IDs
        assert msg1.message_id != msg2.message_id

        # Should be valid UUIDs
        uuid.UUID(msg1.message_id)
        uuid.UUID(msg2.message_id)

    def test_timestamp_generation(self):
        """Test automatic timestamp generation."""
        before = time.time()
        msg = BaseMessage()
        after = time.time()

        assert before <= msg.timestamp <= after

    def test_empty_health_metrics(self):
        """Test health report with empty metrics."""
        report = LayerHealthReport(
            layer_id="test",
            total_seeds=0,
            active_seeds=0,
            health_metrics={},
            performance_summary={},
            telemetry_window=(0, 0)
        )

        assert report.inactive_seeds == 0
        assert report.window_duration == 0

    def test_message_metadata_preservation(self):
        """Test that metadata is preserved through serialization."""
        msg = BaseMessage(
            metadata={
                "custom_field": "value",
                "nested": {"data": [1, 2, 3]}
            }
        )

        data = msg.to_dict()
        restored = BaseMessage.from_dict(data)

        assert restored.metadata == msg.metadata
        assert restored.metadata["nested"]["data"] == [1, 2, 3]

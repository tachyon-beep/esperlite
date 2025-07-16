"""
Tests for message bus contracts validation and performance.
"""

import time
from datetime import datetime

from esper.contracts.messages import BlueprintSubmitted
from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames


class TestTopicNames:
    """Test cases for TopicNames enum."""

    def test_topic_names_values(self):
        """Test TopicNames enum values."""
        assert TopicNames.TELEMETRY_SEED_HEALTH.value == "telemetry.seed.health"
        assert TopicNames.CONTROL_KASMINA_COMMANDS.value == "control.kasmina.commands"
        assert (
            TopicNames.COMPILATION_BLUEPRINT_SUBMITTED.value
            == "compilation.blueprint.submitted"
        )
        assert TopicNames.COMPILATION_KERNEL_READY.value == "compilation.kernel.ready"
        assert (
            TopicNames.VALIDATION_KERNEL_CHARACTERIZED.value
            == "validation.kernel.characterized"
        )
        assert TopicNames.SYSTEM_EVENTS_EPOCH.value == "system.events.epoch"
        assert TopicNames.INNOVATION_FIELD_REPORTS.value == "innovation.field_reports"

    def test_topic_names_hierarchy(self):
        """Test that topic names follow hierarchical structure."""
        for topic in TopicNames:
            assert isinstance(topic.value, str)
            assert "." in topic.value  # Should have hierarchy
            parts = topic.value.split(".")
            assert len(parts) >= 2  # At least category.subcategory

    def test_topic_names_completeness(self):
        """Test that all expected topics are present."""
        expected_topics = {
            "telemetry.seed.health",
            "control.kasmina.commands",
            "compilation.blueprint.submitted",
            "compilation.kernel.ready",
            "validation.kernel.characterized",
            "system.events.epoch",
            "innovation.field_reports",
        }

        actual_topics = {topic.value for topic in TopicNames}
        assert actual_topics == expected_topics


class TestOonaMessage:
    """Test cases for OonaMessage model."""

    def test_oona_message_creation(self):
        """Test OonaMessage creation with required fields."""
        message = OonaMessage(
            sender_id="test-sender",
            trace_id="trace-123",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload={"health_score": 0.95},
        )

        assert message.sender_id == "test-sender"
        assert message.trace_id == "trace-123"
        assert message.topic == TopicNames.TELEMETRY_SEED_HEALTH
        assert message.payload == {"health_score": 0.95}
        assert isinstance(message.event_id, str)
        assert len(message.event_id) == 36  # UUID4 format
        assert isinstance(message.timestamp, datetime)

    def test_oona_message_serialization(self):
        """Test OonaMessage JSON serialization/deserialization."""
        payload = {
            "layer_id": 5,
            "seed_id": 10,
            "health_score": 0.87,
            "metrics": {"accuracy": 0.92, "loss": 0.15},
        }

        original = OonaMessage(
            sender_id="tamiyo-controller",
            trace_id="request-456",
            topic=TopicNames.CONTROL_KASMINA_COMMANDS,
            payload=payload,
        )

        # Serialize and deserialize
        json_data = original.model_dump_json()
        reconstructed = OonaMessage.model_validate_json(json_data)

        # Verify all fields
        assert reconstructed.sender_id == original.sender_id
        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.topic == original.topic
        assert reconstructed.payload == original.payload
        assert reconstructed.event_id == original.event_id

    def test_oona_message_with_complex_payload(self):
        """Test OonaMessage with complex nested payload."""
        complex_payload = {
            "blueprint": {
                "id": "bp-123",
                "architecture": {
                    "layers": [
                        {"type": "Linear", "params": {"in": 512, "out": 256}},
                        {"type": "ReLU"},
                        {"type": "Dropout", "params": {"p": 0.2}},
                    ]
                },
                "metrics": {"validation_accuracy": 0.94, "training_loss": 0.08},
            },
            "metadata": {"created_by": "karn-architect", "priority": "high"},
        }

        message = OonaMessage(
            sender_id="tezzeret-compiler",
            trace_id="compilation-789",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload=complex_payload,
        )

        # Test serialization with complex payload
        json_str = message.model_dump_json()
        reconstructed = OonaMessage.model_validate_json(json_str)

        assert reconstructed.payload == complex_payload
        assert reconstructed.payload["blueprint"]["id"] == "bp-123"
        assert len(reconstructed.payload["blueprint"]["architecture"]["layers"]) == 3

    def test_oona_message_topic_validation(self):
        """Test that topic field accepts TopicNames enum."""
        for topic in TopicNames:
            message = OonaMessage(
                sender_id="test-sender",
                trace_id="test-trace",
                topic=topic,
                payload={"test": "data"},
            )
            assert message.topic == topic


class TestBlueprintSubmitted:
    """Test cases for BlueprintSubmitted model."""

    def test_blueprint_submitted_creation(self):
        """Test BlueprintSubmitted creation."""
        submitted = BlueprintSubmitted(
            blueprint_id="bp-456", submitted_by="tamiyo-policy"
        )

        assert submitted.blueprint_id == "bp-456"
        assert submitted.submitted_by == "tamiyo-policy"

    def test_blueprint_submitted_serialization(self):
        """Test BlueprintSubmitted serialization."""
        original = BlueprintSubmitted(
            blueprint_id="bp-789", submitted_by="karn-architect"
        )

        json_data = original.model_dump_json()
        reconstructed = BlueprintSubmitted.model_validate_json(json_data)

        assert reconstructed.blueprint_id == original.blueprint_id
        assert reconstructed.submitted_by == original.submitted_by

    def test_blueprint_submitted_as_payload(self):
        """Test BlueprintSubmitted used as OonaMessage payload."""
        blueprint_event = BlueprintSubmitted(
            blueprint_id="bp-999", submitted_by="automated-system"
        )

        message = OonaMessage(
            sender_id="urza-library",
            trace_id="submission-321",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload=blueprint_event.model_dump(),
        )

        assert message.payload["blueprint_id"] == "bp-999"
        assert message.payload["submitted_by"] == "automated-system"


class TestMessagePerformance:
    """Performance tests for message contracts."""

    def test_oona_message_serialization_performance(self):
        """Test OonaMessage serialization performance."""
        message = OonaMessage(
            sender_id="performance-test-sender",
            trace_id="perf-trace-123",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload={
                "layer_id": 10,
                "seed_id": 25,
                "health_score": 0.88,
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.89,
                    "recall": 0.93,
                    "f1_score": 0.91,
                },
                "metadata": {"timestamp": "2025-07-16T10:30:00Z", "version": "1.0"},
            },
        )

        # Measure serialization performance
        start_time = time.perf_counter()
        for _ in range(1000):
            json_str = message.model_dump_json()
            OonaMessage.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should serialize/deserialize 1000 times in under 1 second
        assert elapsed < 1.0, (
            f"Message serialization took {elapsed:.3f}s, expected <1.0s"
        )

        # Calculate average per operation
        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0, (
            f"Average operation time {avg_time_ms:.3f}ms exceeds 1ms target"
        )

    def test_blueprint_submitted_performance(self):
        """Test BlueprintSubmitted serialization performance."""
        submitted = BlueprintSubmitted(
            blueprint_id="performance-test-blueprint", submitted_by="perf-tester"
        )

        start_time = time.perf_counter()
        for _ in range(5000):  # More iterations due to simpler structure
            json_str = submitted.model_dump_json()
            BlueprintSubmitted.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should handle 5000 operations in under 1 second
        assert elapsed < 1.0, (
            f"BlueprintSubmitted serialization took {elapsed:.3f}s, expected <1.0s"
        )

    def test_topic_enum_performance(self):
        """Test TopicNames enum access performance."""
        topics = list(TopicNames)

        start_time = time.perf_counter()
        for _ in range(10000):
            for topic in topics:
                # Test common enum operations
                _ = topic.value
                _ = str(topic)
                _ = topic == TopicNames.TELEMETRY_SEED_HEALTH

        elapsed = time.perf_counter() - start_time
        assert elapsed < 0.1, f"TopicNames access took {elapsed:.3f}s, expected <0.1s"

    def test_message_creation_performance(self):
        """Test OonaMessage creation performance."""
        start_time = time.perf_counter()

        for i in range(1000):
            message = OonaMessage(
                sender_id=f"sender-{i}",
                trace_id=f"trace-{i}",
                topic=TopicNames.SYSTEM_EVENTS_EPOCH,
                payload={"epoch": i, "status": "running"},
            )
            # Ensure message is created properly
            assert message.sender_id == f"sender-{i}"

        elapsed = time.perf_counter() - start_time
        assert elapsed < 1.0, f"Message creation took {elapsed:.3f}s, expected <1.0s"

    def test_large_payload_performance(self):
        """Test performance with large message payloads."""
        # Create a large payload (simulating batch data)
        large_payload = {
            "batch_data": [
                {
                    "layer_id": i,
                    "seeds": [
                        {
                            "seed_id": j,
                            "health_score": 0.85 + (j * 0.01),
                            "metrics": {"accuracy": 0.9, "loss": 0.1},
                        }
                        for j in range(10)
                    ],
                }
                for i in range(50)  # 500 total seeds
            ],
            "metadata": {"batch_size": 500, "processing_time": 1.5},
        }

        message = OonaMessage(
            sender_id="batch-processor",
            trace_id="large-batch-001",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload=large_payload,
        )

        # Test serialization performance with large payload
        start_time = time.perf_counter()
        for _ in range(10):  # Fewer iterations due to size
            json_str = message.model_dump_json()
            OonaMessage.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should handle large payloads reasonably fast
        assert elapsed < 1.0, (
            f"Large payload processing took {elapsed:.3f}s, expected <1.0s"
        )

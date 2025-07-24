"""
Tests for OonaClient - Refactored Version.

This version focuses on testing real behavior and integration points
rather than mocking Redis implementation details.
"""

import json
import os
import time
import pytest
from unittest.mock import Mock, patch
import redis.exceptions

from esper.contracts.messages import OonaMessage, TopicNames
from esper.contracts.operational import HealthSignal
from esper.services.oona_client import OonaClient


@pytest.mark.real_components
class TestOonaClientMessaging:
    """Test OonaClient message publishing and consumption behavior."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client with realistic behavior."""
        mock_client = Mock()
        # Successful ping by default
        mock_client.ping.return_value = True
        # Track published messages
        mock_client.published_messages = []

        def mock_xadd(stream_name, message_body):
            mock_client.published_messages.append((stream_name, message_body))
            return f"msg-{len(mock_client.published_messages)}"

        mock_client.xadd.side_effect = mock_xadd
        return mock_client

    @pytest.fixture
    def client_with_mock_redis(self, mock_redis):
        """Create OonaClient with mocked Redis."""
        with patch(
            "esper.services.oona_client.redis.from_url", return_value=mock_redis
        ):
            return OonaClient()

    def test_message_publishing_preserves_structure(
        self, client_with_mock_redis, mock_redis
    ):
        """Test that published messages maintain their structure."""
        # Create a realistic message
        health_signal = HealthSignal(
            layer_id=1,
            seed_id=0,
            chunk_id=0,
            epoch=100,
            activation_variance=0.15,
            dead_neuron_ratio=0.05,
            avg_correlation=0.85,
            health_score=0.92,
            execution_latency=8.5,
            error_count=0,
            active_seeds=4,
            total_seeds=4,
            timestamp=time.time(),
        )

        message = OonaMessage(
            sender_id="kasmina.layer_1",
            trace_id=f"health-{int(time.time())}",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload=health_signal.model_dump(),
        )

        # Publish message
        client_with_mock_redis.publish(message)

        # Verify message was published correctly
        assert len(mock_redis.published_messages) == 1
        stream_name, published_body = mock_redis.published_messages[0]

        assert stream_name == TopicNames.TELEMETRY_SEED_HEALTH.value
        assert published_body["sender_id"] == "kasmina.layer_1"
        assert "event_id" in published_body
        assert "payload" in published_body

        # Verify payload was JSON serialized
        payload_data = json.loads(published_body["payload"])
        assert payload_data["layer_id"] == 1
        assert payload_data["health_score"] == 0.92

    def test_message_consumption_with_parsing(self, client_with_mock_redis, mock_redis):
        """Test consuming and parsing messages correctly."""
        # Mock Redis response with realistic data
        mock_response = [
            (
                TopicNames.CONTROL_KASMINA_COMMANDS.value,
                [
                    (
                        "1234567890-0",
                        {
                            "event_id": "evt-123",
                            "sender_id": "tamiyo",
                            "trace_id": "trace-456",
                            "topic": TopicNames.CONTROL_KASMINA_COMMANDS.value,
                            "payload": json.dumps(
                                {
                                    "command": "add_seed",
                                    "layer_name": "layer_1",
                                    "confidence": 0.85,
                                    "urgency": 0.7,
                                }
                            ),
                        },
                    ),
                    (
                        "1234567890-1",
                        {
                            "event_id": "evt-124",
                            "sender_id": "tamiyo",
                            "trace_id": "trace-457",
                            "topic": TopicNames.CONTROL_KASMINA_COMMANDS.value,
                            "payload": json.dumps(
                                {
                                    "command": "remove_seed",
                                    "layer_name": "layer_2",
                                    "confidence": 0.65,
                                    "urgency": 0.3,
                                }
                            ),
                        },
                    ),
                ],
            )
        ]

        mock_redis.xreadgroup.return_value = mock_response

        # Consume messages
        messages = client_with_mock_redis.consume(
            streams=[TopicNames.CONTROL_KASMINA_COMMANDS.value],
            consumer_group="tolaria-group",
            consumer_name="tolaria-1",
        )

        # Verify message parsing
        assert len(messages) == 2

        # Check first message
        stream, msg_id, fields = messages[0]
        assert stream == TopicNames.CONTROL_KASMINA_COMMANDS.value
        assert msg_id == "1234567890-0"
        assert fields["sender_id"] == "tamiyo"
        assert fields["payload"]["command"] == "add_seed"
        assert fields["payload"]["confidence"] == 0.85

    def test_consumer_group_initialization(self, client_with_mock_redis, mock_redis):
        """Test that consumer groups are created if they don't exist."""
        # Mock group info to simulate non-existent group
        mock_redis.xinfo_groups.side_effect = redis.exceptions.ResponseError(
            "No such group"
        )
        mock_redis.xgroup_create.return_value = True
        mock_redis.xreadgroup.return_value = []  # No messages

        # Consume should create the group
        messages = client_with_mock_redis.consume(
            streams=["test-stream"],
            consumer_group="new-group",
            consumer_name="consumer-1",
        )

        # Should handle gracefully
        assert messages == []

        # Verify group creation was attempted
        mock_redis.xgroup_create.assert_called_with(
            "test-stream", "new-group", id="0", mkstream=True
        )

    def test_publish_error_handling(self, client_with_mock_redis, mock_redis):
        """Test error handling during message publishing."""
        # Simulate Redis error
        mock_redis.xadd.side_effect = redis.exceptions.ConnectionError(
            "Connection lost"
        )

        message = OonaMessage(
            sender_id="test",
            trace_id="test-trace",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload={"test": "data"},
        )

        # Should raise the error
        with pytest.raises(redis.exceptions.ConnectionError):
            client_with_mock_redis.publish(message)

    def test_acknowledge_message(self, client_with_mock_redis, mock_redis):
        """Test message acknowledgment."""
        client_with_mock_redis.acknowledge(
            stream="test-stream", consumer_group="test-group", message_id="12345-0"
        )

        mock_redis.xack.assert_called_once_with("test-stream", "test-group", "12345-0")


@pytest.mark.real_components
class TestOonaClientHealthAndConnection:
    """Test OonaClient health checks and connection handling."""

    def test_connection_failure_handling(self):
        """Test handling of Redis connection failures."""
        with patch("esper.services.oona_client.redis.from_url") as mock_redis_from_url:
            mock_client = Mock()
            mock_client.ping.side_effect = redis.exceptions.ConnectionError(
                "Cannot connect"
            )
            mock_redis_from_url.return_value = mock_client

            # Should raise connection error
            with pytest.raises(redis.exceptions.ConnectionError):
                OonaClient()

    def test_health_check_with_healthy_connection(self):
        """Test health check with working connection."""
        with patch("esper.services.oona_client.redis.from_url") as mock_redis_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis_from_url.return_value = mock_client

            client = OonaClient()
            assert client.health_check() is True
            # Should ping twice - once on init, once on health check
            assert mock_client.ping.call_count == 2

    def test_health_check_with_failed_connection(self):
        """Test health check with failed connection."""
        with patch("esper.services.oona_client.redis.from_url") as mock_redis_from_url:
            mock_client = Mock()
            # Initial ping succeeds
            mock_client.ping.side_effect = [
                True,
                redis.exceptions.ConnectionError("Lost connection"),
            ]
            mock_redis_from_url.return_value = mock_client

            client = OonaClient()
            assert client.health_check() is False

    def test_redis_url_from_environment(self):
        """Test that Redis URL is read from environment."""
        custom_url = "redis://custom-host:6380/1"

        with patch.dict(os.environ, {"REDIS_URL": custom_url}):
            with patch(
                "esper.services.oona_client.redis.from_url"
            ) as mock_redis_from_url:
                mock_client = Mock()
                mock_client.ping.return_value = True
                mock_redis_from_url.return_value = mock_client

                OonaClient()
                mock_redis_from_url.assert_called_with(
                    custom_url, decode_responses=True
                )


@pytest.mark.integration
class TestOonaClientIntegration:
    """Integration tests with mock Redis behavior."""

    @pytest.fixture
    def integration_client(self):
        """Create client with mock Redis that simulates real behavior."""
        with patch("esper.services.oona_client.redis.from_url") as mock_redis_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True

            # Simulate message storage
            mock_client.messages = {}
            mock_client.consumer_groups = {}

            def mock_xadd(stream, body):
                if stream not in mock_client.messages:
                    mock_client.messages[stream] = []
                msg_id = (
                    f"{int(time.time() * 1000)}-{len(mock_client.messages[stream])}"
                )
                mock_client.messages[stream].append((msg_id, body))
                return msg_id

            def mock_xreadgroup(group, consumer, streams, **kwargs):
                result = []
                for stream, last_id in streams.items():
                    if stream in mock_client.messages:
                        # Return all messages (simplified)
                        messages = mock_client.messages[stream]
                        if messages:
                            # Parse the stored messages to match expected format
                            parsed_messages = []
                            for msg_id, body in messages:
                                # Parse JSON strings back to objects
                                parsed_body = {}
                                for key, value in body.items():
                                    try:
                                        parsed_body[key] = (
                                            json.loads(value)
                                            if isinstance(value, str)
                                            and value.startswith("{")
                                            else value
                                        )
                                    except:
                                        parsed_body[key] = value
                                parsed_messages.append((msg_id, parsed_body))
                            result.append((stream, parsed_messages))
                return result

            mock_client.xadd = mock_xadd
            mock_client.xreadgroup = mock_xreadgroup
            mock_client.xinfo_groups = Mock(return_value=[])
            mock_client.xgroup_create = Mock(return_value=True)
            mock_client.xack = Mock(return_value=1)

            mock_redis_from_url.return_value = mock_client
            return OonaClient(), mock_client

    def test_publish_consume_cycle(self, integration_client):
        """Test complete publish-consume cycle."""
        client, mock_redis = integration_client

        # Publish multiple messages
        messages_to_publish = []
        for i in range(3):
            msg = OonaMessage(
                sender_id=f"service-{i}",
                trace_id=f"trace-{i}",
                topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
                payload={"index": i, "data": f"test-{i}"},
            )
            messages_to_publish.append(msg)
            client.publish(msg)

        # Consume messages
        consumed = client.consume(
            streams=[TopicNames.COMPILATION_BLUEPRINT_SUBMITTED.value],
            consumer_group="test-group",
            consumer_name="test-consumer",
            count=10,
        )

        # Verify all messages were consumed
        assert len(consumed) == 3

        # Verify message content
        for i, (stream, msg_id, fields) in enumerate(consumed):
            assert stream == TopicNames.COMPILATION_BLUEPRINT_SUBMITTED.value
            assert fields["sender_id"] == f"service-{i}"
            payload = fields["payload"]  # Already parsed
            assert payload["index"] == i

    def test_multiple_consumer_groups(self, integration_client):
        """Test multiple consumer groups reading same stream."""
        client, mock_redis = integration_client

        # Publish a message
        message = OonaMessage(
            sender_id="producer",
            trace_id="test-trace",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload={"health": "good"},
        )
        client.publish(message)

        # Multiple consumer groups should be able to read the same message
        groups = ["analytics-group", "monitoring-group", "logging-group"]

        for group in groups:
            messages = client.consume(
                streams=[TopicNames.TELEMETRY_SEED_HEALTH.value],
                consumer_group=group,
                consumer_name="consumer-1",
            )
            assert len(messages) == 1
            assert messages[0][2]["sender_id"] == "producer"


@pytest.mark.performance
class TestOonaClientPerformance:
    """Performance tests for OonaClient."""

    def test_message_serialization_performance(self):
        """Test performance of message serialization."""
        import time

        # Create a complex payload
        complex_payload = {
            "layer_stats": {
                f"layer_{i}": {
                    "health": 0.9,
                    "active_seeds": list(range(10)),
                    "metrics": {"loss": 0.1, "accuracy": 0.9},
                }
                for i in range(10)
            },
            "timestamp": time.time(),
            "metadata": {"version": "1.0", "source": "test"},
        }

        message = OonaMessage(
            sender_id="perf-test",
            trace_id="perf-trace",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload=complex_payload,
        )

        # Time serialization
        iterations = 1000
        start = time.perf_counter()

        for _ in range(iterations):
            body = message.model_dump(mode="json")
            # Simulate flattening for Redis
            flattened = {}
            for key, value in body.items():
                if isinstance(value, (dict, list)):
                    flattened[key] = json.dumps(value)
                else:
                    flattened[key] = str(value)

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        # Should be fast (less than 1ms per message)
        assert avg_time < 0.001, (
            f"Serialization too slow: {avg_time * 1000:.2f}ms per message"
        )

    def test_batch_publishing_simulation(self):
        """Test performance of batch publishing."""
        with patch("esper.services.oona_client.redis.from_url") as mock_redis_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True

            # Track publish times
            publish_times = []

            def mock_xadd(stream, body):
                publish_times.append(time.perf_counter())
                return f"msg-{len(publish_times)}"

            mock_client.xadd = mock_xadd
            mock_redis_from_url.return_value = mock_client

            client = OonaClient()

            # Publish many messages
            start = time.perf_counter()
            for i in range(100):
                msg = OonaMessage(
                    sender_id="batch-test",
                    trace_id=f"trace-{i}",
                    topic=TopicNames.COMPILATION_KERNEL_READY,
                    payload={"index": i},
                )
                client.publish(msg)

            total_time = time.perf_counter() - start

            # Should handle 100 messages quickly
            assert total_time < 0.1  # Less than 100ms for 100 messages
            assert len(publish_times) == 100

"""
Tests for OonaClient message bus functionality.
"""

import warnings
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Suppress Pydantic deprecation warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames
from esper.services.oona_client import OonaClient


class TestOonaClient:
    """Test cases for OonaClient."""

    @patch("esper.services.oona_client.redis.from_url")
    def test_init_success(self, mock_redis_from_url):
        """Test successful initialization of OonaClient."""
        mock_redis_client = Mock()
        mock_redis_from_url.return_value = mock_redis_client

        client = OonaClient()

        assert client.redis_client == mock_redis_client
        mock_redis_client.ping.assert_called_once()
        mock_redis_from_url.assert_called_once_with(
            "redis://localhost:6379/0", decode_responses=True
        )

    @patch("esper.services.oona_client.redis.from_url")
    def test_init_connection_error(self, mock_redis_from_url):
        """Test initialization with connection error."""
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        mock_redis_from_url.return_value = mock_redis_client

        with pytest.raises(Exception, match="Connection failed"):
            OonaClient()

    @patch("esper.services.oona_client.redis.from_url")
    def test_publish_success(self, mock_redis_from_url):
        """Test successful message publishing."""
        mock_redis_client = Mock()
        mock_redis_from_url.return_value = mock_redis_client

        client = OonaClient()

        # Create test message
        message = OonaMessage(
            sender_id="test-service",
            trace_id="test-trace-123",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload={"test": "data"},
        )

        # Publish message
        client.publish(message)

        # Verify Redis call
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        stream_name, message_body = call_args[0]

        assert stream_name == TopicNames.COMPILATION_BLUEPRINT_SUBMITTED.value
        assert "event_id" in message_body
        assert "sender_id" in message_body
        assert "topic" in message_body
        assert "payload" in message_body

    @patch("esper.services.oona_client.redis.from_url")
    def test_publish_error(self, mock_redis_from_url):
        """Test publishing with Redis error."""
        mock_redis_client = Mock()
        mock_redis_client.xadd.side_effect = Exception("Redis error")
        mock_redis_from_url.return_value = mock_redis_client

        client = OonaClient()

        message = OonaMessage(
            sender_id="test-service",
            trace_id="test-trace-123",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload={"test": "data"},
        )

        with pytest.raises(Exception, match="Redis error"):
            client.publish(message)

    @patch("esper.services.oona_client.redis.from_url")
    def test_consume_success(self, mock_redis_from_url):
        """Test successful message consumption."""
        mock_redis_client = Mock()
        mock_redis_from_url.return_value = mock_redis_client

        # Mock Redis responses
        mock_redis_client.xreadgroup.return_value = [
            (
                "test-stream",
                [
                    (
                        "msg-1",
                        {
                            "event_id": "test-event-1",
                            "sender_id": "test-sender",
                            "topic": "test-topic",
                            "payload": '{"test": "data"}',
                        },
                    )
                ],
            )
        ]

        client = OonaClient()

        # Test consumption
        messages = client.consume(
            streams=["test-stream"],
            consumer_group="test-group",
            consumer_name="test-consumer",
        )

        assert len(messages) == 1
        stream_name, message_id, fields = messages[0]
        assert stream_name == "test-stream"
        assert message_id == "msg-1"
        assert fields["event_id"] == "test-event-1"
        assert fields["payload"] == {"test": "data"}

    @patch("esper.services.oona_client.redis.from_url")
    def test_acknowledge_success(self, mock_redis_from_url):
        """Test successful message acknowledgment."""
        mock_redis_client = Mock()
        mock_redis_from_url.return_value = mock_redis_client

        client = OonaClient()

        client.acknowledge("test-stream", "test-group", "msg-1")

        mock_redis_client.xack.assert_called_once_with(
            "test-stream", "test-group", "msg-1"
        )

    @patch("esper.services.oona_client.redis.from_url")
    def test_health_check_success(self, mock_redis_from_url):
        """Test successful health check."""
        mock_redis_client = Mock()
        mock_redis_from_url.return_value = mock_redis_client

        client = OonaClient()

        assert client.health_check() is True
        mock_redis_client.ping.assert_called()

    @patch("esper.services.oona_client.redis.from_url")
    def test_health_check_failure(self, mock_redis_from_url):
        """Test health check failure."""
        import redis

        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = redis.exceptions.ConnectionError(
            "Connection failed"
        )
        mock_redis_from_url.return_value = mock_redis_client

        # Skip the initial connection check for this test
        with patch.object(OonaClient, "__init__", lambda x: None):
            client = OonaClient()
            client.redis_client = mock_redis_client

            assert client.health_check() is False

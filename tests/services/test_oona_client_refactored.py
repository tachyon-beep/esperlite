"""
Tests for OonaClient using real Redis infrastructure.

This version uses real Redis as per project guidelines:
"Use real services when available, mock only external dependencies"
"""

import asyncio
import json
import os
import time

import pytest
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError

from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames
from esper.contracts.operational import HealthSignal
from esper.services.oona_client import OonaClient


@pytest.fixture
async def redis_client():
    """Create a real Redis client for testing."""
    client = aioredis.from_url("redis://localhost:6379/15")

    try:
        await client.ping()
        # Clear test database
        await client.flushdb()
        yield client
        # Cleanup
        await client.flushdb()
        await client.aclose()
    except RedisConnectionError:
        pytest.skip("Redis not available")


@pytest.fixture
def oona_client():
    """Create OonaClient with real Redis connection."""
    # Set test Redis URL
    os.environ["REDIS_URL"] = "redis://localhost:6379/15"
    client = OonaClient()
    yield client
    # Cleanup
    if hasattr(client, '_consumer_task') and client._consumer_task:
        client._consumer_task.cancel()


class TestOonaClientWithRealRedis:
    """Test OonaClient with real Redis infrastructure."""

    async def test_message_publishing_end_to_end(self, oona_client, redis_client):
        """Test publishing messages through real Redis."""
        # Create a health signal
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
        oona_client.publish(message)

        # Read directly from Redis to verify
        await asyncio.sleep(0.1)  # Allow time for message to be written

        # Check the stream exists
        streams = await redis_client.keys("*")
        assert TopicNames.TELEMETRY_SEED_HEALTH.value.encode() in streams

        # Read messages from stream
        messages = await redis_client.xread(
            {TopicNames.TELEMETRY_SEED_HEALTH.value: 0},
            count=10
        )

        assert len(messages) > 0
        _, stream_messages = messages[0]
        assert len(stream_messages) > 0

        # Verify message content
        _, msg_data = stream_messages[0]
        assert msg_data[b"sender_id"] == b"kasmina.layer_1"

        # Verify payload
        payload_data = json.loads(msg_data[b"payload"])
        assert payload_data["layer_id"] == 1
        assert payload_data["health_score"] == 0.92

    async def test_message_consumption_real_redis(self, oona_client, redis_client):
        """Test consuming messages from real Redis."""
        received_messages = []

        def callback(message: OonaMessage):
            received_messages.append(message)

        # Subscribe to topic
        oona_client.subscribe(TopicNames.CONTROL_KASMINA_COMMANDS, callback)

        # Publish a command message directly to Redis
        command_data = {
            "sender_id": "test_controller",
            "event_id": "test-123",
            "trace_id": "trace-456",
            "timestamp": str(time.time()),
            "topic": TopicNames.CONTROL_KASMINA_COMMANDS.value,
            "payload": json.dumps({
                "command": "update_seed",
                "layer_id": 1,
                "seed_id": 0,
                "parameters": {"alpha": 0.5}
            })
        }

        await redis_client.xadd(
            TopicNames.CONTROL_KASMINA_COMMANDS.value,
            command_data
        )

        # Allow time for consumption
        await asyncio.sleep(0.2)

        # Verify message was received
        assert len(received_messages) > 0
        msg = received_messages[0]
        assert msg.sender_id == "test_controller"
        assert msg.topic == TopicNames.CONTROL_KASMINA_COMMANDS
        assert msg.payload["command"] == "update_seed"

    async def test_connection_resilience(self, redis_client):
        """Test client handles connection failures gracefully."""
        # Create client with bad URL
        os.environ["REDIS_URL"] = "redis://nonexistent:6379"
        client = OonaClient()

        # Should not crash on publish with bad connection
        message = OonaMessage(
            sender_id="test",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload={"test": "data"}
        )

        # This should handle the error gracefully
        client.publish(message)  # Should log error but not crash

    async def test_multiple_subscribers(self, oona_client, redis_client):
        """Test multiple subscribers receive messages correctly."""
        received_1 = []
        received_2 = []

        def callback1(msg):
            received_1.append(msg)

        def callback2(msg):
            received_2.append(msg)

        # Subscribe multiple callbacks
        oona_client.subscribe(TopicNames.TELEMETRY_SEED_HEALTH, callback1)
        oona_client.subscribe(TopicNames.TELEMETRY_SEED_HEALTH, callback2)

        # Publish message
        message = OonaMessage(
            sender_id="test",
            topic=TopicNames.TELEMETRY_SEED_HEALTH,
            payload={"test": "data"}
        )
        oona_client.publish(message)

        # Allow processing
        await asyncio.sleep(0.2)

        # Both should receive the message
        assert len(received_1) > 0
        assert len(received_2) > 0
        assert received_1[0].payload == received_2[0].payload

    async def test_stream_trimming(self, oona_client, redis_client):
        """Test that streams are trimmed to prevent unbounded growth."""
        # Publish many messages
        for i in range(150):
            message = OonaMessage(
                sender_id="test",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload={"index": i}
            )
            oona_client.publish(message)

        await asyncio.sleep(0.5)

        # Check stream length
        info = await redis_client.xinfo_stream(TopicNames.TELEMETRY_SEED_HEALTH.value)
        # Should be trimmed to around 100 (maxlen in OonaClient)
        assert info[b"length"] <= 110  # Allow some margin

    async def test_error_recovery_in_consumer(self, oona_client, redis_client):
        """Test consumer handles errors gracefully."""
        error_count = 0

        def faulty_callback(msg):
            nonlocal error_count
            error_count += 1
            if error_count < 3:
                raise ValueError("Test error")
            # Success on third attempt

        oona_client.subscribe(TopicNames.CONTROL_KASMINA_COMMANDS, faulty_callback)

        # Publish message
        await redis_client.xadd(
            TopicNames.CONTROL_KASMINA_COMMANDS.value,
            {
                "sender_id": "test",
                "event_id": "123",
                "payload": json.dumps({"test": "data"}),
                "topic": TopicNames.CONTROL_KASMINA_COMMANDS.value
            }
        )

        await asyncio.sleep(0.3)

        # Consumer should continue running despite errors
        assert error_count >= 1  # At least one error occurred

        # Publish another message to verify consumer still works
        await redis_client.xadd(
            TopicNames.CONTROL_KASMINA_COMMANDS.value,
            {
                "sender_id": "test2",
                "event_id": "456",
                "payload": json.dumps({"test": "data2"}),
                "topic": TopicNames.CONTROL_KASMINA_COMMANDS.value
            }
        )

        await asyncio.sleep(0.2)
        # Error count should increase, showing consumer is still running
        assert error_count > 1

"""Unit tests for message bus clients."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import time

from src.esper.morphogenetic_v2.message_bus.clients import (
    MessageBusConfig, RedisStreamClient, MockMessageBusClient,
    MessageBusClient
)
from src.esper.morphogenetic_v2.message_bus.schemas import BaseMessage, LayerHealthReport

# Check if redis async is available
try:
    import redis.asyncio as redis
    HAS_REDIS_ASYNC = True
except ImportError:
    HAS_REDIS_ASYNC = False


class TestMessageBusConfig:
    """Test MessageBusConfig validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = MessageBusConfig(
            instance_id="test",
            redis_url="redis://localhost:6379",
            max_retries=3
        )
        config.validate()  # Should not raise
        
    def test_invalid_max_retries(self):
        """Test invalid max_retries."""
        config = MessageBusConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            config.validate()
            
    def test_invalid_retry_backoff(self):
        """Test invalid retry backoff base."""
        config = MessageBusConfig(retry_backoff_base=0.5)
        with pytest.raises(ValueError, match="retry_backoff_base must be greater than 1"):
            config.validate()
            
    def test_invalid_timeout(self):
        """Test invalid connection timeout."""
        config = MessageBusConfig(connection_timeout=-1)
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            config.validate()
            
    def test_invalid_message_size(self):
        """Test invalid max message size."""
        config = MessageBusConfig(max_message_size=0)
        with pytest.raises(ValueError, match="max_message_size must be positive"):
            config.validate()


class TestMockMessageBusClient:
    """Test MockMessageBusClient functionality."""
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        client = MockMessageBusClient()
        assert not client._connected
        
        await client.connect()
        assert client._connected
        assert await client.is_connected()
        
        await client.disconnect()
        assert not client._connected
        assert not await client.is_connected()
        
    @pytest.mark.asyncio
    async def test_publish_message(self):
        """Test message publishing."""
        client = MockMessageBusClient()
        await client.connect()
        
        msg = BaseMessage(source="test", message_id="123")
        await client.publish("test.topic", msg)
        
        assert len(client.get_messages("test.topic")) == 1
        assert client.get_messages("test.topic")[0] == msg
        
        stats = await client.get_stats()
        assert stats["messages_published"] == 1
        
    @pytest.mark.asyncio
    async def test_publish_not_connected(self):
        """Test publishing when not connected."""
        client = MockMessageBusClient()
        msg = BaseMessage(source="test")
        
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.publish("test.topic", msg)
            
    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self):
        """Test subscription and message delivery."""
        client = MockMessageBusClient()
        await client.connect()
        
        received = []
        
        async def handler(message):
            received.append(message)
            
        sub_id = await client.subscribe("test.*", handler)
        assert sub_id in client._subscriptions
        
        # Publish to matching topic
        msg1 = BaseMessage(source="test1")
        await client.publish("test.topic1", msg1)
        
        # Publish to non-matching topic
        msg2 = BaseMessage(source="test2")
        await client.publish("other.topic", msg2)
        
        # Allow async delivery
        await asyncio.sleep(0.01)
        
        assert len(received) == 1
        assert received[0] == msg1
        
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscription."""
        client = MockMessageBusClient()
        await client.connect()
        
        async def handler(message):
            pass
            
        sub_id = await client.subscribe("test.*", handler)
        assert sub_id in client._subscriptions
        
        await client.unsubscribe(sub_id)
        assert sub_id not in client._subscriptions
        
    @pytest.mark.asyncio
    async def test_wildcard_matching(self):
        """Test topic wildcard matching."""
        client = MockMessageBusClient()
        
        # Test exact match
        assert client._topic_matches_pattern("test.topic", "test.topic")
        assert not client._topic_matches_pattern("test.topic", "test.other")
        
        # Test wildcard
        assert client._topic_matches_pattern("test.topic", "test.*")
        assert client._topic_matches_pattern("test.sub.topic", "test.*.*")
        assert not client._topic_matches_pattern("other.topic", "test.*")
        
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics collection."""
        client = MockMessageBusClient()
        await client.connect()
        
        # Initial stats
        stats = await client.get_stats()
        assert stats["messages_published"] == 0
        assert stats["messages_received"] == 0
        assert stats["active_subscriptions"] == 0
        
        # Add activity
        await client.subscribe("test.*", lambda m: None)
        await client.publish("test.topic", BaseMessage())
        
        stats = await client.get_stats()
        assert stats["messages_published"] == 1
        assert stats["active_subscriptions"] == 1
        assert stats["total_topics"] == 1
        
    def test_test_helpers(self):
        """Test helper methods for testing."""
        client = MockMessageBusClient()
        
        # Test message storage
        msg1 = BaseMessage(source="test1")
        msg2 = BaseMessage(source="test2")
        
        client._messages["topic1"].append(msg1)
        client._messages["topic2"].append(msg2)
        
        assert client.get_messages("topic1") == [msg1]
        assert len(client.get_messages("nonexistent")) == 0  # Nonexistent topic returns empty
        
        # Test clear
        client.clear_messages("topic1")
        assert len(client.get_messages("topic1")) == 0
        assert len(client.get_messages("topic2")) == 1
        
        client.clear_messages()
        assert len(client._messages) == 0


@pytest.mark.skipif(not HAS_REDIS_ASYNC, reason="redis[async] not installed")
class TestRedisStreamClient:
    """Test RedisStreamClient functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(HAS_REDIS_ASYNC, reason="Only test when redis[async] is missing")
    async def test_init_with_aioredis_missing(self):
        """Test initialization when aioredis is not available."""
        config = MessageBusConfig()
        with pytest.raises(ImportError, match="aioredis is required"):
            RedisStreamClient(config)
                
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        config = MessageBusConfig(max_retries=1)
        client = RedisStreamClient(config)
        
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        
        with patch('redis.asyncio.from_url', return_value=mock_redis) as mock_from_url:
            await client.connect()
            
            assert client._connected
            assert client.redis == mock_redis
            mock_from_url.assert_called_once()
            mock_redis.ping.assert_awaited_once()
            
    @pytest.mark.asyncio
    async def test_connect_retry(self):
        """Test connection retry logic."""
        config = MessageBusConfig(max_retries=3, retry_backoff_base=2.0)
        client = RedisStreamClient(config)
        
        call_count = 0
        
        def failing_from_url(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Test error")
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.aclose = AsyncMock()
            return mock_redis
            
        with patch('redis.asyncio.from_url', new=failing_from_url):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                await client.connect()
                
                assert call_count == 3
                assert client._connected
                # Check exponential backoff
                assert mock_sleep.call_count == 2
                
    @pytest.mark.asyncio
    async def test_connect_max_retries_exceeded(self):
        """Test connection failure after max retries."""
        config = MessageBusConfig(max_retries=1)
        client = RedisStreamClient(config)
        
        with patch('redis.asyncio.from_url', side_effect=ConnectionError("Test error")):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                with pytest.raises(ConnectionError, match="Failed to connect after 1 attempts"):
                    await client.connect()
                    
                assert not client._connected
                
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection and cleanup."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        # Setup mock connection
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        client.redis = mock_redis
        client._connected = True
        
        # Add mock tasks
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        client._background_tasks = [mock_task]
        client._subscriptions = {"sub1": mock_task}
        
        await client.disconnect()
        
        assert not client._connected
        assert client.redis is None
        mock_redis.close.assert_awaited_once()
        mock_task.cancel.assert_called()
        
    @pytest.mark.asyncio
    async def test_publish_connected(self):
        """Test publishing when connected."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock()
        client.redis = mock_redis
        client._connected = True
        
        msg = LayerHealthReport(
            layer_id="test",
            total_seeds=100,
            active_seeds=50
        )
        
        await client.publish("test.topic", msg)
        
        mock_redis.xadd.assert_awaited_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "test.topic"
        assert "data" in call_args[0][1]
        assert "type" in call_args[0][1]
        assert client._stats["messages_published"] == 1
        
    @pytest.mark.asyncio
    async def test_publish_disconnected_with_buffer(self):
        """Test publishing to local buffer when disconnected."""
        config = MessageBusConfig(enable_local_buffer=True)
        client = RedisStreamClient(config)
        client._connected = False
        
        msg = BaseMessage(source="test")
        await client.publish("test.topic", msg)
        
        # Should be in local buffer
        assert client._local_buffer.qsize() == 1
        topic, buffered_msg = client._local_buffer.get_nowait()
        assert topic == "test.topic"
        assert buffered_msg == msg
        
    @pytest.mark.asyncio
    async def test_publish_message_too_large(self):
        """Test publishing message that exceeds size limit."""
        config = MessageBusConfig(max_message_size=100)
        client = RedisStreamClient(config)
        client._connected = True
        client.redis = AsyncMock()
        
        # Create large message
        msg = BaseMessage(
            source="test",
            metadata={"data": "x" * 1000}
        )
        
        with pytest.raises(ValueError, match="Message too large"):
            await client.publish("test.topic", msg)
            
    @pytest.mark.asyncio
    async def test_subscribe(self):
        """Test subscription creation."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        async def handler(msg):
            pass
            
        sub_id = await client.subscribe("test.*", handler)
        
        assert sub_id in client._handlers
        assert sub_id in client._subscriptions
        assert client._handlers[sub_id] == handler
        
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscription."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        # Create subscription
        async def handler(msg):
            pass
            
        sub_id = await client.subscribe("test.*", handler)
        
        # Create mock task
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        client._subscriptions[sub_id] = mock_task
        
        await client.unsubscribe(sub_id)
        
        assert sub_id not in client._handlers
        assert sub_id not in client._subscriptions
        mock_task.cancel.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_is_connected(self):
        """Test connection status check."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        # Not connected
        assert not await client.is_connected()
        
        # Connected but no redis
        client._connected = True
        assert not await client.is_connected()
        
        # Connected with redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        client.redis = mock_redis
        assert await client.is_connected()
        mock_redis.ping.assert_awaited_once()
        
        # Redis ping fails
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection lost"))
        assert not await client.is_connected()
        
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics retrieval."""
        config = MessageBusConfig(enable_local_buffer=True)
        client = RedisStreamClient(config)
        client._stats = {
            "messages_published": 10,
            "messages_received": 5,
            "connected_since": time.time() - 60
        }
        client._subscriptions = {"sub1": None, "sub2": None}
        
        stats = await client.get_stats()
        
        assert stats["messages_published"] == 10
        assert stats["messages_received"] == 5
        assert stats["active_subscriptions"] == 2
        assert "uptime_seconds" in stats
        assert stats["uptime_seconds"] >= 60
        assert "local_buffer_size" in stats
        
    def test_serialize_message(self):
        """Test message serialization."""
        config = MessageBusConfig(enable_compression=False)
        client = RedisStreamClient(config)
        
        msg = BaseMessage(source="test", message_id="123")
        serialized = client._serialize_message(msg)
        
        assert isinstance(serialized, bytes)
        data = json.loads(serialized.decode())
        assert data["source"] == "test"
        assert data["message_id"] == "123"
        
    def test_serialize_message_with_compression(self):
        """Test message serialization with compression."""
        config = MessageBusConfig(
            enable_compression=True,
            compression_threshold=10
        )
        client = RedisStreamClient(config)
        
        # Small message - no compression
        small_msg = BaseMessage(source="test")
        serialized = client._serialize_message(small_msg)
        assert not serialized.startswith(b'x\x9c')  # Not compressed
        
        # Large message - compressed
        large_msg = BaseMessage(
            source="test",
            metadata={"data": "x" * 1000}
        )
        serialized = client._serialize_message(large_msg)
        assert serialized.startswith(b'x\x9c')  # zlib magic number
        
    def test_deserialize_message(self):
        """Test message deserialization."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        # Uncompressed
        data = json.dumps({
            "message_type": "BaseMessage",
            "source": "test",
            "message_id": "123"
        }).encode()
        
        msg = client._deserialize_message(data)
        assert isinstance(msg, BaseMessage)
        assert msg.source == "test"
        assert msg.message_id == "123"
        
    def test_deserialize_compressed_message(self):
        """Test compressed message deserialization."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        import zlib
        data = json.dumps({
            "message_type": "BaseMessage",
            "source": "test"
        }).encode()
        compressed = zlib.compress(data)
        
        msg = client._deserialize_message(compressed)
        assert isinstance(msg, BaseMessage)
        assert msg.source == "test"
        
    @pytest.mark.asyncio
    async def test_find_matching_streams_exact(self):
        """Test finding streams with exact match."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        mock_redis = AsyncMock()
        client.redis = mock_redis
        
        streams = await client._find_matching_streams("test.topic")
        
        assert streams == ["test:topic"]
        mock_redis.scan.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_find_matching_streams_wildcard(self):
        """Test finding streams with wildcard."""
        config = MessageBusConfig()
        client = RedisStreamClient(config)
        
        mock_redis = AsyncMock()
        # Mock scan to return stream keys
        mock_redis.scan = AsyncMock(side_effect=[
            (100, [b"test:topic1", b"test:topic2"]),
            (0, [b"test:topic3"])
        ])
        client.redis = mock_redis
        
        streams = await client._find_matching_streams("test.*")
        
        assert len(streams) == 3
        assert "test:topic1" in streams
        assert "test:topic2" in streams
        assert "test:topic3" in streams
        assert mock_redis.scan.call_count == 2
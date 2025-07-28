"""Message bus client implementations.

This module provides abstract and concrete implementations of message bus clients
for the morphogenetic system, including Redis Streams and mock implementations.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from .schemas import BaseMessage
from .schemas import MessageFactory

logger = logging.getLogger(__name__)


@dataclass
class MessageBusConfig:
    """Configuration for message bus clients."""
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    redis_url: str = "redis://localhost:6379"
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # Exponential backoff base
    connection_timeout: float = 10.0
    read_timeout: float = 5.0
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    consumer_group_prefix: str = "morphogenetic"
    enable_local_buffer: bool = True
    local_buffer_size: int = 10000
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress messages larger than 1KB
    health_check_interval: float = 30.0

    def validate(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_base <= 1:
            raise ValueError("retry_backoff_base must be greater than 1")
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")


class MessageBusClient(ABC):
    """Abstract base class for message bus clients."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to message bus."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to message bus."""
        pass

    @abstractmethod
    async def publish(self, topic: str, message: BaseMessage) -> None:
        """Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        pass

    @abstractmethod
    async def subscribe(self, pattern: str, handler: Callable[[BaseMessage], None]) -> str:
        """Subscribe to topics matching a pattern.
        
        Args:
            pattern: Topic pattern to subscribe to (supports wildcards)
            handler: Async function to handle received messages
            
        Returns:
            Subscription ID for later unsubscribe
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a subscription.
        
        Args:
            subscription_id: ID returned from subscribe
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if client is connected."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        pass


class RedisStreamClient(MessageBusClient):
    """Production Redis Streams client with resilience features."""

    def __init__(self, config: MessageBusConfig):
        if redis is None:
            raise ImportError("redis[async] is required for RedisStreamClient")

        config.validate()
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.consumer_group = f"{config.consumer_group_prefix}_{config.instance_id}"
        self._connected = False
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._handlers: Dict[str, Callable] = {}
        self._stats = {
            "messages_published": 0,
            "messages_received": 0,
            "publish_errors": 0,
            "receive_errors": 0,
            "reconnections": 0,
            "last_error": None,
            "connected_since": None
        }

        # Local buffer for resilience
        if config.enable_local_buffer:
            self._local_buffer = asyncio.Queue(maxsize=config.local_buffer_size)
            self._retry_queue = asyncio.Queue(maxsize=config.local_buffer_size // 10)

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

    async def connect(self) -> None:
        """Establish connection with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retries:
            try:
                self.redis = redis.from_url(
                    self.config.redis_url,
                    decode_responses=False,  # We'll handle encoding/decoding
                    socket_connect_timeout=self.config.connection_timeout,
                    socket_keepalive=True,
                    retry_on_timeout=True,
                    health_check_interval=int(self.config.health_check_interval)
                )

                # Test connection
                await self.redis.ping()

                self._connected = True
                self._running = True
                self._stats["connected_since"] = time.time()

                # Start background tasks
                if self.config.enable_local_buffer:
                    self._background_tasks.append(
                        asyncio.create_task(self._flush_local_buffer())
                    )
                    self._background_tasks.append(
                        asyncio.create_task(self._process_retries())
                    )

                self._background_tasks.append(
                    asyncio.create_task(self._health_check_loop())
                )

                logger.info("Connected to Redis at %s", self.config.redis_url)
                return

            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.config.max_retries:
                    wait_time = self.config.retry_backoff_base ** retry_count
                    logger.warning("Connection failed, retrying in %ss: %s", wait_time, e)
                    await asyncio.sleep(wait_time)

        self._stats["last_error"] = str(last_error)
        raise ConnectionError(f"Failed to connect after {retry_count} attempts: {last_error}")

    async def disconnect(self) -> None:
        """Close connection and cleanup."""
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Cancel subscriptions
        for task in self._subscriptions.values():
            task.cancel()
        await asyncio.gather(*self._subscriptions.values(), return_exceptions=True)
        self._subscriptions.clear()
        self._handlers.clear()

        # Close Redis connection
        if self.redis:
            await self.redis.aclose()
            self.redis = None

        self._connected = False
        logger.info("Disconnected from Redis")

    async def publish(self, topic: str, message: BaseMessage) -> None:
        """Publish a message to a topic."""
        if not self._connected and self.config.enable_local_buffer:
            # Buffer locally if disconnected
            await self._local_buffer.put((topic, message))
            return

        try:
            # Serialize message
            data = self._serialize_message(message)

            # Check size
            if len(data) > self.config.max_message_size:
                raise ValueError(f"Message too large: {len(data)} > {self.config.max_message_size}")

            # Publish to Redis stream
            await self.redis.xadd(
                topic,
                {"data": data, "type": message.__class__.__name__},
                maxlen=100000  # Keep last 100k messages per stream
            )

            self._stats["messages_published"] += 1

        except Exception as e:
            self._stats["publish_errors"] += 1
            self._stats["last_error"] = str(e)

            if self.config.enable_local_buffer:
                # Buffer for retry
                await self._retry_queue.put((topic, message))

            raise

    async def subscribe(self, pattern: str, handler: Callable[[BaseMessage], None]) -> str:
        """Subscribe to topics matching a pattern."""
        subscription_id = str(uuid.uuid4())
        self._handlers[subscription_id] = handler

        # Start consumer task
        task = asyncio.create_task(
            self._consumer_loop(subscription_id, pattern)
        )
        self._subscriptions[subscription_id] = task

        logger.info("Subscribed to pattern '%s' with ID %s", pattern, subscription_id)
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a subscription."""
        if subscription_id in self._subscriptions:
            task = self._subscriptions[subscription_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            del self._subscriptions[subscription_id]
            del self._handlers[subscription_id]

            logger.info("Unsubscribed from %s", subscription_id)

    async def is_connected(self) -> bool:
        """Check if client is connected."""
        if not self._connected or not self.redis:
            return False

        try:
            await self.redis.ping()
            return True
        except Exception:
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self._stats.copy()

        if self._stats["connected_since"]:
            stats["uptime_seconds"] = time.time() - self._stats["connected_since"]

        if self.config.enable_local_buffer:
            stats["local_buffer_size"] = self._local_buffer.qsize()
            stats["retry_queue_size"] = self._retry_queue.qsize()

        stats["active_subscriptions"] = len(self._subscriptions)

        return stats

    def _serialize_message(self, message: BaseMessage) -> bytes:
        """Serialize message to bytes."""
        data = json.dumps(message.to_dict())

        # Optionally compress
        if self.config.enable_compression and len(data) > self.config.compression_threshold:
            import zlib
            return zlib.compress(data.encode())

        return data.encode()

    def _deserialize_message(self, data: bytes) -> BaseMessage:
        """Deserialize message from bytes."""
        # Check if compressed
        if data.startswith(b'x\x9c'):  # zlib magic number
            import zlib
            data = zlib.decompress(data)

        message_dict = json.loads(data.decode())
        return MessageFactory.create(message_dict)

    async def _consumer_loop(self, subscription_id: str, pattern: str):
        """Consumer loop for a subscription."""
        handler = self._handlers[subscription_id]

        # For pattern matching, we need to track streams
        streams = await self._find_matching_streams(pattern)

        # Initialize stream positions
        stream_positions = {stream: '0' for stream in streams}

        while self._running and subscription_id in self._subscriptions:
            try:
                # Check for new streams if we don't have any
                if not stream_positions or asyncio.get_event_loop().time() % 5 < 0.1:  # Every ~5 seconds or when empty
                    new_streams = await self._find_matching_streams(pattern)
                    for stream in new_streams:
                        if stream not in stream_positions:
                            stream_positions[stream] = '0'

                if not stream_positions:
                    await asyncio.sleep(1)
                    continue

                # Read from streams
                result = await self.redis.xread(
                    stream_positions,
                    block=1000,  # Block for 1 second
                    count=100    # Read up to 100 messages per stream
                )

                for stream_name, messages in result:
                    stream_name = stream_name.decode() if isinstance(stream_name, bytes) else stream_name

                    for message_id, fields in messages:
                        message_id = message_id.decode() if isinstance(message_id, bytes) else message_id

                        try:
                            # Deserialize and handle message
                            data = fields.get(b'data', fields.get('data'))
                            if data:
                                message = self._deserialize_message(data)
                                await handler(message)
                                self._stats["messages_received"] += 1

                        except Exception as e:
                            self._stats["receive_errors"] += 1
                            logger.error("Error handling message: %s", e)

                        # Update position
                        stream_positions[stream_name] = message_id

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Consumer loop error: %s", e)
                await asyncio.sleep(1)

    async def _find_matching_streams(self, pattern: str) -> List[str]:
        """Find streams matching a pattern."""
        # Use pattern as-is for Redis keys
        redis_pattern = pattern

        # Handle wildcards
        if '*' in redis_pattern:
            # Use SCAN to find matching keys
            matching = []
            cursor = 0

            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=redis_pattern,
                    count=100,
                    _type='stream'
                )

                matching.extend(keys)

                if cursor == 0:
                    break

            return [k.decode() if isinstance(k, bytes) else k for k in matching]
        else:
            # Exact match
            return [redis_pattern]

    async def _flush_local_buffer(self):
        """Background task to flush local buffer."""
        while self._running:
            try:
                if not self._connected:
                    await asyncio.sleep(1)
                    continue

                # Process buffered messages
                batch = []
                deadline = asyncio.get_event_loop().time() + 0.1  # 100ms batch window

                while asyncio.get_event_loop().time() < deadline:
                    try:
                        topic, message = await asyncio.wait_for(
                            self._local_buffer.get(),
                            timeout=0.01
                        )
                        batch.append((topic, message))

                        if len(batch) >= 100:  # Max batch size
                            break

                    except asyncio.TimeoutError:
                        break

                # Publish batch
                for topic, message in batch:
                    try:
                        await self.publish(topic, message)
                    except Exception:
                        # Re-queue for retry
                        await self._retry_queue.put((topic, message))

            except Exception as e:
                logger.error("Buffer flush error: %s", e)
                await asyncio.sleep(1)

    async def _process_retries(self):
        """Background task to process retry queue."""
        while self._running:
            try:
                topic, message = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=1.0
                )

                # Exponential backoff based on retry count
                retry_count = message.metadata.get('_retry_count', 0)
                if retry_count > 0:
                    await asyncio.sleep(min(60, 2 ** retry_count))

                # Attempt republish
                message.metadata['_retry_count'] = retry_count + 1

                try:
                    await self.publish(topic, message)
                except Exception:
                    if retry_count < self.config.max_retries:
                        await self._retry_queue.put((topic, message))
                    else:
                        logger.error("Dropping message after %s retries", retry_count)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Retry processor error: %s", e)
                await asyncio.sleep(1)

    async def _health_check_loop(self):
        """Background health check."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if not await self.is_connected():
                    logger.warning("Health check failed, attempting reconnection")
                    self._stats["reconnections"] += 1
                    await self.connect()

            except Exception as e:
                logger.error("Health check error: %s", e)


class MockMessageBusClient(MessageBusClient):
    """In-memory mock client for testing."""

    def __init__(self, config: Optional[MessageBusConfig] = None):
        self.config = config or MessageBusConfig()
        self._connected = False
        self._messages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._subscriptions: Dict[str, Callable] = {}
        self._subscription_patterns: Dict[str, str] = {}
        self._stats = {
            "messages_published": 0,
            "messages_received": 0,
            "messages_dropped": 0
        }
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.01)  # Simulate connection time
        self._connected = True
        self._running = True

        # Start message processor
        self._processor_task = asyncio.create_task(self._process_messages())

        logger.info("MockMessageBusClient connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info("MockMessageBusClient disconnected")

    async def publish(self, topic: str, message: BaseMessage) -> None:
        """Store message in memory."""
        if not self._connected:
            raise ConnectionError("Not connected")

        self._messages[topic].append(message)
        self._stats["messages_published"] += 1

        # Immediate delivery to subscribers (for testing)
        await self._deliver_message(topic, message)

    async def subscribe(self, pattern: str, handler: Callable[[BaseMessage], None]) -> str:
        """Register handler for pattern."""
        subscription_id = str(uuid.uuid4())
        self._subscriptions[subscription_id] = handler
        self._subscription_patterns[subscription_id] = pattern

        logger.info("Mock subscribed to '%s' with ID %s", pattern, subscription_id)
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            del self._subscription_patterns[subscription_id]
            logger.info("Mock unsubscribed from %s", subscription_id)

    async def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = self._stats.copy()
        stats["active_subscriptions"] = len(self._subscriptions)
        stats["total_topics"] = len(self._messages)
        stats["total_messages"] = sum(len(msgs) for msgs in self._messages.values())
        return stats

    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern."""
        # Simple wildcard matching
        if '*' in pattern:
            import re
            regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
            return bool(re.match(f"^{regex_pattern}$", topic))
        return topic == pattern

    async def _deliver_message(self, topic: str, message: BaseMessage):
        """Deliver message to matching subscribers."""
        for sub_id, handler in self._subscriptions.items():
            pattern = self._subscription_patterns[sub_id]

            if self._topic_matches_pattern(topic, pattern):
                try:
                    await handler(message)
                    self._stats["messages_received"] += 1
                except Exception as e:
                    logger.error("Mock handler error: %s", e)

    async def _process_messages(self):
        """Background processor for delayed operations."""
        while self._running:
            await asyncio.sleep(0.1)
            # Could implement delayed delivery, TTL, etc.

    # Test helper methods

    def get_messages(self, topic: str) -> List[BaseMessage]:
        """Get all messages for a topic (test helper)."""
        return list(self._messages.get(topic, []))

    def clear_messages(self, topic: Optional[str] = None):
        """Clear messages (test helper)."""
        if topic:
            self._messages[topic].clear()
        else:
            self._messages.clear()

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions (test helper)."""
        return len(self._subscriptions)

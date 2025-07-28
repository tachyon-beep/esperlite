"""
Message ordering and Dead Letter Queue (DLQ) support for the message bus.

This module provides:
- Message ordering guarantees using sequence numbers
- Partition-based ordering for scalability
- Dead Letter Queue for failed messages
- Message replay and reprocessing capabilities
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from .clients import MessageBusClient
from .schemas import BaseMessage
from .utils import MetricsCollector

logger = logging.getLogger(__name__)


class OrderingStrategy(Enum):
    """Message ordering strategies."""
    NONE = "none"              # No ordering guarantees
    PARTITION = "partition"    # Order within partition
    GLOBAL = "global"          # Global ordering (slower)
    CAUSAL = "causal"          # Causal ordering


@dataclass
class MessageSequence:
    """Tracks message sequence for ordering."""
    partition_id: str
    sequence_number: int
    timestamp: float
    dependencies: List[str] = field(default_factory=list)

    def __lt__(self, other):
        """Compare by sequence number for ordering."""
        return self.sequence_number < other.sequence_number


@dataclass
class DLQEntry:
    """Dead Letter Queue entry."""
    message: BaseMessage
    topic: str
    error: str
    failure_count: int
    first_failure: float
    last_failure: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderedMessageConfig:
    """Configuration for ordered message processing."""
    ordering_strategy: OrderingStrategy = OrderingStrategy.PARTITION
    max_out_of_order_window: int = 100
    max_pending_messages: int = 10000
    reorder_timeout_ms: int = 5000

    # DLQ configuration
    enable_dlq: bool = True
    max_retries: int = 3
    dlq_ttl_seconds: float = 86400  # 24 hours
    max_dlq_size: int = 10000

    # Partitioning
    partition_count: int = 16
    partition_key_extractor: Optional[Callable[[BaseMessage], str]] = None


class OrderedMessageProcessor:
    """
    Processes messages with ordering guarantees.
    
    Features:
    - Configurable ordering strategies
    - Out-of-order message buffering
    - Automatic reordering within windows
    - Performance optimized for different use cases
    """

    def __init__(self, config: Optional[OrderedMessageConfig] = None):
        self.config = config or OrderedMessageConfig()

        # Sequence tracking
        self.partition_sequences: Dict[str, int] = defaultdict(int)
        self.global_sequence = 0
        self.sequence_lock = asyncio.Lock()

        # Pending messages by partition
        self.pending_queues: Dict[str, deque] = defaultdict(deque)
        self.pending_messages: Dict[str, Dict[int, BaseMessage]] = defaultdict(dict)

        # Message handlers
        self.handlers: Dict[str, Callable] = {}

        # Metrics
        self.metrics = MetricsCollector()

        # Processing state
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self):
        """Start the ordered processor."""
        self._running = True
        logger.info("Ordered message processor started with strategy: %s",
                   self.config.ordering_strategy.value)

    async def stop(self):
        """Stop the ordered processor."""
        self._running = False

        # Cancel processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(),
                               return_exceptions=True)

        logger.info("Ordered message processor stopped")

    def register_handler(self, topic_pattern: str, handler: Callable):
        """Register a message handler for a topic pattern."""
        self.handlers[topic_pattern] = handler
        logger.debug("Registered handler for pattern: %s", topic_pattern)

    async def process_message(self, topic: str, message: BaseMessage) -> bool:
        """
        Process a message with ordering guarantees.
        
        Args:
            topic: Message topic
            message: Message to process
            
        Returns:
            True if processed, False if queued
        """
        # Extract ordering information
        partition_id = self._get_partition_id(message)
        sequence = await self._assign_sequence(message, partition_id)

        # Record metric
        await self.metrics.record("messages_received", 1.0,
                                 {"partition": partition_id})

        # Check ordering strategy
        if self.config.ordering_strategy == OrderingStrategy.NONE:
            # No ordering - process immediately
            return await self._process_immediately(topic, message)

        elif self.config.ordering_strategy == OrderingStrategy.PARTITION:
            # Partition ordering
            return await self._process_with_partition_order(
                topic, message, partition_id, sequence
            )

        elif self.config.ordering_strategy == OrderingStrategy.GLOBAL:
            # Global ordering
            return await self._process_with_global_order(
                topic, message, sequence
            )

        else:  # CAUSAL
            # Causal ordering
            return await self._process_with_causal_order(
                topic, message, sequence
            )

    def _get_partition_id(self, message: BaseMessage) -> str:
        """Get partition ID for message."""
        if self.config.partition_key_extractor:
            key = self.config.partition_key_extractor(message)
        else:
            # Default: use layer_id if available
            key = getattr(message, 'layer_id', message.source)

        # Hash to partition
        partition_num = hash(key) % self.config.partition_count
        return f"partition_{partition_num}"

    async def _assign_sequence(self, message: BaseMessage,
                              partition_id: str) -> MessageSequence:
        """Assign sequence number to message."""
        async with self.sequence_lock:
            if self.config.ordering_strategy == OrderingStrategy.GLOBAL:
                seq_num = self.global_sequence
                self.global_sequence += 1
            else:
                seq_num = self.partition_sequences[partition_id]
                self.partition_sequences[partition_id] += 1

        # Extract dependencies for causal ordering
        dependencies = []
        if hasattr(message, 'metadata') and 'depends_on' in message.metadata:
            dependencies = message.metadata['depends_on']

        return MessageSequence(
            partition_id=partition_id,
            sequence_number=seq_num,
            timestamp=time.time(),
            dependencies=dependencies
        )

    async def _process_immediately(self, topic: str, message: BaseMessage) -> bool:
        """Process message immediately without ordering."""
        handler = self._find_handler(topic)
        if handler:
            try:
                await handler(message)
                await self.metrics.record("messages_processed", 1.0)
                return True
            except Exception as e:
                logger.error("Handler error: %s", e)
                await self.metrics.record("processing_errors", 1.0)
                return False
        return False

    async def _process_with_partition_order(
        self,
        topic: str,
        message: BaseMessage,
        partition_id: str,
        sequence: MessageSequence
    ) -> bool:
        """Process with partition-level ordering."""
        # Get expected sequence for partition
        expected_seq = self._get_expected_sequence(partition_id)

        if sequence.sequence_number == expected_seq:
            # In order - process immediately
            success = await self._process_immediately(topic, message)

            if success:
                # Update expected sequence
                self._update_expected_sequence(partition_id, expected_seq + 1)

                # Process any pending messages
                await self._process_pending_messages(topic, partition_id)

            return success

        elif sequence.sequence_number > expected_seq:
            # Out of order - queue it
            return await self._queue_out_of_order(
                topic, message, partition_id, sequence
            )

        else:
            # Duplicate or old message
            logger.warning("Received old message with seq %d, expected >= %d",
                         sequence.sequence_number, expected_seq)
            await self.metrics.record("duplicate_messages", 1.0)
            return False

    async def _queue_out_of_order(
        self,
        topic: str,
        message: BaseMessage,
        partition_id: str,
        sequence: MessageSequence
    ) -> bool:
        """Queue out-of-order message."""
        # Check if within reorder window
        expected_seq = self._get_expected_sequence(partition_id)

        if sequence.sequence_number - expected_seq > self.config.max_out_of_order_window:
            logger.warning("Message too far out of order: seq=%d, expected=%d",
                         sequence.sequence_number, expected_seq)
            await self.metrics.record("messages_dropped", 1.0)
            return False

        # Queue the message
        self.pending_messages[partition_id][sequence.sequence_number] = (topic, message)

        # Check queue size
        if len(self.pending_messages[partition_id]) > self.config.max_pending_messages:
            logger.warning("Pending queue full for partition %s", partition_id)
            # Remove oldest
            min_seq = min(self.pending_messages[partition_id].keys())
            del self.pending_messages[partition_id][min_seq]

        await self.metrics.record("messages_queued", 1.0)

        # Start reorder timer if needed
        if partition_id not in self.processing_tasks:
            task = asyncio.create_task(
                self._reorder_timeout(partition_id)
            )
            self.processing_tasks[partition_id] = task

        return False  # Not processed yet

    async def _process_pending_messages(self, topic: str, partition_id: str):
        """Process any pending messages that are now in order."""
        expected_seq = self._get_expected_sequence(partition_id)
        pending = self.pending_messages[partition_id]

        while expected_seq in pending:
            topic, message = pending.pop(expected_seq)

            # Process the message
            success = await self._process_immediately(topic, message)

            if success:
                expected_seq += 1
                self._update_expected_sequence(partition_id, expected_seq)
            else:
                # Failed - stop processing
                break

    async def _reorder_timeout(self, partition_id: str):
        """Handle reorder timeout for a partition."""
        try:
            await asyncio.sleep(self.config.reorder_timeout_ms / 1000)

            # Force process pending messages
            logger.warning("Reorder timeout for partition %s", partition_id)

            # Get all pending messages
            pending = self.pending_messages[partition_id]
            if pending:
                # Process in order
                for seq_num in sorted(pending.keys()):
                    topic, message = pending.pop(seq_num)
                    await self._process_immediately(topic, message)

                # Update expected sequence
                self._update_expected_sequence(partition_id, seq_num + 1)

        finally:
            # Remove task
            if partition_id in self.processing_tasks:
                del self.processing_tasks[partition_id]

    def _get_expected_sequence(self, partition_id: str) -> int:
        """Get expected sequence number for partition."""
        # This would typically be persisted
        return self.partition_sequences.get(f"{partition_id}_expected", 0)

    def _update_expected_sequence(self, partition_id: str, seq_num: int):
        """Update expected sequence number."""
        self.partition_sequences[f"{partition_id}_expected"] = seq_num

    def _find_handler(self, topic: str) -> Optional[Callable]:
        """Find handler for topic."""
        for pattern, handler in self.handlers.items():
            if self._topic_matches(topic, pattern):
                return handler
        return None

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern."""
        # Simple wildcard matching
        if pattern.endswith("*"):
            return topic.startswith(pattern[:-1])
        return topic == pattern

    async def _process_with_global_order(
        self,
        topic: str,
        message: BaseMessage,
        sequence: MessageSequence
    ) -> bool:
        """Process with global ordering."""
        # Similar to partition ordering but with single global queue
        return await self._process_with_partition_order(
            topic, message, "global", sequence
        )

    async def _process_with_causal_order(
        self,
        topic: str,
        message: BaseMessage,
        sequence: MessageSequence
    ) -> bool:
        """Process with causal ordering."""
        # Check if dependencies are satisfied
        if sequence.dependencies:
            for dep_id in sequence.dependencies:
                if not self._is_processed(dep_id):
                    # Queue until dependencies satisfied
                    return await self._queue_out_of_order(
                        topic, message, sequence.partition_id, sequence
                    )

        # Dependencies satisfied - process
        return await self._process_immediately(topic, message)

    def _is_processed(self, message_id: str) -> bool:
        """Check if message has been processed."""
        # This would check a persistent store
        return True  # Simplified

    async def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = {
            "ordering_strategy": self.config.ordering_strategy.value,
            "partitions": len(self.partition_sequences),
            "pending_messages": sum(
                len(msgs) for msgs in self.pending_messages.values()
            ),
            "active_reorder_tasks": len(self.processing_tasks)
        }

        # Add metrics
        metrics = await self.metrics.get_all_metrics()
        stats["metrics"] = metrics

        return stats


class DeadLetterQueue:
    """
    Dead Letter Queue for failed messages.
    
    Features:
    - Configurable retry policies
    - TTL-based expiration
    - Manual replay capabilities
    - Metrics and monitoring
    """

    def __init__(
        self,
        message_bus: MessageBusClient,
        config: Optional[OrderedMessageConfig] = None
    ):
        self.message_bus = message_bus
        self.config = config or OrderedMessageConfig()

        # DLQ storage
        self.dlq_entries: Dict[str, DLQEntry] = {}
        self.dlq_order: deque = deque()

        # Retry tracking
        self.retry_counts: Dict[str, int] = defaultdict(int)

        # Metrics
        self.metrics = MetricsCollector()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the DLQ."""
        self._running = True

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Dead Letter Queue started")

    async def stop(self):
        """Stop the DLQ."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Dead Letter Queue stopped")

    async def add_failed_message(
        self,
        topic: str,
        message: BaseMessage,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a failed message to the DLQ.
        
        Args:
            topic: Original topic
            message: Failed message
            error: Error description
            metadata: Additional metadata
            
        Returns:
            True if added, False if rejected
        """
        async with self._lock:
            message_id = message.message_id

            # Check retry count
            self.retry_counts[message_id] += 1

            if self.retry_counts[message_id] > self.config.max_retries:
                logger.warning("Message %s exceeded max retries", message_id)
                await self.metrics.record("dlq_rejected", 1.0)
                return False

            # Check DLQ size
            if len(self.dlq_entries) >= self.config.max_dlq_size:
                # Remove oldest
                if self.dlq_order:
                    oldest_id = self.dlq_order.popleft()
                    del self.dlq_entries[oldest_id]

            # Create DLQ entry
            entry = DLQEntry(
                message=message,
                topic=topic,
                error=error,
                failure_count=self.retry_counts[message_id],
                first_failure=time.time(),
                last_failure=time.time(),
                metadata=metadata or {}
            )

            # Update existing entry
            if message_id in self.dlq_entries:
                existing = self.dlq_entries[message_id]
                entry.first_failure = existing.first_failure

            # Store entry
            self.dlq_entries[message_id] = entry
            self.dlq_order.append(message_id)

            await self.metrics.record("dlq_added", 1.0, {"topic": topic})

            logger.info("Added message %s to DLQ: %s", message_id, error)

            return True

    async def replay_message(self, message_id: str) -> bool:
        """
        Replay a message from the DLQ.
        
        Args:
            message_id: Message to replay
            
        Returns:
            True if replayed successfully
        """
        async with self._lock:
            if message_id not in self.dlq_entries:
                logger.warning("Message %s not found in DLQ", message_id)
                return False

            entry = self.dlq_entries[message_id]

        # Republish to original topic
        try:
            await self.message_bus.publish(entry.topic, entry.message)

            # Remove from DLQ
            async with self._lock:
                del self.dlq_entries[message_id]
                self.dlq_order.remove(message_id)

            await self.metrics.record("dlq_replayed", 1.0)

            logger.info("Replayed message %s to topic %s",
                       message_id, entry.topic)

            return True

        except Exception as e:
            logger.error("Failed to replay message %s: %s", message_id, e)
            await self.metrics.record("dlq_replay_failed", 1.0)
            return False

    async def replay_all(self, topic_filter: Optional[str] = None) -> Dict[str, int]:
        """
        Replay all messages matching filter.
        
        Args:
            topic_filter: Optional topic pattern filter
            
        Returns:
            Replay statistics
        """
        stats = {"attempted": 0, "successful": 0, "failed": 0}

        # Get matching messages
        message_ids = []
        async with self._lock:
            for msg_id, entry in self.dlq_entries.items():
                if not topic_filter or self._matches_filter(entry.topic, topic_filter):
                    message_ids.append(msg_id)

        # Replay messages
        for msg_id in message_ids:
            stats["attempted"] += 1

            if await self.replay_message(msg_id):
                stats["successful"] += 1
            else:
                stats["failed"] += 1

        logger.info("DLQ replay completed: %s", stats)

        return stats

    def _matches_filter(self, topic: str, filter_pattern: str) -> bool:
        """Check if topic matches filter pattern."""
        if filter_pattern.endswith("*"):
            return topic.startswith(filter_pattern[:-1])
        return topic == filter_pattern

    async def get_messages(
        self,
        limit: int = 100,
        topic_filter: Optional[str] = None
    ) -> List[DLQEntry]:
        """
        Get messages from DLQ.
        
        Args:
            limit: Maximum messages to return
            topic_filter: Optional topic filter
            
        Returns:
            List of DLQ entries
        """
        messages = []

        async with self._lock:
            for entry in self.dlq_entries.values():
                if not topic_filter or self._matches_filter(entry.topic, topic_filter):
                    messages.append(entry)

                    if len(messages) >= limit:
                        break

        return messages

    async def remove_message(self, message_id: str) -> bool:
        """Remove a message from DLQ."""
        async with self._lock:
            if message_id in self.dlq_entries:
                del self.dlq_entries[message_id]
                self.dlq_order.remove(message_id)

                await self.metrics.record("dlq_removed", 1.0)
                return True

        return False

    async def _cleanup_loop(self):
        """Background task to clean up expired messages."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                await self._cleanup_expired()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("DLQ cleanup error: %s", e)

    async def _cleanup_expired(self):
        """Remove expired messages from DLQ."""
        current_time = time.time()
        expired = []

        async with self._lock:
            for msg_id, entry in self.dlq_entries.items():
                age = current_time - entry.first_failure

                if age > self.config.dlq_ttl_seconds:
                    expired.append(msg_id)

            # Remove expired
            for msg_id in expired:
                del self.dlq_entries[msg_id]
                self.dlq_order.remove(msg_id)

        if expired:
            await self.metrics.record("dlq_expired", float(len(expired)))
            logger.info("Removed %d expired messages from DLQ", len(expired))

    async def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        async with self._lock:
            # Group by topic
            by_topic = defaultdict(int)
            for entry in self.dlq_entries.values():
                by_topic[entry.topic] += 1

            # Age distribution
            current_time = time.time()
            ages = []
            for entry in self.dlq_entries.values():
                ages.append(current_time - entry.first_failure)

        stats = {
            "total_messages": len(self.dlq_entries),
            "by_topic": dict(by_topic),
            "oldest_age_seconds": max(ages) if ages else 0,
            "average_age_seconds": sum(ages) / len(ages) if ages else 0
        }

        # Add metrics
        metrics = await self.metrics.get_all_metrics()
        stats["metrics"] = metrics

        return stats

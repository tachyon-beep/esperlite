"""
OonaClient: A reliable, typed client for interacting with Redis Streams message bus.

This client provides a simple, high-level interface for all other services to
publish and consume events on the Oona message bus.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import redis

from esper.contracts.messages import OonaMessage

logger = logging.getLogger(__name__)


class OonaClient:
    """A client for publishing and consuming events on the Oona message bus."""

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("OonaClient connected to Redis at %s", redis_url)
        except redis.exceptions.ConnectionError as e:
            logger.error(
                "FATAL: Could not connect to Redis at %s. Error: %s", redis_url, e
            )
            raise

    def publish(self, message: OonaMessage) -> None:
        """Publishes a message to a specific topic (stream)."""
        try:
            stream_name = message.topic.value
            message_body = message.model_dump(mode="json")

            # Convert nested dictionaries to JSON strings for Redis
            flattened_body = {}
            for key, value in message_body.items():
                if isinstance(value, (dict, list)):
                    flattened_body[key] = json.dumps(value)
                else:
                    flattened_body[key] = str(value)

            self.redis_client.xadd(stream_name, flattened_body)
            logger.info("Published event %s to topic %s", message.event_id, stream_name)
        except Exception as e:
            logger.error("Error publishing event %s to Oona: %s", message.event_id, e)
            raise

    def consume(
        self,
        streams: List[str],
        consumer_group: str,
        consumer_name: str,
        count: int = 1,
        block: Optional[int] = 1000,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Consumes messages from specified streams using consumer groups.

        Args:
            streams: List of stream names to consume from
            consumer_group: Consumer group name
            consumer_name: Consumer name within the group
            count: Maximum number of messages to consume
            block: Milliseconds to block for new messages (None for non-blocking)

        Returns:
            List of (stream_name, message_id, message_data) tuples
        """
        try:
            self._ensure_consumer_groups(streams, consumer_group)

            # Prepare stream dictionary for xreadgroup
            stream_dict = dict.fromkeys(streams, ">")

            # Read messages
            result = self.redis_client.xreadgroup(
                consumer_group, consumer_name, stream_dict, count=count, block=block
            )

            return self._parse_messages(result)
        except Exception as e:
            logger.error("Error consuming from streams %s: %s", streams, e)
            raise

    def _ensure_consumer_groups(self, streams: List[str], consumer_group: str) -> None:
        """Ensures consumer groups exist for all streams."""
        for stream in streams:
            try:
                self.redis_client.xgroup_create(
                    stream, consumer_group, id="0", mkstream=True
                )
                logger.debug(
                    "Created consumer group %s for stream %s", consumer_group, stream
                )
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    def _parse_messages(
        self, result: List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Parses Redis stream results into structured format."""
        messages = []
        for stream_name, stream_messages in result:
            for message_id, fields in stream_messages:
                parsed_fields = self._parse_fields(fields)
                messages.append((stream_name, message_id, parsed_fields))
        return messages

    def _parse_fields(self, fields: Dict[str, str]) -> Dict[str, Any]:
        """Parses Redis field values, converting JSON strings back to objects."""
        parsed_fields = {}
        for key, value in fields.items():
            try:
                parsed_fields[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_fields[key] = value
        return parsed_fields

    def acknowledge(self, stream: str, consumer_group: str, message_id: str) -> None:
        """Acknowledges a message has been processed."""
        try:
            self.redis_client.xack(stream, consumer_group, message_id)
            logger.debug("Acknowledged message %s in stream %s", message_id, stream)
        except Exception as e:
            logger.error("Error acknowledging message %s: %s", message_id, e)
            raise

    def health_check(self) -> bool:
        """Checks if the Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except redis.exceptions.ConnectionError:
            return False

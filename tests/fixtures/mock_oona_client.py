"""
Mock OonaClient for testing when Redis is not available.

This provides a mock implementation that simulates the OonaClient
behavior without requiring an actual Redis connection.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from esper.contracts.messages import OonaMessage


class MockOonaClient:
    """Mock OonaClient that simulates Redis behavior without actual connection."""

    def __init__(self):
        """Initialize mock client."""
        self.messages = {}  # Stream name -> list of messages
        self.health = True
        self.published_count = 0
        self.consumed_count = 0

    def publish(self, message: OonaMessage) -> None:
        """Mock publish - stores message in memory."""
        stream_name = message.topic.value
        if stream_name not in self.messages:
            self.messages[stream_name] = []

        # Store the message
        self.messages[stream_name].append({
            "id": f"mock-{self.published_count}",
            "message": message
        })
        self.published_count += 1

    def consume(
        self,
        streams: List[str],
        consumer_group: str,
        consumer_name: str,
        count: int = 1,
        block: Optional[int] = 1000,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Mock consume - returns stored messages."""
        result = []

        for stream in streams:
            if stream in self.messages and self.messages[stream]:
                # Return up to 'count' messages
                messages_to_return = self.messages[stream][:count]

                for msg_data in messages_to_return:
                    msg = msg_data["message"]
                    result.append((
                        stream,
                        msg_data["id"],
                        {
                            "event_id": msg.event_id,
                            "sender_id": msg.sender_id,
                            "trace_id": msg.trace_id,
                            "topic": msg.topic.value,
                            "payload": msg.payload,
                            "timestamp": msg.timestamp.isoformat()
                        }
                    ))
                self.consumed_count += len(messages_to_return)

        return result

    def acknowledge(self, stream: str, consumer_group: str, message_id: str) -> None:
        """Mock acknowledge - just track the call."""
        pass

    def health_check(self) -> bool:
        """Mock health check."""
        return self.health

    def close(self) -> None:
        """Mock close."""
        pass


def create_mock_oona_client():
    """Create a mock OonaClient for testing."""
    return MockOonaClient()


# Pytest fixture
import pytest


@pytest.fixture
def mock_oona_client():
    """Provide mock OonaClient for tests."""
    return create_mock_oona_client()

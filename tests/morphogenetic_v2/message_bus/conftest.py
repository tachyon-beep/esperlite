"""Test fixtures for message bus tests."""

import asyncio
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List

import pytest

from src.esper.morphogenetic_v2.message_bus.clients import MessageBusConfig
from src.esper.morphogenetic_v2.message_bus.clients import MockMessageBusClient
from src.esper.morphogenetic_v2.message_bus.schemas import BaseMessage


@pytest.fixture
async def mock_message_bus():
    """In-memory message bus for testing."""
    client = MockMessageBusClient()
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def redis_test_client():
    """Real Redis client for integration tests."""
    try:
        import redis.asyncio as aioredis

        # Use test database
        redis = aioredis.from_url("redis://localhost:6379/15")

        # Test connection
        await redis.ping()

        # Clear test database
        await redis.flushdb()

        yield redis

        # Cleanup
        await redis.flushdb()
        await redis.aclose()

    except Exception:
        pytest.skip("Redis not available")


@pytest.fixture
def message_bus_config():
    """Default message bus configuration for tests."""
    return MessageBusConfig(
        instance_id="test_instance",
        redis_url="redis://localhost:6379/15",
        max_retries=1,
        connection_timeout=1.0,
        enable_local_buffer=True,
        local_buffer_size=100
    )


@pytest.fixture
async def test_layer_registry():
    """Mock layer registry for testing."""
    class MockLayer:
        def __init__(self, layer_id: str, num_seeds: int = 100):
            self.layer_id = layer_id
            self.num_seeds = num_seeds
            self.seed_states = {}
            self.seed_blueprints = {}
            self.emergency_stopped = False

        def get_seed_state(self, seed_id: int):
            from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle
            return self.seed_states.get(seed_id, ExtendedLifecycle.DORMANT)

        def set_seed_state(self, seed_id: int, state):
            self.seed_states[seed_id] = state

        async def transition_seed(self, seed_id: int, target_state, params, force):
            # Simple transition logic
            self.seed_states[seed_id] = target_state
            return True

        async def update_seed_blueprint(self, seed_id: int, blueprint_id: str,
                                      strategy: str, config: Dict[str, Any]):
            self.seed_blueprints[seed_id] = blueprint_id
            return True

        async def get_seed_metrics(self, seed_id: int) -> Dict[str, float]:
            return {
                "loss": 0.1,
                "accuracy": 0.95,
                "compute_time_ms": 10.0
            }

        async def emergency_stop(self):
            self.emergency_stopped = True

    return {
        "test_layer_1": MockLayer("test_layer_1", 100),
        "test_layer_2": MockLayer("test_layer_2", 200)
    }


@pytest.fixture
def telemetry_config():
    """Default telemetry configuration for tests."""
    from src.esper.morphogenetic_v2.message_bus.publishers import TelemetryConfig

    return TelemetryConfig(
        batch_size=10,
        batch_window_ms=50,
        compression=None,  # Disable for tests
        enable_aggregation=False,
        anomaly_detection=False
    )


@pytest.fixture
async def message_collector():
    """Collects messages for verification in tests."""
    class MessageCollector:
        def __init__(self):
            self.messages: Dict[str, List[BaseMessage]] = defaultdict(list)
            self.all_messages: List[BaseMessage] = []

        async def collect(self, message: BaseMessage):
            """Handler that collects messages."""
            topic = getattr(message, '_topic', 'unknown')
            self.messages[topic].append(message)
            self.all_messages.append(message)

        def get_messages(self, topic: str = None) -> List[BaseMessage]:
            """Get collected messages."""
            if topic:
                return self.messages.get(topic, [])
            return self.all_messages

        def clear(self):
            """Clear collected messages."""
            self.messages.clear()
            self.all_messages.clear()

        async def wait_for_messages(self, count: int, timeout: float = 1.0) -> bool:
            """Wait for specific number of messages."""
            start = asyncio.get_event_loop().time()

            while len(self.all_messages) < count:
                if asyncio.get_event_loop().time() - start > timeout:
                    return False
                await asyncio.sleep(0.01)

            return True

    return MessageCollector()


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from src.esper.morphogenetic_v2.message_bus.schemas import AlertSeverity
    from src.esper.morphogenetic_v2.message_bus.schemas import AlertType
    from src.esper.morphogenetic_v2.message_bus.schemas import LayerHealthReport
    from src.esper.morphogenetic_v2.message_bus.schemas import (
        LifecycleTransitionCommand,
    )
    from src.esper.morphogenetic_v2.message_bus.schemas import PerformanceAlert
    from src.esper.morphogenetic_v2.message_bus.schemas import SeedMetricsSnapshot
    from src.esper.morphogenetic_v2.message_bus.schemas import StateTransitionEvent

    return {
        "health_report": LayerHealthReport(
            layer_id="test_layer",
            total_seeds=100,
            active_seeds=30,
            health_metrics={
                0: {"loss": 0.1, "accuracy": 0.95},
                1: {"loss": 0.2, "accuracy": 0.90}
            },
            performance_summary={"loss_mean": 0.15, "accuracy_mean": 0.925},
            telemetry_window=(0.0, 10.0)
        ),

        "seed_metrics": SeedMetricsSnapshot(
            layer_id="test_layer",
            seed_id=0,
            lifecycle_state="TRAINING",
            blueprint_id=42,
            metrics={"loss": 0.1, "accuracy": 0.95}
        ),

        "lifecycle_command": LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=0,
            target_state="GRAFTING",
            reason="Performance threshold reached"
        ),

        "state_event": StateTransitionEvent(
            layer_id="test_layer",
            seed_id=0,
            from_state="TRAINING",
            to_state="GRAFTING",
            reason="Automatic transition"
        ),

        "alert": PerformanceAlert(
            layer_id="test_layer",
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.WARNING,
            metric_name="loss",
            metric_value=0.5,
            threshold=0.2
        )
    }


@pytest.fixture
async def cleanup_tasks():
    """Cleanup async tasks after tests."""
    tasks = []

    yield tasks

    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for cancellation
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

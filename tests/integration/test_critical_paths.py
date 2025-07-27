"""
Critical path integration tests for Esperlite.

Tests the core functionality paths identified in the specifications,
using real services and infrastructure without mocking.
"""

import pytest
import asyncio
import torch
import time

from esper.morphogenetic_v2.lifecycle.extended_lifecycle import ExtendedLifecycle
from esper.morphogenetic_v2.kasmina.hybrid_layer import HybridKasminaLayer
from esper.morphogenetic_v2.kasmina.chunked_layer import ChunkedKasminaLayer
from esper.morphogenetic_v2.message_bus.clients import RedisStreamClient, MessageBusConfig
from esper.morphogenetic_v2.message_bus.publishers import TelemetryPublisher, TelemetryConfig
from esper.morphogenetic_v2.message_bus.handlers import CommandHandler
from esper.morphogenetic_v2.message_bus.schemas import (
    LifecycleTransitionCommand, BlueprintUpdateCommand,
    StateTransitionEvent
)
from esper.services.clients.tamiyo_client import TamiyoClient
from esper.services.clients.tolaria_client import TolariaClient
from esper.services.clients.nissa_client import NissaClient
from esper.utils.config import ServiceConfig
from esper.contracts.operational import HealthSignal


@pytest.fixture
async def redis_client():
    """Real Redis client for testing."""
    config = MessageBusConfig(
        redis_url="redis://localhost:6379/15",
        instance_id="test_critical_paths"
    )
    client = RedisStreamClient(config)

    try:
        await client.connect()
        yield client
        await client.disconnect()
    except Exception:
        pytest.skip("Redis not available")


@pytest.fixture
def service_config():
    """Configuration for services."""
    return ServiceConfig(
        tamiyo_url="http://localhost:8102",
        tolaria_url="http://localhost:8101",
        nissa_url="http://localhost:8103",
        timeout=5.0
    )


class TestThreeServiceIntegration:
    """Test integration of all three services (Tolaria, Tamiyo, Nissa)."""

    @pytest.mark.asyncio
    async def test_full_adaptation_flow(self, redis_client, service_config):
        """Test complete adaptation flow through all services."""
        # Initialize services
        try:
            tamiyo = TamiyoClient(config=service_config)
            tolaria = TolariaClient(config=service_config)
            nissa = NissaClient(config=service_config)
        except Exception:
            pytest.skip("Services not available")

        # Create test layer with real components
        base_layer = torch.nn.Linear(128, 256)
        kasmina_layer = HybridKasminaLayer(
            base_layer=base_layer,
            layer_id="test_layer",
            num_seeds=4
        )

        # 1. Nissa collects telemetry
        health_data = kasmina_layer.get_health_tensor()
        telemetry = {
            "layer_id": "test_layer",
            "health_data": health_data.tolist(),
            "timestamp": time.time()
        }

        await nissa.report_telemetry(telemetry)

        # 2. Tamiyo analyzes and recommends adaptations
        health_signals = [
            HealthSignal(
                layer_id=0,
                seed_id=i,
                chunk_id=0,
                epoch=100,
                activation_variance=0.9 if i == 0 else 0.1,  # First seed unhealthy
                dead_neuron_ratio=0.4 if i == 0 else 0.05,
                avg_correlation=0.3 if i == 0 else 0.8,
                health_score=0.4 if i == 0 else 0.9,
                execution_latency=50.0 if i == 0 else 5.0,
                error_count=10 if i == 0 else 0,
                active_seeds=1,
                total_seeds=4,
                timestamp=time.time()
            )
            for i in range(4)
        ]

        decisions = await tamiyo.analyze_model_state(health_signals)
        assert len(decisions) > 0

        # 3. Tolaria trains new kernels based on recommendations
        for decision in decisions:
            if decision.action == "update_kernel":
                blueprint_id = decision.kernel_id or "default_blueprint"
                kernel_data = await tolaria.train_kernel(
                    blueprint_id=blueprint_id,
                    layer_config={
                        "input_dim": 128,
                        "output_dim": 256,
                        "adaptation_params": decision.parameters
                    }
                )

                # Load new kernel into layer
                success = await kasmina_layer.load_kernel(
                    seed_id=decision.seed_id,
                    kernel_module=kernel_data.module
                )
                assert success

        # 4. Verify adaptation improved health
        new_health = kasmina_layer.get_health_tensor()
        # Health should improve for adapted seeds
        assert torch.mean(new_health).item() > torch.mean(health_data).item()


class TestElevenStateLifecycle:
    """Test the complete 11-state lifecycle system."""

    @pytest.mark.asyncio
    async def test_valid_state_transitions(self, redis_client):
        """Test all valid state transitions in the lifecycle."""
        # Create layer registry with test layers
        layer_registry = {}

        # Create handler
        handler = CommandHandler(layer_registry, redis_client)
        await handler.start()

        # Create test layer
        base_layer = torch.nn.Linear(64, 128)
        test_layer = HybridKasminaLayer(
            base_layer=base_layer,
            layer_id="lifecycle_test",
            num_seeds=2
        )
        layer_registry["lifecycle_test"] = test_layer

        seed_id = 0

        # Test valid transition paths
        valid_transitions = [
            # Basic lifecycle
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.EVALUATING),
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.GRAFTING),
            (ExtendedLifecycle.GRAFTING, ExtendedLifecycle.FLOWERING),
            (ExtendedLifecycle.FLOWERING, ExtendedLifecycle.SEEDING),

            # Decline path
            (ExtendedLifecycle.FLOWERING, ExtendedLifecycle.WITHERING),
            (ExtendedLifecycle.WITHERING, ExtendedLifecycle.COMPOSTING),
            (ExtendedLifecycle.COMPOSTING, ExtendedLifecycle.DORMANT),

            # Hibernation path
            (ExtendedLifecycle.FLOWERING, ExtendedLifecycle.DORMANT),
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.HIBERNATING),
            (ExtendedLifecycle.HIBERNATING, ExtendedLifecycle.DORMANT)
        ]

        # Reset to DORMANT
        test_layer.state_manager.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.DORMANT])
        )

        for from_state, to_state in valid_transitions:
            # Ensure we're in the correct starting state
            if test_layer.state_manager.get_state(seed_id) != from_state:
                # Transition to starting state
                test_layer.state_manager.set_state(
                    torch.tensor([seed_id]),
                    torch.tensor([from_state])
                )

            # Test transition
            command = LifecycleTransitionCommand(
                layer_id="lifecycle_test",
                seed_id=seed_id,
                target_state=ExtendedLifecycle.state_to_str(to_state),
                reason=f"Testing {from_state} to {to_state}"
            )

            result = await handler.handle_command(command)
            assert result.success, f"Failed transition {from_state} to {to_state}: {result.error}"
            assert test_layer.state_manager.get_state(seed_id) == to_state

        await handler.stop()

    @pytest.mark.asyncio
    async def test_invalid_state_transitions(self, redis_client):
        """Test that invalid transitions are rejected."""
        layer_registry = {}
        handler = CommandHandler(layer_registry, redis_client)
        await handler.start()

        # Create test layer
        base_layer = torch.nn.Linear(64, 128)
        test_layer = HybridKasminaLayer(
            base_layer=base_layer,
            layer_id="invalid_test",
            num_seeds=1
        )
        layer_registry["invalid_test"] = test_layer

        # Invalid transitions
        invalid_transitions = [
            # Can't skip stages
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.TRAINING),
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.FLOWERING),

            # Can't go backwards in main flow
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GERMINATED),
            (ExtendedLifecycle.FLOWERING, ExtendedLifecycle.TRAINING),

            # Can't resurrect from COMPOSTING
            (ExtendedLifecycle.COMPOSTING, ExtendedLifecycle.FLOWERING)
        ]

        for from_state, to_state in invalid_transitions:
            # Set starting state
            test_layer.state_manager.set_state(
                torch.tensor([0]),
                torch.tensor([from_state])
            )

            command = LifecycleTransitionCommand(
                layer_id="invalid_test",
                seed_id=0,
                target_state=ExtendedLifecycle.state_to_str(to_state),
                reason=f"Testing invalid {from_state} to {to_state}"
            )

            result = await handler.handle_command(command)
            assert not result.success, f"Invalid transition {from_state} to {to_state} should fail"
            assert "Cannot transition" in result.error

        await handler.stop()


class TestChunkedArchitecture:
    """Test the chunked architecture split->process->concatenate flow."""

    @pytest.mark.asyncio
    async def test_chunked_processing_flow(self):
        """Test complete chunked processing pipeline."""
        # Create base layer
        base_layer = torch.nn.Linear(1000, 2000)  # Large layer for chunking

        # Create chunked layer
        chunked_layer = ChunkedKasminaLayer(
            base_layer=base_layer,
            num_seeds=4,
            num_chunks=10,  # Split into 10 chunks
            layer_id="chunked_test"
        )

        # Create input
        batch_size = 32
        input_tensor = torch.randn(batch_size, 1000)

        # 1. Test splitting
        chunks = chunked_layer.chunk_manager.split_input(input_tensor)
        assert len(chunks) == 10
        assert all(chunk.shape[1] == 100 for chunk in chunks)  # 1000/10 = 100

        # 2. Test processing
        outputs = []
        for i, chunk in enumerate(chunks):
            # Process chunk with seed selection
            seed_id = i % 4  # Rotate through seeds
            if chunked_layer.state_manager.get_state(seed_id) == ExtendedLifecycle.DORMANT:
                # Activate seed
                chunked_layer.state_manager.set_state(
                    torch.tensor([seed_id]),
                    torch.tensor([ExtendedLifecycle.FLOWERING])
                )

            # Process chunk
            chunk_out = await chunked_layer._process_chunk_async(
                chunk,
                chunk_id=i,
                seed_assignments=[seed_id]
            )
            outputs.append(chunk_out)

        # 3. Test concatenation
        final_output = chunked_layer.chunk_manager.concatenate_outputs(outputs)
        assert final_output.shape == (batch_size, 2000)

        # 4. Verify against base layer
        base_output = base_layer(input_tensor)
        # Outputs should be similar but not identical due to seed processing
        assert final_output.shape == base_output.shape

        # 5. Test telemetry collection
        telemetry = chunked_layer.get_chunk_telemetry()
        assert "chunk_processing_times" in telemetry
        assert "chunk_health_scores" in telemetry
        assert len(telemetry["chunk_processing_times"]) == 10


class TestBlueprintManagement:
    """Test blueprint registry and management."""

    @pytest.mark.asyncio
    async def test_blueprint_lifecycle(self, redis_client):
        """Test blueprint creation, update, and application."""
        # Create registry
        layer_registry = {}
        handler = CommandHandler(layer_registry, redis_client)
        await handler.start()

        # Create test layer
        base_layer = torch.nn.Conv2d(3, 64, kernel_size=3)
        test_layer = HybridKasminaLayer(
            base_layer=base_layer,
            layer_id="blueprint_test",
            num_seeds=4
        )
        layer_registry["blueprint_test"] = test_layer

        # 1. Create blueprint
        blueprint_id = "conv_adaptation_v1"
        blueprint_command = BlueprintUpdateCommand(
            layer_id="blueprint_test",
            seed_id=0,
            blueprint_id=blueprint_id,
            strategy="progressive_adaptation",
            config={
                "learning_rate": 0.001,
                "adaptation_strength": 0.5,
                "kernel_regularization": 0.1
            }
        )

        result = await handler.handle_command(blueprint_command)
        assert result.success

        # 2. Verify blueprint stored
        seed_blueprint = test_layer.state_manager.get_blueprint_id(0)
        assert seed_blueprint == hash(blueprint_id) % (2**31)  # Blueprint ID is hashed

        # 3. Update blueprint
        updated_command = BlueprintUpdateCommand(
            layer_id="blueprint_test",
            seed_id=0,
            blueprint_id=blueprint_id,
            strategy="aggressive_adaptation",
            config={
                "learning_rate": 0.01,
                "adaptation_strength": 0.8,
                "kernel_regularization": 0.05,
                "dropout_rate": 0.2
            }
        )

        result = await handler.handle_command(updated_command)
        assert result.success

        # 4. Apply blueprint to multiple seeds
        for seed_id in range(1, 4):
            apply_command = BlueprintUpdateCommand(
                layer_id="blueprint_test",
                seed_id=seed_id,
                blueprint_id=blueprint_id,
                strategy="progressive_adaptation",
                config={}  # Use defaults from blueprint
            )

            result = await handler.handle_command(apply_command)
            assert result.success

            # Verify blueprint applied
            assert test_layer.state_manager.get_blueprint_id(seed_id) == hash(blueprint_id) % (2**31)

        await handler.stop()


class TestMessageBusIntegration:
    """Test message bus integration with real components."""

    @pytest.mark.asyncio
    async def test_telemetry_and_command_flow(self, redis_client):
        """Test complete telemetry collection and command execution flow."""
        # Setup telemetry publisher
        telemetry_config = TelemetryConfig(
            batch_size=5,
            batch_window_ms=100,
            anomaly_detection=True,
            anomaly_threshold_stddev=2.0
        )

        publisher = TelemetryPublisher(redis_client, telemetry_config)
        await publisher.start()

        # Setup command handler
        layer_registry = {}
        handler = CommandHandler(layer_registry, redis_client)
        await handler.start()

        # Create test layer
        base_layer = torch.nn.Linear(128, 256)
        test_layer = HybridKasminaLayer(
            base_layer=base_layer,
            layer_id="telemetry_test",
            num_seeds=10
        )
        layer_registry["telemetry_test"] = test_layer

        # Collect events
        events_received = []

        async def event_handler(event):
            events_received.append(event)

        await redis_client.subscribe(
            "morphogenetic.event.*",
            event_handler
        )

        # 1. Publish health telemetry
        health_tensor = test_layer.get_health_tensor()
        await publisher.publish_layer_health("telemetry_test", health_tensor)

        # 2. Simulate unhealthy seed
        for i in range(20):
            await publisher.publish_seed_metrics(
                layer_id="telemetry_test",
                seed_id=3,
                metrics={
                    "loss": 10.0 if i > 10 else 0.5,  # Sudden degradation
                    "accuracy": 0.1 if i > 10 else 0.9,
                    "latency": 100.0 if i > 10 else 10.0
                },
                lifecycle_state="TRAINING"
            )

        # Wait for anomaly detection
        await asyncio.sleep(0.5)

        # 3. Send lifecycle transition based on anomaly
        transition_command = LifecycleTransitionCommand(
            layer_id="telemetry_test",
            seed_id=3,
            target_state="WITHERING",
            reason="Performance degradation detected"
        )

        result = await handler.handle_command(transition_command)
        assert result.success

        # 4. Verify event propagation
        await asyncio.sleep(0.2)
        assert len(events_received) > 0

        # Find transition event
        transition_events = [
            e for e in events_received
            if isinstance(e, StateTransitionEvent)
        ]
        assert len(transition_events) > 0

        # Cleanup
        await publisher.stop()
        await handler.stop()


@pytest.mark.asyncio
async def test_performance_under_load():
    """Test system performance with realistic load."""
    # Create multiple layers
    layers = []
    for i in range(10):
        base = torch.nn.Linear(512, 512)
        layer = HybridKasminaLayer(
            base_layer=base,
            layer_id=f"load_test_{i}",
            num_seeds=20
        )
        layers.append(layer)

    # Simulate concurrent operations
    batch_size = 64
    input_tensor = torch.randn(batch_size, 512)

    start_time = time.time()

    # Process through all layers concurrently
    tasks = []
    for layer in layers:
        # Activate some seeds
        for seed_id in range(5):
            layer.state_manager.set_state(
                torch.tensor([seed_id]),
                torch.tensor([ExtendedLifecycle.FLOWERING])
            )

        # Process input
        task = asyncio.create_task(layer.forward_async(input_tensor))
        tasks.append(task)

    outputs = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Performance assertions
    assert len(outputs) == 10
    assert all(out.shape == (batch_size, 512) for out in outputs)
    assert elapsed < 5.0  # Should complete within 5 seconds

    # Check telemetry
    total_active_seeds = sum(
        len(layer.get_active_seeds())
        for layer in layers
    )
    assert total_active_seeds == 50  # 5 per layer * 10 layers
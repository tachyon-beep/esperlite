"""
Performance benchmarks for contract serialization and validation.
"""

import time

from esper.contracts.assets import Blueprint
from esper.contracts.assets import Seed
from esper.contracts.assets import TrainingSession
from esper.contracts.enums import BlueprintState
from esper.contracts.enums import SeedState
from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames
from esper.contracts.operational import HealthSignal
from esper.contracts.operational import SystemStatePacket


class TestContractPerformanceBenchmarks:
    """Comprehensive performance benchmarks for all contract models."""

    def test_seed_batch_serialization_performance(self):
        """Test batch serialization performance for Seed models."""
        # Create a batch of seeds
        seeds = [
            Seed(
                layer_id=i % 10,
                position=i,
                state=SeedState.TRAINING if i % 2 == 0 else SeedState.DORMANT,
                metadata={
                    "batch_id": f"batch-{i // 100}",
                    "performance": 0.85 + (i % 15) * 0.01,
                },
            )
            for i in range(1000)
        ]

        # Test batch serialization
        start_time = time.perf_counter()
        serialized_seeds = []
        for seed in seeds:
            serialized_seeds.append(seed.model_dump_json())
        elapsed_serialize = time.perf_counter() - start_time

        # Test batch deserialization
        start_time = time.perf_counter()
        for json_str in serialized_seeds:
            Seed.model_validate_json(json_str)
        elapsed_deserialize = time.perf_counter() - start_time

        # Performance assertions
        assert (
            elapsed_serialize < 1.0
        ), f"Batch serialization took {elapsed_serialize:.3f}s, expected <1.0s"
        assert (
            elapsed_deserialize < 1.0
        ), f"Batch deserialization took {elapsed_deserialize:.3f}s, expected <1.0s"

        # Total throughput check
        total_operations = len(seeds) * 2  # serialize + deserialize
        total_time = elapsed_serialize + elapsed_deserialize
        ops_per_second = total_operations / total_time
        assert (
            ops_per_second > 1000
        ), f"Throughput {ops_per_second:.0f} ops/s, expected >1000 ops/s"

    def test_blueprint_complex_architecture_performance(self):
        """Test performance with complex blueprint architectures."""
        # Create a complex architecture (simulating a large neural network)
        complex_architecture = {
            "model_type": "transformer",
            "layers": [
                {
                    "type": "Embedding",
                    "params": {"vocab_size": 50000, "embed_dim": 768},
                },
                *[
                    {
                        "type": "TransformerBlock",
                        "params": {
                            "embed_dim": 768,
                            "num_heads": 12,
                            "ff_dim": 3072,
                            "dropout": 0.1,
                        },
                        "layer_id": i,
                    }
                    for i in range(12)  # 12 transformer layers
                ],
                {"type": "LayerNorm", "params": {"normalized_shape": 768}},
                {
                    "type": "Linear",
                    "params": {"in_features": 768, "out_features": 50000},
                },
            ],
            "total_parameters": 117000000,  # ~117M parameters
            "memory_usage_mb": 450,
        }

        hyperparameters = {
            "learning_rate": 0.0001,
            "batch_size": 32,
            "sequence_length": 512,
            "warmup_steps": 4000,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }

        performance_metrics = {
            "training_loss": 3.24,
            "validation_loss": 3.31,
            "training_accuracy": 0.42,
            "validation_accuracy": 0.41,
            "perplexity": 27.3,
            "bleu_score": 0.28,
            "training_time_minutes": 1440,  # 24 hours
            "memory_peak_gb": 12.8,
        }

        blueprint = Blueprint(
            name="large-transformer-model",
            description="Large transformer model for language modeling",
            architecture=complex_architecture,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            created_by="automated-architecture-search",
            state=BlueprintState.CHARACTERIZED,
        )

        # Test serialization performance
        start_time = time.perf_counter()
        for _ in range(100):  # Fewer iterations due to complexity
            json_str = blueprint.model_dump_json()
            Blueprint.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        assert (
            elapsed < 2.0
        ), f"Complex Blueprint serialization took {elapsed:.3f}s, expected <2.0s"

        # Test that the complex data is preserved
        json_str = blueprint.model_dump_json()
        reconstructed = Blueprint.model_validate_json(json_str)
        assert (
            len(reconstructed.architecture["layers"]) == 15
        )  # 1 embedding + 12 transformer + 1 norm + 1 linear
        assert reconstructed.architecture["total_parameters"] == 117000000
        assert abs(reconstructed.performance_metrics["bleu_score"] - 0.28) < 1e-10

    def test_health_signal_high_frequency_performance(self):
        """Test high-frequency health signal processing performance."""
        # Simulate high-frequency health signals from multiple layers/seeds
        health_signals = []
        for epoch in range(10):  # 10 epochs
            for layer_id in range(20):  # 20 layers
                for seed_id in range(5):  # 5 seeds per layer
                    signal = HealthSignal(
                        layer_id=layer_id,
                        seed_id=seed_id,
                        chunk_id=epoch * 100 + layer_id,
                        epoch=epoch,
                        activation_variance=0.5 + (layer_id * 0.01),
                        dead_neuron_ratio=max(0.0, 0.1 - (seed_id * 0.02)),
                        avg_correlation=0.7 + (epoch * 0.01),
                        health_score=min(1.0, 0.8 + (layer_id * 0.01)),
                        execution_latency=0.001 + (seed_id * 0.0001),
                        error_count=max(0, 2 - seed_id),
                        active_seeds=seed_id + 1,
                        total_seeds=5,
                    )
                    health_signals.append(signal)

        # Total: 10 epochs * 20 layers * 5 seeds = 1000 signals
        assert len(health_signals) == 1000

        # Test batch processing performance
        start_time = time.perf_counter()
        serialized_signals = []
        for signal in health_signals:
            serialized_signals.append(signal.model_dump_json())
        elapsed_serialize = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for json_str in serialized_signals:
            HealthSignal.model_validate_json(json_str)
        elapsed_deserialize = time.perf_counter() - start_time

        # Performance requirements for high-frequency data
        assert (
            elapsed_serialize < 0.5
        ), f"Health signal serialization took {elapsed_serialize:.3f}s, expected <0.5s"
        assert (
            elapsed_deserialize < 0.5
        ), f"Health signal deserialization took {elapsed_deserialize:.3f}s, expected <0.5s"

        # Throughput requirement for real-time processing
        total_time = elapsed_serialize + elapsed_deserialize
        signals_per_second = len(health_signals) / total_time
        assert (
            signals_per_second > 2000
        ), f"Throughput {signals_per_second:.0f} signals/s, expected >2000 signals/s"

    def test_oona_message_bus_performance(self):
        """Test message bus performance with realistic message patterns."""
        # Create messages for different topics
        messages = []

        # Health telemetry messages (high frequency)
        for i in range(500):
            message = OonaMessage(
                sender_id=f"kasmina-layer-{i % 10}",
                trace_id=f"health-trace-{i}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload={
                    "layer_id": i % 10,
                    "seed_id": i % 5,
                    "health_score": 0.85 + (i % 20) * 0.005,
                    "metrics": {
                        "activation_variance": 0.3 + (i % 10) * 0.01,
                        "dead_neurons": i % 3,
                        "correlation": 0.7 + (i % 15) * 0.01,
                    },
                },
            )
            messages.append(message)

        # Control commands (medium frequency)
        for i in range(100):
            message = OonaMessage(
                sender_id="tamiyo-controller",
                trace_id=f"control-trace-{i}",
                topic=TopicNames.CONTROL_KASMINA_COMMANDS,
                payload={
                    "command": "adapt_seed",
                    "target_layer": i % 10,
                    "target_seed": i % 5,
                    "new_state": "training",
                    "parameters": {"learning_rate": 0.001, "momentum": 0.9},
                },
            )
            messages.append(message)

        # Blueprint events (low frequency)
        for i in range(50):
            message = OonaMessage(
                sender_id="urza-library",
                trace_id=f"blueprint-trace-{i}",
                topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
                payload={
                    "blueprint_id": f"bp-{i}",
                    "architecture_hash": f"arch-hash-{i:08x}",
                    "submitted_by": "karn-architect",
                    "priority": "normal" if i % 3 != 0 else "high",
                },
            )
            messages.append(message)

        # Total: 650 messages
        assert len(messages) == 650

        # Test message bus processing performance
        start_time = time.perf_counter()
        serialized_messages = []
        for message in messages:
            serialized_messages.append(message.model_dump_json())
        elapsed_serialize = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for json_str in serialized_messages:
            OonaMessage.model_validate_json(json_str)
        elapsed_deserialize = time.perf_counter() - start_time

        # Message bus performance requirements
        assert (
            elapsed_serialize < 1.0
        ), f"Message serialization took {elapsed_serialize:.3f}s, expected <1.0s"
        assert (
            elapsed_deserialize < 1.0
        ), f"Message deserialization took {elapsed_deserialize:.3f}s, expected <1.0s"

        # Throughput requirement for message bus
        total_time = elapsed_serialize + elapsed_deserialize
        messages_per_second = len(messages) / total_time
        assert (
            messages_per_second > 500
        ), f"Throughput {messages_per_second:.0f} msgs/s, expected >500 msgs/s"

    def test_training_session_with_large_state_performance(self):
        """Test performance with large training session state."""
        # Create a large training session with many seeds and blueprints
        seeds = [
            Seed(
                layer_id=i // 20,  # 20 seeds per layer
                position=i % 20,
                state=SeedState.TRAINING if i % 3 == 0 else SeedState.DORMANT,
                metadata={
                    "batch_created": i // 100,
                    "performance_score": 0.7 + (i % 30) * 0.01,
                    "last_updated_epoch": i % 50,
                },
            )
            for i in range(1000)  # 1000 seeds across 50 layers
        ]

        blueprints = [
            Blueprint(
                name=f"blueprint-{i}",
                description=f"Generated blueprint {i} for architecture exploration",
                architecture={
                    "type": "feedforward" if i % 2 == 0 else "convolutional",
                    "layers": [
                        {"type": "Linear", "in": 128 * (j + 1), "out": 128 * (j + 2)}
                        for j in range(3 + (i % 5))  # Variable depth
                    ],
                    "activation": "ReLU",
                    "dropout": 0.1 + (i % 10) * 0.01,
                },
                hyperparameters={
                    "learning_rate": 0.001 * (1 + i % 10),
                    "batch_size": 32 * (1 + i % 4),
                    "weight_decay": 0.0001 * (1 + i % 5),
                },
                performance_metrics={
                    "accuracy": 0.7 + (i % 25) * 0.01,
                    "loss": 0.5 - (i % 20) * 0.01,
                    "training_time": 300 + (i % 100) * 10,
                },
                created_by=f"generator-{i % 5}",
                state=(
                    BlueprintState.CHARACTERIZED
                    if i % 3 == 0
                    else BlueprintState.PROPOSED
                ),
            )
            for i in range(200)  # 200 blueprints
        ]

        training_session = TrainingSession(
            name="large-morphogenetic-session",
            description="Large-scale morphogenetic training with 1000 seeds and 200 blueprints",
            training_model_config={
                "base_architecture": "ResNet50",
                "num_classes": 1000,
                "input_size": [3, 224, 224],
                "morphogenetic_enabled": True,
                "adaptation_frequency": 10,
            },
            training_config={
                "epochs": 100,
                "batch_size": 256,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_schedule": "cosine",
                "warmup_epochs": 5,
                "distributed": True,
                "num_gpus": 8,
            },
            seeds=seeds,
            blueprints=blueprints,
            status="running",
        )

        # Test large session serialization performance
        start_time = time.perf_counter()
        json_str = training_session.model_dump_json()
        elapsed_serialize = time.perf_counter() - start_time

        start_time = time.perf_counter()
        reconstructed = TrainingSession.model_validate_json(json_str)
        elapsed_deserialize = time.perf_counter() - start_time

        # Performance requirements for large sessions
        assert (
            elapsed_serialize < 3.0
        ), f"Large session serialization took {elapsed_serialize:.3f}s, expected <3.0s"
        assert (
            elapsed_deserialize < 3.0
        ), f"Large session deserialization took {elapsed_deserialize:.3f}s, expected <3.0s"

        # Verify data integrity
        assert len(reconstructed.seeds) == 1000
        assert len(reconstructed.blueprints) == 200
        assert reconstructed.training_config["num_gpus"] == 8
        assert reconstructed.status == "running"

    def test_cross_contract_integration_performance(self):
        """Test performance when contracts interact (realistic workflow)."""
        # Simulate a realistic workflow: health monitoring -> decision -> blueprint creation

        # 1. Health signals from multiple layers
        health_signals = [
            HealthSignal(
                layer_id=i % 5,
                seed_id=j,
                chunk_id=i * 10 + j,
                epoch=i,
                activation_variance=0.4 + (i * 0.01),
                dead_neuron_ratio=0.05 + (j * 0.01),
                avg_correlation=0.8 - (i * 0.005),
                health_score=0.9 - (i * 0.01),
            )
            for i in range(20)  # 20 epochs
            for j in range(3)  # 3 seeds per layer
        ]  # Total: 60 signals

        # 2. System state packets
        system_states = [
            SystemStatePacket(
                epoch=i,
                total_seeds=15,  # 5 layers * 3 seeds
                active_seeds=10 + (i % 5),
                training_loss=2.5 - (i * 0.05),
                validation_loss=2.7 - (i * 0.04),
                system_load=0.6 + (i % 10) * 0.02,
                memory_usage=0.7 + (i % 8) * 0.01,
            )
            for i in range(20)
        ]

        # 3. Messages for each signal and state
        messages = []
        for signal in health_signals:
            message = OonaMessage(
                sender_id=f"kasmina-layer-{signal.layer_id}",
                trace_id=f"health-{signal.chunk_id}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload=signal.model_dump(),
            )
            messages.append(message)

        for state in system_states:
            message = OonaMessage(
                sender_id="tolaria-orchestrator",
                trace_id=f"system-state-{state.epoch}",
                topic=TopicNames.SYSTEM_EVENTS_EPOCH,
                payload=state.model_dump(),
            )
            messages.append(message)

        # Total workflow items: 60 signals + 20 states + 80 messages = 160 operations
        total_items = len(health_signals) + len(system_states) + len(messages)
        assert total_items == 160

        # Test integrated workflow performance
        start_time = time.perf_counter()

        # Serialize all components
        serialized_data = []
        for signal in health_signals:
            serialized_data.append(("health", signal.model_dump_json()))

        for state in system_states:
            serialized_data.append(("state", state.model_dump_json()))

        for message in messages:
            serialized_data.append(("message", message.model_dump_json()))

        elapsed_serialize = time.perf_counter() - start_time

        # Deserialize all components
        start_time = time.perf_counter()
        reconstructed_items = []
        for item_type, json_str in serialized_data:
            if item_type == "health":
                reconstructed_items.append(HealthSignal.model_validate_json(json_str))
            elif item_type == "state":
                reconstructed_items.append(
                    SystemStatePacket.model_validate_json(json_str)
                )
            elif item_type == "message":
                reconstructed_items.append(OonaMessage.model_validate_json(json_str))

        elapsed_deserialize = time.perf_counter() - start_time

        # Integrated workflow performance requirements
        assert (
            elapsed_serialize < 1.0
        ), f"Integrated serialization took {elapsed_serialize:.3f}s, expected <1.0s"
        assert (
            elapsed_deserialize < 1.0
        ), f"Integrated deserialization took {elapsed_deserialize:.3f}s, expected <1.0s"

        # Verify all items processed correctly
        assert len(reconstructed_items) == total_items

        # Calculate overall throughput
        total_time = elapsed_serialize + elapsed_deserialize
        items_per_second = total_items / total_time
        assert (
            items_per_second > 100
        ), f"Integrated throughput {items_per_second:.0f} items/s, expected >100 items/s"

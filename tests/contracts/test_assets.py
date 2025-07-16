"""
Tests for asset models validation logic and performance.
"""

import time
from datetime import datetime

from esper.contracts.assets import Blueprint
from esper.contracts.assets import Seed
from esper.contracts.assets import TrainingSession
from esper.contracts.enums import BlueprintState
from esper.contracts.enums import SeedState


class TestSeed:
    """Test cases for the Seed model."""

    def test_seed_creation_defaults(self):
        """Test Seed creation with minimal required fields."""
        seed = Seed(layer_id=1, position=0)

        assert seed.layer_id == 1
        assert seed.position == 0
        assert seed.state == SeedState.DORMANT
        assert seed.blueprint_id is None
        assert isinstance(seed.seed_id, str)
        assert len(seed.seed_id) == 36  # UUID4 format
        assert isinstance(seed.created_at, datetime)
        assert isinstance(seed.updated_at, datetime)
        assert not seed.metadata

    def test_seed_serialization(self):
        """Test Seed JSON serialization/deserialization."""
        original = Seed(
            layer_id=5,
            position=10,
            state=SeedState.GERMINATED,
            blueprint_id="test-blueprint-123",
        )

        # Serialize to JSON
        json_data = original.model_dump_json()
        assert isinstance(json_data, str)

        # Deserialize from JSON
        reconstructed = Seed.model_validate_json(json_data)

        # Verify all fields match
        assert reconstructed.layer_id == original.layer_id
        assert reconstructed.position == original.position
        assert reconstructed.state == original.state
        assert reconstructed.blueprint_id == original.blueprint_id
        assert reconstructed.seed_id == original.seed_id

    def test_seed_state_transitions(self):
        """Test valid seed state transitions."""
        seed = Seed(layer_id=1, position=0)

        # Test state progression
        seed.state = SeedState.DORMANT
        assert seed.state == SeedState.DORMANT

        seed.state = SeedState.GERMINATED
        assert seed.state == SeedState.GERMINATED

        seed.state = SeedState.TRAINING
        assert seed.state == SeedState.TRAINING

    def test_seed_metadata_handling(self):
        """Test seed metadata field functionality."""
        metadata = {
            "performance_score": 0.95,
            "last_activation": "2025-07-16T10:30:00Z",
            "tags": ["high-performance", "stable"],
        }

        seed = Seed(layer_id=1, position=0, metadata=metadata)
        assert seed.metadata == metadata

        # Test metadata serialization
        json_str = seed.model_dump_json()
        reconstructed = Seed.model_validate_json(json_str)
        assert reconstructed.metadata == metadata

    def test_seed_state_display(self):
        """Test Seed state_display method."""
        seed = Seed(layer_id=1, position=0, state=SeedState.DORMANT)
        assert seed.state_display() == "Dormant"

        seed.state = SeedState.TRAINING
        assert seed.state_display() == "Training"

        seed.state = SeedState.GRAFTING
        assert seed.state_display() == "Grafting"

        seed.state = SeedState.GERMINATED
        assert seed.state_display() == "Germinated"

    def test_seed_is_active(self):
        """Test Seed is_active method."""
        # Test inactive states
        seed = Seed(layer_id=1, position=0, state=SeedState.DORMANT)
        assert not seed.is_active()

        seed.state = SeedState.GERMINATED
        assert not seed.is_active()

        # Test active states
        seed.state = SeedState.TRAINING
        assert seed.is_active()

        seed.state = SeedState.GRAFTING
        assert seed.is_active()


class TestBlueprint:
    """Test cases for the Blueprint model."""

    def test_blueprint_creation(self):
        """Test Blueprint creation with required fields."""
        architecture = {
            "layers": [
                {"type": "Linear", "in_features": 512, "out_features": 256},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": 256, "out_features": 128},
            ]
        }

        blueprint = Blueprint(
            name="test-blueprint",
            description="Test architectural blueprint",
            architecture=architecture,
            created_by="test-system",
        )

        assert blueprint.name == "test-blueprint"
        assert blueprint.description == "Test architectural blueprint"
        assert blueprint.state == BlueprintState.PROPOSED
        assert blueprint.architecture == architecture
        assert not blueprint.hyperparameters
        assert not blueprint.performance_metrics
        assert blueprint.created_by == "test-system"
        assert isinstance(blueprint.blueprint_id, str)

    def test_blueprint_serialization(self):
        """Test Blueprint JSON serialization/deserialization."""
        architecture = {"layer_type": "conv2d", "filters": 64}
        hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        metrics = {"accuracy": 0.95, "loss": 0.05}

        original = Blueprint(
            name="conv-blueprint",
            description="Convolutional layer blueprint",
            architecture=architecture,
            hyperparameters=hyperparams,
            performance_metrics=metrics,
            created_by="tamiyo-policy",
            state=BlueprintState.CHARACTERIZED,
        )

        # Serialize and deserialize
        json_data = original.model_dump_json()
        reconstructed = Blueprint.model_validate_json(json_data)

        # Verify all fields
        assert reconstructed.name == original.name
        assert reconstructed.description == original.description
        assert reconstructed.architecture == original.architecture
        assert reconstructed.hyperparameters == original.hyperparameters
        assert reconstructed.performance_metrics == original.performance_metrics
        assert reconstructed.state == original.state
        assert reconstructed.created_by == original.created_by

    def test_blueprint_state_management(self):
        """Test blueprint state transitions."""
        blueprint = Blueprint(
            name="test", description="test", architecture={}, created_by="test"
        )

        # Test state progression
        assert blueprint.state == BlueprintState.PROPOSED

        blueprint.state = BlueprintState.COMPILING
        assert blueprint.state == BlueprintState.COMPILING

        blueprint.state = BlueprintState.CHARACTERIZED
        assert blueprint.state == BlueprintState.CHARACTERIZED

    def test_blueprint_state_display(self):
        """Test Blueprint state_display method."""
        blueprint = Blueprint(
            name="test", description="test", architecture={}, created_by="test"
        )

        blueprint.state = BlueprintState.PROPOSED
        assert blueprint.state_display() == "Proposed"

        blueprint.state = BlueprintState.COMPILING
        assert blueprint.state_display() == "Compiling"

        blueprint.state = BlueprintState.CHARACTERIZED
        assert blueprint.state_display() == "Characterized"

    def test_blueprint_is_ready_for_deployment(self):
        """Test Blueprint is_ready_for_deployment method."""
        blueprint = Blueprint(
            name="test", description="test", architecture={}, created_by="test"
        )

        # Not ready in PROPOSED state
        blueprint.state = BlueprintState.PROPOSED
        assert not blueprint.is_ready_for_deployment()

        # Not ready in COMPILING state
        blueprint.state = BlueprintState.COMPILING
        assert not blueprint.is_ready_for_deployment()

        # Ready in CHARACTERIZED state
        blueprint.state = BlueprintState.CHARACTERIZED
        assert blueprint.is_ready_for_deployment()

    def test_blueprint_get_performance_summary(self):
        """Test Blueprint get_performance_summary method."""
        blueprint = Blueprint(
            name="test", description="test", architecture={}, created_by="test"
        )

        # Test with no metrics
        assert blueprint.get_performance_summary() == "No metrics available"

        # Test with metrics
        blueprint.performance_metrics = {
            "accuracy": 0.95,
            "latency": 0.12,
            "memory_usage": 45.678,
        }
        summary = blueprint.get_performance_summary()
        assert "Performance:" in summary
        assert "accuracy: 0.950" in summary
        assert "latency: 0.120" in summary
        assert "memory_usage: 45.678" in summary


class TestTrainingSession:
    """Test cases for the TrainingSession model."""

    def test_training_session_creation(self):
        """Test TrainingSession creation."""
        model_config = {"architecture": "ResNet18", "layers": 18}
        training_config = {"epochs": 100, "learning_rate": 0.001}

        session = TrainingSession(
            name="cifar10-experiment",
            description="CIFAR-10 morphogenetic training",
            training_model_config=model_config,
            training_config=training_config,
        )

        assert session.name == "cifar10-experiment"
        assert session.description == "CIFAR-10 morphogenetic training"
        assert session.training_model_config == model_config
        assert session.training_config == training_config
        assert session.seeds == []
        assert session.blueprints == []
        assert session.status == "initialized"
        assert isinstance(session.session_id, str)

    def test_training_session_with_seeds_and_blueprints(self):
        """Test TrainingSession with seeds and blueprints."""
        seed1 = Seed(layer_id=1, position=0)
        seed2 = Seed(layer_id=2, position=1)

        blueprint1 = Blueprint(
            name="bp1", description="test", architecture={}, created_by="test"
        )

        session = TrainingSession(
            name="test-session",
            description="Test session with seeds",
            training_model_config={},
            training_config={},
            seeds=[seed1, seed2],
            blueprints=[blueprint1],
        )

        assert len(session.seeds) == 2
        assert len(session.blueprints) == 1
        assert session.seeds[0].layer_id == 1
        assert session.seeds[1].layer_id == 2
        assert session.blueprints[0].name == "bp1"

    def test_training_session_serialization(self):
        """Test TrainingSession serialization with nested objects."""
        seed = Seed(layer_id=1, position=0, state=SeedState.TRAINING)
        blueprint = Blueprint(
            name="test-bp",
            description="test",
            architecture={"type": "linear"},
            created_by="test-system",
        )

        session = TrainingSession(
            name="complex-session",
            description="Session with nested objects",
            training_model_config={"type": "ResNet"},
            training_config={"epochs": 10},
            seeds=[seed],
            blueprints=[blueprint],
            status="running",
        )

        # Serialize and deserialize
        json_data = session.model_dump_json()
        reconstructed = TrainingSession.model_validate_json(json_data)

        # Verify nested objects
        assert len(reconstructed.seeds) == 1
        assert len(reconstructed.blueprints) == 1
        assert reconstructed.seeds[0].layer_id == 1
        assert reconstructed.seeds[0].state == SeedState.TRAINING
        assert reconstructed.blueprints[0].name == "test-bp"
        assert reconstructed.status == "running"


class TestAssetPerformance:
    """Performance tests for asset models."""

    def test_seed_serialization_performance(self):
        """Test Seed serialization performance."""
        seed = Seed(
            layer_id=100,
            position=50,
            state=SeedState.GERMINATED,
            metadata={"score": 0.95, "tags": ["fast", "stable"]},
        )

        # Measure serialization performance
        start_time = time.perf_counter()
        for _ in range(1000):
            json_str = seed.model_dump_json()
            Seed.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should serialize/deserialize 1000 times in under 1 second
        assert elapsed < 1.0, f"Serialization took {elapsed:.3f}s, expected <1.0s"

        # Calculate average per operation
        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0, (
            f"Average operation time {avg_time_ms:.3f}ms exceeds 1ms target"
        )

    def test_blueprint_serialization_performance(self):
        """Test Blueprint serialization performance with complex architecture."""
        complex_architecture = {
            "layers": [
                {
                    "type": "Conv2d",
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 3,
                },
                {"type": "BatchNorm2d", "num_features": 64},
                {"type": "ReLU"},
                {"type": "MaxPool2d", "kernel_size": 2},
                {
                    "type": "Conv2d",
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                },
                {"type": "BatchNorm2d", "num_features": 128},
                {"type": "ReLU"},
                {"type": "AdaptiveAvgPool2d", "output_size": (1, 1)},
                {"type": "Flatten"},
                {"type": "Linear", "in_features": 128, "out_features": 10},
            ]
        }

        blueprint = Blueprint(
            name="complex-cnn",
            description="Complex CNN architecture",
            architecture=complex_architecture,
            hyperparameters={"lr": 0.001, "momentum": 0.9, "weight_decay": 1e-4},
            performance_metrics={"accuracy": 0.92, "f1_score": 0.91, "precision": 0.93},
            created_by="tamiyo-policy",
        )

        # Measure performance
        start_time = time.perf_counter()
        for _ in range(500):  # Fewer iterations due to complexity
            json_str = blueprint.model_dump_json()
            Blueprint.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should handle 500 complex serializations in under 1 second
        assert elapsed < 1.0, (
            f"Complex Blueprint serialization took {elapsed:.3f}s, expected <1.0s"
        )

    def test_training_session_performance(self):
        """Test TrainingSession performance with multiple nested objects."""
        # Create multiple seeds and blueprints
        seeds = [Seed(layer_id=i, position=j) for i in range(5) for j in range(3)]
        blueprints = [
            Blueprint(
                name=f"bp-{i}",
                description=f"Blueprint {i}",
                architecture={"layer": i, "neurons": 128 * (i + 1)},
                created_by="test-system",
            )
            for i in range(3)
        ]

        session = TrainingSession(
            name="performance-test-session",
            description="Session for performance testing",
            training_model_config={"type": "ResNet18", "num_classes": 10},
            training_config={"epochs": 100, "batch_size": 32, "lr": 0.001},
            seeds=seeds,
            blueprints=blueprints,
        )

        # Measure serialization performance
        start_time = time.perf_counter()
        for _ in range(100):  # Fewer iterations due to complexity
            json_str = session.model_dump_json()
            TrainingSession.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        # Should handle 100 complex session serializations in under 1 second
        assert elapsed < 1.0, (
            f"TrainingSession serialization took {elapsed:.3f}s, expected <1.0s"
        )

    def test_training_session_get_active_seed_count(self):
        """Test TrainingSession.get_active_seed_count() method."""
        # Create seeds with different states
        training_seed = Seed(layer_id=1, position=0, state=SeedState.TRAINING)
        grafting_seed = Seed(layer_id=2, position=1, state=SeedState.GRAFTING)
        germinated_seed = Seed(layer_id=3, position=2, state=SeedState.GERMINATED)
        dormant_seed = Seed(layer_id=4, position=3, state=SeedState.DORMANT)
        culled_seed = Seed(layer_id=5, position=4, state=SeedState.CULLED)

        session = TrainingSession(
            name="seed-count-test",
            description="Test session for seed counting",
            training_model_config={"type": "TestModel"},
            training_config={"epochs": 10},
            seeds=[
                training_seed,
                grafting_seed,
                germinated_seed,
                dormant_seed,
                culled_seed,
            ],
        )

        # Test active seed count (only TRAINING and GRAFTING are active)
        active_count = session.get_active_seed_count()
        assert active_count == 2

        # Test with empty seeds list
        empty_session = TrainingSession(
            name="empty-session",
            description="Session with no seeds",
            training_model_config={"type": "TestModel"},
            training_config={"epochs": 10},
            seeds=[],
        )
        assert empty_session.get_active_seed_count() == 0

    def test_training_session_get_session_summary(self):
        """Test TrainingSession.get_session_summary() method."""
        # Create test seeds and blueprints
        seeds = [
            Seed(layer_id=1, position=0, state=SeedState.TRAINING),
            Seed(layer_id=2, position=1, state=SeedState.GRAFTING),
            Seed(layer_id=3, position=2, state=SeedState.DORMANT),
        ]

        blueprints = [
            Blueprint(
                name="bp1",
                description="First blueprint",
                architecture={"layers": [1, 2, 3]},
                performance_metrics={"accuracy": 0.95},
                created_by="test",
            ),
            Blueprint(
                name="bp2",
                description="Second blueprint",
                architecture={"layers": [4, 5]},
                performance_metrics={"accuracy": 0.87},
                created_by="test",
            ),
        ]

        session = TrainingSession(
            name="summary-test",
            description="Test session for summary generation",
            training_model_config={"type": "TestModel", "layers": 5},
            training_config={"epochs": 100, "batch_size": 32},
            seeds=seeds,
            blueprints=blueprints,
            status="running",
        )

        # Get summary
        summary = session.get_session_summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "summary-test" in summary
        assert "3 seeds" in summary
        assert "2 active" in summary  # TRAINING + GRAFTING
        assert "2 blueprints" in summary
        assert "status: running" in summary

        # Test with minimal session
        minimal_session = TrainingSession(
            name="minimal",
            description="Minimal session",
            training_model_config={},
            training_config={},
        )

        minimal_summary = minimal_session.get_session_summary()
        assert isinstance(minimal_summary, str)
        assert "minimal" in minimal_summary
        assert "0 seeds" in minimal_summary
        assert "0 active" in minimal_summary
        assert "0 blueprints" in minimal_summary

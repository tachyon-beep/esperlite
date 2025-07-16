"""
Comprehensive tests for operational data models.
"""

import time

import pytest
from pydantic import ValidationError

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.contracts.operational import ModelGraphState
from esper.contracts.operational import SystemStatePacket


class TestHealthSignal:
    """Test suite for HealthSignal model."""

    def test_health_signal_creation_basic(self):
        """Test basic HealthSignal creation with required fields."""
        signal = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
        )

        assert signal.layer_id == 1
        assert signal.seed_id == 123
        assert signal.chunk_id == 456
        assert signal.epoch == 10
        assert signal.activation_variance == pytest.approx(0.25)
        assert signal.dead_neuron_ratio == pytest.approx(0.05)
        assert signal.avg_correlation == pytest.approx(0.75)
        assert signal.is_ready_for_transition is False
        assert signal.health_score == pytest.approx(1.0)  # Default value
        assert signal.execution_latency == pytest.approx(0.0)  # Default value
        assert signal.error_count == 0  # Default value
        assert signal.active_seeds == 1  # Default value
        assert signal.total_seeds == 1  # Default value

    def test_health_signal_validation_constraints(self):
        """Test HealthSignal field validation constraints."""
        # Test valid ranges
        signal = HealthSignal(
            layer_id=0,
            seed_id=0,
            chunk_id=0,
            epoch=0,
            activation_variance=0.0,
            dead_neuron_ratio=0.0,
            avg_correlation=-1.0,
            health_score=0.5,
            execution_latency=10.5,
        )
        assert signal.layer_id == 0
        assert signal.dead_neuron_ratio == pytest.approx(0.0)
        assert signal.avg_correlation == pytest.approx(-1.0)

        # Test constraint violations - negative layer_id
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            HealthSignal(
                layer_id=-1,
                seed_id=123,
                chunk_id=456,
                epoch=10,
                activation_variance=0.25,
                dead_neuron_ratio=0.05,
                avg_correlation=0.75,
            )

        # Test constraint violations - negative seed_id
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            HealthSignal(
                layer_id=1,
                seed_id=-5,
                chunk_id=100,
                epoch=10,
                activation_variance=0.5,
                dead_neuron_ratio=0.1,
                avg_correlation=0.8,
            )

        # Test ratio constraints - dead_neuron_ratio > 1.0
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 1"
        ):
            HealthSignal(
                layer_id=1,
                seed_id=123,
                chunk_id=456,
                epoch=10,
                activation_variance=0.25,
                dead_neuron_ratio=1.5,  # Invalid: > 1.0
                avg_correlation=0.75,
            )

        # Test correlation constraints - avg_correlation > 1.0
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 1"
        ):
            HealthSignal(
                layer_id=1,
                seed_id=123,
                chunk_id=456,
                epoch=10,
                activation_variance=0.25,
                dead_neuron_ratio=0.05,
                avg_correlation=1.5,  # Invalid: > 1.0
            )

    def test_health_signal_computed_properties(self):
        """Test HealthSignal computed property methods."""
        # Test healthy status
        healthy_signal = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.9,
            error_count=2,
        )

        assert healthy_signal.health_status() == "Healthy"
        assert healthy_signal.is_healthy() is True

        # Test boundary case: exactly 0.8 health_score
        boundary_healthy = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.8,  # Boundary case
            error_count=1,
        )

        assert boundary_healthy.health_status() == "Healthy"
        assert boundary_healthy.is_healthy() is True

        # Test warning status
        warning_signal = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.65,
        )

        assert warning_signal.health_status() == "Warning"
        assert warning_signal.is_healthy() is False

        # Test boundary case: exactly 0.6 health_score
        boundary_warning = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.6,  # Boundary case
            error_count=3,
        )

        assert boundary_warning.health_status() == "Warning"
        assert boundary_warning.is_healthy() is False

        # Test critical status
        critical_signal = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.3,
            error_count=10,
        )

        assert critical_signal.health_status() == "Critical"
        assert critical_signal.is_healthy() is False

        # Test boundary case: exactly 5 errors (unhealthy)
        boundary_errors = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.8,  # Good score but too many errors
            error_count=5,  # Boundary case
        )

        assert boundary_errors.is_healthy() is False


class TestSystemStatePacket:
    """Test suite for SystemStatePacket model."""

    def test_system_state_packet_creation(self):
        """Test basic SystemStatePacket creation."""
        packet = SystemStatePacket(
            epoch=100,
            total_seeds=50,
            active_seeds=35,
            training_loss=0.25,
            validation_loss=0.32,
            system_load=0.65,
            memory_usage=0.78,
        )

        assert packet.epoch == 100
        assert packet.total_seeds == 50
        assert packet.active_seeds == 35
        assert packet.training_loss == pytest.approx(0.25)
        assert packet.validation_loss == pytest.approx(0.32)
        assert packet.system_load == pytest.approx(0.65)
        assert packet.memory_usage == pytest.approx(0.78)

    def test_system_state_packet_validation(self):
        """Test SystemStatePacket validation constraints."""
        # Test valid minimum values
        packet = SystemStatePacket(
            epoch=0,
            total_seeds=0,
            active_seeds=0,
            training_loss=0.0,
            validation_loss=0.0,
            system_load=0.0,
            memory_usage=0.0,
        )
        assert packet.epoch == 0

        # Test constraint violations - negative epoch
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            SystemStatePacket(
                epoch=-1,
                total_seeds=100,
                active_seeds=50,
                training_loss=0.5,
                validation_loss=0.6,
                system_load=0.7,
                memory_usage=0.8,
            )

    def test_system_state_packet_computed_properties(self):
        """Test SystemStatePacket computed property methods."""
        # Test normal system health
        normal_packet = SystemStatePacket(
            epoch=100,
            total_seeds=50,
            active_seeds=35,
            training_loss=0.25,
            validation_loss=0.32,
            system_load=0.5,
            memory_usage=0.6,
        )

        assert normal_packet.system_health() == "Normal"
        assert normal_packet.is_overloaded() is False

        # Test warning system health
        warning_packet = SystemStatePacket(
            epoch=100,
            total_seeds=50,
            active_seeds=35,
            training_loss=0.25,
            validation_loss=0.32,
            system_load=0.75,
            memory_usage=0.6,
        )

        assert warning_packet.system_health() == "Warning"
        assert warning_packet.is_overloaded() is False

        # Test critical system health
        critical_packet = SystemStatePacket(
            epoch=100,
            total_seeds=50,
            active_seeds=35,
            training_loss=0.25,
            validation_loss=0.32,
            system_load=0.95,
            memory_usage=0.92,
        )

        assert critical_packet.system_health() == "Critical"
        assert critical_packet.is_overloaded() is True


class TestAdaptationDecision:
    """Test suite for AdaptationDecision model."""

    def test_adaptation_decision_creation(self):
        """Test basic AdaptationDecision creation."""
        decision = AdaptationDecision(
            layer_name="transformer_block_5",
            adaptation_type="add_seed",
            confidence=0.85,
            urgency=0.6,
            metadata={"reason": "performance_bottleneck", "suggested_count": 3},
        )

        assert decision.layer_name == "transformer_block_5"
        assert decision.adaptation_type == "add_seed"
        assert decision.confidence == pytest.approx(0.85)
        assert decision.urgency == pytest.approx(0.6)
        assert decision.metadata["reason"] == "performance_bottleneck"
        assert isinstance(decision.timestamp, float)

    def test_adaptation_decision_validation(self):
        """Test AdaptationDecision validation constraints."""
        # Test valid adaptation types
        valid_types = [
            "add_seed",
            "remove_seed",
            "modify_architecture",
            "optimize_parameters",
        ]
        for adaptation_type in valid_types:
            decision = AdaptationDecision(
                layer_name="test_layer",
                adaptation_type=adaptation_type,
                confidence=0.8,
                urgency=0.5,
            )
            assert decision.adaptation_type == adaptation_type

        # Test invalid adaptation type
        with pytest.raises(ValueError):
            AdaptationDecision(
                layer_name="test_layer",
                adaptation_type="invalid_type",
                confidence=0.8,
                urgency=0.5,
            )

    def test_adaptation_decision_computed_properties(self):
        """Test AdaptationDecision computed property methods."""
        # Test high priority decision
        high_priority = AdaptationDecision(
            layer_name="critical_layer",
            adaptation_type="add_seed",
            confidence=0.9,
            urgency=0.8,
        )

        assert high_priority.decision_priority() == "High"
        assert high_priority.should_execute_immediately() is True

        # Test medium priority decision
        medium_priority = AdaptationDecision(
            layer_name="normal_layer",
            adaptation_type="optimize_parameters",
            confidence=0.6,
            urgency=0.4,
        )

        assert medium_priority.decision_priority() == "Medium"
        assert medium_priority.should_execute_immediately() is False

        # Test low priority decision
        low_priority = AdaptationDecision(
            layer_name="stable_layer",
            adaptation_type="remove_seed",
            confidence=0.3,
            urgency=0.2,
        )

        assert low_priority.decision_priority() == "Low"
        assert low_priority.should_execute_immediately() is False

    def test_adaptation_decision_timestamp(self):
        """Test AdaptationDecision timestamp functionality."""
        # Record time before creating decision
        before = time.time()

        decision = AdaptationDecision(
            layer_name="test_layer",
            adaptation_type="add_seed",
            confidence=0.8,
            urgency=0.5,
        )

        # Record time after creating decision
        after = time.time()

        # Timestamp should be between before and after
        assert before <= decision.timestamp <= after
        assert isinstance(decision.timestamp, float)


class TestModelGraphState:
    """Test suite for ModelGraphState dataclass."""

    def test_model_graph_state_creation(self):
        """Test ModelGraphState dataclass creation."""
        # Create sample health signals
        health_signals = {
            "layer_1": HealthSignal(
                layer_id=1,
                seed_id=123,
                chunk_id=456,
                epoch=10,
                activation_variance=0.25,
                dead_neuron_ratio=0.05,
                avg_correlation=0.75,
            ),
            "layer_2": HealthSignal(
                layer_id=2,
                seed_id=124,
                chunk_id=457,
                epoch=10,
                activation_variance=0.35,
                dead_neuron_ratio=0.08,
                avg_correlation=0.65,
            ),
        }

        state = ModelGraphState(
            topology="mock_topology",
            health_signals=health_signals,
            health_trends={"layer_1": 0.8, "layer_2": 0.6},
            problematic_layers={"layer_2"},
            overall_health=0.7,
            analysis_timestamp=time.time(),
        )

        assert state.topology == "mock_topology"
        assert len(state.health_signals) == 2
        assert "layer_1" in state.health_signals
        assert "layer_2" in state.health_signals
        assert state.health_trends["layer_1"] == pytest.approx(0.8)
        assert "layer_2" in state.problematic_layers
        assert state.overall_health == pytest.approx(0.7)
        assert isinstance(state.analysis_timestamp, float)


class TestOperationalPerformance:
    """Performance tests for operational models."""

    def test_health_signal_serialization_performance(self):
        """Test HealthSignal serialization performance."""
        signal = HealthSignal(
            layer_id=1,
            seed_id=123,
            chunk_id=456,
            epoch=10,
            activation_variance=0.25,
            dead_neuron_ratio=0.05,
            avg_correlation=0.75,
            health_score=0.85,
            execution_latency=2.5,
            error_count=1,
            active_seeds=5,
            total_seeds=10,
        )

        # Test serialization speed
        start_time = time.perf_counter()
        for _ in range(100):
            json_str = signal.model_dump_json()
            HealthSignal.model_validate_json(json_str)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.1, f"Serialization took {elapsed:.3f}s, expected <0.1s"

    def test_batch_operational_models_performance(self):
        """Test batch processing performance for operational models."""
        # Create mixed batch
        models = []

        # Add health signals
        for i in range(100):
            models.append(
                HealthSignal(
                    layer_id=i % 10,
                    seed_id=i,
                    chunk_id=i + 1000,
                    epoch=i // 10,
                    activation_variance=0.25,
                    dead_neuron_ratio=0.05,
                    avg_correlation=0.75,
                )
            )

        # Add system state packets
        for i in range(50):
            models.append(
                SystemStatePacket(
                    epoch=i,
                    total_seeds=100,
                    active_seeds=80 + i % 20,
                    training_loss=0.5 - i * 0.001,
                    validation_loss=0.6 - i * 0.001,
                    system_load=0.5 + (i % 5) * 0.1,
                    memory_usage=0.6 + (i % 4) * 0.1,
                )
            )

        # Test serialization performance
        start_time = time.perf_counter()
        for model in models:
            model.model_dump_json()
        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.3, (
            f"Mixed model serialization took {elapsed:.3f}s, expected <0.3s"
        )

        # Test throughput
        ops_per_second = len(models) / elapsed
        assert ops_per_second > 500, (
            f"Mixed model throughput {ops_per_second:.0f} ops/s, expected >500 ops/s"
        )

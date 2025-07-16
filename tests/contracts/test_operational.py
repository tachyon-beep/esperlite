"""
Tests for operational models validation logic.
"""

import pytest

from esper.contracts.operational import HealthSignal, SystemStatePacket


def test_health_signal_validation():
    """Test HealthSignal validation."""
    # Test negative layer_id
    with pytest.raises(ValueError, match="IDs must be non-negative"):
        HealthSignal(
            layer_id=-1,
            seed_id=5,
            chunk_id=100,
            epoch=10,
            activation_variance=0.5,
            dead_neuron_ratio=0.1,
            avg_correlation=0.8,
        )

    # Test negative seed_id
    with pytest.raises(ValueError, match="IDs must be non-negative"):
        HealthSignal(
            layer_id=1,
            seed_id=-5,
            chunk_id=100,
            epoch=10,
            activation_variance=0.5,
            dead_neuron_ratio=0.1,
            avg_correlation=0.8,
        )


def test_system_state_packet_validation():
    """Test SystemStatePacket validation."""
    # Test negative epoch
    with pytest.raises(ValueError, match="Epoch must be non-negative"):
        SystemStatePacket(
            epoch=-1,
            total_seeds=100,
            active_seeds=50,
            training_loss=0.5,
            validation_loss=0.6,
            system_load=0.7,
            memory_usage=0.8,
        )

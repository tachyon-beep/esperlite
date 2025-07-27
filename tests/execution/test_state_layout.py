"""
Unit tests for KasminaStateLayout.
"""

import pytest
import torch

from esper.execution.state_layout import KasminaStateLayout
from esper.execution.state_layout import SeedLifecycleState


class TestKasminaStateLayout:
    """Test cases for KasminaStateLayout."""

    # Removed trivial initialization test - it only checked constructor parameters
    # and default values without testing any meaningful behavior

    def test_get_active_seeds(self):
        """Test active seed detection."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Initially no active seeds
        active_seeds = layout.get_active_seeds()
        assert torch.all(~active_seeds)

        # Transition one seed to active
        layout.transition_seed_state(1, SeedLifecycleState.ACTIVE)
        active_seeds = layout.get_active_seeds()

        expected = torch.tensor([False, True, False, False])
        assert torch.all(active_seeds == expected)

    def test_get_dormant_seeds(self):
        """Test dormant seed detection."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Initially all dormant
        dormant_seeds = layout.get_dormant_seeds()
        assert torch.all(dormant_seeds)

        # Transition one seed to active
        layout.transition_seed_state(1, SeedLifecycleState.ACTIVE)
        dormant_seeds = layout.get_dormant_seeds()

        expected = torch.tensor([True, False, True, True])
        assert torch.all(dormant_seeds == expected)

    def test_transition_seed_state(self):
        """Test seed state transitions."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Test transition to active
        layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=12345)

        assert layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
        assert layout.active_kernel_id[0] == 12345
        assert layout.error_count[0] == 0
        assert not layout.fallback_active[0]

        # Test transition to error recovery
        layout.transition_seed_state(0, SeedLifecycleState.ERROR_RECOVERY)
        assert layout.lifecycle_states[0] == SeedLifecycleState.ERROR_RECOVERY

    def test_transition_seed_state_invalid_index(self):
        """Test error handling for invalid seed index."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        with pytest.raises(ValueError, match="Invalid seed index"):
            layout.transition_seed_state(10, SeedLifecycleState.ACTIVE)

        with pytest.raises(ValueError, match="Invalid seed index"):
            layout.transition_seed_state(-1, SeedLifecycleState.ACTIVE)

    def test_increment_error_count(self):
        """Test error count increment and fallback activation."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Increment error count
        count = layout.increment_error_count(0)
        assert count == 1
        assert layout.error_count[0] == 1
        assert not layout.fallback_active[0]

        # Second increment
        count = layout.increment_error_count(0)
        assert count == 2
        assert not layout.fallback_active[0]

        # Third increment should trigger fallback
        count = layout.increment_error_count(0)
        assert count == 3
        assert layout.fallback_active[0]
        assert layout.lifecycle_states[0] == SeedLifecycleState.ERROR_RECOVERY

    def test_increment_error_count_invalid_index(self):
        """Test error handling for invalid seed index in error count."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        with pytest.raises(ValueError, match="Invalid seed index"):
            layout.increment_error_count(10)

    def test_update_telemetry(self):
        """Test telemetry update."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Update telemetry
        layout.update_telemetry(0, latency_us=500, health_score=0.8)

        assert layout.exec_latency_us[0] == 500
        assert (
            abs(layout.health_accumulator[0].item() - 0.8) < 0.01
        )  # First update, no smoothing

        # Second update should apply smoothing
        layout.update_telemetry(0, latency_us=300, health_score=0.6)

        assert layout.exec_latency_us[0] == 300
        # Health should be smoothed: 0.9 * 0.8 + 0.1 * 0.6 = 0.78
        assert abs(layout.health_accumulator[0].item() - 0.78) < 0.01

    def test_update_telemetry_latency_clamping(self):
        """Test latency clamping to uint16 range."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Test large latency value
        layout.update_telemetry(0, latency_us=100000, health_score=0.5)
        assert layout.exec_latency_us[0] == 65535  # Clamped to uint16 max

    def test_update_telemetry_invalid_index(self):
        """Test error handling for invalid seed index in telemetry update."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        with pytest.raises(ValueError, match="Invalid seed index"):
            layout.update_telemetry(10, latency_us=100, health_score=0.5)

    def test_get_stats(self):
        """Test statistics collection."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Set up some state
        layout.transition_seed_state(0, SeedLifecycleState.ACTIVE)
        layout.transition_seed_state(1, SeedLifecycleState.LOADING)
        layout.increment_error_count(2)
        layout.update_telemetry(0, latency_us=100, health_score=0.9)

        stats = layout.get_stats()

        assert stats["num_seeds"] == 4
        assert stats["active_seeds"] == 1
        assert stats["loading_seeds"] == 1
        assert stats["dormant_seeds"] == 2
        assert stats["error_recovery_seeds"] == 0
        assert stats["total_errors"] == 1
        assert stats["fallback_active_count"] == 0
        assert stats["avg_health"] > 0  # Should be positive

    def test_reset_seed(self):
        """Test seed reset functionality."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        # Set up seed with some state
        layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=12345)
        layout.alpha_blend[0] = 0.5
        layout.update_telemetry(0, latency_us=100, health_score=0.8)
        layout.increment_error_count(0)

        # Reset the seed
        layout.reset_seed(0)

        # Check all values are reset
        assert layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert layout.active_kernel_id[0] == 0
        assert abs(layout.alpha_blend[0].item() - 0.0) < 0.01
        assert abs(layout.health_accumulator[0].item() - 0.0) < 0.01
        assert layout.last_update_epoch[0] == 0
        assert layout.exec_latency_us[0] == 0
        assert layout.error_count[0] == 0
        assert not layout.fallback_active[0]

    def test_reset_seed_invalid_index(self):
        """Test error handling for invalid seed index in reset."""
        layout = KasminaStateLayout(4, torch.device("cpu"))

        with pytest.raises(ValueError, match="Invalid seed index"):
            layout.reset_seed(10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_initialization(self):
        """Test initialization on GPU."""
        device = torch.device("cuda")
        layout = KasminaStateLayout(4, device)

        assert layout.device == device
        assert layout.lifecycle_states.is_cuda
        assert layout.alpha_blend.is_cuda
        assert layout.health_accumulator.is_cuda

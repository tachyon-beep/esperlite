"""
Tests for StateTensor module.

Validates GPU-resident state management for massively parallel seeds.
"""

import pytest
import torch
# Removed typing imports - not needed after removing test_initialization

from esper.morphogenetic_v2.kasmina.state_tensor import StateTensor
from esper.morphogenetic_v2.kasmina.logical_seed import SeedLifecycle, LogicalSeedState


class TestStateTensor:
    """Test suite for StateTensor functionality."""

    @pytest.fixture
    def state_tensor(self):
        """Create a StateTensor instance for testing."""
        num_seeds = 100
        return StateTensor(num_seeds)

    @pytest.fixture
    def gpu_state_tensor(self):
        """Create GPU StateTensor if available."""
        if torch.cuda.is_available():
            return StateTensor(100, device=torch.device("cuda"))
        return None

    # Removed test_initialization - only checked constructor parameters

    def test_lifecycle_state_management(self, state_tensor):
        """Test lifecycle state operations."""
        # Get initial states
        states = state_tensor.get_lifecycle_states()
        assert (states == SeedLifecycle.DORMANT).all()

        # Set single seed state
        state_tensor.set_lifecycle_state(5, SeedLifecycle.LOADING)
        assert state_tensor.get_lifecycle_states()[5] == SeedLifecycle.LOADING
        assert state_tensor.state_tensor[5, StateTensor.EPOCHS_IN_STATE_IDX] == 0

        # Batch set states
        seed_ids = torch.tensor([10, 15, 20])
        state_tensor.batch_set_lifecycle_state(seed_ids, SeedLifecycle.ACTIVE)
        states = state_tensor.get_lifecycle_states()
        assert states[10] == SeedLifecycle.ACTIVE
        assert states[15] == SeedLifecycle.ACTIVE
        assert states[20] == SeedLifecycle.ACTIVE

    def test_active_dormant_masks(self, state_tensor):
        """Test active and dormant seed masks."""
        # Initially all dormant
        assert not state_tensor.get_active_seeds().any()
        assert state_tensor.get_dormant_seeds().all()

        # Activate some seeds
        state_tensor.set_lifecycle_state(0, SeedLifecycle.LOADING)
        state_tensor.set_lifecycle_state(5, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(10, SeedLifecycle.ERROR_RECOVERY)

        active_mask = state_tensor.get_active_seeds()
        assert active_mask[0]  # LOADING is active
        assert active_mask[5]  # ACTIVE is active
        assert not active_mask[10]  # ERROR_RECOVERY is not active

        dormant_mask = state_tensor.get_dormant_seeds()
        assert not dormant_mask[0]
        assert not dormant_mask[5]
        assert not dormant_mask[10]

    def test_seed_state_retrieval(self, state_tensor):
        """Test getting complete state for a seed."""
        # Setup seed state
        seed_id = 7
        state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.ACTIVE)
        state_tensor.set_blueprint(seed_id, 42, grafting_strategy=2)
        state_tensor.health_scores[seed_id] = 0.85
        state_tensor.error_counts[seed_id] = 1
        state_tensor.alpha_blend[seed_id] = 0.75

        # Get state
        state = state_tensor.get_seed_state(seed_id)

        assert state["lifecycle_state"] == SeedLifecycle.ACTIVE
        assert state["blueprint_id"] == 42
        assert state["grafting_strategy"] == 2
        assert abs(state["health_score"] - 0.85) < 0.01
        assert state["error_count"] == 1
        assert state["alpha_blend"] == 0.75
        assert state["epochs_in_state"] == 0

    def test_blueprint_management(self, state_tensor):
        """Test blueprint and grafting strategy setting."""
        seed_id = 3
        blueprint_id = 123
        grafting_strategy = 1

        state_tensor.set_blueprint(seed_id, blueprint_id, grafting_strategy)

        assert state_tensor.state_tensor[seed_id, StateTensor.BLUEPRINT_ID_IDX] == blueprint_id
        assert state_tensor.state_tensor[seed_id, StateTensor.GRAFTING_STRATEGY_IDX] == grafting_strategy

    def test_epoch_increment(self, state_tensor):
        """Test epoch counter increment."""
        # Set some seeds to different states
        state_tensor.set_lifecycle_state(0, SeedLifecycle.LOADING)
        state_tensor.set_lifecycle_state(1, SeedLifecycle.ACTIVE)

        # Increment epochs
        state_tensor.increment_epochs()

        # Check all epochs incremented
        epochs = state_tensor.state_tensor[:, StateTensor.EPOCHS_IN_STATE_IDX]
        assert (epochs == 1).all()

        # Increment again
        state_tensor.increment_epochs()
        assert (epochs == 2).all()

    def test_health_score_updates(self, state_tensor):
        """Test health score management."""
        seed_ids = torch.tensor([1, 5, 10])
        health_scores = torch.tensor([0.9, 0.5, 0.3])

        state_tensor.update_health_scores(seed_ids, health_scores.to(state_tensor.device))

        assert state_tensor.health_scores[1] == 0.9
        assert state_tensor.health_scores[5] == 0.5
        assert state_tensor.health_scores[10] == 0.3

        # Find unhealthy seeds
        unhealthy = state_tensor.find_unhealthy_seeds(threshold=0.7)
        assert 5 in unhealthy
        assert 10 in unhealthy
        assert 1 not in unhealthy

    def test_error_counting(self, state_tensor):
        """Test error count management."""
        seed_id = 8

        # Increment errors
        count1 = state_tensor.increment_error_count(seed_id)
        assert count1 == 1

        count2 = state_tensor.increment_error_count(seed_id)
        assert count2 == 2

        count3 = state_tensor.increment_error_count(seed_id)
        assert count3 == 3

        assert state_tensor.error_counts[seed_id] == 3

    def test_alpha_blend_updates(self, state_tensor):
        """Test alpha blending factor updates."""
        seed_ids = torch.tensor([2, 4, 6])
        alphas = torch.tensor([0.0, 0.5, 1.0])

        state_tensor.update_alpha_blend(seed_ids, alphas.to(state_tensor.device))

        assert state_tensor.alpha_blend[2] == 0.0
        assert state_tensor.alpha_blend[4] == 0.5
        assert state_tensor.alpha_blend[6] == 1.0

    def test_telemetry_accumulation(self, state_tensor):
        """Test telemetry statistics accumulation."""
        seed_id = 12

        # Accumulate some statistics
        state_tensor.accumulate_telemetry(seed_id, chunk_sum=10.0, chunk_sum_sq=50.0, count=4)
        state_tensor.accumulate_telemetry(seed_id, chunk_sum=6.0, chunk_sum_sq=18.0, count=2)

        # Get statistics
        mean, variance = state_tensor.get_telemetry_stats(seed_id)

        # Total: sum=16, sum_sq=68, count=6
        # Mean = 16/6 = 2.667
        # E[X^2] = 68/6 = 11.333
        # Var = E[X^2] - E[X]^2 = 11.333 - 7.111 = 4.222
        assert abs(mean - 2.667) < 0.01
        assert abs(variance - 4.222) < 0.01

        # Test reset
        state_tensor.reset_telemetry()
        mean2, var2 = state_tensor.get_telemetry_stats(seed_id)
        assert mean2 == 0.0
        assert var2 == 0.0

    def test_state_distribution(self, state_tensor):
        """Test getting state distribution counts."""
        # Set various states
        state_tensor.set_lifecycle_state(0, SeedLifecycle.LOADING)
        state_tensor.set_lifecycle_state(1, SeedLifecycle.LOADING)
        state_tensor.set_lifecycle_state(2, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(3, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(4, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(5, SeedLifecycle.ERROR_RECOVERY)
        state_tensor.set_lifecycle_state(6, SeedLifecycle.FOSSILIZED)

        dist = state_tensor.get_state_distribution()

        assert dist["DORMANT"] == 93  # 100 - 7 set above
        assert dist["LOADING"] == 2
        assert dist["ACTIVE"] == 3
        assert dist["ERROR_RECOVERY"] == 1
        assert dist["FOSSILIZED"] == 1

    def test_find_seeds_by_state(self, state_tensor):
        """Test finding seeds in specific states."""
        # Set some states
        state_tensor.set_lifecycle_state(10, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(20, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(30, SeedLifecycle.LOADING)

        # Find active seeds
        active_seeds = state_tensor.find_seeds_by_state(SeedLifecycle.ACTIVE)
        assert len(active_seeds) == 2
        assert 10 in active_seeds
        assert 20 in active_seeds

        # Find loading seeds
        loading_seeds = state_tensor.find_seeds_by_state(SeedLifecycle.LOADING)
        assert len(loading_seeds) == 1
        assert 30 in loading_seeds

    def test_grafting_ramp(self, state_tensor):
        """Test grafting ramp application."""
        # Set up loading seeds with different epochs
        loading_seeds = [5, 10, 15]
        for i, seed_id in enumerate(loading_seeds):
            state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.LOADING)
            # Manually set epochs for testing
            state_tensor.state_tensor[seed_id, StateTensor.EPOCHS_IN_STATE_IDX] = i * 10

        # Apply grafting ramp
        state_tensor.apply_grafting_ramp(ramp_duration=30)

        # Check alpha values
        assert abs(state_tensor.alpha_blend[5] - 0.0) < 0.01  # 0/30
        assert abs(state_tensor.alpha_blend[10] - 0.333) < 0.01  # 10/30
        assert abs(state_tensor.alpha_blend[15] - 0.667) < 0.01  # 20/30

        # Non-loading seeds should not be affected
        assert state_tensor.alpha_blend[0] == 0.0

    def test_transition_detection(self, state_tensor):
        """Test finding seeds ready for transitions."""
        # Set up seeds in various states
        state_tensor.set_lifecycle_state(1, SeedLifecycle.LOADING)
        state_tensor.alpha_blend[1] = 1.0  # Ramp complete

        state_tensor.set_lifecycle_state(2, SeedLifecycle.LOADING)
        state_tensor.alpha_blend[2] = 0.5  # Ramp incomplete

        state_tensor.set_lifecycle_state(3, SeedLifecycle.ERROR_RECOVERY)
        state_tensor.state_tensor[3, StateTensor.EPOCHS_IN_STATE_IDX] = 150  # Past cooldown

        state_tensor.set_lifecycle_state(4, SeedLifecycle.ERROR_RECOVERY)
        state_tensor.state_tensor[4, StateTensor.EPOCHS_IN_STATE_IDX] = 50  # Before cooldown

        # Get transitions
        transitions = state_tensor.transition_ready_seeds()

        assert 1 in transitions["loading_to_active"]
        assert 2 not in transitions["loading_to_active"]
        assert 3 in transitions["error_to_dormant"]
        assert 4 not in transitions["error_to_dormant"]

    def test_summary_statistics(self, state_tensor):
        """Test summary statistics generation."""
        # Set up some state
        state_tensor.set_lifecycle_state(0, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(1, SeedLifecycle.ACTIVE)
        state_tensor.health_scores[0] = 0.8
        state_tensor.health_scores[1] = 0.6
        state_tensor.error_counts[5] = 10
        state_tensor.alpha_blend[0] = 0.5
        state_tensor.alpha_blend[1] = 0.7

        stats = state_tensor.get_summary_stats()

        assert stats["total_seeds"] == 100
        assert stats["active_count"] == 2
        assert stats["max_error_count"] == 10
        assert abs(stats["avg_alpha_blend"] - 0.6) < 0.01  # (0.5 + 0.7) / 2

    def test_to_logical_seeds(self, state_tensor):
        """Test conversion to LogicalSeedState objects."""
        # Set up some seeds
        state_tensor.set_lifecycle_state(0, SeedLifecycle.ACTIVE)
        state_tensor.set_blueprint(0, 42, 1)
        state_tensor.health_scores[0] = 0.9

        # Convert
        layer_id = "test_layer"
        chunk_sizes = [10] * state_tensor.num_seeds
        logical_seeds = state_tensor.to_logical_seeds(layer_id, chunk_sizes)

        # Check conversion
        assert len(logical_seeds) == state_tensor.num_seeds

        seed0 = logical_seeds[0]
        assert isinstance(seed0, LogicalSeedState)
        assert seed0.layer_id == layer_id
        assert seed0.seed_id == 0
        assert seed0.lifecycle_state == SeedLifecycle.ACTIVE
        assert seed0.blueprint_id == 42
        assert seed0.grafting_strategy == 1
        assert abs(seed0.health_score - 0.9) < 0.01

    def test_checkpointing(self, state_tensor):
        """Test checkpoint save/load."""
        # Set up state
        state_tensor.set_lifecycle_state(5, SeedLifecycle.ACTIVE)
        state_tensor.health_scores[5] = 0.75
        state_tensor.error_counts[5] = 2
        state_tensor.alpha_blend[5] = 0.5

        # Save checkpoint
        checkpoint = state_tensor.save_checkpoint()

        # Create new state tensor and load
        new_st = StateTensor(state_tensor.num_seeds)
        new_st.load_checkpoint(checkpoint)

        # Verify state restored
        assert new_st.get_lifecycle_states()[5] == SeedLifecycle.ACTIVE
        assert new_st.health_scores[5] == 0.75
        assert new_st.error_counts[5] == 2
        assert new_st.alpha_blend[5] == 0.5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_operations(self, gpu_state_tensor):
        """Test operations on GPU."""
        st = gpu_state_tensor

        # Verify on GPU
        assert st.device.type == "cuda"
        assert st.state_tensor.device.type == "cuda"

        # Test operations
        st.set_lifecycle_state(0, SeedLifecycle.ACTIVE)
        st.increment_epochs()

        seed_ids = torch.tensor([1, 2, 3], device="cuda")
        health_scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
        st.update_health_scores(seed_ids, health_scores)

        # Get statistics
        stats = st.get_summary_stats()
        assert stats["active_count"] == 1

        # Move to CPU
        st_cpu = st.to(torch.device("cpu"))
        assert st_cpu.device.type == "cpu"
        assert st_cpu.state_tensor.device.type == "cpu"

    def test_error_handling(self, state_tensor):
        """Test error conditions."""
        # Invalid seed ID
        with pytest.raises(ValueError):
            state_tensor.get_seed_state(1000)

        # Edge case - empty telemetry
        mean, var = state_tensor.get_telemetry_stats(0)
        assert mean == 0.0
        assert var == 0.0

    def test_concurrent_modifications(self, state_tensor):
        """Test concurrent-style modifications."""
        # Simulate multiple updates to same seed
        seed_id = 10

        # Multiple state changes
        state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.LOADING)
        state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.ACTIVE)
        state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.ERROR_RECOVERY)

        # Verify final state
        assert state_tensor.get_lifecycle_states()[seed_id] == SeedLifecycle.ERROR_RECOVERY

        # Verify epoch counter reset on state change
        assert state_tensor.state_tensor[seed_id, StateTensor.EPOCHS_IN_STATE_IDX] == 0
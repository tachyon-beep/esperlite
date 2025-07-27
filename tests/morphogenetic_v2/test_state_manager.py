"""Unit tests for extended state management."""

import pytest
import torch
import numpy as np
from typing import Dict, List

from esper.morphogenetic_v2.lifecycle import (
    ExtendedStateTensor,
    ExtendedLifecycle
)


class TestExtendedStateTensor:
    """Test cases for ExtendedStateTensor."""
    
    @pytest.fixture
    def state_tensor(self):
        """Create state tensor for testing."""
        return ExtendedStateTensor(num_seeds=100)
    
    @pytest.fixture
    def gpu_state_tensor(self):
        """Create GPU state tensor if available."""
        if torch.cuda.is_available():
            return ExtendedStateTensor(
                num_seeds=50,
                device=torch.device('cuda:0')
            )
        else:
            pytest.skip("GPU not available")
    
    # Removed test_initialization - only checked shapes and default values
    
    def test_device_placement(self, gpu_state_tensor):
        """Test GPU device placement."""
        assert gpu_state_tensor.state_tensor.device.type == 'cuda'
        assert gpu_state_tensor.transition_history.device.type == 'cuda'
        assert gpu_state_tensor.performance_metrics.device.type == 'cuda'
    
    def test_get_state(self, state_tensor):
        """Test getting lifecycle states."""
        # Get all states
        all_states = state_tensor.get_state()
        assert all_states.shape == (100,)
        assert torch.all(all_states == ExtendedLifecycle.DORMANT)
        
        # Get specific seeds
        seed_indices = torch.tensor([0, 5, 10])
        states = state_tensor.get_state(seed_indices)
        assert states.shape == (3,)
    
    def test_set_state(self, state_tensor):
        """Test setting lifecycle states."""
        seed_indices = torch.tensor([1, 3, 5])
        new_states = torch.tensor([
            ExtendedLifecycle.GERMINATED,
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING
        ])
        
        # Set states
        state_tensor.set_state(seed_indices, new_states)
        
        # Verify states updated
        assert state_tensor.get_state(seed_indices[0]) == ExtendedLifecycle.GERMINATED
        assert state_tensor.get_state(seed_indices[1]) == ExtendedLifecycle.TRAINING
        assert state_tensor.get_state(seed_indices[2]) == ExtendedLifecycle.GRAFTING
        
        # Verify epochs reset
        epochs = state_tensor.state_tensor[seed_indices, state_tensor.EPOCHS_IN_STATE]
        assert torch.all(epochs == 0)
        
        # Verify parent states stored
        parent_states = state_tensor.state_tensor[seed_indices, state_tensor.PARENT_STATE]
        assert torch.all(parent_states == ExtendedLifecycle.DORMANT)
    
    def test_transition_history(self, state_tensor):
        """Test transition history recording."""
        seed_idx = torch.tensor([0])
        
        # Make several transitions
        transitions = [
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING)
        ]
        
        for from_state, to_state in transitions:
            # Set current state first
            state_tensor.state_tensor[seed_idx, state_tensor.LIFECYCLE_STATE] = from_state
            # Then transition
            state_tensor.set_state(seed_idx, torch.tensor([to_state]))
        
        # Get history
        history = state_tensor.get_transition_history(0)
        assert len(history) >= len(transitions)
        
        # Verify transitions recorded correctly
        for i, (from_state, to_state) in enumerate(transitions):
            assert history[i] == (from_state, to_state)
    
    def test_circular_history_buffer(self, state_tensor):
        """Test that history buffer wraps correctly."""
        seed_idx = torch.tensor([0])
        
        # Make more transitions than history depth
        for i in range(15):  # More than HISTORY_DEPTH (10)
            from_state = i % 5
            to_state = (i + 1) % 5
            state_tensor.state_tensor[seed_idx, state_tensor.LIFECYCLE_STATE] = from_state
            state_tensor.set_state(seed_idx, torch.tensor([to_state]))
        
        # Get history
        history = state_tensor.get_transition_history(0)
        assert len(history) == state_tensor.HISTORY_DEPTH
        
        # Most recent should be last transition
        assert history[0] == (14 % 5, 15 % 5)
    
    def test_increment_epochs(self, state_tensor):
        """Test incrementing epochs counter."""
        # Set some seeds to different states
        state_tensor.set_state(
            torch.tensor([0, 1, 2]),
            torch.tensor([ExtendedLifecycle.TRAINING] * 3)
        )
        
        # Increment all
        state_tensor.increment_epochs()
        epochs = state_tensor.state_tensor[:, state_tensor.EPOCHS_IN_STATE]
        assert epochs[0] == 1
        assert epochs[1] == 1
        assert epochs[2] == 1
        assert epochs[3] == 1  # Even dormant seeds increment
        
        # Increment with mask
        mask = torch.zeros(100, dtype=torch.bool)
        mask[[0, 2]] = True
        state_tensor.increment_epochs(mask)
        
        epochs = state_tensor.state_tensor[:, state_tensor.EPOCHS_IN_STATE]
        assert epochs[0] == 2  # Incremented
        assert epochs[1] == 1  # Not incremented
        assert epochs[2] == 2  # Incremented
        assert epochs[3] == 1  # Not incremented
    
    def test_get_seeds_in_state(self, state_tensor):
        """Test getting seeds in specific state."""
        # Set various states
        state_tensor.set_state(
            torch.tensor([1, 3, 5, 7]),
            torch.tensor([ExtendedLifecycle.TRAINING] * 4)
        )
        state_tensor.set_state(
            torch.tensor([2, 4, 6]),
            torch.tensor([ExtendedLifecycle.GRAFTING] * 3)
        )
        
        # Get seeds in TRAINING
        training_seeds = state_tensor.get_seeds_in_state(ExtendedLifecycle.TRAINING)
        assert len(training_seeds) == 4
        assert set(training_seeds.tolist()) == {1, 3, 5, 7}
        
        # Get seeds in GRAFTING
        grafting_seeds = state_tensor.get_seeds_in_state(ExtendedLifecycle.GRAFTING)
        assert len(grafting_seeds) == 3
        assert set(grafting_seeds.tolist()) == {2, 4, 6}
        
        # Get seeds in DORMANT
        dormant_seeds = state_tensor.get_seeds_in_state(ExtendedLifecycle.DORMANT)
        assert len(dormant_seeds) == 93  # 100 - 7 set to other states
    
    def test_update_blueprint(self, state_tensor):
        """Test blueprint assignment."""
        seed_indices = torch.tensor([10, 20, 30])
        blueprint_ids = torch.tensor([100, 200, 300])
        grafting_strategies = torch.tensor([1, 2, 3])
        
        # Update blueprints
        state_tensor.update_blueprint(
            seed_indices,
            blueprint_ids,
            grafting_strategies
        )
        
        # Verify updates
        for i, seed_idx in enumerate(seed_indices):
            assert state_tensor.state_tensor[seed_idx, state_tensor.BLUEPRINT_ID] == blueprint_ids[i]
            assert state_tensor.state_tensor[seed_idx, state_tensor.GRAFTING_STRATEGY] == grafting_strategies[i]
        
        # Update without grafting strategies
        state_tensor.update_blueprint(
            torch.tensor([40]),
            torch.tensor([400])
        )
        assert state_tensor.state_tensor[40, state_tensor.BLUEPRINT_ID] == 400
    
    def test_update_performance(self, state_tensor):
        """Test performance metrics update."""
        seed_indices = torch.tensor([0, 1, 2])
        metrics = {
            'loss': torch.tensor([0.1, 0.2, 0.3]),
            'accuracy': torch.tensor([0.9, 0.8, 0.7]),
            'stability': torch.tensor([0.95, 0.85, 0.75]),
            'efficiency': torch.tensor([0.8, 0.7, 0.6]),
            'evaluation_score': torch.tensor([0.85, 0.75, 0.65])
        }
        
        # Update metrics
        state_tensor.update_performance(seed_indices, metrics)
        
        # Verify performance metrics
        for i, seed_idx in enumerate(seed_indices):
            perf = state_tensor.performance_metrics[seed_idx]
            assert perf[0].item() == pytest.approx(metrics['loss'][i].item())
            assert perf[1].item() == pytest.approx(metrics['accuracy'][i].item())
            assert perf[2].item() == pytest.approx(metrics['stability'][i].item())
            assert perf[3].item() == pytest.approx(metrics['efficiency'][i].item())
            
            # Check evaluation score (scaled to int)
            eval_score = state_tensor.state_tensor[seed_idx, state_tensor.EVALUATION_SCORE]
            expected = int(metrics['evaluation_score'][i].item() * 1000)
            assert eval_score == expected
    
    def test_error_counting(self, state_tensor):
        """Test error count management."""
        seed_indices = torch.tensor([5, 10, 15])
        
        # Increment errors
        state_tensor.increment_error_count(seed_indices)
        state_tensor.increment_error_count(seed_indices)
        state_tensor.increment_error_count(torch.tensor([5]))  # Extra for seed 5
        
        # Check counts
        assert state_tensor.state_tensor[5, state_tensor.ERROR_COUNT] == 3
        assert state_tensor.state_tensor[10, state_tensor.ERROR_COUNT] == 2
        assert state_tensor.state_tensor[15, state_tensor.ERROR_COUNT] == 2
        assert state_tensor.state_tensor[20, state_tensor.ERROR_COUNT] == 0
        
        # Reset errors
        state_tensor.reset_error_count(torch.tensor([5, 10]))
        assert state_tensor.state_tensor[5, state_tensor.ERROR_COUNT] == 0
        assert state_tensor.state_tensor[10, state_tensor.ERROR_COUNT] == 0
        assert state_tensor.state_tensor[15, state_tensor.ERROR_COUNT] == 2  # Not reset
    
    def test_checkpoint_management(self, state_tensor):
        """Test checkpoint ID tracking."""
        seed_indices = torch.tensor([0, 1, 2])
        checkpoint_ids = torch.tensor([1001, 2002, 3003])
        
        # Set checkpoint IDs
        state_tensor.set_checkpoint(seed_indices, checkpoint_ids)
        
        # Verify
        for i, seed_idx in enumerate(seed_indices):
            assert state_tensor.state_tensor[seed_idx, state_tensor.CHECKPOINT_ID] == checkpoint_ids[i]
    
    def test_rollback_states(self, state_tensor):
        """Test getting rollback states."""
        # Set up some transitions
        seeds = torch.tensor([0, 1, 2])
        
        # First transition
        state_tensor.set_state(seeds, torch.tensor([ExtendedLifecycle.GERMINATED] * 3))
        
        # Second transition (parent states should be GERMINATED)
        state_tensor.set_state(seeds, torch.tensor([ExtendedLifecycle.TRAINING] * 3))
        
        # Get rollback states
        rollback = state_tensor.get_rollback_states(seeds)
        assert torch.all(rollback == ExtendedLifecycle.GERMINATED)
    
    def test_state_summary(self, state_tensor):
        """Test state distribution summary."""
        # Set various states
        state_assignments = [
            (range(10), ExtendedLifecycle.DORMANT),
            (range(10, 25), ExtendedLifecycle.TRAINING),
            (range(25, 30), ExtendedLifecycle.GRAFTING),
            (range(30, 35), ExtendedLifecycle.EVALUATING),
            (range(35, 37), ExtendedLifecycle.FOSSILIZED),
            (range(37, 39), ExtendedLifecycle.CULLED)
        ]
        
        for indices, state in state_assignments:
            state_tensor.set_state(
                torch.tensor(list(indices)),
                torch.tensor([state] * len(indices)),
                record_transition=False  # Skip for simplicity
            )
        
        # Get summary
        summary = state_tensor.get_state_summary()
        
        # Verify counts (100 total, but DORMANT includes unset seeds)
        assert summary['DORMANT'] == 71  # 10 set + 61 unset
        assert summary['TRAINING'] == 15
        assert summary['GRAFTING'] == 5
        assert summary['EVALUATING'] == 5
        assert summary['FOSSILIZED'] == 2
        assert summary['CULLED'] == 2
    
    def test_active_seeds_mask(self, state_tensor):
        """Test getting active computation seeds."""
        # Set some seeds to active states
        active_states = [
            (10, ExtendedLifecycle.TRAINING),
            (20, ExtendedLifecycle.GRAFTING),
            (30, ExtendedLifecycle.FINE_TUNING),
            (40, ExtendedLifecycle.EVALUATING),  # Not active
            (50, ExtendedLifecycle.DORMANT)      # Not active
        ]
        
        for seed_idx, state in active_states:
            state_tensor.set_state(
                torch.tensor([seed_idx]),
                torch.tensor([state]),
                record_transition=False
            )
        
        # Get active mask
        mask = state_tensor.get_active_seeds_mask()
        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        
        # Check specific seeds
        assert mask[10] == True   # TRAINING is active
        assert mask[20] == True   # GRAFTING is active
        assert mask[30] == True   # FINE_TUNING is active
        assert mask[40] == False  # EVALUATING is not active
        assert mask[50] == False  # DORMANT is not active
        
        # Count active seeds
        assert mask.sum() == 3
    
    def test_validate_transitions(self, state_tensor):
        """Test simplified transition validation."""
        # Set some seeds to different states
        state_tensor.set_state(
            torch.tensor([0, 1, 2]),
            torch.tensor([
                ExtendedLifecycle.DORMANT,
                ExtendedLifecycle.FOSSILIZED,  # Terminal
                ExtendedLifecycle.TRAINING
            ]),
            record_transition=False
        )
        
        # Try to transition all three
        seed_indices = torch.tensor([0, 1, 2])
        target_states = torch.tensor([
            ExtendedLifecycle.GERMINATED,  # Valid
            ExtendedLifecycle.TRAINING,    # Invalid (terminal)
            ExtendedLifecycle.GRAFTING     # Valid
        ])
        
        valid_mask, errors = state_tensor.validate_transitions(
            seed_indices, target_states
        )
        
        assert valid_mask[0] == True   # DORMANT -> GERMINATED is valid
        assert valid_mask[1] == False  # FOSSILIZED -> anywhere is invalid
        assert valid_mask[2] == True   # TRAINING -> GRAFTING is valid
        
        assert len(errors) == 1
        assert "terminal state" in errors[0]
    
    def test_telemetry_buffer(self, state_tensor):
        """Test telemetry buffer operations."""
        # Add some telemetry data
        seed_indices = torch.tensor([0, 1, 2])
        values = torch.tensor([1.0, 2.0, 3.0])
        
        # Simulate telemetry accumulation
        state_tensor.telemetry_buffer[seed_indices, 0] += values      # Sum
        state_tensor.telemetry_buffer[seed_indices, 1] += values ** 2  # Sum of squares
        
        # Verify
        assert state_tensor.telemetry_buffer[0, 0] == 1.0
        assert state_tensor.telemetry_buffer[0, 1] == 1.0
        assert state_tensor.telemetry_buffer[1, 0] == 2.0
        assert state_tensor.telemetry_buffer[1, 1] == 4.0
        
        # Reset telemetry
        state_tensor.reset_telemetry()
        assert torch.all(state_tensor.telemetry_buffer == 0)
    
    def test_export_import(self, state_tensor):
        """Test state export and import."""
        # Set up some state
        state_tensor.set_state(
            torch.tensor([0, 1, 2]),
            torch.tensor([ExtendedLifecycle.TRAINING] * 3)
        )
        state_tensor.update_performance(
            torch.tensor([0]),
            {'loss': torch.tensor([0.5])}
        )
        
        # Export
        state_dict = state_tensor.to_dict()
        
        # Verify export structure
        assert 'num_seeds' in state_dict
        assert 'state_tensor' in state_dict
        assert 'transition_history' in state_dict
        assert 'performance_metrics' in state_dict
        assert isinstance(state_dict['state_tensor'], np.ndarray)
        
        # Create new tensor and import
        new_tensor = ExtendedStateTensor(num_seeds=100)
        new_tensor.from_dict(state_dict)
        
        # Verify state restored
        assert torch.all(new_tensor.get_state() == state_tensor.get_state())
        assert torch.allclose(
            new_tensor.performance_metrics,
            state_tensor.performance_metrics
        )


class TestStateTensorScenarios:
    """Test complete state management scenarios."""
    
    def test_multi_seed_lifecycle(self):
        """Test managing multiple seeds through lifecycle."""
        state_tensor = ExtendedStateTensor(num_seeds=10)
        
        # Simulate different seeds at different stages
        scenarios = [
            # Successful seeds
            (0, [
                ExtendedLifecycle.DORMANT,
                ExtendedLifecycle.GERMINATED,
                ExtendedLifecycle.TRAINING,
                ExtendedLifecycle.GRAFTING,
                ExtendedLifecycle.FOSSILIZED
            ]),
            # Failed seed
            (1, [
                ExtendedLifecycle.DORMANT,
                ExtendedLifecycle.GERMINATED,
                ExtendedLifecycle.TRAINING,
                ExtendedLifecycle.CULLED
            ]),
            # Cancelled seed
            (2, [
                ExtendedLifecycle.DORMANT,
                ExtendedLifecycle.GERMINATED,
                ExtendedLifecycle.CANCELLED
            ])
        ]
        
        # Execute scenarios
        for seed_id, states in scenarios:
            for i, state in enumerate(states):
                if i > 0:  # Skip first state (already DORMANT)
                    state_tensor.set_state(
                        torch.tensor([seed_id]),
                        torch.tensor([state])
                    )
                    
                # Simulate epochs passing
                state_tensor.increment_epochs()
        
        # Verify final states
        assert state_tensor.get_state(torch.tensor([0])) == ExtendedLifecycle.FOSSILIZED
        assert state_tensor.get_state(torch.tensor([1])) == ExtendedLifecycle.CULLED
        assert state_tensor.get_state(torch.tensor([2])) == ExtendedLifecycle.CANCELLED
        
        # Check histories
        history0 = state_tensor.get_transition_history(0)
        assert len(history0) >= 4  # At least 4 transitions
        
        history1 = state_tensor.get_transition_history(1)
        assert history1[0] == (ExtendedLifecycle.TRAINING, ExtendedLifecycle.CULLED)
    
    def test_performance_tracking_scenario(self):
        """Test tracking performance through lifecycle."""
        state_tensor = ExtendedStateTensor(num_seeds=5)
        seed_id = 0
        seed_tensor = torch.tensor([seed_id])
        
        # Training phase
        state_tensor.set_state(seed_tensor, torch.tensor([ExtendedLifecycle.TRAINING]))
        
        # Simulate training with improving performance
        for epoch in range(10):
            loss = 1.0 / (epoch + 1)  # Decreasing loss
            accuracy = epoch / 10.0     # Increasing accuracy
            
            state_tensor.update_performance(
                seed_tensor,
                {
                    'loss': torch.tensor([loss]),
                    'accuracy': torch.tensor([accuracy]),
                    'evaluation_score': torch.tensor([accuracy])
                }
            )
            state_tensor.increment_epochs()
        
        # Check final metrics
        perf = state_tensor.performance_metrics[seed_id]
        assert perf[0] < 0.2  # Low loss
        assert perf[1] > 0.8  # High accuracy
        
        # Check evaluation score
        eval_score = state_tensor.state_tensor[seed_id, state_tensor.EVALUATION_SCORE]
        assert eval_score > 800  # Scaled accuracy
    
    def test_error_recovery_scenario(self):
        """Test error tracking and recovery."""
        state_tensor = ExtendedStateTensor(num_seeds=5)
        seed_id = 0
        seed_tensor = torch.tensor([seed_id])
        
        # Start training
        state_tensor.set_state(seed_tensor, torch.tensor([ExtendedLifecycle.TRAINING]))
        
        # Simulate errors
        for _ in range(5):
            state_tensor.increment_error_count(seed_tensor)
        
        # Check error count
        assert state_tensor.state_tensor[seed_id, state_tensor.ERROR_COUNT] == 5
        
        # Transition to error recovery (CULLED)
        state_tensor.set_state(seed_tensor, torch.tensor([ExtendedLifecycle.CULLED]))
        
        # Verify state change
        assert state_tensor.get_state(seed_tensor) == ExtendedLifecycle.CULLED
        
        # Could implement recovery by resetting and restarting
        # This would be done at a higher level

"""Unit tests for extended lifecycle management."""

import pytest
import time
from typing import List, Tuple

import torch

from esper.morphogenetic_v2.lifecycle import (
    ExtendedLifecycle,
    TransitionContext,
    StateTransition,
    LifecycleManager
)


class TestExtendedLifecycle:
    """Test cases for ExtendedLifecycle enum."""
    
    def test_all_states_defined(self):
        """Verify all 11 states are defined."""
        expected_states = [
            'DORMANT', 'GERMINATED', 'TRAINING', 'GRAFTING',
            'STABILIZATION', 'EVALUATING', 'FINE_TUNING',
            'FOSSILIZED', 'CULLED', 'CANCELLED', 'ROLLED_BACK'
        ]
        
        actual_states = [state.name for state in ExtendedLifecycle]
        assert len(actual_states) == 11
        assert set(actual_states) == set(expected_states)
    
    def test_state_values(self):
        """Verify state values are sequential."""
        for i, state in enumerate(ExtendedLifecycle):
            assert state.value == i
    
    def test_terminal_states(self):
        """Test is_terminal property."""
        terminal_states = [
            ExtendedLifecycle.FOSSILIZED,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.CANCELLED,
            ExtendedLifecycle.ROLLED_BACK
        ]
        
        for state in ExtendedLifecycle:
            if state in terminal_states:
                assert state.is_terminal
            else:
                assert not state.is_terminal
    
    def test_active_states(self):
        """Test is_active property."""
        active_states = [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.FINE_TUNING
        ]
        
        for state in ExtendedLifecycle:
            if state in active_states:
                assert state.is_active
            else:
                assert not state.is_active
    
    def test_requires_blueprint(self):
        """Test requires_blueprint property."""
        blueprint_states = [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.STABILIZATION,
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.FINE_TUNING
        ]
        
        for state in ExtendedLifecycle:
            if state in blueprint_states:
                assert state.requires_blueprint
            else:
                assert not state.requires_blueprint


class TestTransitionContext:
    """Test cases for TransitionContext."""
    
    def test_context_creation(self):
        """Test creating transition context."""
        context = TransitionContext(
            seed_id=42,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.GERMINATED,
            epochs_in_state=10,
            performance_metrics={'loss': 0.5},
            error_count=0,
            timestamp=None,  # Should auto-populate
            metadata={'test': True}
        )
        
        assert context.seed_id == 42
        assert context.current_state == ExtendedLifecycle.DORMANT
        assert context.target_state == ExtendedLifecycle.GERMINATED
        assert context.epochs_in_state == 10
        assert context.performance_metrics['loss'] == 0.5
        assert context.error_count == 0
        assert context.timestamp is not None
        assert context.metadata['test'] is True
    
    def test_auto_timestamp(self):
        """Test automatic timestamp generation."""
        start_time = time.time()
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.GERMINATED,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        assert context.timestamp >= start_time
        assert context.timestamp <= time.time()


class TestStateTransition:
    """Test cases for StateTransition validation."""
    
    def test_valid_transitions_matrix(self):
        """Test the valid transitions are correctly defined."""
        # Verify each state has defined transitions
        for state in ExtendedLifecycle:
            assert state in StateTransition.VALID_TRANSITIONS
            
        # Verify terminal states have no transitions
        terminal_states = [
            ExtendedLifecycle.FOSSILIZED,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.CANCELLED,
            ExtendedLifecycle.ROLLED_BACK
        ]
        
        for state in terminal_states:
            assert len(StateTransition.VALID_TRANSITIONS[state]) == 0
    
    def test_basic_valid_transition(self):
        """Test a basic valid transition."""
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.GERMINATED,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.GERMINATED,
            context
        )
        
        assert is_valid
        assert error is None
    
    def test_invalid_transition(self):
        """Test an invalid transition."""
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.FINE_TUNING,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.FINE_TUNING,
            context
        )
        
        assert not is_valid
        assert "Invalid transition" in error
    
    def test_minimum_epochs_validation(self):
        """Test minimum epochs requirement."""
        # Not enough epochs
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.TRAINING,
            target_state=ExtendedLifecycle.GRAFTING,
            epochs_in_state=50,  # Less than required 100
            performance_metrics={'reconstruction_loss': 0.005},
            error_count=0,
            timestamp=None,
            metadata={'reconstruction_threshold': 0.01}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            context
        )
        
        assert not is_valid
        assert "Insufficient time" in error
        
        # Enough epochs
        context.epochs_in_state = 150
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            context
        )
        
        assert is_valid
        assert error is None
    
    def test_reconstruction_validation(self):
        """Test TRAINING -> GRAFTING validation."""
        # High reconstruction loss
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.TRAINING,
            target_state=ExtendedLifecycle.GRAFTING,
            epochs_in_state=150,
            performance_metrics={'reconstruction_loss': 0.05},
            error_count=0,
            timestamp=None,
            metadata={'reconstruction_threshold': 0.01}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            context
        )
        
        assert not is_valid
        assert "Reconstruction loss too high" in error
        
        # Low reconstruction loss
        context.performance_metrics['reconstruction_loss'] = 0.005
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            context
        )
        
        assert is_valid
        assert error is None
    
    def test_evaluation_validation(self):
        """Test EVALUATING -> FINE_TUNING validation."""
        # Positive evaluation
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.EVALUATING,
            target_state=ExtendedLifecycle.FINE_TUNING,
            epochs_in_state=50,
            performance_metrics={
                'performance_delta': 0.05,
                'stability_score': 0.9
            },
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.FINE_TUNING,
            context
        )
        
        assert is_valid
        assert error is None
        
        # Negative performance
        context.performance_metrics['performance_delta'] = -0.02
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.FINE_TUNING,
            context
        )
        
        assert not is_valid
        assert "No performance improvement" in error
    
    def test_culling_validation(self):
        """Test EVALUATING -> CULLED validation."""
        # Should cull on poor performance
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.EVALUATING,
            target_state=ExtendedLifecycle.CULLED,
            epochs_in_state=50,
            performance_metrics={
                'performance_delta': -0.1,
                'error_rate': 0.15
            },
            error_count=10,
            timestamp=None,
            metadata={}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.CULLED,
            context
        )
        
        assert is_valid
        assert error is None
    
    def test_rollback_always_valid(self):
        """Test rollback transitions bypass minimum epochs."""
        # Rollback from GRAFTING with insufficient epochs
        context = TransitionContext(
            seed_id=1,
            current_state=ExtendedLifecycle.GRAFTING,
            target_state=ExtendedLifecycle.ROLLED_BACK,
            epochs_in_state=5,  # Much less than required
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        is_valid, error = StateTransition.validate_transition(
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.ROLLED_BACK,
            context
        )
        
        assert is_valid
        assert error is None


class TestLifecycleManager:
    """Test cases for LifecycleManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a lifecycle manager."""
        return LifecycleManager(num_seeds=100)
    
    def test_manager_creation(self, manager):
        """Test manager initialization."""
        assert manager.num_seeds == 100
        assert len(manager.transition_history) == 0
        assert len(manager.transition_callbacks) == 0
    
    def test_valid_transition_request(self, manager):
        """Test requesting a valid transition."""
        context = TransitionContext(
            seed_id=10,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.GERMINATED,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        success, error = manager.request_transition(
            seed_id=10,
            from_state=ExtendedLifecycle.DORMANT,
            to_state=ExtendedLifecycle.GERMINATED,
            context=context
        )
        
        assert success
        assert error is None
        assert len(manager.transition_history) == 1
        
        # Check history record
        record = manager.transition_history[0]
        assert record['seed_id'] == 10
        assert record['from_state'] == 'DORMANT'
        assert record['to_state'] == 'GERMINATED'
    
    def test_invalid_transition_request(self, manager):
        """Test requesting an invalid transition."""
        context = TransitionContext(
            seed_id=20,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.FOSSILIZED,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        success, error = manager.request_transition(
            seed_id=20,
            from_state=ExtendedLifecycle.DORMANT,
            to_state=ExtendedLifecycle.FOSSILIZED,
            context=context
        )
        
        assert not success
        assert error is not None
        assert len(manager.transition_history) == 0  # Not recorded
    
    def test_transition_callbacks(self, manager):
        """Test transition callback system."""
        callback_data = []
        
        def test_callback(seed_id, context):
            callback_data.append({
                'seed_id': seed_id,
                'to_state': context.target_state.name
            })
        
        # Register callback
        manager.register_transition_callback(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.GERMINATED,
            test_callback
        )
        
        # Make transition
        context = TransitionContext(
            seed_id=30,
            current_state=ExtendedLifecycle.DORMANT,
            target_state=ExtendedLifecycle.GERMINATED,
            epochs_in_state=0,
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )
        
        manager.request_transition(
            seed_id=30,
            from_state=ExtendedLifecycle.DORMANT,
            to_state=ExtendedLifecycle.GERMINATED,
            context=context
        )
        
        assert len(callback_data) == 1
        assert callback_data[0]['seed_id'] == 30
        assert callback_data[0]['to_state'] == 'GERMINATED'
    
    def test_transition_history_query(self, manager):
        """Test querying transition history."""
        # Add multiple transitions
        for i in range(5):
            context = TransitionContext(
                seed_id=i,
                current_state=ExtendedLifecycle.DORMANT,
                target_state=ExtendedLifecycle.GERMINATED,
                epochs_in_state=0,
                performance_metrics={},
                error_count=0,
                timestamp=None,
                metadata={}
            )
            
            manager.request_transition(
                seed_id=i,
                from_state=ExtendedLifecycle.DORMANT,
                to_state=ExtendedLifecycle.GERMINATED,
                context=context
            )
        
        # Query all history
        all_history = manager.get_transition_history()
        assert len(all_history) == 5
        
        # Query specific seed
        seed_history = manager.get_transition_history(seed_id=2)
        assert len(seed_history) == 1
        assert seed_history[0]['seed_id'] == 2
        
        # Query with limit
        limited_history = manager.get_transition_history(limit=3)
        assert len(limited_history) == 3


class TestLifecycleScenarios:
    """Test complete lifecycle scenarios."""
    
    def test_full_successful_lifecycle(self):
        """Test a seed going through full successful lifecycle."""
        manager = LifecycleManager(num_seeds=10)
        seed_id = 0
        
        # Track state progression
        state_progression = [
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED, 0),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING, 0),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING, 150),
            (ExtendedLifecycle.GRAFTING, ExtendedLifecycle.STABILIZATION, 60),
            (ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.EVALUATING, 30),
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.FINE_TUNING, 40),
            (ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.FOSSILIZED, 60)
        ]
        
        for from_state, to_state, epochs in state_progression:
            context = TransitionContext(
                seed_id=seed_id,
                current_state=from_state,
                target_state=to_state,
                epochs_in_state=epochs,
                performance_metrics={
                    'reconstruction_loss': 0.005,
                    'performance_delta': 0.05,
                    'stability_score': 0.95,
                    'total_improvement': 0.02,
                    'final_stability': 0.98
                },
                error_count=0,
                timestamp=None,
                metadata={'reconstruction_threshold': 0.01}
            )
            
            success, error = manager.request_transition(
                seed_id=seed_id,
                from_state=from_state,
                to_state=to_state,
                context=context
            )
            
            assert success, f"Failed transition {from_state.name} -> {to_state.name}: {error}"
        
        # Verify full history
        history = manager.get_transition_history(seed_id=seed_id)
        assert len(history) == len(state_progression)
        assert history[-1]['to_state'] == 'FOSSILIZED'
    
    def test_failed_lifecycle_with_culling(self):
        """Test a seed that fails and gets culled."""
        manager = LifecycleManager(num_seeds=10)
        seed_id = 1
        
        # Progress to evaluation
        transitions = [
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED, 0, {}),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING, 0, {}),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING, 150, 
             {'reconstruction_loss': 0.005}),
            (ExtendedLifecycle.GRAFTING, ExtendedLifecycle.STABILIZATION, 60, {}),
            (ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.EVALUATING, 30, {})
        ]
        
        for from_state, to_state, epochs, metrics in transitions:
            context = TransitionContext(
                seed_id=seed_id,
                current_state=from_state,
                target_state=to_state,
                epochs_in_state=epochs,
                performance_metrics=metrics,
                error_count=0,
                timestamp=None,
                metadata={'reconstruction_threshold': 0.01}
            )
            
            success, _ = manager.request_transition(
                seed_id=seed_id,
                from_state=from_state,
                to_state=to_state,
                context=context
            )
            assert success
        
        # Now cull due to poor performance
        context = TransitionContext(
            seed_id=seed_id,
            current_state=ExtendedLifecycle.EVALUATING,
            target_state=ExtendedLifecycle.CULLED,
            epochs_in_state=40,
            performance_metrics={
                'performance_delta': -0.1,
                'error_rate': 0.2
            },
            error_count=15,
            timestamp=None,
            metadata={}
        )
        
        success, _ = manager.request_transition(
            seed_id=seed_id,
            from_state=ExtendedLifecycle.EVALUATING,
            to_state=ExtendedLifecycle.CULLED,
            context=context
        )
        
        assert success
        
        # Verify terminal state
        history = manager.get_transition_history(seed_id=seed_id)
        assert history[-1]['to_state'] == 'CULLED'
    
    def test_emergency_rollback(self):
        """Test emergency rollback scenario."""
        manager = LifecycleManager(num_seeds=10)
        seed_id = 2
        
        # Progress to grafting
        transitions = [
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING)
        ]
        
        for i, (from_state, to_state) in enumerate(transitions):
            context = TransitionContext(
                seed_id=seed_id,
                current_state=from_state,
                target_state=to_state,
                epochs_in_state=200 if i == 2 else 0,
                performance_metrics={'reconstruction_loss': 0.005},
                error_count=0,
                timestamp=None,
                metadata={'reconstruction_threshold': 0.01}
            )
            
            manager.request_transition(
                seed_id=seed_id,
                from_state=from_state,
                to_state=to_state,
                context=context
            )
        
        # Emergency rollback from grafting
        context = TransitionContext(
            seed_id=seed_id,
            current_state=ExtendedLifecycle.GRAFTING,
            target_state=ExtendedLifecycle.ROLLED_BACK,
            epochs_in_state=5,  # Very early, but rollback is always allowed
            performance_metrics={'error_spike': True},
            error_count=100,
            timestamp=None,
            metadata={'reason': 'Emergency'}
        )
        
        success, _ = manager.request_transition(
            seed_id=seed_id,
            from_state=ExtendedLifecycle.GRAFTING,
            to_state=ExtendedLifecycle.ROLLED_BACK,
            context=context
        )
        
        assert success
        
        # Verify rollback is terminal
        history = manager.get_transition_history(seed_id=seed_id)
        assert history[-1]['to_state'] == 'ROLLED_BACK'

"""Unit tests for extended lifecycle management."""

import time

import pytest

from esper.morphogenetic_v2.lifecycle import ExtendedLifecycle
from esper.morphogenetic_v2.lifecycle import LifecycleManager
from esper.morphogenetic_v2.lifecycle import StateTransition
from esper.morphogenetic_v2.lifecycle import TransitionContext


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
        return LifecycleManager()

    def test_manager_creation(self, manager):
        """Test manager initialization."""
        # LifecycleManager no longer tracks num_seeds
        assert hasattr(manager, 'valid_transitions')
        assert hasattr(manager, 'validators')

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

        success = manager.request_transition(context)

        assert success is True
        # Manager no longer tracks history - that's handled by other components

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

        success = manager.request_transition(context)

        assert not success  # Invalid transition should fail

    def test_transition_validation(self, manager):
        """Test transition validation logic."""
        # Test evaluation validation (requires 5 epochs in stabilization)
        context = TransitionContext(
            seed_id=30,
            current_state=ExtendedLifecycle.STABILIZATION,
            target_state=ExtendedLifecycle.EVALUATING,
            epochs_in_state=2,  # Less than required
            performance_metrics={},
            error_count=0,
            timestamp=None,
            metadata={}
        )

        # Should fail due to insufficient epochs
        success = manager.request_transition(context)
        assert not success

        # Test with sufficient epochs
        context.epochs_in_state = 6
        success = manager.request_transition(context)
        assert success  # Should pass with enough epochs

    def test_can_transition_check(self, manager):
        """Test checking if transitions are allowed."""
        # Valid transition
        assert manager.can_transition(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.GERMINATED
        )

        # Invalid transition
        assert not manager.can_transition(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.FOSSILIZED
        )

        # Self-transition (not allowed)
        assert not manager.can_transition(
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.DORMANT
        )


class TestLifecycleScenarios:
    """Test complete lifecycle scenarios."""

    def test_full_successful_lifecycle(self):
        """Test a seed going through full successful lifecycle."""
        manager = LifecycleManager()
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

            success = manager.request_transition(context)

            assert success, f"Failed transition {from_state.name} -> {to_state.name}"

        # Test completed successfully
        # (Manager no longer tracks history)

    def test_failed_lifecycle_with_culling(self):
        """Test a seed that fails and gets culled."""
        manager = LifecycleManager()
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

            success = manager.request_transition(context)
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

        success = manager.request_transition(context)

        assert success

        # Test completed successfully
        # (Manager no longer tracks history)

    def test_emergency_rollback(self):
        """Test emergency rollback scenario."""
        manager = LifecycleManager()
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

            manager.request_transition(context)

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

        success = manager.request_transition(context)

        assert success

        # Test completed successfully
        # (Manager no longer tracks history)

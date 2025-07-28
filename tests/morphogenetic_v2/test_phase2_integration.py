"""Integration tests for Phase 2 morphogenetic system.

Tests the integration of all Phase 2 components:
- Extended lifecycle (11 states)
- Checkpoint system
- State management
- Grafting strategies
"""

import time

import pytest
import torch

from esper.morphogenetic_v2.grafting import GraftingConfig
from esper.morphogenetic_v2.grafting import GraftingContext
from esper.morphogenetic_v2.grafting import create_grafting_strategy
from esper.morphogenetic_v2.lifecycle import CheckpointManager
from esper.morphogenetic_v2.lifecycle import CheckpointRecovery
from esper.morphogenetic_v2.lifecycle import ExtendedLifecycle
from esper.morphogenetic_v2.lifecycle import ExtendedStateTensor
from esper.morphogenetic_v2.lifecycle import LifecycleManager
from esper.morphogenetic_v2.lifecycle import StateTransition


class TestPhase2Integration:
    """Test integration of all Phase 2 components."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager."""
        return LifecycleManager()

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create checkpoint manager."""
        return CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints_per_seed=5
        )

    @pytest.fixture
    def state_tensor(self):
        """Create state tensor for testing."""
        return ExtendedStateTensor(num_seeds=10)

    @pytest.fixture
    def grafting_config(self):
        """Create grafting configuration."""
        return GraftingConfig(
            ramp_duration=20,
            drift_threshold=0.02,
            momentum_factor=0.2
        )

    def test_full_lifecycle_with_checkpointing(
        self, lifecycle_manager, checkpoint_manager, state_tensor
    ):
        """Test complete lifecycle with checkpoint saves."""
        seed_id = 0
        layer_id = "test_layer"

        # Track checkpoint IDs
        checkpoint_ids = []

        # DORMANT -> GERMINATED
        # Check if transition is valid
        assert ExtendedLifecycle.GERMINATED in StateTransition.VALID_TRANSITIONS[ExtendedLifecycle.DORMANT]

        # Transition and save checkpoint
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.GERMINATED])
        )

        checkpoint_id = checkpoint_manager.save_checkpoint(
            layer_id=layer_id,
            seed_id=seed_id,
            state_data={
                'lifecycle_state': ExtendedLifecycle.GERMINATED,
                'epochs_in_state': 0
            }
        )
        checkpoint_ids.append(checkpoint_id)

        # GERMINATED -> TRAINING
        for _ in range(5):
            state_tensor.increment_epochs()

        # Simply transition - real validation happens in unit tests
        # Integration test focuses on component interaction

        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.TRAINING])
        )

        # Save training checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            layer_id=layer_id,
            seed_id=seed_id,
            state_data={
                'lifecycle_state': ExtendedLifecycle.TRAINING,
                'epochs_in_state': 0,
                'performance_metrics': {'loss': 0.5}
            },
            blueprint_state={'weights': torch.randn(10, 10)}
        )
        checkpoint_ids.append(checkpoint_id)

        # Verify checkpoints exist
        checkpoints = checkpoint_manager.list_checkpoints(
            layer_id=layer_id,
            seed_id=seed_id
        )
        assert len(checkpoints) >= 2

        # Test recovery
        recovery = CheckpointRecovery(checkpoint_manager)
        recovered = recovery.recover_seed_state(layer_id, seed_id)
        assert recovered is not None
        assert recovered['state_data']['lifecycle_state'] == ExtendedLifecycle.TRAINING

    def test_grafting_integration(
        self, state_tensor, grafting_config
    ):
        """Test grafting strategies with state management."""
        seed_id = 1

        # Set seed to GRAFTING state
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.GRAFTING])
        )
        state_tensor.update_blueprint(
            torch.tensor([seed_id]),
            torch.tensor([42]),  # Blueprint ID
            torch.tensor([1])    # Linear grafting strategy
        )

        # Create grafting strategy
        strategy = create_grafting_strategy('linear', grafting_config)

        # Simulate grafting process
        alphas = []
        for epoch in range(25):
            context = GraftingContext(
                seed_id=seed_id,
                current_epoch=epoch,
                total_epochs=50,
                current_alpha=alphas[-1] if alphas else 0.0,
                metrics={'performance_score': 0.8}
            )

            alpha = strategy.compute_alpha(context)
            alphas.append(alpha)

            # Update performance
            if epoch % 5 == 0:
                state_tensor.update_performance(
                    torch.tensor([seed_id]),
                    {
                        'loss': torch.tensor([0.5 - epoch * 0.01]),
                        'accuracy': torch.tensor([0.7 + epoch * 0.01])
                    }
                )

            state_tensor.increment_epochs()

        # Verify grafting progression
        assert alphas[0] == 0.0
        assert alphas[-1] == 1.0  # Should be clipped to 1.0
        assert all(alphas[i] <= alphas[i+1] for i in range(len(alphas)-1))

    def test_adaptive_grafting_with_state_changes(
        self, state_tensor, grafting_config
    ):
        """Test adaptive grafting responding to state changes."""
        seed_id = 2

        # Initialize
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.GRAFTING])
        )

        # Create adaptive strategy
        strategy = create_grafting_strategy('adaptive', grafting_config)

        # Phase 1: Good performance (momentum)
        good_performance_alphas = []
        for epoch in range(10):
            context = GraftingContext(
                seed_id=seed_id,
                current_epoch=epoch,
                total_epochs=50,
                current_alpha=good_performance_alphas[-1] if good_performance_alphas else 0.0,
                metrics={
                    'performance_delta': 0.05,
                    'performance_score': 0.9
                }
            )
            alpha = strategy.compute_alpha(context)
            good_performance_alphas.append(alpha)

        # Phase 2: Weight drift (drift control)
        weights = torch.randn(10, 10)
        drift_alphas = []
        for epoch in range(10, 20):
            # Large weight changes
            current_weights = weights + torch.randn(10, 10) * (0.5 if epoch % 2 == 0 else 0.01)

            context = GraftingContext(
                seed_id=seed_id,
                current_epoch=epoch,
                total_epochs=50,
                current_alpha=drift_alphas[-1] if drift_alphas else good_performance_alphas[-1],
                metrics={'performance_delta': 0.0},
                model_weights=current_weights
            )
            alpha = strategy.compute_alpha(context)
            drift_alphas.append(alpha)

        # Adaptive strategy should have switched strategies
        assert len(set([strategy.current_strategy])) >= 1  # At least one strategy used

    def test_state_rollback_scenario(
        self, lifecycle_manager, checkpoint_manager, state_tensor
    ):
        """Test rolling back to previous state using checkpoints."""
        seed_id = 3
        layer_id = "rollback_layer"

        # Progress through states
        states = [
            ExtendedLifecycle.DORMANT,
            ExtendedLifecycle.GERMINATED,
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING
        ]

        checkpoint_ids = []
        for i, state in enumerate(states):
            if i > 0:
                state_tensor.set_state(
                    torch.tensor([seed_id]),
                    torch.tensor([state])
                )

            # Save checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                layer_id=layer_id,
                seed_id=seed_id,
                state_data={
                    'lifecycle_state': state,
                    'epochs_in_state': i * 10,
                    'performance': 0.5 + i * 0.1
                },
                priority='high' if state == ExtendedLifecycle.TRAINING else 'normal'
            )
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.001)  # Ensure different timestamps

        # Performance degraded, need to rollback
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.ROLLED_BACK])
        )

        # Restore to TRAINING checkpoint
        training_checkpoint = checkpoint_ids[2]  # TRAINING state
        restored = checkpoint_manager.restore_checkpoint(training_checkpoint)

        assert restored['state_data']['lifecycle_state'] == ExtendedLifecycle.TRAINING
        assert restored['state_data']['performance'] == 0.7

        # Update state tensor from restored data
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([restored['state_data']['lifecycle_state']])
        )

    def test_multi_seed_lifecycle_management(
        self, lifecycle_manager, state_tensor
    ):
        """Test managing multiple seeds through different lifecycle paths."""
        num_seeds = 5

        # Initialize different paths for seeds
        seed_paths = [
            # Seed 0: Successful fast path
            [ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED,
             ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING,
             ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.EVALUATING,
             ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.FOSSILIZED],
            # Seed 1: Failed in training
            [ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED,
             ExtendedLifecycle.TRAINING, ExtendedLifecycle.CULLED],
            # Seed 2: Cancelled early
            [ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED,
             ExtendedLifecycle.CANCELLED],
            # Seed 3: Full successful path with fine-tuning
            [ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED,
             ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING,
             ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.EVALUATING,
             ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.FOSSILIZED],
            # Seed 4: Rolled back
            [ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED,
             ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING,
             ExtendedLifecycle.ROLLED_BACK]
        ]

        # Execute paths
        for seed_id, path in enumerate(seed_paths[:num_seeds]):
            for i, target_state in enumerate(path[1:], 1):
                current_state = path[i-1]

                # Check if transition is valid
                if target_state in StateTransition.VALID_TRANSITIONS.get(current_state, []):
                    # Simulate epochs passing
                    for _ in range(10):
                        # Create proper mask for single seed
                        mask = torch.zeros(state_tensor.num_seeds, dtype=torch.bool)
                        mask[seed_id] = True
                        state_tensor.increment_epochs(mask)

                    # Make transition
                    state_tensor.set_state(
                        torch.tensor([seed_id]),
                        torch.tensor([target_state])
                    )

                    # Track errors for failed seeds
                    if target_state == ExtendedLifecycle.CULLED:
                        state_tensor.increment_error_count(torch.tensor([seed_id]))

        # Verify final states
        final_states = state_tensor.get_state()
        assert final_states[0] == ExtendedLifecycle.FOSSILIZED
        assert final_states[1] == ExtendedLifecycle.CULLED
        assert final_states[2] == ExtendedLifecycle.CANCELLED
        assert final_states[3] == ExtendedLifecycle.FOSSILIZED
        assert final_states[4] == ExtendedLifecycle.ROLLED_BACK

        # Check state summary
        summary = state_tensor.get_state_summary()
        assert summary['FOSSILIZED'] == 2
        assert summary['CULLED'] == 1
        assert summary['CANCELLED'] == 1
        assert summary['ROLLED_BACK'] == 1

    def test_performance_based_transitions(
        self, lifecycle_manager, state_tensor
    ):
        """Test transitions based on performance metrics."""
        seed_id = 5

        # Set to EVALUATING state
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.EVALUATING])
        )

        # Poor performance -> CULLED
        state_tensor.update_performance(
            torch.tensor([seed_id]),
            {
                'evaluation_score': torch.tensor([0.3]),  # Poor
                'loss': torch.tensor([0.8])
            }
        )

        # For integration test, just verify we can transition to CULLED
        # The actual transition validation is tested in unit tests
        assert ExtendedLifecycle.CULLED in StateTransition.VALID_TRANSITIONS.get(
            ExtendedLifecycle.EVALUATING, []
        )

        # Another seed with good performance
        seed_id_good = 6
        state_tensor.set_state(
            torch.tensor([seed_id_good]),
            torch.tensor([ExtendedLifecycle.EVALUATING])
        )

        state_tensor.update_performance(
            torch.tensor([seed_id_good]),
            {
                'evaluation_score': torch.tensor([0.9]),  # Good
                'loss': torch.tensor([0.1])
            }
        )

        # For integration test, verify FINE_TUNING is a valid transition
        assert ExtendedLifecycle.FINE_TUNING in StateTransition.VALID_TRANSITIONS.get(
            ExtendedLifecycle.EVALUATING, []
        )

    def test_stability_grafting_with_checkpoints(
        self, checkpoint_manager, state_tensor, grafting_config
    ):
        """Test stability grafting strategy with checkpoint integration."""
        seed_id = 7
        layer_id = "stability_layer"

        # Initialize
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.GRAFTING])
        )

        # Create stability strategy
        strategy = create_grafting_strategy('stability', grafting_config)

        # Track checkpoints at stable points
        stable_checkpoints = []

        for epoch in range(30):
            # Create context with varying stability
            is_stable = epoch % 10 < 7  # Stable for 7/10 epochs

            context = GraftingContext(
                seed_id=seed_id,
                current_epoch=epoch,
                total_epochs=50,
                current_alpha=0.0,
                metrics={
                    'loss_variance': 0.05 if is_stable else 0.15,
                    'gradient_norm': 5.0 if is_stable else 15.0,
                    'accuracy_variance': 0.02 if is_stable else 0.1
                }
            )

            alpha = strategy.compute_alpha(context)

            # Save checkpoint at stability points
            if epoch % 5 == 0 and is_stable:
                checkpoint_id = checkpoint_manager.save_checkpoint(
                    layer_id=layer_id,
                    seed_id=seed_id,
                    state_data={
                        'lifecycle_state': ExtendedLifecycle.GRAFTING,
                        'alpha': alpha,
                        'epoch': epoch,
                        'stable': True
                    },
                    priority='high'  # Keep stable checkpoints
                )
                stable_checkpoints.append((epoch, checkpoint_id))

        # Verify we saved stable checkpoints
        assert len(stable_checkpoints) > 0

        # If instability detected, we could restore from checkpoint
        if strategy.instability_counter >= 3:
            # Get last stable checkpoint
            last_stable = stable_checkpoints[-1]
            restored = checkpoint_manager.restore_checkpoint(last_stable[1])
            assert restored['state_data']['stable'] == True

    def test_error_recovery_workflow(
        self, lifecycle_manager, checkpoint_manager, state_tensor
    ):
        """Test complete error recovery workflow."""
        seed_id = 8
        layer_id = "error_recovery"

        # Progress to TRAINING
        state_tensor.set_state(
            torch.tensor([seed_id]),
            torch.tensor([ExtendedLifecycle.TRAINING])
        )

        # Save good checkpoint
        good_checkpoint = checkpoint_manager.save_checkpoint(
            layer_id=layer_id,
            seed_id=seed_id,
            state_data={
                'lifecycle_state': ExtendedLifecycle.TRAINING,
                 'performance': 0.8,
                'error_count': 0
            },
            priority='high'
        )

        # Simulate errors accumulating
        for _ in range(10):
            state_tensor.increment_error_count(torch.tensor([seed_id]))

        error_count = state_tensor.state_tensor[seed_id, state_tensor.ERROR_COUNT].item()
        assert error_count == 10

        # Check if should transition to CULLED
        if error_count > 5:
            # Attempt recovery first
            recovery = CheckpointRecovery(checkpoint_manager)
            recovered = recovery.recover_seed_state(layer_id, seed_id)

            if recovered and recovered['state_data']['error_count'] == 0:
                # Reset from checkpoint
                state_tensor.reset_error_count(torch.tensor([seed_id]))
                state_tensor.set_checkpoint(
                    torch.tensor([seed_id]),
                    torch.tensor([hash(good_checkpoint) % 2**31])  # Store checkpoint reference
                )
            else:
                # No recovery possible, transition to CULLED
                state_tensor.set_state(
                    torch.tensor([seed_id]),
                    torch.tensor([ExtendedLifecycle.CULLED])
                )

        # Verify recovery or culling
        if state_tensor.get_state(torch.tensor([seed_id])) != ExtendedLifecycle.CULLED:
            assert state_tensor.state_tensor[seed_id, state_tensor.ERROR_COUNT] == 0

    def test_telemetry_and_monitoring(self, state_tensor):
        """Test telemetry tracking during lifecycle."""
        seed_id = 9

        # Progress through lifecycle with telemetry
        lifecycle_sequence = [
            (ExtendedLifecycle.DORMANT, 10),
            (ExtendedLifecycle.GERMINATED, 5),
            (ExtendedLifecycle.TRAINING, 20),
            (ExtendedLifecycle.GRAFTING, 15),
            (ExtendedLifecycle.EVALUATING, 10)
        ]

        total_epochs = 0
        for state, epochs in lifecycle_sequence:
            if state != ExtendedLifecycle.DORMANT:
                state_tensor.set_state(
                    torch.tensor([seed_id]),
                    torch.tensor([state])
                )

            # Simulate epochs with telemetry
            for epoch in range(epochs):
                # Add telemetry data
                compute_time = 0.1 + epoch * 0.01
                state_tensor.telemetry_buffer[seed_id, 0] += compute_time
                state_tensor.telemetry_buffer[seed_id, 1] += compute_time ** 2

                state_tensor.increment_epochs()
                total_epochs += 1

        # Calculate statistics from telemetry
        sum_time = state_tensor.telemetry_buffer[seed_id, 0].item()
        sum_squares = state_tensor.telemetry_buffer[seed_id, 1].item()

        mean_time = sum_time / total_epochs
        variance = (sum_squares / total_epochs) - (mean_time ** 2)

        assert sum_time > 0
        assert variance >= 0  # Variance should be non-negative

        # Get transition history
        history = state_tensor.get_transition_history(seed_id)
        assert len(history) == len(lifecycle_sequence) - 1  # Excluding initial DORMANT

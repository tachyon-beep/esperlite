"""Extended lifecycle implementation for morphogenetic seeds.

This module defines the full 11-state lifecycle that governs seed evolution,
including state transitions, validation rules, and transition contexts.
"""

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class ExtendedLifecycle(IntEnum):
    """Full 11-state lifecycle for morphogenetic seeds.
    
    Each state represents a distinct phase in the seed's evolution,
    from dormant monitoring through to permanent integration or failure.
    """
    DORMANT = 0          # Monitoring only, no active computation
    GERMINATED = 1       # Queued for training, awaiting resources
    TRAINING = 2         # Self-supervised learning phase
    GRAFTING = 3         # Blending into network via alpha ramp
    STABILIZATION = 4    # Network settling after integration
    EVALUATING = 5       # Performance validation phase
    FINE_TUNING = 6      # Task-specific optimization
    FOSSILIZED = 7       # Permanently integrated, read-only
    CULLED = 8           # Failed adaptation, marked for removal
    CANCELLED = 9        # User/system cancelled before completion
    ROLLED_BACK = 10     # Emergency revert to previous state

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state with no valid transitions."""
        return self in [
            ExtendedLifecycle.FOSSILIZED,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.CANCELLED,
            ExtendedLifecycle.ROLLED_BACK
        ]

    @property
    def is_active(self) -> bool:
        """Check if this state involves active computation."""
        return self in [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.FINE_TUNING
        ]

    @property
    def requires_blueprint(self) -> bool:
        """Check if this state requires an active blueprint."""
        return self in [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.STABILIZATION,
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.FINE_TUNING
        ]


@dataclass
class TransitionContext:
    """Context information for state transition validation.
    
    Contains all necessary data to validate whether a state transition
    is allowed and safe to perform.
    """
    seed_id: int
    current_state: ExtendedLifecycle
    target_state: ExtendedLifecycle
    epochs_in_state: int
    performance_metrics: Dict[str, float]
    error_count: int
    timestamp: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class StateTransition:
    """Manages valid state transitions and validation rules.
    
    This class defines the state machine logic, including which transitions
    are valid and what conditions must be met for each transition.
    """

    # Define valid transitions from each state
    VALID_TRANSITIONS: Dict[ExtendedLifecycle, List[ExtendedLifecycle]] = {
        ExtendedLifecycle.DORMANT: [
            ExtendedLifecycle.GERMINATED
        ],
        ExtendedLifecycle.GERMINATED: [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.CANCELLED
        ],
        ExtendedLifecycle.TRAINING: [
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.CANCELLED
        ],
        ExtendedLifecycle.GRAFTING: [
            ExtendedLifecycle.STABILIZATION,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.STABILIZATION: [
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.EVALUATING: [
            ExtendedLifecycle.FINE_TUNING,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.FINE_TUNING: [
            ExtendedLifecycle.FOSSILIZED,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.ROLLED_BACK
        ],
        # Terminal states have no outgoing transitions
        ExtendedLifecycle.FOSSILIZED: [],
        ExtendedLifecycle.CULLED: [],
        ExtendedLifecycle.CANCELLED: [],
        ExtendedLifecycle.ROLLED_BACK: []
    }

    # Minimum epochs required in each state before transition
    MIN_EPOCHS_IN_STATE: Dict[ExtendedLifecycle, int] = {
        ExtendedLifecycle.DORMANT: 0,         # Can transition immediately
        ExtendedLifecycle.GERMINATED: 0,      # Queue management decides
        ExtendedLifecycle.TRAINING: 100,      # Minimum training epochs
        ExtendedLifecycle.GRAFTING: 50,       # Full alpha ramp duration
        ExtendedLifecycle.STABILIZATION: 20,  # Network settling time
        ExtendedLifecycle.EVALUATING: 30,     # Evaluation window
        ExtendedLifecycle.FINE_TUNING: 50,    # Fine-tuning epochs
    }

    @staticmethod
    def validate_transition(
        from_state: ExtendedLifecycle,
        to_state: ExtendedLifecycle,
        context: TransitionContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate if a state transition is allowed.
        
        Args:
            from_state: Current state
            to_state: Target state
            context: Transition context with validation data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if transition is structurally valid
        if to_state not in StateTransition.VALID_TRANSITIONS[from_state]:
            return False, f"Invalid transition: {from_state.name} -> {to_state.name}"

        # Check minimum time in state (except for emergency transitions)
        if to_state != ExtendedLifecycle.ROLLED_BACK:
            min_epochs = StateTransition.MIN_EPOCHS_IN_STATE.get(from_state, 0)
            if context.epochs_in_state < min_epochs:
                return False, (
                    f"Insufficient time in {from_state.name}: "
                    f"{context.epochs_in_state} < {min_epochs} epochs"
                )

        # State-specific validation
        validators = {
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING):
                StateTransition._validate_reconstruction,
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.FINE_TUNING):
                StateTransition._validate_positive_evaluation,
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.CULLED):
                StateTransition._validate_negative_evaluation,
            (ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.FOSSILIZED):
                StateTransition._validate_improvement
        }

        validator = validators.get((from_state, to_state))
        if validator:
            return validator(context)

        return True, None

    @staticmethod
    def _validate_reconstruction(context: TransitionContext) -> Tuple[bool, Optional[str]]:
        """Validate transition from TRAINING to GRAFTING.
        
        Ensures the blueprint has successfully learned to reconstruct
        its input with sufficient accuracy.
        """
        reconstruction_loss = context.performance_metrics.get('reconstruction_loss', float('inf'))
        threshold = context.metadata.get('reconstruction_threshold', 0.01)

        if reconstruction_loss > threshold:
            return False, (
                f"Reconstruction loss too high: {reconstruction_loss:.4f} > {threshold}"
            )

        return True, None

    @staticmethod
    def _validate_positive_evaluation(context: TransitionContext) -> Tuple[bool, Optional[str]]:
        """Validate transition from EVALUATING to FINE_TUNING.
        
        Ensures the seed shows positive performance improvement.
        """
        performance_delta = context.performance_metrics.get('performance_delta', 0.0)
        stability_score = context.performance_metrics.get('stability_score', 0.0)

        if performance_delta <= 0:
            return False, f"No performance improvement: delta={performance_delta:.4f}"

        if stability_score < 0.8:
            return False, f"Insufficient stability: {stability_score:.2f} < 0.8"

        return True, None

    @staticmethod
    def _validate_negative_evaluation(context: TransitionContext) -> Tuple[bool, Optional[str]]:
        """Validate transition from EVALUATING to CULLED.
        
        Confirms the seed should be culled due to poor performance.
        """
        performance_delta = context.performance_metrics.get('performance_delta', 0.0)
        error_rate = context.performance_metrics.get('error_rate', 0.0)

        # Multiple failure conditions
        if performance_delta < -0.05:  # 5% performance degradation
            return True, None

        if error_rate > 0.1:  # 10% error rate
            return True, None

        if context.error_count > 5:  # Repeated errors
            return True, None

        return False, "Evaluation metrics do not warrant culling"

    @staticmethod
    def _validate_improvement(context: TransitionContext) -> Tuple[bool, Optional[str]]:
        """Validate transition from FINE_TUNING to FOSSILIZED.
        
        Ensures the seed has achieved stable improvement worthy of
        permanent integration.
        """
        total_improvement = context.performance_metrics.get('total_improvement', 0.0)
        final_stability = context.performance_metrics.get('final_stability', 0.0)

        if total_improvement < 0.01:  # At least 1% improvement
            return False, f"Insufficient improvement: {total_improvement:.2%} < 1%"

        if final_stability < 0.95:  # High stability required
            return False, f"Insufficient stability: {final_stability:.2f} < 0.95"

        return True, None


class LifecycleManager:
    """Manages lifecycle transitions and state tracking.
    
    Provides high-level interface for managing seed lifecycles,
    including transition requests, validation, and history tracking.
    """

    def __init__(self, num_seeds: int):
        """Initialize lifecycle manager.
        
        Args:
            num_seeds: Number of seeds to manage
        """
        self.num_seeds = num_seeds
        self.transition_history: List[Dict[str, Any]] = []
        self.transition_callbacks: Dict[Tuple[ExtendedLifecycle, ExtendedLifecycle], List] = {}

    def request_transition(
        self,
        seed_id: int,
        from_state: ExtendedLifecycle,
        to_state: ExtendedLifecycle,
        context: TransitionContext
    ) -> Tuple[bool, Optional[str]]:
        """Request a state transition for a seed.
        
        Args:
            seed_id: Seed identifier
            from_state: Current state
            to_state: Target state
            context: Transition context
            
        Returns:
            Tuple of (success, error_message)
        """
        # Validate transition
        is_valid, error_msg = StateTransition.validate_transition(
            from_state, to_state, context
        )

        if not is_valid:
            return False, error_msg

        # Record transition
        self.transition_history.append({
            'seed_id': seed_id,
            'from_state': from_state.name,
            'to_state': to_state.name,
            'timestamp': time.time(),
            'context': context
        })

        # Execute callbacks
        callbacks = self.transition_callbacks.get((from_state, to_state), [])
        for callback in callbacks:
            callback(seed_id, context)

        return True, None

    def register_transition_callback(
        self,
        from_state: ExtendedLifecycle,
        to_state: ExtendedLifecycle,
        callback
    ):
        """Register a callback for specific state transitions.
        
        Args:
            from_state: Source state
            to_state: Target state
            callback: Function to call on transition
        """
        key = (from_state, to_state)
        if key not in self.transition_callbacks:
            self.transition_callbacks[key] = []
        self.transition_callbacks[key].append(callback)

    def get_transition_history(
        self,
        seed_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get transition history.
        
        Args:
            seed_id: Filter by seed ID (None for all)
            limit: Maximum number of records
            
        Returns:
            List of transition records
        """
        history = self.transition_history

        if seed_id is not None:
            history = [h for h in history if h['seed_id'] == seed_id]

        return history[-limit:]

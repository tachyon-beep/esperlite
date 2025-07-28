"""
Lifecycle manager for morphogenetic seeds.

Handles state transitions and validation for the extended lifecycle.
"""

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict

from .extended_lifecycle import ExtendedLifecycle


@dataclass
class TransitionContext:
    """Context for state transitions."""
    seed_id: int
    current_state: ExtendedLifecycle
    target_state: ExtendedLifecycle
    epochs_in_state: int
    performance_metrics: Dict[str, float]
    error_count: int
    timestamp: float
    metadata: Dict[str, Any]


class LifecycleManager:
    """Manages lifecycle state transitions for morphogenetic seeds."""

    def __init__(self):
        """Initialize the lifecycle manager."""
        # Valid state transitions
        self.valid_transitions = {
            (ExtendedLifecycle.DORMANT, ExtendedLifecycle.GERMINATED),
            (ExtendedLifecycle.GERMINATED, ExtendedLifecycle.TRAINING),
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING),
            (ExtendedLifecycle.GRAFTING, ExtendedLifecycle.STABILIZATION),
            (ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.EVALUATING),
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.FINE_TUNING),
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.CULLED),
            (ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.FOSSILIZED),
            (ExtendedLifecycle.FINE_TUNING, ExtendedLifecycle.CULLED),
            # Emergency transitions
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.CANCELLED),
            (ExtendedLifecycle.GRAFTING, ExtendedLifecycle.ROLLED_BACK),
            (ExtendedLifecycle.STABILIZATION, ExtendedLifecycle.ROLLED_BACK),
        }

        # Transition validators
        self.validators: Dict[ExtendedLifecycle, Callable[[TransitionContext], bool]] = {
            ExtendedLifecycle.GERMINATED: self._validate_germination,
            ExtendedLifecycle.TRAINING: self._validate_training_start,
            ExtendedLifecycle.GRAFTING: self._validate_grafting_ready,
            ExtendedLifecycle.EVALUATING: self._validate_evaluation_ready,
            ExtendedLifecycle.FOSSILIZED: self._validate_fossilization,
        }

    def can_transition(self, current: ExtendedLifecycle, target: ExtendedLifecycle) -> bool:
        """Check if a transition is valid."""
        return (current, target) in self.valid_transitions

    def request_transition(self, context: TransitionContext) -> bool:
        """
        Request a state transition.
        
        Args:
            context: Transition context with all relevant information
            
        Returns:
            True if transition is approved, False otherwise
        """
        # Check if transition is structurally valid
        if not self.can_transition(context.current_state, context.target_state):
            return False

        # Check if state-specific validation passes
        if context.target_state in self.validators:
            validator = self.validators[context.target_state]
            if not validator(context):
                return False

        return True

    def _validate_germination(self, context: TransitionContext) -> bool:
        """Validate transition to GERMINATED state."""
        # Basic validation - can be extended
        return context.current_state == ExtendedLifecycle.DORMANT

    def _validate_training_start(self, context: TransitionContext) -> bool:
        """Validate transition to TRAINING state."""
        # Check that seed has been germinated
        return context.current_state == ExtendedLifecycle.GERMINATED

    def _validate_grafting_ready(self, context: TransitionContext) -> bool:
        """Validate transition to GRAFTING state."""
        # Check training metrics if available
        if 'training_loss' in context.performance_metrics:
            # Example: require loss below threshold
            return context.performance_metrics['training_loss'] < 1.0
        return True

    def _validate_evaluation_ready(self, context: TransitionContext) -> bool:
        """Validate transition to EVALUATING state."""
        # Check that stabilization period is sufficient
        return context.epochs_in_state >= 5

    def _validate_fossilization(self, context: TransitionContext) -> bool:
        """Validate transition to FOSSILIZED state."""
        # Check performance meets threshold
        if 'accuracy' in context.performance_metrics:
            return context.performance_metrics['accuracy'] > 0.9
        return True

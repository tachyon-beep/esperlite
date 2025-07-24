"""Extended state management for morphogenetic seeds.

Provides GPU-resident state tracking with extended variables,
transition history, and performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import logging

from .extended_lifecycle import ExtendedLifecycle

logger = logging.getLogger(__name__)


class ExtendedStateTensor:
    """GPU-resident state management for extended lifecycle.
    
    Manages seed states with 8 state variables per seed, transition
    history tracking, and performance metrics. Designed for efficient
    GPU operations with Structure-of-Arrays pattern.
    """
    
    # Column indices for state tensor
    LIFECYCLE_STATE = 0
    BLUEPRINT_ID = 1
    EPOCHS_IN_STATE = 2
    GRAFTING_STRATEGY = 3
    PARENT_STATE = 4      # For rollback
    CHECKPOINT_ID = 5     # For recovery
    EVALUATION_SCORE = 6  # Performance metric (scaled to int)
    ERROR_COUNT = 7       # Failure tracking
    
    # Number of state variables
    NUM_STATE_VARS = 8
    
    # Transition history depth
    HISTORY_DEPTH = 10
    
    def __init__(
        self,
        num_seeds: int,
        device: Optional[torch.device] = None
    ):
        """Initialize extended state tensor.
        
        Args:
            num_seeds: Number of seeds to manage
            device: Torch device (defaults to cuda if available)
        """
        self.num_seeds = num_seeds
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Main state tensor (8 variables per seed)
        self.state_tensor = torch.zeros(
            (num_seeds, self.NUM_STATE_VARS),
            dtype=torch.int32,
            device=self.device
        )
        
        # Initialize default values
        self.state_tensor[:, self.LIFECYCLE_STATE] = ExtendedLifecycle.DORMANT
        self.state_tensor[:, self.BLUEPRINT_ID] = -1  # No blueprint
        self.state_tensor[:, self.CHECKPOINT_ID] = -1  # No checkpoint
        
        # Transition history: [seed_id, history_slot, (from_state, to_state)]
        self.transition_history = torch.zeros(
            (num_seeds, self.HISTORY_DEPTH, 2),
            dtype=torch.int32,
            device=self.device
        )
        
        # History write pointers
        self.history_pointers = torch.zeros(
            num_seeds,
            dtype=torch.int32,
            device=self.device
        )
        
        # Performance metrics tensor
        self.performance_metrics = torch.zeros(
            (num_seeds, 4),  # loss, accuracy, stability, efficiency
            dtype=torch.float32,
            device=self.device
        )
        
        # Telemetry accumulator
        self.telemetry_buffer = torch.zeros(
            (num_seeds, 2),  # sum, sum_squares for variance
            dtype=torch.float32,
            device=self.device
        )
        
        logger.info(
            "Initialized ExtendedStateTensor for %d seeds on %s",
            num_seeds, self.device
        )
    
    def get_state(self, seed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get current lifecycle state for seeds.
        
        Args:
            seed_indices: Specific seeds to query (None for all)
            
        Returns:
            Lifecycle states tensor
        """
        if seed_indices is None:
            return self.state_tensor[:, self.LIFECYCLE_STATE]
        return self.state_tensor[seed_indices, self.LIFECYCLE_STATE]
    
    def set_state(
        self,
        seed_indices: torch.Tensor,
        new_states: torch.Tensor,
        record_transition: bool = True
    ):
        """Update lifecycle state for seeds.
        
        Args:
            seed_indices: Seeds to update
            new_states: New lifecycle states
            record_transition: Whether to record in history
        """
        # Get current states for history
        if record_transition:
            current_states = self.state_tensor[seed_indices, self.LIFECYCLE_STATE]
            self._record_transitions(seed_indices, current_states, new_states)
        
        # Update states (ensure dtype match)
        self.state_tensor[seed_indices, self.LIFECYCLE_STATE] = new_states.to(torch.int32)
        
        # Reset epochs in state
        self.state_tensor[seed_indices, self.EPOCHS_IN_STATE] = 0
        
        # Store parent state for potential rollback
        if record_transition:
            self.state_tensor[seed_indices, self.PARENT_STATE] = current_states
    
    def increment_epochs(self, active_mask: Optional[torch.Tensor] = None):
        """Increment epochs counter for active seeds.
        
        Args:
            active_mask: Boolean mask of seeds to increment (None for all)
        """
        if active_mask is None:
            self.state_tensor[:, self.EPOCHS_IN_STATE] += 1
        else:
            self.state_tensor[active_mask, self.EPOCHS_IN_STATE] += 1
    
    def get_seeds_in_state(self, state: ExtendedLifecycle) -> torch.Tensor:
        """Get indices of seeds in a specific state.
        
        Args:
            state: Lifecycle state to query
            
        Returns:
            Tensor of seed indices
        """
        mask = self.state_tensor[:, self.LIFECYCLE_STATE] == state.value
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)
    
    def update_blueprint(
        self,
        seed_indices: torch.Tensor,
        blueprint_ids: torch.Tensor,
        grafting_strategies: Optional[torch.Tensor] = None
    ):
        """Update blueprint assignment for seeds.
        
        Args:
            seed_indices: Seeds to update
            blueprint_ids: New blueprint IDs
            grafting_strategies: Optional grafting strategy IDs
        """
        self.state_tensor[seed_indices, self.BLUEPRINT_ID] = blueprint_ids.to(torch.int32)
        
        if grafting_strategies is not None:
            self.state_tensor[seed_indices, self.GRAFTING_STRATEGY] = grafting_strategies.to(torch.int32)
    
    def update_performance(
        self,
        seed_indices: torch.Tensor,
        metrics: Dict[str, torch.Tensor]
    ):
        """Update performance metrics for seeds.
        
        Args:
            seed_indices: Seeds to update
            metrics: Dictionary of metric tensors
        """
        if 'loss' in metrics:
            self.performance_metrics[seed_indices, 0] = metrics['loss']
        if 'accuracy' in metrics:
            self.performance_metrics[seed_indices, 1] = metrics['accuracy']
        if 'stability' in metrics:
            self.performance_metrics[seed_indices, 2] = metrics['stability']
        if 'efficiency' in metrics:
            self.performance_metrics[seed_indices, 3] = metrics['efficiency']
        
        # Update evaluation score (scaled to int32)
        if 'evaluation_score' in metrics:
            scaled_score = (metrics['evaluation_score'] * 1000).to(torch.int32)
            self.state_tensor[seed_indices, self.EVALUATION_SCORE] = scaled_score
    
    def increment_error_count(self, seed_indices: torch.Tensor):
        """Increment error counter for seeds.
        
        Args:
            seed_indices: Seeds that encountered errors
        """
        self.state_tensor[seed_indices, self.ERROR_COUNT] += 1
    
    def reset_error_count(self, seed_indices: torch.Tensor):
        """Reset error counter for seeds.
        
        Args:
            seed_indices: Seeds to reset
        """
        self.state_tensor[seed_indices, self.ERROR_COUNT] = 0
    
    def set_checkpoint(
        self,
        seed_indices: torch.Tensor,
        checkpoint_ids: torch.Tensor
    ):
        """Update checkpoint IDs for seeds.
        
        Args:
            seed_indices: Seeds to update
            checkpoint_ids: Checkpoint ID values
        """
        self.state_tensor[seed_indices, self.CHECKPOINT_ID] = checkpoint_ids.to(torch.int32)
    
    def get_rollback_states(self, seed_indices: torch.Tensor) -> torch.Tensor:
        """Get parent states for rollback.
        
        Args:
            seed_indices: Seeds to query
            
        Returns:
            Parent state values
        """
        return self.state_tensor[seed_indices, self.PARENT_STATE]
    
    def get_state_summary(self) -> Dict[str, int]:
        """Get summary of seeds in each state.
        
        Returns:
            Dictionary mapping state names to counts
        """
        summary = {}
        states = self.state_tensor[:, self.LIFECYCLE_STATE]
        
        for lifecycle in ExtendedLifecycle:
            count = (states == lifecycle.value).sum().item()
            if count > 0:
                summary[lifecycle.name] = count
        
        return summary
    
    def get_transition_history(
        self,
        seed_id: int,
        limit: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Get transition history for a seed.
        
        Args:
            seed_id: Seed to query
            limit: Maximum transitions to return
            
        Returns:
            List of (from_state, to_state) tuples
        """
        history = self.transition_history[seed_id].cpu().numpy()
        pointer = self.history_pointers[seed_id].item()
        
        # Extract valid transitions
        transitions = []
        for i in range(self.HISTORY_DEPTH):
            idx = (pointer - i - 1) % self.HISTORY_DEPTH
            from_state, to_state = history[idx]
            
            # Skip uninitialized entries
            if from_state == 0 and to_state == 0 and i > 0:
                break
            
            transitions.append((int(from_state), int(to_state)))
            
            if limit and len(transitions) >= limit:
                break
        
        return transitions
    
    def _record_transitions(
        self,
        seed_indices: torch.Tensor,
        from_states: torch.Tensor,
        to_states: torch.Tensor
    ):
        """Record state transitions in history.
        
        Args:
            seed_indices: Seeds that transitioned
            from_states: Previous states
            to_states: New states
        """
        for i, seed_id in enumerate(seed_indices):
            pointer = self.history_pointers[seed_id]
            
            # Write transition (ensure dtype match)
            self.transition_history[seed_id, pointer, 0] = from_states[i].to(torch.int32) if hasattr(from_states[i], 'to') else int(from_states[i])
            self.transition_history[seed_id, pointer, 1] = to_states[i].to(torch.int32) if hasattr(to_states[i], 'to') else int(to_states[i])
            
            # Update pointer (circular buffer)
            self.history_pointers[seed_id] = (pointer + 1) % self.HISTORY_DEPTH
    
    def to_dict(self) -> Dict[str, Any]:
        """Export state data as dictionary.
        
        Returns:
            Dictionary containing all state data
        """
        return {
            'num_seeds': self.num_seeds,
            'state_tensor': self.state_tensor.cpu().numpy(),
            'transition_history': self.transition_history.cpu().numpy(),
            'history_pointers': self.history_pointers.cpu().numpy(),
            'performance_metrics': self.performance_metrics.cpu().numpy(),
            'telemetry_buffer': self.telemetry_buffer.cpu().numpy()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load state data from dictionary.
        
        Args:
            data: Dictionary containing state data
        """
        self.state_tensor = torch.tensor(
            data['state_tensor'],
            dtype=torch.int32,
            device=self.device
        )
        self.transition_history = torch.tensor(
            data['transition_history'],
            dtype=torch.int32,
            device=self.device
        )
        self.history_pointers = torch.tensor(
            data['history_pointers'],
            dtype=torch.int32,
            device=self.device
        )
        self.performance_metrics = torch.tensor(
            data['performance_metrics'],
            dtype=torch.float32,
            device=self.device
        )
        self.telemetry_buffer = torch.tensor(
            data['telemetry_buffer'],
            dtype=torch.float32,
            device=self.device
        )
    
    def reset_telemetry(self):
        """Reset telemetry buffer to zeros."""
        self.telemetry_buffer.zero_()
    
    def get_active_seeds_mask(self) -> torch.Tensor:
        """Get boolean mask of seeds in active computation states.
        
        Returns:
            Boolean tensor indicating active seeds
        """
        states = self.state_tensor[:, self.LIFECYCLE_STATE]
        
        active_states = torch.tensor([
            ExtendedLifecycle.TRAINING.value,
            ExtendedLifecycle.GRAFTING.value,
            ExtendedLifecycle.FINE_TUNING.value
        ], device=self.device)
        
        # Check if state is in active states
        mask = torch.zeros(self.num_seeds, dtype=torch.bool, device=self.device)
        for state in active_states:
            mask |= (states == state)
        
        return mask
    
    def validate_transitions(
        self,
        seed_indices: torch.Tensor,
        target_states: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """Validate proposed state transitions.
        
        Args:
            seed_indices: Seeds to transition
            target_states: Proposed new states
            
        Returns:
            Tuple of (valid_mask, error_messages)
        """
        current_states = self.state_tensor[seed_indices, self.LIFECYCLE_STATE]
        valid_mask = torch.ones_like(seed_indices, dtype=torch.bool)
        errors = []
        
        # This is a simplified validation - full validation
        # should use StateTransition.validate_transition
        for i, (current, target) in enumerate(zip(current_states, target_states)):
            current_enum = ExtendedLifecycle(current.item())
            
            if current_enum.is_terminal:
                valid_mask[i] = False
                errors.append(
                    f"Seed {seed_indices[i]}: Cannot transition from "
                    f"terminal state {current_enum.name}"
                )
        
        return valid_mask, errors

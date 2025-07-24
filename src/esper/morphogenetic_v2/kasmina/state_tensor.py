"""
StateTensor: GPU-resident state management for massively parallel seeds.

This module provides efficient state management for thousands of logical seeds
using GPU-optimized tensor operations.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from .logical_seed import SeedLifecycle, LogicalSeedState

logger = logging.getLogger(__name__)


class StateTensor:
    """
    Efficient GPU-resident state management for all seeds in a layer.
    
    Uses Structure-of-Arrays (SoA) pattern for optimal GPU memory access.
    All state updates are atomic to prevent race conditions.
    """
    
    # State tensor column indices
    LIFECYCLE_STATE_IDX = 0
    BLUEPRINT_ID_IDX = 1
    EPOCHS_IN_STATE_IDX = 2
    GRAFTING_STRATEGY_IDX = 3
    
    def __init__(self, num_seeds: int, device: Optional[torch.device] = None):
        """
        Initialize StateTensor.
        
        Args:
            num_seeds: Number of seeds to manage
            device: Device for tensor storage (default: cuda if available)
        """
        self.num_seeds = num_seeds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Primary state tensor: (num_seeds, 4)
        # Columns: [lifecycle_state, blueprint_id, epochs_in_state, grafting_strategy]
        self.state_tensor = torch.zeros(
            (num_seeds, 4),
            dtype=torch.int32,
            device=self.device
        )
        
        # Additional state arrays for values that don't fit in int32
        self.health_scores = torch.ones(num_seeds, dtype=torch.float32, device=self.device)
        self.error_counts = torch.zeros(num_seeds, dtype=torch.int16, device=self.device)
        self.alpha_blend = torch.zeros(num_seeds, dtype=torch.float32, device=self.device)
        
        # Telemetry accumulators
        self.telemetry_sum = torch.zeros(num_seeds, dtype=torch.float32, device=self.device)
        self.telemetry_sum_sq = torch.zeros(num_seeds, dtype=torch.float32, device=self.device)
        self.telemetry_count = torch.zeros(num_seeds, dtype=torch.int32, device=self.device)
        
        logger.info("StateTensor initialized: num_seeds=%d, device=%s", num_seeds, device)
    
    def get_lifecycle_states(self) -> torch.Tensor:
        """Get lifecycle states for all seeds."""
        return self.state_tensor[:, self.LIFECYCLE_STATE_IDX]
    
    def get_active_seeds(self) -> torch.Tensor:
        """Get boolean mask of active seeds."""
        states = self.get_lifecycle_states()
        return (states == SeedLifecycle.LOADING) | (states == SeedLifecycle.ACTIVE)
    
    def get_dormant_seeds(self) -> torch.Tensor:
        """Get boolean mask of dormant seeds."""
        return self.get_lifecycle_states() == SeedLifecycle.DORMANT
    
    def get_seed_state(self, seed_id: int) -> Dict[str, Any]:
        """Get complete state for a single seed."""
        if seed_id >= self.num_seeds:
            raise ValueError(f"Seed ID {seed_id} out of range [0, {self.num_seeds})")
        
        return {
            "lifecycle_state": self.state_tensor[seed_id, self.LIFECYCLE_STATE_IDX].item(),
            "blueprint_id": self.state_tensor[seed_id, self.BLUEPRINT_ID_IDX].item(),
            "epochs_in_state": self.state_tensor[seed_id, self.EPOCHS_IN_STATE_IDX].item(),
            "grafting_strategy": self.state_tensor[seed_id, self.GRAFTING_STRATEGY_IDX].item(),
            "health_score": self.health_scores[seed_id].item(),
            "error_count": self.error_counts[seed_id].item(),
            "alpha_blend": self.alpha_blend[seed_id].item()
        }
    
    def set_lifecycle_state(self, seed_id: int, state: SeedLifecycle):
        """Set lifecycle state for a single seed."""
        self.state_tensor[seed_id, self.LIFECYCLE_STATE_IDX] = state.value
        self.state_tensor[seed_id, self.EPOCHS_IN_STATE_IDX] = 0  # Reset epoch counter
    
    def batch_set_lifecycle_state(self, seed_ids: torch.Tensor, state: SeedLifecycle):
        """Set lifecycle state for multiple seeds."""
        self.state_tensor[seed_ids, self.LIFECYCLE_STATE_IDX] = state.value
        self.state_tensor[seed_ids, self.EPOCHS_IN_STATE_IDX] = 0
    
    def set_blueprint(self, seed_id: int, blueprint_id: int, grafting_strategy: int = 0):
        """Set blueprint and grafting strategy for a seed."""
        self.state_tensor[seed_id, self.BLUEPRINT_ID_IDX] = blueprint_id
        self.state_tensor[seed_id, self.GRAFTING_STRATEGY_IDX] = grafting_strategy
    
    def increment_epochs(self):
        """Increment epoch counter for all seeds."""
        self.state_tensor[:, self.EPOCHS_IN_STATE_IDX] += 1
    
    def update_health_scores(self, seed_ids: torch.Tensor, health_scores: torch.Tensor):
        """Update health scores for specified seeds."""
        self.health_scores[seed_ids] = health_scores
    
    def increment_error_count(self, seed_id: int) -> int:
        """Increment error count for a seed and return new count."""
        self.error_counts[seed_id] += 1
        return self.error_counts[seed_id].item()
    
    def update_alpha_blend(self, seed_ids: torch.Tensor, alpha_values: torch.Tensor):
        """Update alpha blending factors."""
        self.alpha_blend[seed_ids] = alpha_values
    
    def accumulate_telemetry(self, seed_id: int, chunk_sum: float, chunk_sum_sq: float, count: int):
        """Accumulate telemetry statistics for a seed."""
        self.telemetry_sum[seed_id] += chunk_sum
        self.telemetry_sum_sq[seed_id] += chunk_sum_sq
        self.telemetry_count[seed_id] += count
    
    def get_telemetry_stats(self, seed_id: int) -> Tuple[float, float]:
        """Compute mean and variance from accumulated telemetry."""
        count = self.telemetry_count[seed_id].item()
        if count == 0:
            return 0.0, 0.0
        
        mean = self.telemetry_sum[seed_id].item() / count
        mean_sq = self.telemetry_sum_sq[seed_id].item() / count
        variance = mean_sq - mean * mean
        
        return mean, variance
    
    def reset_telemetry(self):
        """Reset telemetry accumulators."""
        self.telemetry_sum.zero_()
        self.telemetry_sum_sq.zero_()
        self.telemetry_count.zero_()
    
    def get_state_distribution(self) -> Dict[str, int]:
        """Get count of seeds in each lifecycle state."""
        states = self.get_lifecycle_states()
        distribution = {}
        
        for state in SeedLifecycle:
            count = (states == state.value).sum().item()
            distribution[state.name] = count
        
        return distribution
    
    def find_seeds_by_state(self, state: SeedLifecycle) -> torch.Tensor:
        """Find all seed IDs in a given state."""
        states = self.get_lifecycle_states()
        return torch.where(states == state.value)[0]
    
    def find_unhealthy_seeds(self, threshold: float = 0.7) -> torch.Tensor:
        """Find seeds with health scores below threshold."""
        return torch.where(self.health_scores < threshold)[0]
    
    def apply_grafting_ramp(self, ramp_duration: int = 30):
        """Update alpha blend values based on grafting ramp."""
        # Find seeds in LOADING state
        loading_seeds = self.find_seeds_by_state(SeedLifecycle.LOADING)
        
        if len(loading_seeds) > 0:
            # Get epochs in state for these seeds
            epochs = self.state_tensor[loading_seeds, self.EPOCHS_IN_STATE_IDX].float()
            
            # Compute alpha values (linear ramp)
            alphas = torch.clamp(epochs / ramp_duration, 0.0, 1.0)
            
            # Update alpha blend values
            self.alpha_blend[loading_seeds] = alphas
    
    def transition_ready_seeds(self) -> Dict[str, List[int]]:
        """
        Find seeds ready for state transitions.
        
        Returns:
            Dictionary mapping transition types to seed IDs
        """
        transitions = {
            "loading_to_active": [],
            "active_to_fossilized": [],
            "error_to_dormant": []
        }
        
        # Loading -> Active (after ramp complete)
        loading_seeds = self.find_seeds_by_state(SeedLifecycle.LOADING)
        if len(loading_seeds) > 0:
            ramp_complete = self.alpha_blend[loading_seeds] >= 1.0
            ready_seeds = loading_seeds[ramp_complete]
            transitions["loading_to_active"] = ready_seeds.tolist()
        
        # Active -> Fossilized (based on health/performance)
        # This is a placeholder - actual logic will be in Tamiyo
        
        # Error Recovery -> Dormant (after cooldown)
        error_seeds = self.find_seeds_by_state(SeedLifecycle.ERROR_RECOVERY)
        if len(error_seeds) > 0:
            cooldown = 100  # epochs
            epochs = self.state_tensor[error_seeds, self.EPOCHS_IN_STATE_IDX]
            ready_seeds = error_seeds[epochs >= cooldown]
            transitions["error_to_dormant"] = ready_seeds.tolist()
        
        return transitions
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        return {
            "total_seeds": self.num_seeds,
            "state_distribution": self.get_state_distribution(),
            "active_count": self.get_active_seeds().sum().item(),
            "avg_health_score": self.health_scores.mean().item(),
            "min_health_score": self.health_scores.min().item(),
            "max_error_count": self.error_counts.max().item(),
            "avg_alpha_blend": self.alpha_blend[self.get_active_seeds()].mean().item() if self.get_active_seeds().any() else 0.0
        }
    
    def to_logical_seeds(self, layer_id: str, chunk_sizes: List[int]) -> List[LogicalSeedState]:
        """Convert state tensor to list of LogicalSeedState objects."""
        seeds = []
        
        for i in range(self.num_seeds):
            seed_state = LogicalSeedState(
                layer_id=layer_id,
                seed_id=i,
                chunk_idx=i,
                lifecycle_state=SeedLifecycle(self.state_tensor[i, self.LIFECYCLE_STATE_IDX].item()),
                blueprint_id=self.state_tensor[i, self.BLUEPRINT_ID_IDX].item(),
                epochs_in_state=self.state_tensor[i, self.EPOCHS_IN_STATE_IDX].item(),
                grafting_strategy=self.state_tensor[i, self.GRAFTING_STRATEGY_IDX].item(),
                health_score=self.health_scores[i].item(),
                error_count=self.error_counts[i].item()
            )
            seeds.append(seed_state)
        
        return seeds
    
    def save_checkpoint(self) -> Dict[str, torch.Tensor]:
        """Save state for checkpointing."""
        return {
            "state_tensor": self.state_tensor.cpu(),
            "health_scores": self.health_scores.cpu(),
            "error_counts": self.error_counts.cpu(),
            "alpha_blend": self.alpha_blend.cpu()
        }
    
    def load_checkpoint(self, checkpoint: Dict[str, torch.Tensor]):
        """Load state from checkpoint."""
        self.state_tensor = checkpoint["state_tensor"].to(self.device)
        self.health_scores = checkpoint["health_scores"].to(self.device)
        self.error_counts = checkpoint["error_counts"].to(self.device)
        self.alpha_blend = checkpoint["alpha_blend"].to(self.device)
    
    def to(self, device: torch.device) -> "StateTensor":
        """Move StateTensor to a different device."""
        self.device = device
        self.state_tensor = self.state_tensor.to(device)
        self.health_scores = self.health_scores.to(device)
        self.error_counts = self.error_counts.to(device)
        self.alpha_blend = self.alpha_blend.to(device)
        self.telemetry_sum = self.telemetry_sum.to(device)
        self.telemetry_sum_sq = self.telemetry_sum_sq.to(device)
        self.telemetry_count = self.telemetry_count.to(device)
        return self
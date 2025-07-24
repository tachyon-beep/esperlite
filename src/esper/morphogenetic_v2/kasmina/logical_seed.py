"""
LogicalSeed: Abstraction for independent morphogenetic agents.

This module provides the logical view of seeds as independent agents, while
the actual implementation is handled efficiently by the physical layer.
"""

import torch
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SeedLifecycle(IntEnum):
    """Seed lifecycle states (Phase 1 uses simplified 5-state model)."""
    DORMANT = 0
    LOADING = 1  # Will become GERMINATED in Phase 2
    ACTIVE = 2   # Will expand to multiple states in Phase 2
    ERROR_RECOVERY = 3
    FOSSILIZED = 4
    
    # Phase 2 will add:
    # GERMINATED = 1
    # TRAINING = 2
    # GRAFTING = 3
    # STABILIZATION = 4
    # EVALUATING = 5
    # FINE_TUNING = 6
    # FOSSILIZED = 7
    # CULLED = 8
    # CANCELLED = 9
    # ROLLED_BACK = 10


@dataclass
class LogicalSeedState:
    """
    Read-only view of a logical seed's state.
    
    This provides a convenient interface for examining seed state without
    direct tensor manipulation.
    """
    layer_id: str
    seed_id: int
    chunk_idx: int
    lifecycle_state: SeedLifecycle
    blueprint_id: int
    epochs_in_state: int
    grafting_strategy: int
    health_score: float = 1.0
    error_count: int = 0
    
    @property
    def is_active(self) -> bool:
        """Check if seed is in an active state."""
        return self.lifecycle_state in [SeedLifecycle.LOADING, SeedLifecycle.ACTIVE]
    
    @property
    def is_terminal(self) -> bool:
        """Check if seed is in a terminal state."""
        return self.lifecycle_state in [SeedLifecycle.FOSSILIZED, SeedLifecycle.ERROR_RECOVERY]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_id": self.layer_id,
            "seed_id": self.seed_id,
            "chunk_idx": self.chunk_idx,
            "lifecycle_state": self.lifecycle_state.name,
            "lifecycle_value": self.lifecycle_state.value,
            "blueprint_id": self.blueprint_id,
            "epochs_in_state": self.epochs_in_state,
            "grafting_strategy": self.grafting_strategy,
            "health_score": self.health_score,
            "error_count": self.error_count,
            "is_active": self.is_active,
            "is_terminal": self.is_terminal
        }


class LogicalSeed:
    """
    Logical representation of a morphogenetic seed.
    
    This class provides the abstraction of an independent agent that monitors
    and adapts a chunk of the neural network. While seeds appear independent,
    they are actually managed efficiently by the StateTensor in the physical layer.
    """
    
    def __init__(
        self,
        layer_id: str,
        seed_id: int,
        chunk_idx: int,
        chunk_size: int,
        device: torch.device
    ):
        """
        Initialize a logical seed.
        
        Args:
            layer_id: Identifier of the parent layer
            seed_id: Unique seed identifier within the layer
            chunk_idx: Index of the chunk this seed manages
            chunk_size: Size of the managed chunk
            device: Device for tensor operations
        """
        self.layer_id = layer_id
        self.seed_id = seed_id
        self.chunk_idx = chunk_idx
        self.chunk_size = chunk_size
        self.device = device
        
        # Logical state (will be backed by StateTensor in practice)
        self._lifecycle_state = SeedLifecycle.DORMANT
        self._blueprint_id = 0
        self._epochs_in_state = 0
        self._grafting_strategy = 0
        self._health_score = 1.0
        self._error_count = 0
        
        logger.debug(
            "LogicalSeed created: layer=%s, seed=%d, chunk=%d, size=%d",
            layer_id, seed_id, chunk_idx, chunk_size
        )
    
    def get_state(self) -> LogicalSeedState:
        """Get current state as a read-only snapshot."""
        return LogicalSeedState(
            layer_id=self.layer_id,
            seed_id=self.seed_id,
            chunk_idx=self.chunk_idx,
            lifecycle_state=self._lifecycle_state,
            blueprint_id=self._blueprint_id,
            epochs_in_state=self._epochs_in_state,
            grafting_strategy=self._grafting_strategy,
            health_score=self._health_score,
            error_count=self._error_count
        )
    
    def process_chunk(self, chunk: torch.Tensor, blueprint: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        Process a chunk of activations through this seed.
        
        Args:
            chunk: Input chunk tensor
            blueprint: Optional blueprint module for transformation
            
        Returns:
            Processed chunk tensor
        """
        if self._lifecycle_state == SeedLifecycle.DORMANT:
            # Identity operation for dormant seeds
            return chunk
        
        elif self._lifecycle_state == SeedLifecycle.LOADING:
            # Gradual blending during loading
            if blueprint is not None:
                alpha = min(1.0, self._epochs_in_state / 10.0)  # Simple ramp
                blueprint_output = blueprint(chunk)
                return (1 - alpha) * chunk + alpha * blueprint_output
            return chunk
        
        elif self._lifecycle_state == SeedLifecycle.ACTIVE:
            # Full blueprint application
            if blueprint is not None:
                return blueprint(chunk)
            return chunk
        
        elif self._lifecycle_state == SeedLifecycle.ERROR_RECOVERY:
            # Fallback to identity during error recovery
            logger.warning("Seed %d in error recovery, using identity", self.seed_id)
            return chunk
        
        else:  # FOSSILIZED
            # Frozen transformation
            if blueprint is not None:
                with torch.no_grad():
                    return blueprint(chunk)
            return chunk
    
    def compute_health_metrics(self, chunk: torch.Tensor) -> Dict[str, float]:
        """
        Compute health metrics for the managed chunk.
        
        Args:
            chunk: Activation chunk to analyze
            
        Returns:
            Dictionary of health metrics
        """
        with torch.no_grad():
            # Basic health metrics for Phase 1
            chunk_flat = chunk.flatten()
            
            metrics = {
                "mean": chunk_flat.mean().item(),
                "std": chunk_flat.std().item(),
                "sparsity": (chunk_flat.abs() < 1e-6).float().mean().item(),
                "magnitude": chunk_flat.norm().item(),
            }
            
            # Compute health score (simple heuristic for Phase 1)
            # Low variance or high sparsity indicates potential issues
            variance_score = min(1.0, metrics["std"] / 0.1)  # Normalize by expected std
            activity_score = 1.0 - metrics["sparsity"]
            
            self._health_score = 0.5 * variance_score + 0.5 * activity_score
            metrics["health_score"] = self._health_score
            
            return metrics
    
    def request_adaptation(self) -> Optional[Dict[str, Any]]:
        """
        Determine if this seed needs adaptation.
        
        Returns:
            Adaptation request or None
        """
        if self._lifecycle_state != SeedLifecycle.DORMANT:
            return None  # Already adapting
        
        if self._health_score < 0.7:  # Threshold from original implementation
            return {
                "seed_id": self.seed_id,
                "chunk_idx": self.chunk_idx,
                "health_score": self._health_score,
                "reason": "low_health",
                "urgency": 1.0 - self._health_score
            }
        
        return None
    
    def transition_state(self, new_state: SeedLifecycle) -> bool:
        """
        Transition to a new lifecycle state.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition was valid
        """
        # Simple validation for Phase 1
        valid_transitions = {
            SeedLifecycle.DORMANT: [SeedLifecycle.LOADING],
            SeedLifecycle.LOADING: [SeedLifecycle.ACTIVE, SeedLifecycle.ERROR_RECOVERY],
            SeedLifecycle.ACTIVE: [SeedLifecycle.FOSSILIZED, SeedLifecycle.ERROR_RECOVERY],
            SeedLifecycle.ERROR_RECOVERY: [SeedLifecycle.DORMANT],
            SeedLifecycle.FOSSILIZED: [SeedLifecycle.DORMANT]
        }
        
        if new_state in valid_transitions.get(self._lifecycle_state, []):
            logger.info(
                "Seed %d transitioning: %s -> %s",
                self.seed_id,
                self._lifecycle_state.name,
                new_state.name
            )
            self._lifecycle_state = new_state
            self._epochs_in_state = 0
            return True
        
        logger.warning(
            "Invalid transition for seed %d: %s -> %s",
            self.seed_id,
            self._lifecycle_state.name,
            new_state.name
        )
        return False
    
    def increment_epoch(self):
        """Increment epoch counter for the current state."""
        self._epochs_in_state += 1
    
    def set_blueprint(self, blueprint_id: int, grafting_strategy: int = 0):
        """Set the active blueprint and grafting strategy."""
        self._blueprint_id = blueprint_id
        self._grafting_strategy = grafting_strategy
    
    def record_error(self):
        """Record an error occurrence."""
        self._error_count += 1
        if self._error_count >= 3:  # Threshold from original implementation
            self.transition_state(SeedLifecycle.ERROR_RECOVERY)
    
    def reset(self):
        """Reset seed to dormant state."""
        self._lifecycle_state = SeedLifecycle.DORMANT
        self._blueprint_id = 0
        self._epochs_in_state = 0
        self._grafting_strategy = 0
        self._health_score = 1.0
        self._error_count = 0
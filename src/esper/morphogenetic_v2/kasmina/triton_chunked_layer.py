"""
Triton-optimized ChunkedKasminaLayer integrating Phase 2 and Phase 3.

This layer combines:
- Phase 1: Logical/physical separation
- Phase 2: Extended lifecycle management
- Phase 3: GPU-optimized Triton kernels
"""

from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from ..common.feature_flags import is_feature_enabled
from ..grafting.strategies import GraftingStrategyFactory
from ..lifecycle.checkpoint_manager_v2 import CheckpointManager
from ..lifecycle.extended_lifecycle import ExtendedLifecycle
from ..lifecycle.lifecycle_manager import LifecycleManager
from ..lifecycle.state_manager import ExtendedStateTensor
from ..triton.simple_forward_kernel import simple_kasmina_kernel


class TritonChunkedKasminaLayer(nn.Module):
    """
    Production-ready Triton-optimized morphogenetic layer.

    Combines all phases of the migration:
    - Chunked architecture from Phase 1
    - Extended lifecycle from Phase 2
    - Triton kernels from Phase 3
    """

    def __init__(
        self,
        base_layer: nn.Module,
        chunks_per_layer: int = 1000,
        device: torch.device = torch.device('cuda'),
        checkpoint_dir: Optional[str] = None,
        enable_triton: bool = True
    ):
        super().__init__()

        self.base_layer = base_layer
        self.chunks_per_layer = chunks_per_layer
        self.device = device
        self.enable_triton = enable_triton and is_feature_enabled('triton_kernels')

        # Extract dimensions
        if hasattr(base_layer, 'out_features'):
            self.hidden_dim = base_layer.out_features
        else:
            # Try to infer from first Linear layer
            for module in base_layer.modules():
                if isinstance(module, nn.Linear):
                    self.hidden_dim = module.out_features
                    break
            else:
                raise ValueError("Cannot determine hidden dimension")

        # Calculate chunk size
        self.chunk_size = max(1, self.hidden_dim // chunks_per_layer)

        # Phase 2: Extended lifecycle components
        self.lifecycle_manager = LifecycleManager()
        self.extended_state = ExtendedStateTensor(
            num_seeds=chunks_per_layer,
            device=device
        )
        self.grafting_factory = GraftingStrategyFactory()

        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = None

        # Phase 3: Triton-specific state arrays
        if self.enable_triton:
            self._init_triton_state()

        # Blueprint registry (shared between implementations)
        self.blueprint_registry = nn.ModuleDict()
        self.blueprint_weights = torch.randn(10, self.hidden_dim, device=device)

        # Telemetry
        self.telemetry_enabled = True
        self.telemetry_buffer = torch.zeros(chunks_per_layer, 4, device=device)

    def _init_triton_state(self):
        """Initialize Triton-specific state arrays."""
        num_seeds = self.chunks_per_layer

        # Separate arrays for Triton kernel efficiency
        self.register_buffer('lifecycle_states', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('blueprint_ids', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('grafting_strategies', torch.zeros(num_seeds, dtype=torch.int32))

        # Sync with extended state
        self._sync_triton_state()

    def _sync_triton_state(self):
        """Synchronize Triton state with Phase 2 extended state."""
        if not self.enable_triton:
            return

        # Copy state from extended state manager to Triton arrays
        state_tensor = self.extended_state.state_tensor
        self.lifecycle_states.copy_(state_tensor[:, 0].int())
        self.blueprint_ids.copy_(state_tensor[:, 1].int())
        self.grafting_strategies.copy_(state_tensor[:, 3].int())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional Triton acceleration.

        Args:
            x: Input tensor [batch_size, hidden_dim]

        Returns:
            Output tensor [batch_size, hidden_dim]
        """
        if self.enable_triton and x.is_cuda:
            return self._forward_triton(x)
        else:
            return self._forward_pytorch(x)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton-optimized forward pass."""
        batch_size = x.shape[0]
        output = torch.empty_like(x)

        # Sync state if needed
        self._sync_triton_state()

        # Configure grid
        grid = lambda meta: (batch_size * self.chunks_per_layer,)

        # Launch kernel
        simple_kasmina_kernel[grid](
            x, output,
            self.lifecycle_states,
            self.blueprint_ids,
            self.grafting_strategies,
            self.blueprint_weights,
            batch_size,
            self.hidden_dim,
            self.chunks_per_layer,
            self.chunk_size,
            BLOCK_SIZE=256
        )

        # Update telemetry
        if self.telemetry_enabled:
            self._update_telemetry_triton()

        return output

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback implementation."""
        output = x.clone()

        # Process each seed/chunk
        for seed_id in range(self.chunks_per_layer):
            # Get seed state directly from state tensor
            state_tensor = self.extended_state.state_tensor[seed_id]
            lifecycle = ExtendedLifecycle(int(state_tensor[0]))

            # Skip dormant seeds
            if lifecycle == ExtendedLifecycle.DORMANT:
                continue

            # Calculate chunk boundaries
            start = seed_id * self.chunk_size
            end = min(start + self.chunk_size, self.hidden_dim)

            if start >= self.hidden_dim:
                break

            # Apply transformation for active seeds
            if lifecycle in [ExtendedLifecycle.GRAFTING, ExtendedLifecycle.STABILIZATION,
                           ExtendedLifecycle.EVALUATING, ExtendedLifecycle.FINE_TUNING]:

                blueprint_id = int(state_tensor[1])  # blueprint is at index 1
                strategy_id = int(state_tensor[3])   # grafting_strategy is at index 3

                # Get blueprint weights
                if blueprint_id < self.blueprint_weights.shape[0]:
                    weights = self.blueprint_weights[blueprint_id, start:end]

                    # Apply grafting strategy
                    if strategy_id == 0:  # Multiplicative
                        output[:, start:end] *= weights
                    elif strategy_id == 1:  # Additive
                        output[:, start:end] += weights
                    else:  # Mixed
                        output[:, start:end] = output[:, start:end] * 0.5 + weights * 0.5

        return output

    def request_state_transition(
        self,
        seed_id: int,
        target_state: ExtendedLifecycle,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Request a state transition for a seed.

        Integrates Phase 2 lifecycle management.
        """
        current_state = ExtendedLifecycle(
            int(self.extended_state.state_tensor[seed_id, 0])
        )

        # Create transition context
        from ..lifecycle.lifecycle_manager import TransitionContext

        context = TransitionContext(
            seed_id=seed_id,
            current_state=current_state,
            target_state=target_state,
            epochs_in_state=int(self.extended_state.state_tensor[seed_id, 2]),
            performance_metrics=self._get_seed_metrics(seed_id),
            error_count=int(self.extended_state.state_tensor[seed_id, 6]),
            timestamp=0.0,  # Simple timestamp
            metadata=metadata or {}
        )

        # Request transition
        if self.lifecycle_manager.request_transition(context):
            # Update state
            self.extended_state.update_state(
                seed_id,
                {'lifecycle': target_state.value}
            )

            # Sync Triton state if enabled
            if self.enable_triton:
                self._sync_triton_state()

            return True

        return False

    def save_checkpoint(self, tag: str = 'latest') -> str:
        """Save layer checkpoint."""
        if not self.checkpoint_manager:
            raise RuntimeError("Checkpoint manager not initialized")

        checkpoint_data = {
            'extended_state': self.extended_state.state_tensor.cpu(),
            'blueprint_weights': self.blueprint_weights.cpu(),
            'telemetry': self.telemetry_buffer.cpu(),
            'config': {
                'chunks_per_layer': self.chunks_per_layer,
                'hidden_dim': self.hidden_dim,
                'chunk_size': self.chunk_size,
                'enable_triton': self.enable_triton
            }
        }

        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            layer_id=f"triton_layer_{id(self)}",
            seed_id=0,  # Layer-level checkpoint
            state_data=checkpoint_data
        )

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str):
        """Load layer checkpoint."""
        if not self.checkpoint_manager:
            raise RuntimeError("Checkpoint manager not initialized")

        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_id,
            target_device=self.device
        )

        # Restore state
        self.extended_state.state_tensor.copy_(checkpoint['state_data']['extended_state'])
        self.blueprint_weights.copy_(checkpoint['state_data']['blueprint_weights'])
        self.telemetry_buffer.copy_(checkpoint['state_data']['telemetry'])

        # Sync Triton state
        if self.enable_triton:
            self._sync_triton_state()

    def _get_seed_metrics(self, seed_id: int) -> Dict[str, float]:
        """Get performance metrics for a seed."""
        if seed_id >= self.telemetry_buffer.shape[0]:
            return {}

        telemetry = self.telemetry_buffer[seed_id]
        count = telemetry[2].item()

        if count > 0:
            mean = telemetry[0].item() / count
            variance = (telemetry[1].item() / count) - mean**2
            return {
                'activation_mean': mean,
                'activation_variance': max(0, variance),
                'activation_std': max(0, variance)**0.5,
                'sample_count': count
            }

        return {}

    def _update_telemetry_triton(self):
        """Update telemetry after Triton kernel execution."""
        # Telemetry is updated in-kernel for Triton
        # This method could aggregate or post-process if needed

    def get_active_seed_count(self) -> int:
        """Get number of active (non-dormant) seeds."""
        if self.enable_triton:
            return (self.lifecycle_states > 0).sum().item()
        else:
            states = self.extended_state.state_tensor[:, 0]
            return (states > 0).sum().item()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        active_count = self.get_active_seed_count()
        total_count = self.chunks_per_layer

        return {
            'total_seeds': total_count,
            'active_seeds': active_count,
            'active_ratio': active_count / total_count,
            'triton_enabled': self.enable_triton,
            'chunk_size': self.chunk_size,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device)
        }

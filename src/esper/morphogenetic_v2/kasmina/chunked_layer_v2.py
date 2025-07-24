"""
ChunkedKasminaLayer V2: Updated with Phase 2 Extended Lifecycle support.

This module provides the high-performance implementation that manages thousands
of logical seeds through efficient GPU operations, now with 11-state lifecycle,
checkpoint support, and advanced grafting strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from pathlib import Path

from .chunk_manager import ChunkManager
from .logical_seed import LogicalSeed
from .state_tensor import StateTensor

# Import Phase 2 components
from ..lifecycle import (
    ExtendedLifecycle,
    LifecycleManager,
    CheckpointManager,
    CheckpointRecovery,
    ExtendedStateTensor
)
from ..grafting import (
    create_grafting_strategy,
    GraftingConfig,
    GraftingContext
)

logger = logging.getLogger(__name__)


class ChunkedKasminaLayerV2(nn.Module):
    """
    High-performance chunked implementation with Phase 2 Extended Lifecycle.
    
    This layer manages thousands of logical seeds efficiently through:
    - Chunked tensor operations
    - GPU-resident state management with 11-state lifecycle
    - Advanced grafting strategies
    - Checkpoint/recovery system
    - Zero-copy operations where possible
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        num_seeds: int = 1000,
        layer_id: Optional[str] = None,
        enable_telemetry: bool = True,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        grafting_config: Optional[GraftingConfig] = None
    ):
        """
        Initialize ChunkedKasminaLayerV2.
        
        Args:
            base_layer: The underlying neural network layer
            num_seeds: Number of logical seeds (chunks)
            layer_id: Unique identifier for this layer
            enable_telemetry: Whether to collect telemetry
            device: Device for operations
            checkpoint_dir: Directory for checkpoints
            grafting_config: Configuration for grafting strategies
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.num_seeds = num_seeds
        self.layer_id = layer_id or f"kasmina_layer_{id(self)}"
        self.enable_telemetry = enable_telemetry
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine layer dimensions
        if hasattr(base_layer, 'out_features'):
            self.output_dim = base_layer.out_features
            self.input_dim = base_layer.in_features
        elif hasattr(base_layer, 'weight'):
            self.output_dim = base_layer.weight.shape[0]
            self.input_dim = base_layer.weight.shape[1]
        else:
            raise ValueError("Cannot determine dimensions from base layer")
        
        # Initialize Phase 1 components
        self.chunk_manager = ChunkManager(self.output_dim, num_seeds, self.device)
        
        # Initialize Phase 2 components
        self.extended_state = ExtendedStateTensor(num_seeds, self.device)
        self.lifecycle_manager = LifecycleManager(num_seeds)
        
        # Checkpoint management
        if checkpoint_dir is None:
            checkpoint_dir = Path(f"./checkpoints/{self.layer_id}")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints_per_seed=5
        )
        self.checkpoint_recovery = CheckpointRecovery(self.checkpoint_manager)
        
        # Grafting configuration
        self.grafting_config = grafting_config or GraftingConfig()
        self.grafting_strategies = {}  # seed_id -> strategy instance
        
        # Blueprint registry (compatible with Phase 1)
        self.blueprints: Dict[int, nn.Module] = {}
        self.next_blueprint_id = 1
        
        # Performance tracking
        self.forward_count = 0
        self.total_latency = 0.0
        
        # Move base layer to device
        self.base_layer = self.base_layer.to(self.device)
        
        logger.info(
            "ChunkedKasminaLayerV2 initialized: layer_id=%s, num_seeds=%d, dims=(%d, %d)",
            self.layer_id, num_seeds, self.input_dim, self.output_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunked processing and extended lifecycle support.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        start_time = time.perf_counter()
        
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # Get active seeds based on extended lifecycle
        active_mask = self.extended_state.get_active_seeds_mask()
        if not active_mask.any():
            # Fast path: no active seeds
            if self.enable_telemetry:
                self._collect_dormant_telemetry(base_output)
            
            self.forward_count += 1
            self.total_latency += time.perf_counter() - start_time
            return base_output
        
        # Chunked processing for active seeds
        output = self._process_chunks_v2(base_output, active_mask)
        
        # Update telemetry
        if self.enable_telemetry:
            self._update_telemetry_v2(output)
        
        # Update epochs for all seeds
        self.extended_state.increment_epochs()
        
        # Process grafting for seeds in GRAFTING state
        self._update_grafting_progress()
        
        self.forward_count += 1
        self.total_latency += time.perf_counter() - start_time
        
        return output
    
    def _process_chunks_v2(self, base_output: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Process chunks with active seeds using extended lifecycle."""
        # Split output into chunks
        chunks = self.chunk_manager.split_tensor(base_output)
        
        # Get all relevant states at once for efficiency
        states = self.extended_state.get_state()
        blueprint_ids = self.extended_state.state_tensor[:, self.extended_state.BLUEPRINT_ID]
        grafting_strategies = self.extended_state.state_tensor[:, self.extended_state.GRAFTING_STRATEGY]
        
        # Process each chunk
        processed_chunks = []
        for i in range(self.num_seeds):
            chunk = chunks[i]
            
            if active_mask[i]:
                state = ExtendedLifecycle(states[i].item())
                blueprint_id = blueprint_ids[i].item()
                
                # Apply blueprint if available and seed is in appropriate state
                if blueprint_id > 0 and blueprint_id in self.blueprints:
                    blueprint = self.blueprints[blueprint_id]
                    
                    try:
                        # Different processing based on state
                        if state == ExtendedLifecycle.TRAINING:
                            # Full gradient computation
                            blueprint_output = blueprint(chunk)
                        elif state == ExtendedLifecycle.GRAFTING:
                            # Apply grafting strategy
                            alpha = self._get_grafting_alpha(i)
                            with torch.no_grad():
                                blueprint_output = blueprint(chunk)
                            chunk = (1 - alpha) * chunk + alpha * blueprint_output
                        elif state == ExtendedLifecycle.FINE_TUNING:
                            # Limited gradient computation
                            blueprint_output = blueprint(chunk)
                            chunk = 0.1 * chunk + 0.9 * blueprint_output
                        elif state == ExtendedLifecycle.FOSSILIZED:
                            # No gradients, full blueprint
                            with torch.no_grad():
                                chunk = blueprint(chunk)
                                
                    except Exception as e:
                        logger.error("Blueprint execution failed for seed %d: %s", i, e)
                        self.extended_state.increment_error_count(torch.tensor([i]))
                        
                        # Check if should transition to CULLED
                        error_count = self.extended_state.state_tensor[i, self.extended_state.ERROR_COUNT].item()
                        if error_count >= 5:
                            self._request_transition(i, ExtendedLifecycle.CULLED)
            
            processed_chunks.append(chunk)
        
        # Concatenate chunks
        return self.chunk_manager.concatenate_chunks(processed_chunks)
    
    def _get_grafting_alpha(self, seed_id: int) -> float:
        """Get current grafting alpha for a seed."""
        if seed_id not in self.grafting_strategies:
            # Create strategy based on seed's configuration
            strategy_id = self.extended_state.state_tensor[seed_id, self.extended_state.GRAFTING_STRATEGY].item()
            strategy_name = ['linear', 'drift_controlled', 'momentum', 'adaptive', 'stability'][strategy_id]
            self.grafting_strategies[seed_id] = create_grafting_strategy(strategy_name, self.grafting_config)
        
        # Get performance metrics
        perf_metrics = self.extended_state.performance_metrics[seed_id]
        
        # Create grafting context
        context = GraftingContext(
            seed_id=seed_id,
            current_epoch=self.extended_state.state_tensor[seed_id, self.extended_state.EPOCHS_IN_STATE].item(),
            total_epochs=self.grafting_config.ramp_duration,
            current_alpha=0.0,  # Will be tracked by strategy
            metrics={
                'loss': perf_metrics[0].item(),
                'accuracy': perf_metrics[1].item(),
                'stability': perf_metrics[2].item(),
                'efficiency': perf_metrics[3].item()
            }
        )
        
        return self.grafting_strategies[seed_id].compute_alpha(context)
    
    def _update_grafting_progress(self):
        """Update grafting progress for seeds in GRAFTING state."""
        grafting_seeds = self.extended_state.get_seeds_in_state(ExtendedLifecycle.GRAFTING)
        
        for seed_id in grafting_seeds:
            epochs_in_state = self.extended_state.state_tensor[seed_id, self.extended_state.EPOCHS_IN_STATE].item()
            
            # Check if grafting is complete
            if epochs_in_state >= self.grafting_config.ramp_duration:
                self._request_transition(seed_id.item(), ExtendedLifecycle.STABILIZATION)
    
    def _collect_dormant_telemetry(self, output: torch.Tensor):
        """Collect telemetry for dormant seeds."""
        chunks = self.chunk_manager.split_tensor(output)
        dormant_seeds = self.extended_state.get_seeds_in_state(ExtendedLifecycle.DORMANT)
        
        for seed_id in dormant_seeds:
            chunk = chunks[seed_id]
            chunk_sum = chunk.sum().item()
            chunk_sum_sq = (chunk * chunk).sum().item()
            
            # Update telemetry buffer
            self.extended_state.telemetry_buffer[seed_id, 0] += chunk_sum
            self.extended_state.telemetry_buffer[seed_id, 1] += chunk_sum_sq
    
    def _update_telemetry_v2(self, output: torch.Tensor):
        """Update telemetry and performance metrics for all seeds."""
        chunks = self.chunk_manager.split_tensor(output)
        
        # Compute performance metrics for each chunk
        for i in range(self.num_seeds):
            chunk = chunks[i]
            
            # Compute metrics
            chunk_var = chunk.var().item()
            chunk_sparsity = (chunk.abs() < 1e-6).float().mean().item()
            chunk_mean = chunk.mean().item()
            chunk_norm = chunk.norm().item()
            
            # Simple loss proxy (reconstruction error)
            loss_proxy = chunk_var + 0.1 * chunk_sparsity
            
            # Update performance metrics
            self.extended_state.update_performance(
                torch.tensor([i]),
                {
                    'loss': torch.tensor([loss_proxy]),
                    'accuracy': torch.tensor([1.0 - chunk_sparsity]),  # Activity as proxy
                    'stability': torch.tensor([1.0 / (1.0 + chunk_var)]),
                    'efficiency': torch.tensor([min(1.0, chunk_norm)])
                }
            )
    
    def request_germination(self, seed_id: int, blueprint_id: Optional[int] = None, 
                          grafting_strategy: str = 'linear') -> bool:
        """
        Request germination for a specific seed with Phase 2 lifecycle.
        
        Args:
            seed_id: Seed to germinate
            blueprint_id: Blueprint to use (will create if None)
            grafting_strategy: Strategy name for grafting phase
            
        Returns:
            Success status
        """
        # Input validation
        if not isinstance(seed_id, int) or seed_id < 0:
            logger.error("Invalid seed_id: must be non-negative integer")
            return False
        if seed_id >= self.num_seeds:
            logger.error("Invalid seed_id %d (max: %d)", seed_id, self.num_seeds - 1)
            return False
        
        current_state = ExtendedLifecycle(self.extended_state.get_state(torch.tensor([seed_id])).item())
        if current_state != ExtendedLifecycle.DORMANT:
            logger.warning("Seed %d not dormant (state: %s)", seed_id, current_state.name)
            return False
        
        # Create blueprint if needed
        if blueprint_id is None:
            blueprint_id = self._create_default_blueprint(seed_id)
        
        # Validate grafting strategy
        valid_strategies = ['linear', 'drift_controlled', 'momentum', 'adaptive', 'stability']
        if grafting_strategy not in valid_strategies:
            logger.warning(
                "Invalid grafting_strategy '%s', using 'linear'. Valid options: %s",
                grafting_strategy, valid_strategies
            )
            grafting_strategy = 'linear'
        
        # Map strategy name to ID
        strategy_map = {
            'linear': 0,
            'drift_controlled': 1,
            'momentum': 2,
            'adaptive': 3,
            'stability': 4
        }
        strategy_id = strategy_map.get(grafting_strategy, 0)
        
        # Request transition through lifecycle manager
        success = self._request_transition(seed_id, ExtendedLifecycle.GERMINATED)
        
        if success:
            # Update blueprint and strategy
            self.extended_state.update_blueprint(
                torch.tensor([seed_id]),
                torch.tensor([blueprint_id]),
                torch.tensor([strategy_id])
            )
            
            logger.info(
                "Germination requested: seed=%d, blueprint=%d, strategy=%s",
                seed_id, blueprint_id, grafting_strategy
            )
        
        return success
    
    def _request_transition(self, seed_id: int, target_state: ExtendedLifecycle) -> bool:
        """Request state transition through lifecycle manager."""
        current_state = ExtendedLifecycle(self.extended_state.get_state(torch.tensor([seed_id])).item())
        
        # Create transition context
        from ..lifecycle import TransitionContext
        
        # Get current performance metrics
        perf_metrics = self.extended_state.performance_metrics[seed_id]
        
        context = TransitionContext(
            seed_id=seed_id,
            current_state=current_state,
            target_state=target_state,
            epochs_in_state=self.extended_state.state_tensor[seed_id, self.extended_state.EPOCHS_IN_STATE].item(),
            performance_metrics={
                'loss': perf_metrics[0].item(),
                'accuracy': perf_metrics[1].item(),
                'stability': perf_metrics[2].item(),
                'efficiency': perf_metrics[3].item()
            },
            error_count=self.extended_state.state_tensor[seed_id, self.extended_state.ERROR_COUNT].item(),
            timestamp=time.time(),
            metadata={'layer_id': self.layer_id}
        )
        
        # Use lifecycle manager to validate transition
        approved, reason = self.lifecycle_manager.request_transition(
            seed_id=seed_id,
            from_state=current_state,
            to_state=target_state,
            context=context
        )
        
        if approved:
            self.extended_state.set_state(
                torch.tensor([seed_id]),
                torch.tensor([target_state.value])
            )
            return True
        else:
            logger.warning(
                "Transition denied for seed %d: %s -> %s (%s)",
                seed_id, current_state.name, target_state.name, reason
            )
            return False
    
    def save_checkpoint(self, seed_id: int, priority: str = 'normal') -> Optional[str]:
        """Save checkpoint for a seed."""
        # Get current state
        state_data = {
            'lifecycle_state': self.extended_state.get_state(torch.tensor([seed_id])).item(),
            'epochs_in_state': self.extended_state.state_tensor[seed_id, self.extended_state.EPOCHS_IN_STATE].item(),
            'performance_metrics': {
                'loss': self.extended_state.performance_metrics[seed_id, 0].item(),
                'accuracy': self.extended_state.performance_metrics[seed_id, 1].item(),
                'stability': self.extended_state.performance_metrics[seed_id, 2].item(),
                'efficiency': self.extended_state.performance_metrics[seed_id, 3].item()
            },
            'error_count': self.extended_state.state_tensor[seed_id, self.extended_state.ERROR_COUNT].item()
        }
        
        # Get blueprint if exists
        blueprint_id = self.extended_state.state_tensor[seed_id, self.extended_state.BLUEPRINT_ID].item()
        blueprint_state = None
        if blueprint_id > 0 and blueprint_id in self.blueprints:
            blueprint_state = self.blueprints[blueprint_id].state_dict()
        
        # Save checkpoint
        try:
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                layer_id=self.layer_id,
                seed_id=seed_id,
                state_data=state_data,
                blueprint_state=blueprint_state,
                priority=priority
            )
            
            # Update checkpoint reference in state
            self.extended_state.set_checkpoint(
                torch.tensor([seed_id]),
                torch.tensor([hash(checkpoint_id) % 2**31])
            )
            
            return checkpoint_id
        except Exception as e:
            logger.error("Failed to save checkpoint for seed %d: %s", seed_id, e)
            return None
    
    def restore_checkpoint(self, seed_id: int, checkpoint_id: Optional[str] = None) -> bool:
        """Restore seed state from checkpoint."""
        try:
            if checkpoint_id is None:
                # Try to recover latest checkpoint
                recovered = self.checkpoint_recovery.recover_seed_state(self.layer_id, seed_id)
            else:
                recovered = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
            
            if recovered is None:
                return False
            
            # Restore state
            state_data = recovered['state_data']
            self.extended_state.set_state(
                torch.tensor([seed_id]),
                torch.tensor([state_data['lifecycle_state']])
            )
            
            # Restore performance metrics
            if 'performance_metrics' in state_data:
                metrics = state_data['performance_metrics']
                self.extended_state.update_performance(
                    torch.tensor([seed_id]),
                    {k: torch.tensor([v]) for k, v in metrics.items()}
                )
            
            # Restore blueprint if present
            if recovered.get('blueprint_state'):
                blueprint_id = self.extended_state.state_tensor[seed_id, self.extended_state.BLUEPRINT_ID].item()
                if blueprint_id in self.blueprints:
                    self.blueprints[blueprint_id].load_state_dict(recovered['blueprint_state'])
            
            logger.info("Restored checkpoint for seed %d", seed_id)
            return True
            
        except Exception as e:
            logger.error("Failed to restore checkpoint for seed %d: %s", seed_id, e)
            return False
    
    def _create_default_blueprint(self, seed_id: int) -> int:
        """Create a default blueprint for a seed."""
        # Get chunk size for this seed
        chunk_size = self.chunk_manager.get_chunk_size(seed_id)
        
        # More sophisticated blueprint for Phase 2
        blueprint = nn.Sequential(
            nn.Linear(chunk_size, chunk_size * 2),
            nn.LayerNorm(chunk_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(chunk_size * 2, chunk_size * 2),
            nn.LayerNorm(chunk_size * 2),
            nn.ReLU(),
            nn.Linear(chunk_size * 2, chunk_size)
        ).to(self.device)
        
        # Initialize weights
        for module in blueprint.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Register blueprint
        blueprint_id = self.next_blueprint_id
        self.blueprints[blueprint_id] = blueprint
        self.next_blueprint_id += 1
        
        return blueprint_id
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive layer statistics."""
        # Get state distribution
        state_summary = self.extended_state.get_state_summary()
        
        # Performance stats
        stats = {
            "layer_id": self.layer_id,
            "forward_count": self.forward_count,
            "avg_latency_ms": (self.total_latency / max(1, self.forward_count)) * 1000,
            "num_blueprints": len(self.blueprints),
            "output_dim": self.output_dim,
            "chunks_per_seed": self.output_dim / self.num_seeds,
            "state_distribution": state_summary,
            "active_seeds": self.extended_state.get_active_seeds_mask().sum().item(),
            "total_errors": self.extended_state.state_tensor[:, self.extended_state.ERROR_COUNT].sum().item()
        }
        
        return stats
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Get per-seed health metrics
        seed_reports = []
        for i in range(self.num_seeds):
            state = ExtendedLifecycle(self.extended_state.get_state(torch.tensor([i])).item())
            
            # Get transition history
            history = self.extended_state.get_transition_history(i, limit=5)
            
            seed_report = {
                "seed_id": i,
                "state": state.name,
                "epochs_in_state": self.extended_state.state_tensor[i, self.extended_state.EPOCHS_IN_STATE].item(),
                "blueprint_id": self.extended_state.state_tensor[i, self.extended_state.BLUEPRINT_ID].item(),
                "error_count": self.extended_state.state_tensor[i, self.extended_state.ERROR_COUNT].item(),
                "performance": {
                    "loss": self.extended_state.performance_metrics[i, 0].item(),
                    "accuracy": self.extended_state.performance_metrics[i, 1].item(),
                    "stability": self.extended_state.performance_metrics[i, 2].item(),
                    "efficiency": self.extended_state.performance_metrics[i, 3].item()
                },
                "recent_transitions": [
                    {"from": ExtendedLifecycle(f).name, "to": ExtendedLifecycle(t).name}
                    for f, t in history
                ]
            }
            seed_reports.append(seed_report)
        
        return {
            "layer_id": self.layer_id,
            "timestamp": time.time(),
            "seeds": seed_reports,
            "performance": {
                "forward_count": self.forward_count,
                "avg_latency_ms": (self.total_latency / max(1, self.forward_count)) * 1000
            },
            "checkpoints": {
                "total_saved": len(self.checkpoint_manager.list_checkpoints(layer_id=self.layer_id)),
                "checkpoint_dir": str(self.checkpoint_manager.checkpoint_dir)
            }
        }
    
    def to(self, device: torch.device) -> "ChunkedKasminaLayerV2":
        """Move layer to device."""
        super().to(device)
        self.device = device
        self.base_layer = self.base_layer.to(device)
        self.chunk_manager = self.chunk_manager.to(device)
        self.extended_state = ExtendedStateTensor(self.num_seeds, device)
        
        # Move blueprints
        for blueprint_id, blueprint in self.blueprints.items():
            self.blueprints[blueprint_id] = blueprint.to(device)
        
        return self
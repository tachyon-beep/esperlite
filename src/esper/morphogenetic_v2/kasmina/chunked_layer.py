"""
ChunkedKasminaLayer: The physical implementation of the chunked architecture.

This module provides the high-performance implementation that manages thousands
of logical seeds through efficient GPU operations.
"""

import logging
import time
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from .chunk_manager import ChunkManager
from .logical_seed import SeedLifecycle
from .state_tensor import StateTensor

logger = logging.getLogger(__name__)


class ChunkedKasminaLayer(nn.Module):
    """
    High-performance chunked implementation of KasminaLayer.
    
    This layer manages thousands of logical seeds efficiently through:
    - Chunked tensor operations
    - GPU-resident state management
    - Vectorized forward pass
    - Zero-copy operations where possible
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_seeds: int = 1000,
        layer_id: Optional[str] = None,
        enable_telemetry: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ChunkedKasminaLayer.
        
        Args:
            base_layer: The underlying neural network layer
            num_seeds: Number of logical seeds (chunks)
            layer_id: Unique identifier for this layer
            enable_telemetry: Whether to collect telemetry
            device: Device for operations
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

        # Initialize components
        self.chunk_manager = ChunkManager(self.output_dim, num_seeds, self.device)
        self.state_tensor = StateTensor(num_seeds, self.device)

        # Blueprint registry (Phase 1 uses simple dict, Phase 3 will optimize)
        self.blueprints: Dict[int, nn.Module] = {}
        self.next_blueprint_id = 1

        # Performance tracking
        self.forward_count = 0
        self.total_latency = 0.0

        # Move base layer to device
        self.base_layer = self.base_layer.to(self.device)

        logger.info(
            "ChunkedKasminaLayer initialized: layer_id=%s, num_seeds=%d, dims=(%d, %d)",
            self.layer_id, num_seeds, self.input_dim, self.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with chunked processing.
        
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

        # Check if any seeds are active
        active_seeds = self.state_tensor.get_active_seeds()
        if not active_seeds.any():
            # Fast path: no active seeds
            if self.enable_telemetry:
                self._collect_dormant_telemetry(base_output)

            self.forward_count += 1
            self.total_latency += time.perf_counter() - start_time
            return base_output

        # Chunked processing for active seeds
        output = self._process_chunks(base_output, active_seeds)

        # Update telemetry
        if self.enable_telemetry:
            self._update_telemetry(output)

        # Update state
        self.state_tensor.increment_epochs()
        self.state_tensor.apply_grafting_ramp()

        self.forward_count += 1
        self.total_latency += time.perf_counter() - start_time

        return output

    def _process_chunks(self, base_output: torch.Tensor, active_seeds: torch.Tensor) -> torch.Tensor:
        """Process chunks with active seeds."""
        # Split output into chunks
        chunks = self.chunk_manager.split_tensor(base_output)

        # Process each chunk
        processed_chunks = []
        for i in range(self.num_seeds):
            chunk = chunks[i]

            if active_seeds[i]:
                # Get seed state
                state = self.state_tensor.get_seed_state(i)
                blueprint_id = state["blueprint_id"]
                alpha = state["alpha_blend"]

                # Apply blueprint if available
                if blueprint_id in self.blueprints:
                    blueprint = self.blueprints[blueprint_id]
                    try:
                        with torch.no_grad() if state["lifecycle_state"] == SeedLifecycle.FOSSILIZED else torch.enable_grad():
                            blueprint_output = blueprint(chunk)

                        # Blend outputs
                        chunk = (1 - alpha) * chunk + alpha * blueprint_output
                    except Exception as e:
                        logger.error("Blueprint execution failed for seed %d: %s", i, e)
                        error_count = self.state_tensor.increment_error_count(i)
                        if error_count >= 3:
                            self.state_tensor.set_lifecycle_state(i, SeedLifecycle.ERROR_RECOVERY)

            processed_chunks.append(chunk)

        # Concatenate chunks
        return self.chunk_manager.concatenate_chunks(processed_chunks)

    def _collect_dormant_telemetry(self, output: torch.Tensor):
        """Collect telemetry for dormant seeds."""
        chunks = self.chunk_manager.split_tensor(output)

        for i in range(self.num_seeds):
            if self.state_tensor.get_lifecycle_states()[i] == SeedLifecycle.DORMANT:
                chunk = chunks[i]
                chunk_sum = chunk.sum().item()
                chunk_sum_sq = (chunk * chunk).sum().item()
                count = chunk.numel()

                self.state_tensor.accumulate_telemetry(i, chunk_sum, chunk_sum_sq, count)

    def _update_telemetry(self, output: torch.Tensor):
        """Update telemetry for all seeds."""
        chunks = self.chunk_manager.split_tensor(output)

        # Compute health metrics for each chunk
        health_scores = torch.zeros(self.num_seeds, device=self.device)

        for i in range(self.num_seeds):
            chunk = chunks[i]

            # Simple health metric (variance-based)
            chunk_var = chunk.var().item()
            chunk_sparsity = (chunk.abs() < 1e-6).float().mean().item()

            # Health score heuristic
            variance_score = min(1.0, chunk_var / 0.01)  # Normalize
            activity_score = 1.0 - chunk_sparsity
            health_scores[i] = 0.5 * variance_score + 0.5 * activity_score

            # Accumulate statistics
            chunk_sum = chunk.sum().item()
            chunk_sum_sq = (chunk * chunk).sum().item()
            count = chunk.numel()
            self.state_tensor.accumulate_telemetry(i, chunk_sum, chunk_sum_sq, count)

        # Update health scores
        self.state_tensor.update_health_scores(torch.arange(self.num_seeds, device=self.device), health_scores)

    def request_germination(self, seed_id: int, blueprint_id: Optional[int] = None, grafting_strategy: int = 0) -> bool:
        """
        Request germination for a specific seed (Tamiyo interface).
        
        Args:
            seed_id: Seed to germinate
            blueprint_id: Blueprint to use (will create if None)
            grafting_strategy: Strategy for integration
            
        Returns:
            Success status
        """
        if seed_id >= self.num_seeds:
            logger.error("Invalid seed_id %d (max: %d)", seed_id, self.num_seeds - 1)
            return False

        current_state = self.state_tensor.get_lifecycle_states()[seed_id].item()
        if current_state != SeedLifecycle.DORMANT:
            logger.warning("Seed %d not dormant (state: %d)", seed_id, current_state)
            return False

        # Create blueprint if needed
        if blueprint_id is None:
            blueprint_id = self._create_default_blueprint(seed_id)

        # Update state
        self.state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.LOADING)
        self.state_tensor.set_blueprint(seed_id, blueprint_id, grafting_strategy)

        logger.info(
            "Germination requested: seed=%d, blueprint=%d, strategy=%d",
            seed_id, blueprint_id, grafting_strategy
        )
        return True

    def cancel_germination(self, seed_id: int) -> bool:
        """Cancel ongoing germination."""
        current_state = self.state_tensor.get_lifecycle_states()[seed_id].item()

        if current_state in [SeedLifecycle.LOADING, SeedLifecycle.ACTIVE]:
            self.state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.DORMANT)
            self.state_tensor.set_blueprint(seed_id, 0, 0)
            self.state_tensor.update_alpha_blend(
                torch.tensor([seed_id], device=self.device),
                torch.zeros(1, device=self.device)
            )
            logger.info("Germination cancelled for seed %d", seed_id)
            return True

        return False

    def _create_default_blueprint(self, seed_id: int) -> int:
        """Create a default blueprint for a seed."""
        # Get chunk size for this seed
        chunk_size = self.chunk_manager.get_chunk_size(seed_id)

        # Simple feedforward blueprint for Phase 1
        blueprint = nn.Sequential(
            nn.Linear(chunk_size, chunk_size * 2),
            nn.ReLU(),
            nn.Linear(chunk_size * 2, chunk_size)
        ).to(self.device)

        # Register blueprint
        blueprint_id = self.next_blueprint_id
        self.blueprints[blueprint_id] = blueprint
        self.next_blueprint_id += 1

        return blueprint_id

    def register_blueprint(self, blueprint: nn.Module) -> int:
        """Register a pre-created blueprint."""
        blueprint_id = self.next_blueprint_id
        self.blueprints[blueprint_id] = blueprint.to(self.device)
        self.next_blueprint_id += 1
        return blueprint_id

    def get_layer_stats(self) -> Dict[str, Any]:
        """Get layer statistics for monitoring."""
        stats = self.state_tensor.get_summary_stats()

        # Add performance stats
        stats.update({
            "layer_id": self.layer_id,
            "forward_count": self.forward_count,
            "avg_latency_ms": (self.total_latency / max(1, self.forward_count)) * 1000,
            "num_blueprints": len(self.blueprints),
            "output_dim": self.output_dim,
            "chunks_per_seed": self.output_dim / self.num_seeds
        })

        return stats

    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report for Tamiyo."""
        # Get logical seed states
        chunk_sizes = [self.chunk_manager.get_chunk_size(i) for i in range(self.num_seeds)]
        logical_seeds = self.state_tensor.to_logical_seeds(self.layer_id, chunk_sizes)

        # Get telemetry stats
        telemetry_stats = []
        for i in range(self.num_seeds):
            mean, variance = self.state_tensor.get_telemetry_stats(i)
            telemetry_stats.append({"mean": mean, "variance": variance})

        # Reset telemetry for next epoch
        self.state_tensor.reset_telemetry()

        return {
            "layer_id": self.layer_id,
            "timestamp": time.time(),
            "seeds": [seed.to_dict() for seed in logical_seeds],
            "telemetry": telemetry_stats,
            "performance": {
                "forward_count": self.forward_count,
                "avg_latency_ms": (self.total_latency / max(1, self.forward_count)) * 1000
            }
        }

    def to(self, device: torch.device) -> "ChunkedKasminaLayer":
        """Move layer to device."""
        super().to(device)
        self.device = device
        self.base_layer = self.base_layer.to(device)
        self.chunk_manager = self.chunk_manager.to(device)
        self.state_tensor = self.state_tensor.to(device)

        # Move blueprints
        for blueprint_id, blueprint in self.blueprints.items():
            self.blueprints[blueprint_id] = blueprint.to(device)

        return self

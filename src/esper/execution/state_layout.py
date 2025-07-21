"""
GPU-optimized state management for KasminaLayer.

This module implements the Structure-of-Arrays (SoA) memory layout for optimal
GPU memory coalescing during kernel execution.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class SeedLifecycleState(IntEnum):
    """Enumeration of seed lifecycle states."""

    DORMANT = 0
    LOADING = 1
    ACTIVE = 2
    ERROR_RECOVERY = 3
    FOSSILIZED = 4


@dataclass
class KasminaStateLayout:
    """
    Structure-of-Arrays layout optimized for GPU memory coalescing.

    This class manages the state tensors for all seeds in a KasminaLayer,
    using a Structure-of-Arrays (SoA) layout for optimal GPU performance.
    """

    # Core lifecycle management
    lifecycle_states: torch.Tensor  # uint8: SeedLifecycleState values
    active_kernel_id: torch.Tensor  # uint64: Hash of currently loaded kernel artifact
    alpha_blend: torch.Tensor  # float16: Blending coefficient for grafting

    # Performance tracking
    health_accumulator: torch.Tensor  # float32: Running statistics for telemetry
    last_update_epoch: torch.Tensor  # uint32: For staleness tracking
    exec_latency_us: torch.Tensor  # uint16: Per-seed execution time measurement

    # Error handling
    error_count: torch.Tensor  # uint16: Count of consecutive failures
    fallback_active: torch.Tensor  # bool: Whether using fallback execution

    def __init__(self, num_seeds: int, device: torch.device):
        """
        Initialize state tensors for the specified number of seeds.

        Args:
            num_seeds: Number of seeds to manage
            device: PyTorch device (CPU or GPU)
        """
        self.num_seeds = num_seeds
        self.device = device

        # CPU-based active seed tracking for performance optimization
        self._active_seed_count = 0

        # Initialize all state tensors
        self.lifecycle_states = torch.zeros(num_seeds, dtype=torch.uint8, device=device)
        self.active_kernel_id = torch.zeros(num_seeds, dtype=torch.int64, device=device)
        self.alpha_blend = torch.zeros(num_seeds, dtype=torch.float16, device=device)
        self.health_accumulator = torch.zeros(
            num_seeds, dtype=torch.float32, device=device
        )
        self.last_update_epoch = torch.zeros(
            num_seeds, dtype=torch.int32, device=device
        )
        self.exec_latency_us = torch.zeros(num_seeds, dtype=torch.uint16, device=device)
        self.error_count = torch.zeros(num_seeds, dtype=torch.int16, device=device)
        self.fallback_active = torch.zeros(num_seeds, dtype=torch.bool, device=device)

        logger.debug(
            f"Initialized KasminaStateLayout with {num_seeds} seeds on {device}"
        )

    def get_active_seeds(self) -> torch.Tensor:
        """
        Get mask of seeds that are currently active.

        Returns:
            Boolean tensor indicating which seeds are active
        """
        return self.lifecycle_states == SeedLifecycleState.ACTIVE

    def get_dormant_seeds(self) -> torch.Tensor:
        """
        Get mask of seeds that are currently dormant.

        Returns:
            Boolean tensor indicating which seeds are dormant
        """
        return self.lifecycle_states == SeedLifecycleState.DORMANT

    def transition_seed_state(
        self,
        seed_idx: int,
        new_state: SeedLifecycleState,
        kernel_id: Optional[int] = None,
    ) -> None:
        """
        Transition a seed to a new lifecycle state.

        Args:
            seed_idx: Index of the seed to transition
            new_state: New lifecycle state
            kernel_id: Optional kernel ID to associate with the seed
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        old_state = SeedLifecycleState(self.lifecycle_states[seed_idx].item())
        self.lifecycle_states[seed_idx] = new_state

        # Update CPU-based active seed counter
        if (
            old_state != SeedLifecycleState.ACTIVE
            and new_state == SeedLifecycleState.ACTIVE
        ):
            self._active_seed_count += 1
        elif (
            old_state == SeedLifecycleState.ACTIVE
            and new_state != SeedLifecycleState.ACTIVE
        ):
            self._active_seed_count -= 1

        if kernel_id is not None:
            self.active_kernel_id[seed_idx] = kernel_id

        # Reset error count on successful transition to ACTIVE
        if new_state == SeedLifecycleState.ACTIVE:
            self.error_count[seed_idx] = 0
            self.fallback_active[seed_idx] = False

        logger.debug(
            f"Seed {seed_idx} transitioned from {old_state.name} to {new_state.name}"
        )

    def increment_error_count(self, seed_idx: int) -> int:
        """
        Increment error count for a seed and return the new count.

        Args:
            seed_idx: Index of the seed

        Returns:
            New error count
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        current_count = self.error_count[seed_idx].item()
        self.error_count[seed_idx] = current_count + 1
        new_count = current_count + 1

        # Activate fallback if too many errors
        if new_count >= 3:
            self.fallback_active[seed_idx] = True
            self.transition_seed_state(seed_idx, SeedLifecycleState.ERROR_RECOVERY)

        return new_count

    def update_telemetry(
        self, seed_idx: int, latency_us: int, health_score: float
    ) -> None:
        """
        Update telemetry data for a seed.

        Args:
            seed_idx: Index of the seed
            latency_us: Execution latency in microseconds
            health_score: Health score (0.0 to 1.0)
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        self.exec_latency_us[seed_idx] = min(latency_us, 65535)  # Clamp to uint16 max

        # Update health accumulator with exponential moving average
        alpha = 0.1  # Smoothing factor
        current_health = self.health_accumulator[seed_idx].item()

        # For first update (when current_health is near 0), use the health_score directly
        if abs(current_health) < 1e-6:
            self.health_accumulator[seed_idx] = health_score
        else:
            self.health_accumulator[seed_idx] = (
                1 - alpha
            ) * current_health + alpha * health_score

    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive statistics about the current state.

        Returns:
            Dictionary containing various state statistics
        """
        states = self.lifecycle_states.cpu().numpy()

        return {
            "num_seeds": self.num_seeds,
            "dormant_seeds": int((states == SeedLifecycleState.DORMANT).sum()),
            "active_seeds": int((states == SeedLifecycleState.ACTIVE).sum()),
            "loading_seeds": int((states == SeedLifecycleState.LOADING).sum()),
            "error_recovery_seeds": int(
                (states == SeedLifecycleState.ERROR_RECOVERY).sum()
            ),
            "fossilized_seeds": int((states == SeedLifecycleState.FOSSILIZED).sum()),
            "avg_health": float(self.health_accumulator.mean().item()),
            "avg_latency_us": float(self.exec_latency_us.float().mean().item()),
            "total_errors": int(self.error_count.sum().item()),
            "fallback_active_count": int(self.fallback_active.sum().item()),
        }

    def reset_seed(self, seed_idx: int) -> None:
        """
        Reset a seed to its initial state.

        Args:
            seed_idx: Index of the seed to reset
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        self.lifecycle_states[seed_idx] = SeedLifecycleState.DORMANT
        self.active_kernel_id[seed_idx] = 0
        self.alpha_blend[seed_idx] = 0.0
        self.health_accumulator[seed_idx] = 0.0
        self.last_update_epoch[seed_idx] = 0
        self.exec_latency_us[seed_idx] = 0
        self.error_count[seed_idx] = 0
        self.fallback_active[seed_idx] = False

        logger.debug("Reset seed %d to initial state", seed_idx)

    def has_active_seeds(self) -> bool:
        """
        Fast check if any seeds are currently active.

        This uses a CPU-based counter for maximum performance and avoids
        any GPU operations in the common case where all seeds are dormant.

        Returns:
            True if any seeds are active, False otherwise
        """
        return self._active_seed_count > 0

    def get_active_count(self) -> int:
        """
        Get the count of currently active seeds.

        Uses CPU-based tracking for optimal performance.

        Returns:
            Number of currently active seeds
        """
        return self._active_seed_count

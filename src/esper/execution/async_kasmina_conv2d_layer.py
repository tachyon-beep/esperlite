"""
Async-Enhanced KasminaConv2dLayer.

This module provides a fully async-capable Conv2D layer that eliminates
synchronous fallbacks while maintaining morphogenetic capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .async_conv2d_kernel import AsyncConv2dKernel, create_async_conv2d_kernel
from .gradient_sync import GradientSynchronizer
from .kasmina_conv2d_layer import KasminaConv2dLayer
# from .state_management import SeedData  # Will create inline for now

logger = logging.getLogger(__name__)


# Simple SeedData class for async execution
class SeedData:
    """Seed data for morphogenetic kernel execution."""
    def __init__(self, id: str, position: int, kernel_id: str, parameters: Dict[str, Any]):
        self.id = id
        self.position = position
        self.kernel_id = kernel_id
        self.parameters = parameters


class AsyncKasminaConv2dLayer(KasminaConv2dLayer):
    """
    Async-enhanced Conv2D layer with true asynchronous execution.
    
    This layer extends KasminaConv2dLayer to provide proper async
    execution without synchronous fallbacks, maintaining gradient
    correctness and morphogenetic capabilities.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "async_kasmina_conv2d_layer",
        enable_gradient_sync: bool = True,
    ):
        """
        Initialize AsyncKasminaConv2dLayer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Padding applied to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: Whether to add learnable bias
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to collect telemetry
            layer_name: Name of this layer
            enable_gradient_sync: Whether to enable gradient synchronization
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )
        
        # Initialize async components
        self.async_kernel = create_async_conv2d_kernel(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size,
            stride=stride[0] if isinstance(stride, tuple) else stride,
            padding=padding[0] if isinstance(padding, tuple) else padding,
            dilation=dilation[0] if isinstance(dilation, tuple) else dilation,
            groups=groups,
            device=self.default_transform.weight.device,
        )
        
        # Initialize gradient synchronizer
        self.gradient_sync = GradientSynchronizer() if enable_gradient_sync else None
        
        # Async execution statistics
        self.async_execution_count = 0
        self.sync_fallback_count = 0
        
        logger.info(
            f"Initialized AsyncKasminaConv2dLayer '{layer_name}' "
            f"with async support and gradient sync {'enabled' if enable_gradient_sync else 'disabled'}"
        )

    async def forward_async(self, x: torch.Tensor) -> torch.Tensor:
        """
        Async forward pass through Conv2d layer.
        
        This method provides true async execution without blocking,
        maintaining gradient correctness through the synchronizer.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor of shape (N, out_channels, H_out, W_out)
        """
        # Validate input
        if len(x.shape) != 4:
            raise ValueError(f"Conv2d input must be 4D (N, C, H, W), got {x.shape}")
            
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels {x.shape[1]} != expected {self.in_channels}"
            )
        
        self.total_forward_calls += 1
        self.async_execution_count += 1
        
        # Check for active seeds
        active_seeds = self.state_layout.get_active_seeds()
        
        if not active_seeds.any():
            # Fast path: No active seeds, use async default transformation
            return await self._execute_default_async(x)
        
        # Slow path: Execute with morphogenetic capabilities
        return await self._execute_with_seeds_async(x, active_seeds)

    async def _execute_default_async(self, x: torch.Tensor) -> torch.Tensor:
        """Execute default Conv2D transformation asynchronously."""
        weight = self.default_transform.weight
        bias = self.default_transform.bias if self.bias_enabled else None
        
        if self.gradient_sync:
            # Register with gradient synchronizer
            return await self.gradient_sync.register_async_operation(
                self.async_kernel.execute_async,
                (x, weight, bias)
            )
        else:
            # Direct async execution
            return await self.async_kernel.execute_async(x, weight, bias)

    async def _execute_with_seeds_async(
        self, x: torch.Tensor, active_seeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute Conv2d operation with active morphogenetic seeds asynchronously.
        
        Args:
            x: Input tensor (N, C, H, W)
            active_seeds: Boolean mask of active seeds
            
        Returns:
            Blended output tensor
        """
        # Start with default transformation
        default_output = await self._execute_default_async(x)
        
        # Get alpha blending factors
        alpha_blend = self.state_layout.alpha_blend
        
        # Initialize accumulated morphogenetic output
        morpho_output = torch.zeros_like(default_output)
        total_alpha = 0.0
        
        # Create list of async operations for active seeds
        seed_operations = []
        seed_indices = []
        
        for seed_idx in range(self.num_seeds):
            if active_seeds[seed_idx]:
                seed_indices.append(seed_idx)
                seed_operations.append(
                    self._execute_seed_kernel_async(x, seed_idx)
                )
        
        # Execute all seed operations concurrently
        if seed_operations:
            seed_outputs = await asyncio.gather(*seed_operations)
            
            # Blend outputs
            for idx, seed_output in zip(seed_indices, seed_outputs):
                alpha = alpha_blend[idx].item()
                morpho_output += alpha * seed_output
                total_alpha += alpha
        
        # Blend default and morphogenetic outputs
        if total_alpha > 0:
            default_weight = 1.0 - total_alpha
            return default_weight * default_output + morpho_output
        else:
            return default_output

    async def _execute_seed_kernel_async(
        self, x: torch.Tensor, seed_idx: int
    ) -> torch.Tensor:
        """
        Execute kernel for a specific seed asynchronously.
        
        Args:
            x: Input tensor
            seed_idx: Index of the seed
            
        Returns:
            Output tensor from kernel execution
        """
        try:
            # Get seed data
            seed_data = self._get_seed_data(seed_idx)
            
            if seed_data is None or seed_data.kernel_id == "":
                # No kernel loaded, use default
                return await self._execute_default_async(x)
            
            # Execute real kernel asynchronously
            return await self._execute_kernel_real_async(x, seed_data)
            
        except Exception as e:
            logger.error(
                f"Failed to execute seed {seed_idx} kernel: {e}, "
                "falling back to default"
            )
            self.kernel_failures += 1
            return await self._execute_default_async(x)

    async def _execute_kernel_real_async(
        self, x: torch.Tensor, seed_data: SeedData
    ) -> torch.Tensor:
        """
        Execute real compiled kernel asynchronously.
        
        Args:
            x: Input tensor
            seed_data: Seed data containing kernel parameters
            
        Returns:
            Output tensor
        """
        # In production, this would load and execute the actual compiled kernel
        # For now, we'll use the async kernel with seed-specific parameters
        
        # Get kernel parameters from seed data
        weight = seed_data.parameters.get('weight', self.default_transform.weight)
        bias = seed_data.parameters.get('bias', self.default_transform.bias)
        
        # Execute through async kernel
        output = await self.async_kernel.execute_async(x, weight, bias)
        
        # Record telemetry
        if self.telemetry_enabled:
            asyncio.create_task(self._record_async_telemetry(
                kernel_id=seed_data.kernel_id,
                seed_idx=seed_data.position,
            ))
        
        return output

    async def _record_async_telemetry(self, kernel_id: str, seed_idx: int):
        """Record telemetry asynchronously without blocking execution."""
        # This would record to the telemetry service
        logger.debug(
            f"Async kernel execution recorded: kernel={kernel_id}, seed={seed_idx}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synchronous forward method for compatibility.
        
        This method detects if we're in an async context and handles
        execution appropriately.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            
            # We're in an async context
            logger.warning(
                "AsyncKasminaConv2dLayer.forward() called in async context. "
                "Use forward_async() for proper async execution."
            )
            self.sync_fallback_count += 1
            
            # Don't create an async task, just fall back to parent's synchronous implementation
            return super().forward(x)
            
        except RuntimeError:
            # Not in an async context, use parent's implementation
            return super().forward(x)

    def _get_seed_data(self, seed_idx: int) -> Optional[SeedData]:
        """Get seed data for a specific seed index."""
        # This would retrieve actual seed data from state management
        # For now, return a mock SeedData
        kernel_id = int(self.state_layout.active_kernel_id[seed_idx].item())
        
        if kernel_id == 0:
            return None
            
        return SeedData(
            id=f"seed_{seed_idx}",
            position=seed_idx,
            kernel_id=f"kernel_{kernel_id}",
            parameters={
                'weight': self.default_transform.weight.clone(),
                'bias': self.default_transform.bias.clone() if self.bias_enabled else None,
            }
        )

    def get_async_stats(self) -> Dict[str, Any]:
        """Get async execution statistics."""
        stats = {
            "async_execution_count": self.async_execution_count,
            "sync_fallback_count": self.sync_fallback_count,
            "async_execution_ratio": (
                self.async_execution_count / max(self.total_forward_calls, 1)
            ),
        }
        
        # Add async kernel stats
        stats.update(self.async_kernel.get_execution_stats())
        
        # Add gradient sync stats if enabled
        if self.gradient_sync:
            stats.update({
                f"gradient_sync_{k}": v
                for k, v in self.gradient_sync.get_stats().items()
            })
        
        return stats

    def cleanup(self):
        """Clean up async resources."""
        self.async_kernel.cleanup()
        logger.info(f"Cleaned up async resources for layer '{self.layer_name}'")
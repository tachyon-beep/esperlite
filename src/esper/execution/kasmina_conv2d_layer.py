"""
KasminaConv2dLayer: Conv2d-specific execution layer for morphogenetic kernels.

This module implements a specialized KasminaLayer that preserves spatial semantics
for convolutional operations while enabling dynamic kernel loading.
"""

import logging
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

from .kasmina_layer import KasminaLayer

logger = logging.getLogger(__name__)


class KasminaConv2dLayer(KasminaLayer):
    """
    Conv2d-specific KasminaLayer that preserves spatial semantics.

    This layer maintains the convolutional structure while enabling
    morphogenetic capabilities through dynamic kernel loading.
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
        layer_name: str = "kasmina_conv2d_layer",
    ):
        """
        Initialize KasminaConv2dLayer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Padding applied to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: Whether to add learnable bias
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to collect telemetry
            layer_name: Name of this layer for telemetry
        """
        # Call parent constructor with channel counts as size dimensions
        super().__init__(
            input_size=in_channels,
            output_size=out_channels,
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )

        # Store convolution parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.bias_enabled = bias

        # Replace default Linear layer with Conv2d
        self.default_transform = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Store conv parameters for kernel adaptation
        self.conv_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": groups,
            "bias": bias,
        }

        logger.info(
            f"Initialized KasminaConv2dLayer '{layer_name}' "
            f"({in_channels}→{out_channels}, kernel={kernel_size})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conv2d layer with morphogenetic capabilities.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, out_channels, H_out, W_out)
        """
        # Validate input shape for Conv2d
        if len(x.shape) != 4:
            raise ValueError(f"Conv2d input must be 4D (N, C, H, W), got {x.shape}")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels {x.shape[1]} != expected {self.in_channels}"
            )

        self.total_forward_calls += 1

        # Check for active seeds
        active_seeds = self.state_layout.get_active_seeds()

        if not active_seeds.any():
            # Fast path: No active seeds, just use default transformation
            return self.default_transform(x)

        # Slow path: Execute with morphogenetic capabilities
        return self._execute_with_seeds(x, active_seeds)

    def _execute_with_seeds(
        self, x: torch.Tensor, active_seeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute Conv2d operation with active morphogenetic seeds.

        Args:
            x: Input tensor (N, C, H, W)
            active_seeds: Boolean mask of active seeds

        Returns:
            Blended output tensor
        """
        _, _, _, _ = x.shape

        # Start with default transformation
        default_output = self.default_transform(x)

        # Get alpha blending factors for active seeds
        alpha_blend = self.state_layout.alpha_blend

        # Initialize accumulated morphogenetic output
        morpho_output = torch.zeros_like(default_output)
        total_alpha = 0.0

        # Execute each active seed
        for seed_idx in range(self.num_seeds):
            if active_seeds[seed_idx]:
                try:
                    # Execute morphogenetic kernel for this seed
                    seed_output = self._execute_kernel_placeholder(x, seed_idx)

                    # Ensure output shape consistency
                    if seed_output.shape != default_output.shape:
                        logger.warning(
                            f"Seed {seed_idx} output shape mismatch: "
                            f"{seed_output.shape} != {default_output.shape}"
                        )
                        # Try to adapt the shape
                        seed_output = self._adapt_output_shape(
                            seed_output, default_output.shape
                        )

                    # Blend with alpha factor
                    alpha = alpha_blend[seed_idx].item()
                    morpho_output += alpha * seed_output
                    total_alpha += alpha

                    self.total_kernel_executions += 1

                except Exception as e:
                    logger.error("Seed %d execution failed: %s", seed_idx, e)
                    # Mark seed as error and continue
                    self.state_layout.increment_error_count(seed_idx)

        # Blend default and morphogenetic outputs
        if total_alpha > 0:
            # Normalize morphogenetic contribution
            morpho_output = morpho_output / total_alpha

            # Blend: (1 - total_alpha) * default + total_alpha * morpho
            final_alpha = min(total_alpha, 1.0)
            output = (1.0 - final_alpha) * default_output + final_alpha * morpho_output
        else:
            output = default_output

        return output

    def _execute_kernel_placeholder(
        self, x: torch.Tensor, seed_idx: int
    ) -> torch.Tensor:
        """
        Placeholder kernel execution for Conv2d operations.

        This will be replaced with actual loaded kernels in production.

        Args:
            x: Input tensor (N, C, H, W)
            seed_idx: Index of the seed to execute

        Returns:
            Transformed tensor maintaining Conv2d spatial semantics
        """
        # Apply convolution-based transformation that preserves spatial structure
        # This is still a placeholder but maintains proper Conv2d semantics

        # Create a simple perturbation based on seed index
        seed_factor = 0.5 + seed_idx * 0.1

        # Apply the default convolution with a small modification
        # In practice, this would load and execute a compiled kernel
        output = self.default_transform(x) * seed_factor

        # Add small spatial-aware noise to simulate morphogenetic adaptation
        if self.training:
            noise_scale = 0.01 * (seed_idx + 1)
            spatial_noise = torch.randn_like(output) * noise_scale
            output += spatial_noise

        return output

    def _adapt_output_shape(
        self, tensor: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        """
        Adapt tensor shape to match target shape.

        Args:
            tensor: Input tensor to reshape
            target_shape: Target shape to match

        Returns:
            Reshaped tensor
        """
        if tensor.shape == target_shape:
            return tensor

        # Handle different adaptation strategies
        if len(tensor.shape) == len(target_shape):
            if tensor.shape[0] == target_shape[0]:  # Same batch size
                # Use adaptive pooling to match spatial dimensions
                if len(target_shape) == 4:  # Conv2d output
                    target_h, target_w = target_shape[2], target_shape[3]
                    adapted = nn.functional.adaptive_avg_pool2d(
                        tensor, (target_h, target_w)
                    )

                    # Handle channel dimension mismatch
                    if adapted.shape[1] != target_shape[1]:
                        if adapted.shape[1] > target_shape[1]:
                            # Reduce channels by taking first N channels
                            adapted = adapted[:, : target_shape[1], :, :]
                        else:
                            # Pad channels
                            padding_channels = target_shape[1] - adapted.shape[1]
                            channel_padding = torch.zeros(
                                adapted.shape[0],
                                padding_channels,
                                adapted.shape[2],
                                adapted.shape[3],
                                device=adapted.device,
                                dtype=adapted.dtype,
                            )
                            adapted = torch.cat([adapted, channel_padding], dim=1)

                    return adapted

        # Fallback: reshape/view if possible
        try:
            return tensor.view(target_shape)
        except RuntimeError:
            logger.warning(
                f"Cannot adapt shape {tensor.shape} to {target_shape}, " f"using zeros"
            )
            return torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)

    def get_layer_stats(self) -> Dict[str, Any]:
        """
        Get layer statistics including Conv2d-specific information.

        Returns:
            Dictionary containing layer statistics
        """
        base_stats = super().get_layer_stats()

        # Add Conv2d-specific statistics
        conv_stats = {
            "conv_params": self.conv_params,
            "input_channels": self.in_channels,
            "output_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        }

        base_stats.update(conv_stats)
        return base_stats

    def copy_weights_from_conv2d(self, source_conv: nn.Conv2d) -> None:
        """
        Copy weights from a source Conv2d layer.

        Args:
            source_conv: Source Conv2d layer to copy weights from
        """
        if not isinstance(source_conv, nn.Conv2d):
            raise ValueError("Source must be a Conv2d layer")

        # Verify compatibility
        if (
            source_conv.in_channels != self.in_channels
            or source_conv.out_channels != self.out_channels
        ):
            raise ValueError(
                f"Channel mismatch: source({source_conv.in_channels}, "
                f"{source_conv.out_channels}) != target({self.in_channels}, "
                f"{self.out_channels})"
            )

        # Copy weights exactly
        with torch.no_grad():
            self.default_transform.weight.copy_(source_conv.weight)
            if source_conv.bias is not None and self.default_transform.bias is not None:
                self.default_transform.bias.copy_(source_conv.bias)

        logger.info(
            f"Copied weights from Conv2d({source_conv.in_channels}, "
            f"{source_conv.out_channels}) to {self.layer_name}"
        )

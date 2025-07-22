"""
KasminaLayerNormLayer for Layer Normalization with morphogenetic capabilities.

This module provides a specialized KasminaLayer for PyTorch's LayerNorm
that preserves normalization semantics while enabling dynamic parameter adaptation.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.execution.kasmina_layer import KasminaLayer

logger = logging.getLogger(__name__)


class KasminaLayerNormLayer(KasminaLayer):
    """
    KasminaLayer specialized for Layer Normalization.

    This layer preserves LayerNorm behavior while adding morphogenetic
    capabilities for adaptive scale and shift parameters.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "layernorm_layer",
    ):
        """
        Initialize KasminaLayerNormLayer.

        Args:
            normalized_shape: Input shape from an expected input
            eps: A value added to denominator for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
            bias: Whether to use bias parameter
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to enable telemetry
            layer_name: Name of the layer for logging
        """
        # For LayerNorm, we adapt the scale and bias parameters
        # Input/output size is the same as normalized_shape
        super().__init__(
            input_size=normalized_shape,
            output_size=(
                normalized_shape * 2 if elementwise_affine else normalized_shape
            ),  # Scale + bias
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Create LayerNorm parameters
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        logger.info(
            f"Created KasminaLayerNormLayer: normalized_shape={normalized_shape}, "
            f"eps={eps}, elementwise_affine={elementwise_affine}"
        )

    def copy_weights_from_layernorm(self, original_layer: nn.LayerNorm) -> None:
        """
        Copy weights from original LayerNorm layer.

        Args:
            original_layer: The original LayerNorm layer
        """
        with torch.no_grad():
            if self.elementwise_affine and original_layer.elementwise_affine:
                if self.weight is not None and original_layer.weight is not None:
                    self.weight.copy_(original_layer.weight)
                if self.bias is not None and original_layer.bias is not None:
                    self.bias.copy_(original_layer.bias)

        logger.info("Copied weights from LayerNorm to %s", self.layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through layer normalization.

        Args:
            x: Input tensor [..., normalized_shape]

        Returns:
            Normalized tensor with same shape as input
        """
        # Check input shape
        if x.shape[-1] != self.normalized_shape[0]:
            raise ValueError(
                f"Expected input with last dimension {self.normalized_shape[0]}, "
                f"got {x.shape[-1]}"
            )

        # Apply layer normalization
        normalized = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

        # Apply morphogenetic adaptation if any seeds are active
        if self._has_active_seeds():
            normalized = self._apply_morphogenetic_adaptation(x, normalized)

        # Update telemetry
        if self.telemetry_enabled:
            self._update_telemetry(x, normalized)

        return normalized

    def _has_active_seeds(self) -> bool:
        """Check if any morphogenetic seeds are active."""
        return self.state_layout.get_active_seeds().any()

    def _apply_morphogenetic_adaptation(
        self, input_tensor: torch.Tensor, normalized: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply morphogenetic adaptations to the normalized output.

        Args:
            input_tensor: Original input tensor
            normalized: Normalized tensor from standard LayerNorm

        Returns:
            Adapted normalized tensor
        """
        input_tensor.shape[0]

        # Get active seeds and their alpha blending factors
        active_seeds = self.state_layout.get_active_seeds()
        alpha_factors = self.state_layout.alpha_blend[active_seeds]

        # Start with the standard normalized output
        adapted_output = normalized.clone()

        # Apply each active seed's adaptation
        for seed_idx, alpha in zip(active_seeds.nonzero().squeeze(-1), alpha_factors):
            if alpha > 0:
                # Apply morphogenetic kernel adaptation
                seed_adaptation = self._compute_seed_adaptation(
                    input_tensor, seed_idx.item()
                )

                # Blend with existing output
                adapted_output = (1 - alpha) * adapted_output + alpha * seed_adaptation

        return adapted_output

    def _compute_seed_adaptation(self, x: torch.Tensor, seed_idx: int) -> torch.Tensor:
        """
        Compute adaptation for a specific seed.

        Args:
            x: Input tensor
            seed_idx: Index of the seed

        Returns:
            Adapted tensor
        """
        # In a real implementation, this would use loaded kernels
        # For now, we apply a simple learned transformation
        if self.elementwise_affine:
            # Apply adaptive scale and bias
            adaptive_scale = torch.ones_like(self.weight) * (1.0 + 0.1 * seed_idx)
            adaptive_bias = (
                torch.zeros_like(self.bias) if self.bias is not None else None
            )

            # Normalize with adaptive parameters
            adapted = F.layer_norm(
                x,
                self.normalized_shape,
                self.weight * adaptive_scale,
                self.bias + adaptive_bias if adaptive_bias is not None else self.bias,
                self.eps,
            )
        else:
            # If no elementwise affine, just return normalized input
            adapted = F.layer_norm(x, self.normalized_shape, None, None, self.eps)

        return adapted

    def _update_telemetry(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> None:
        """
        Update telemetry data for this layer.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor
        """
        # Compute statistics for telemetry
        input_tensor.mean().item()
        input_tensor.std().item()
        output_tensor.mean().item()
        output_tensor.std().item()

        # Update state layout telemetry for active seeds
        active_seeds = self.state_layout.get_active_seeds()
        for seed_idx in active_seeds.nonzero().squeeze(-1):
            health_score = self._compute_health_score(input_tensor, output_tensor)
            self.state_layout.update_telemetry(
                seed_idx.item(),
                latency_us=0,  # Would be measured in real implementation
                health_score=health_score,
            )

    def _compute_health_score(
        self, _input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> float:
        """
        Compute health score based on normalization quality.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor

        Returns:
            Health score between 0 and 1
        """
        # Check if output is properly normalized (mean~0, std~1)
        output_mean = torch.abs(output_tensor.mean())
        output_std = torch.abs(output_tensor.std() - 1.0)

        # Good normalization should have mean close to 0 and std close to 1
        normalization_quality = 1.0 / (1.0 + output_mean.item() + output_std.item())

        # Check for numerical stability (no NaNs or infinities)
        stability_score = 1.0 if torch.isfinite(output_tensor).all() else 0.0

        # Combine scores
        health_score = 0.7 * normalization_quality + 0.3 * stability_score

        return max(0.0, min(1.0, health_score))

    def get_adaptation_stats(self) -> dict:
        """
        Get statistics about current adaptations.

        Returns:
            Dictionary containing adaptation statistics
        """
        active_seeds = self.state_layout.get_active_seeds()

        return {
            "active_adaptations": active_seeds.sum().item(),
            "total_seeds": len(active_seeds),
            "adaptation_strength": (
                self.state_layout.alpha_blend[active_seeds].mean().item()
                if active_seeds.any()
                else 0.0
            ),
            "normalized_shape": self.normalized_shape,
            "eps": self.eps,
            "elementwise_affine": self.elementwise_affine,
        }

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, num_seeds={self.num_seeds}"
        )

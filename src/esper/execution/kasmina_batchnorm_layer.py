"""
KasminaBatchNormLayer for Batch Normalization with morphogenetic capabilities.

This module provides specialized KasminaLayers for PyTorch's BatchNorm variants
that preserve normalization semantics while enabling dynamic parameter adaptation.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.execution.kasmina_layer import KasminaLayer

logger = logging.getLogger(__name__)


class KasminaBatchNorm1dLayer(KasminaLayer):
    """
    KasminaLayer specialized for 1D Batch Normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "batchnorm1d_layer",
    ):
        """
        Initialize KasminaBatchNorm1dLayer.

        Args:
            num_features: Number of features (C from input of size (N, C, L))
            eps: A value added to denominator for numerical stability
            momentum: Value used for running mean and variance computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to enable telemetry
            layer_name: Name of the layer for logging
        """
        super().__init__(
            input_size=num_features,
            output_size=num_features * 2 if affine else num_features,  # Weight + bias
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        logger.info("Created KasminaBatchNorm1dLayer: num_features=%d", num_features)

    def copy_weights_from_batchnorm(self, original_layer: nn.BatchNorm1d) -> None:
        """Copy weights from original BatchNorm1d layer."""
        with torch.no_grad():
            if self.affine and original_layer.affine:
                if self.weight is not None and original_layer.weight is not None:
                    self.weight.copy_(original_layer.weight)
                if self.bias is not None and original_layer.bias is not None:
                    self.bias.copy_(original_layer.bias)

            if self.track_running_stats and original_layer.track_running_stats:
                if (
                    self.running_mean is not None
                    and original_layer.running_mean is not None
                ):
                    self.running_mean.copy_(original_layer.running_mean)
                if (
                    self.running_var is not None
                    and original_layer.running_var is not None
                ):
                    self.running_var.copy_(original_layer.running_var)
                if (
                    self.num_batches_tracked is not None
                    and original_layer.num_batches_tracked is not None
                ):
                    self.num_batches_tracked.copy_(original_layer.num_batches_tracked)

        logger.info("Copied weights from BatchNorm1d to %s", self.layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through batch normalization."""
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Apply batch normalization
        normalized = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

        # Apply morphogenetic adaptation if any seeds are active
        if self._has_active_seeds():
            normalized = self._apply_morphogenetic_adaptation(x, normalized)

        return normalized

    def _has_active_seeds(self) -> bool:
        """Check if any morphogenetic seeds are active."""
        return self.state_layout.get_active_seeds().any()

    def _apply_morphogenetic_adaptation(
        self, input_tensor: torch.Tensor, normalized: torch.Tensor
    ) -> torch.Tensor:
        """Apply morphogenetic adaptations to the normalized output."""
        active_seeds = self.state_layout.get_active_seeds()
        alpha_factors = self.state_layout.alpha_blend[active_seeds]

        adapted_output = normalized.clone()

        for seed_idx, alpha in zip(active_seeds.nonzero().squeeze(-1), alpha_factors):
            if alpha > 0:
                seed_adaptation = self._compute_seed_adaptation(
                    input_tensor, seed_idx.item()
                )
                adapted_output = (1 - alpha) * adapted_output + alpha * seed_adaptation

        return adapted_output

    def _compute_seed_adaptation(self, x: torch.Tensor, seed_idx: int) -> torch.Tensor:
        """Compute adaptation for a specific seed."""
        if self.affine:
            # Apply adaptive scale and bias
            adaptive_scale = torch.ones_like(self.weight) * (1.0 + 0.05 * seed_idx)
            adaptive_bias = torch.zeros_like(self.bias) * (0.01 * seed_idx)

            adapted = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight * adaptive_scale,
                self.bias + adaptive_bias,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )
        else:
            adapted = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                None,
                None,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )

        return adapted


class KasminaBatchNorm2dLayer(KasminaLayer):
    """
    KasminaLayer specialized for 2D Batch Normalization (most common for CNNs).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "batchnorm2d_layer",
    ):
        """
        Initialize KasminaBatchNorm2dLayer.

        Args:
            num_features: Number of features (C from input of size (N, C, H, W))
            eps: A value added to denominator for numerical stability
            momentum: Value used for running mean and variance computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to enable telemetry
            layer_name: Name of the layer for logging
        """
        super().__init__(
            input_size=num_features,
            output_size=num_features * 2 if affine else num_features,
            num_seeds=num_seeds,
            cache_size_mb=cache_size_mb,
            telemetry_enabled=telemetry_enabled,
            layer_name=layer_name,
        )

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        logger.info("Created KasminaBatchNorm2dLayer: num_features=%d", num_features)

    def copy_weights_from_batchnorm(self, original_layer: nn.BatchNorm2d) -> None:
        """Copy weights from original BatchNorm2d layer."""
        with torch.no_grad():
            if self.affine and original_layer.affine:
                if self.weight is not None and original_layer.weight is not None:
                    self.weight.copy_(original_layer.weight)
                if self.bias is not None and original_layer.bias is not None:
                    self.bias.copy_(original_layer.bias)

            if self.track_running_stats and original_layer.track_running_stats:
                if (
                    self.running_mean is not None
                    and original_layer.running_mean is not None
                ):
                    self.running_mean.copy_(original_layer.running_mean)
                if (
                    self.running_var is not None
                    and original_layer.running_var is not None
                ):
                    self.running_var.copy_(original_layer.running_var)
                if (
                    self.num_batches_tracked is not None
                    and original_layer.num_batches_tracked is not None
                ):
                    self.num_batches_tracked.copy_(original_layer.num_batches_tracked)

        logger.info("Copied weights from BatchNorm2d to %s", self.layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2D batch normalization."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {x.dim()}D")

        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} channels, got {x.shape[1]}"
            )

        # Apply batch normalization
        normalized = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
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
        """Apply morphogenetic adaptations to the normalized output."""
        active_seeds = self.state_layout.get_active_seeds()
        alpha_factors = self.state_layout.alpha_blend[active_seeds]

        adapted_output = normalized.clone()

        for seed_idx, alpha in zip(active_seeds.nonzero().squeeze(-1), alpha_factors):
            if alpha > 0:
                seed_adaptation = self._compute_seed_adaptation(
                    input_tensor, seed_idx.item()
                )
                adapted_output = (1 - alpha) * adapted_output + alpha * seed_adaptation

        return adapted_output

    def _compute_seed_adaptation(self, x: torch.Tensor, seed_idx: int) -> torch.Tensor:
        """Compute adaptation for a specific seed."""
        if self.affine:
            # Apply adaptive scale and bias with smaller perturbations for stability
            adaptive_scale = torch.ones_like(self.weight) * (1.0 + 0.02 * seed_idx)
            adaptive_bias = torch.zeros_like(self.bias) + (0.005 * seed_idx)

            adapted = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight * adaptive_scale,
                self.bias + adaptive_bias,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )
        else:
            adapted = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                None,
                None,
                self.training or not self.track_running_stats,
                self.momentum,
                self.eps,
            )

        return adapted

    def _update_telemetry(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> None:
        """Update telemetry data for this layer."""
        # Compute statistics for telemetry
        input_tensor.mean(dim=(0, 2, 3))  # Mean across batch, height, width
        input_tensor.std(dim=(0, 2, 3))  # Std across batch, height, width

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
        """Compute health score based on normalization quality."""
        # Check if output channels are properly normalized
        channel_means = output_tensor.mean(dim=(0, 2, 3))
        channel_stds = output_tensor.std(dim=(0, 2, 3))

        # Good batch normalization should have means close to 0 and stds close to 1
        mean_quality = 1.0 / (1.0 + torch.abs(channel_means).mean().item())
        std_quality = 1.0 / (1.0 + torch.abs(channel_stds - 1.0).mean().item())

        # Check for numerical stability
        stability_score = 1.0 if torch.isfinite(output_tensor).all() else 0.0

        # Combine scores
        health_score = 0.4 * mean_quality + 0.4 * std_quality + 0.2 * stability_score

        return max(0.0, min(1.0, health_score))

    def get_adaptation_stats(self) -> dict:
        """Get statistics about current adaptations."""
        active_seeds = self.state_layout.get_active_seeds()

        return {
            "active_adaptations": active_seeds.sum().item(),
            "total_seeds": len(active_seeds),
            "adaptation_strength": (
                self.state_layout.alpha_blend[active_seeds].mean().item()
                if active_seeds.any()
                else 0.0
            ),
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "track_running_stats": self.track_running_stats,
        }

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}, "
            f"num_seeds={self.num_seeds}"
        )

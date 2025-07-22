"""
Model wrapping utility for injecting morphogenetic capabilities.

This module provides the core esper.wrap() function that transforms
standard PyTorch models into morphogenetic models.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

from esper.execution.kasmina_attention_layer import KasminaAttentionLayer
from esper.execution.kasmina_batchnorm_layer import (
    KasminaBatchNorm1dLayer,
    KasminaBatchNorm2dLayer,
)
from esper.execution.kasmina_conv2d_layer import KasminaConv2dLayer
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.kasmina_layernorm_layer import KasminaLayerNormLayer

logger = logging.getLogger(__name__)


class MorphableModel(nn.Module):
    """
    A PyTorch model enhanced with morphogenetic capabilities.

    This wrapper class maintains the original model while adding
    KasminaLayers that can load and execute dynamic kernels.
    """

    def __init__(
        self,
        wrapped_model: nn.Module,
        kasmina_layers: Dict[str, KasminaLayer],
        original_model: Optional[nn.Module] = None,
    ):
        """
        Initialize MorphableModel.

        Args:
            wrapped_model: The model with KasminaLayers injected
            kasmina_layers: Dictionary mapping layer names to KasminaLayer instances
            original_model: Optional reference to original model for comparison
        """
        super().__init__()

        self.wrapped_model = wrapped_model
        self.kasmina_layers = nn.ModuleDict(kasmina_layers)
        self.original_model = original_model

        # Statistics
        self.total_forward_calls = 0
        self.morphogenetic_active = False

        logger.info(f"Created MorphableModel with {len(kasmina_layers)} KasminaLayers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the morphable model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        self.total_forward_calls += 1
        return self.wrapped_model(x)

    async def load_kernel(
        self, layer_name: str, seed_idx: int, artifact_id: str
    ) -> bool:
        """
        Load a compiled kernel into a specific layer and seed.

        Args:
            layer_name: Name of the layer
            seed_idx: Index of the seed within the layer
            artifact_id: ID of the kernel artifact

        Returns:
            True if kernel was loaded successfully
        """
        if layer_name not in self.kasmina_layers:
            raise ValueError(f"Layer '{layer_name}' not found in morphable model")

        kasmina_layer = self.kasmina_layers[layer_name]
        success = await kasmina_layer.load_kernel(seed_idx, artifact_id)

        if success:
            self.morphogenetic_active = True
            logger.info(f"Loaded kernel {artifact_id} into {layer_name}[{seed_idx}]")
        else:
            logger.warning(
                f"Failed to load kernel {artifact_id} into {layer_name}[{seed_idx}]"
            )

        return success

    async def unload_kernel(self, layer_name: str, seed_idx: int) -> bool:
        """
        Unload kernel from a specific layer and seed.

        Args:
            layer_name: Name of the layer
            seed_idx: Index of the seed within the layer

        Returns:
            True if kernel was unloaded successfully
        """
        if layer_name not in self.kasmina_layers:
            raise ValueError(f"Layer '{layer_name}' not found in morphable model")

        kasmina_layer = self.kasmina_layers[layer_name]
        success = await kasmina_layer.unload_kernel(seed_idx)

        if success:
            logger.info(f"Unloaded kernel from {layer_name}[{seed_idx}]")

            # Check if any kernels are still active
            self.morphogenetic_active = self._check_morphogenetic_active()

        return success

    def _check_morphogenetic_active(self) -> bool:
        """
        Check if any KasminaLayers have active seeds.

        Returns:
            True if any morphogenetic capabilities are active
        """
        return any(
            layer.state_layout.get_active_seeds().any()
            for layer in self.kasmina_layers.values()
        )

    def get_layer_names(self) -> List[str]:
        """
        Get list of all KasminaLayer names in the model.

        Returns:
            List of layer names
        """
        return list(self.kasmina_layers.keys())

    def get_layer_stats(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific layer or all layers.

        Args:
            layer_name: Name of specific layer, or None for all layers

        Returns:
            Dictionary containing layer statistics
        """
        if layer_name is not None:
            if layer_name not in self.kasmina_layers:
                raise ValueError(f"Layer '{layer_name}' not found in morphable model")
            return self.kasmina_layers[layer_name].get_layer_stats()
        else:
            return {
                name: layer.get_layer_stats()
                for name, layer in self.kasmina_layers.items()
            }

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics.

        Returns:
            Dictionary containing model statistics
        """
        layer_stats = self.get_layer_stats()

        # Aggregate statistics
        total_seeds = sum(layer.num_seeds for layer in self.kasmina_layers.values())
        active_seeds = sum(
            layer.state_layout.get_active_seeds().sum().item()
            for layer in self.kasmina_layers.values()
        )
        total_kernel_executions = sum(
            layer.total_kernel_executions for layer in self.kasmina_layers.values()
        )

        return {
            "total_forward_calls": self.total_forward_calls,
            "morphogenetic_active": self.morphogenetic_active,
            "total_kasmina_layers": len(self.kasmina_layers),
            "total_seeds": total_seeds,
            "active_seeds": active_seeds,
            "total_kernel_executions": total_kernel_executions,
            "layer_stats": layer_stats,
        }

    def set_seed_alpha(self, layer_name: str, seed_idx: int, alpha: float) -> None:
        """
        Set the alpha blend factor for a specific seed.

        Args:
            layer_name: Name of the layer
            seed_idx: Index of the seed
            alpha: Blend factor (0.0 to 1.0)
        """
        if layer_name not in self.kasmina_layers:
            raise ValueError(f"Layer '{layer_name}' not found in morphable model")

        self.kasmina_layers[layer_name].set_seed_alpha(seed_idx, alpha)

    def enable_telemetry(self, enabled: bool = True) -> None:
        """
        Enable or disable telemetry for all KasminaLayers.

        Args:
            enabled: Whether to enable telemetry
        """
        for layer in self.kasmina_layers.values():
            layer.telemetry_enabled = enabled

        logger.info(f"Telemetry {'enabled' if enabled else 'disabled'} for all layers")

    def compare_with_original(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compare output with original model (if available).

        Args:
            x: Input tensor

        Returns:
            Dictionary containing comparison results
        """
        if self.original_model is None:
            raise ValueError("Original model not available for comparison")

        # Get outputs
        morphable_output = self.forward(x)
        original_output = self.original_model(x)

        # Compute differences
        with torch.no_grad():
            diff = morphable_output - original_output
            mse = torch.mean(diff**2).item()
            max_diff = torch.max(torch.abs(diff)).item()

        return {
            "mse": mse,
            "max_absolute_difference": max_diff,
            "output_shape": morphable_output.shape,
            "morphogenetic_active": self.morphogenetic_active,
        }


def wrap(
    model: nn.Module,
    target_layers: Optional[List[Type[nn.Module]]] = None,
    seeds_per_layer: int = 4,
    cache_size_mb: int = 128,
    telemetry_enabled: bool = True,
    preserve_original: bool = True,
) -> MorphableModel:
    """
    Wrap a PyTorch model with morphogenetic capabilities.

    This function automatically identifies target layers and replaces them
    with KasminaLayers that preserve the original behavior while enabling
    dynamic kernel loading.

    Args:
        model: The PyTorch model to wrap
        target_layers: List of layer types to replace (default: [nn.Linear, nn.Conv2d])
        seeds_per_layer: Number of morphogenetic seeds per layer
        cache_size_mb: Kernel cache size in MB per layer
        telemetry_enabled: Whether to enable telemetry collection
        preserve_original: Whether to keep a reference to the original model

    Returns:
        MorphableModel with KasminaLayers injected
    """
    if target_layers is None:
        target_layers = [
            nn.Linear,
            nn.Conv2d,
            nn.MultiheadAttention,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
        ]  # Default to common transformable layers

    logger.info("Wrapping model with morphogenetic capabilities")
    logger.info(f"Target layers: {[cls.__name__ for cls in target_layers]}")

    # Deep copy the model to avoid modifying the original
    wrapped_model = copy.deepcopy(model)
    original_model = copy.deepcopy(model) if preserve_original else None

    # Dictionary to store KasminaLayers
    kasmina_layers = {}

    # Recursively replace target layers
    def replace_layers(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            child_name = f"{prefix}_{name}" if prefix else name

            # Check if this child should be replaced
            if type(child) in target_layers:
                try:
                    # Create KasminaLayer replacement
                    kasmina_layer = _create_kasmina_layer(
                        child,
                        child_name,
                        seeds_per_layer,
                        cache_size_mb,
                        telemetry_enabled,
                    )

                    # Replace the child
                    setattr(module, name, kasmina_layer)
                    kasmina_layers[child_name] = kasmina_layer

                    logger.info(
                        f"Replaced {type(child).__name__} at '{child_name}' with KasminaLayer"
                    )
                except NotImplementedError as e:
                    logger.warning(
                        f"Skipping {type(child).__name__} at '{child_name}': {e}"
                    )
                    # Don't replace this layer, just continue
                    replace_layers(child, child_name)
            else:
                # Recursively process children
                replace_layers(child, child_name)

    # Perform the replacement
    replace_layers(wrapped_model)

    logger.info(f"Created {len(kasmina_layers)} KasminaLayers")

    return MorphableModel(wrapped_model, kasmina_layers, original_model)


def _create_kasmina_layer(
    original_layer: nn.Module,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaLayer:
    """
    Create a KasminaLayer replacement for an original layer.

    Args:
        original_layer: The original layer to replace
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaLayer configured to replace the original layer
    """
    if isinstance(original_layer, nn.Linear):
        return _create_kasmina_layer_linear(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    elif isinstance(original_layer, nn.Conv2d):
        return _create_kasmina_layer_conv2d(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    elif isinstance(original_layer, nn.MultiheadAttention):
        return _create_kasmina_layer_attention(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    elif isinstance(original_layer, nn.LayerNorm):
        return _create_kasmina_layer_layernorm(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    elif isinstance(original_layer, nn.BatchNorm1d):
        return _create_kasmina_layer_batchnorm1d(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    elif isinstance(original_layer, nn.BatchNorm2d):
        return _create_kasmina_layer_batchnorm2d(
            original_layer,
            layer_name,
            seeds_per_layer,
            cache_size_mb,
            telemetry_enabled,
        )
    else:
        # For other layer types, we'll need to infer dimensions
        raise NotImplementedError(
            f"Layer type {type(original_layer)} not yet supported"
        )


def _create_kasmina_layer_linear(
    original_layer: nn.Linear,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaLayer:
    """
    Create KasminaLayer for Linear layers with proper weight handling.

    Args:
        original_layer: The original Linear layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaLayer configured to replace the Linear layer
    """
    # Create KasminaLayer with Linear layer dimensions
    kasmina_layer = KasminaLayer(
        input_size=original_layer.in_features,
        output_size=original_layer.out_features,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original behavior
    with torch.no_grad():
        kasmina_layer.default_transform.weight.copy_(original_layer.weight)
        if original_layer.bias is not None:
            kasmina_layer.default_transform.bias.copy_(original_layer.bias)

    logger.info(
        f"Created KasminaLayer for Linear({original_layer.in_features}, "
        f"{original_layer.out_features}) at '{layer_name}'"
    )

    return kasmina_layer


def _create_kasmina_layer_conv2d(
    original_layer: nn.Conv2d,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaConv2dLayer:
    """
    Create KasminaConv2dLayer for Conv2d layers with proper weight handling.

    Args:
        original_layer: The original Conv2d layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaConv2dLayer configured to replace the Conv2d layer
    """
    # Create Conv2d-aware KasminaLayer
    kasmina_layer = KasminaConv2dLayer(
        in_channels=original_layer.in_channels,
        out_channels=original_layer.out_channels,
        kernel_size=original_layer.kernel_size,
        stride=original_layer.stride,
        padding=original_layer.padding,
        dilation=original_layer.dilation,
        groups=original_layer.groups,
        bias=original_layer.bias is not None,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original convolution behavior
    kasmina_layer.copy_weights_from_conv2d(original_layer)

    logger.info(
        f"Created KasminaConv2dLayer for Conv2d({original_layer.in_channels}, "
        f"{original_layer.out_channels}, kernel_size={original_layer.kernel_size}) "
        f"at '{layer_name}'"
    )

    return kasmina_layer


def _create_kasmina_layer_attention(
    original_layer: nn.MultiheadAttention,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaAttentionLayer:
    """
    Create KasminaAttentionLayer for MultiheadAttention layers.

    Args:
        original_layer: The original MultiheadAttention layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaAttentionLayer configured to replace the MultiheadAttention layer
    """
    # Create attention-aware KasminaLayer
    kasmina_layer = KasminaAttentionLayer(
        embed_dim=original_layer.embed_dim,
        num_heads=original_layer.num_heads,
        dropout=original_layer.dropout,
        bias=original_layer.in_proj_bias is not None,
        add_bias_kv=original_layer.bias_k is not None,
        add_zero_attn=original_layer.add_zero_attn,
        kdim=original_layer.kdim,
        vdim=original_layer.vdim,
        batch_first=original_layer.batch_first,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original attention behavior
    kasmina_layer.copy_weights_from_attention(original_layer)

    logger.info(
        f"Created KasminaAttentionLayer for MultiheadAttention(embed_dim={original_layer.embed_dim}, "
        f"num_heads={original_layer.num_heads}) at '{layer_name}'"
    )

    return kasmina_layer


def _create_kasmina_layer_layernorm(
    original_layer: nn.LayerNorm,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaLayerNormLayer:
    """
    Create KasminaLayerNormLayer for LayerNorm layers.

    Args:
        original_layer: The original LayerNorm layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaLayerNormLayer configured to replace the LayerNorm layer
    """
    # Create LayerNorm-aware KasminaLayer
    # Handle both single-dimensional and multi-dimensional normalized shapes
    if isinstance(original_layer.normalized_shape, int):
        normalized_shape = original_layer.normalized_shape
    elif len(original_layer.normalized_shape) == 1:
        normalized_shape = original_layer.normalized_shape[0]
    else:
        # For multi-dimensional shapes, we'll need to handle this differently
        raise NotImplementedError(
            f"Multi-dimensional normalized shapes not yet supported: {original_layer.normalized_shape}"
        )

    kasmina_layer = KasminaLayerNormLayer(
        normalized_shape=normalized_shape,
        eps=original_layer.eps,
        elementwise_affine=original_layer.elementwise_affine,
        bias=original_layer.bias is not None,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original normalization behavior
    kasmina_layer.copy_weights_from_layernorm(original_layer)

    logger.info(
        f"Created KasminaLayerNormLayer for LayerNorm(normalized_shape={original_layer.normalized_shape}) "
        f"at '{layer_name}'"
    )

    return kasmina_layer


def _create_kasmina_layer_batchnorm1d(
    original_layer: nn.BatchNorm1d,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaBatchNorm1dLayer:
    """
    Create KasminaBatchNorm1dLayer for BatchNorm1d layers.

    Args:
        original_layer: The original BatchNorm1d layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaBatchNorm1dLayer configured to replace the BatchNorm1d layer
    """
    # Create BatchNorm1d-aware KasminaLayer
    kasmina_layer = KasminaBatchNorm1dLayer(
        num_features=original_layer.num_features,
        eps=original_layer.eps,
        momentum=original_layer.momentum,
        affine=original_layer.affine,
        track_running_stats=original_layer.track_running_stats,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original normalization behavior
    kasmina_layer.copy_weights_from_batchnorm(original_layer)

    logger.info(
        f"Created KasminaBatchNorm1dLayer for BatchNorm1d(num_features={original_layer.num_features}) "
        f"at '{layer_name}'"
    )

    return kasmina_layer


def _create_kasmina_layer_batchnorm2d(
    original_layer: nn.BatchNorm2d,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool,
) -> KasminaBatchNorm2dLayer:
    """
    Create KasminaBatchNorm2dLayer for BatchNorm2d layers.

    Args:
        original_layer: The original BatchNorm2d layer
        layer_name: Name of the layer
        seeds_per_layer: Number of seeds
        cache_size_mb: Cache size in MB
        telemetry_enabled: Whether to enable telemetry

    Returns:
        KasminaBatchNorm2dLayer configured to replace the BatchNorm2d layer
    """
    # Create BatchNorm2d-aware KasminaLayer
    kasmina_layer = KasminaBatchNorm2dLayer(
        num_features=original_layer.num_features,
        eps=original_layer.eps,
        momentum=original_layer.momentum,
        affine=original_layer.affine,
        track_running_stats=original_layer.track_running_stats,
        num_seeds=seeds_per_layer,
        cache_size_mb=cache_size_mb,
        telemetry_enabled=telemetry_enabled,
        layer_name=layer_name,
    )

    # Copy weights exactly to preserve original normalization behavior
    kasmina_layer.copy_weights_from_batchnorm(original_layer)

    logger.info(
        f"Created KasminaBatchNorm2dLayer for BatchNorm2d(num_features={original_layer.num_features}) "
        f"at '{layer_name}'"
    )

    return kasmina_layer


def unwrap(morphable_model: MorphableModel) -> nn.Module:
    """
    Extract the original model from a MorphableModel.

    Args:
        morphable_model: The MorphableModel to unwrap

    Returns:
        The original model (or best approximation)
    """
    if morphable_model.original_model is not None:
        return morphable_model.original_model
    else:
        # Return the wrapped model (it will behave like the original when no kernels are loaded)
        return morphable_model.wrapped_model

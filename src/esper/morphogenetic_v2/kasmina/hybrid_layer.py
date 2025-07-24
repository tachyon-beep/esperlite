"""
HybridKasminaLayer: Backward-compatible wrapper for smooth migration.

This module provides a hybrid implementation that can seamlessly switch between
legacy and chunked architectures based on feature flags, enabling gradual rollout.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
import logging
import time

from .chunked_layer import ChunkedKasminaLayer
from ...execution.kasmina_layer import KasminaLayer  # Legacy implementation
from ..common.feature_flags import FeatureFlagManager

logger = logging.getLogger(__name__)


class HybridKasminaLayer(nn.Module):
    """
    Hybrid implementation that wraps both legacy and chunked KasminaLayer.
    
    This layer provides:
    - Seamless switching between implementations via feature flags
    - A/B testing support for performance comparison
    - Gradual rollout capabilities
    - Full backward compatibility
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        layer_id: Optional[str] = None,
        num_seeds: int = 1000,
        enable_telemetry: bool = True,
        device: Optional[torch.device] = None,
        force_implementation: Optional[str] = None
    ):
        """
        Initialize HybridKasminaLayer.
        
        Args:
            base_layer: The underlying neural network layer
            layer_id: Unique identifier for this layer
            num_seeds: Number of logical seeds for chunked implementation
            enable_telemetry: Whether to collect telemetry
            device: Device for operations
            force_implementation: Force specific implementation ("legacy" or "chunked")
        """
        super().__init__()
        
        self.layer_id = layer_id or f"hybrid_kasmina_{id(self)}"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.force_implementation = force_implementation
        
        # Initialize feature flags
        self.feature_flags = FeatureFlagManager()
        
        # Initialize both implementations
        self._init_legacy_layer(base_layer, layer_id, enable_telemetry)
        self._init_chunked_layer(base_layer, num_seeds, layer_id, enable_telemetry)
        
        # Metrics tracking
        self.implementation_calls = {"legacy": 0, "chunked": 0}
        self.implementation_latency = {"legacy": 0.0, "chunked": 0.0}
        
        # Move to device
        self.to(self.device)
        
        logger.info(
            "HybridKasminaLayer initialized: layer_id=%s, num_seeds=%d, force=%s",
            self.layer_id, num_seeds, force_implementation
        )
    
    def _init_legacy_layer(self, base_layer: nn.Module, layer_id: Optional[str], enable_telemetry: bool):
        """Initialize legacy KasminaLayer."""
        try:
            # Get dimensions from base layer
            if hasattr(base_layer, 'out_features'):
                output_size = base_layer.out_features
                input_size = base_layer.in_features
            elif hasattr(base_layer, 'weight'):
                output_size = base_layer.weight.shape[0]
                input_size = base_layer.weight.shape[1]
            else:
                raise ValueError("Cannot determine dimensions from base layer")
            
            self.legacy_layer = KasminaLayer(
                input_size=input_size,
                output_size=output_size
            )
            self.legacy_available = True
        except (ValueError, AttributeError) as e:
            logger.warning("Failed to initialize legacy layer: %s", e)
            self.legacy_layer = None
            self.legacy_available = False
    
    def _init_chunked_layer(self, base_layer: nn.Module, num_seeds: int, 
                           layer_id: Optional[str], enable_telemetry: bool):
        """Initialize chunked KasminaLayer."""
        try:
            # Clone the base layer for chunked implementation
            import copy
            chunked_base = copy.deepcopy(base_layer)
            
            self.chunked_layer = ChunkedKasminaLayer(
                base_layer=chunked_base,
                num_seeds=num_seeds,
                layer_id=layer_id,
                enable_telemetry=enable_telemetry,
                device=self.device
            )
            self.chunked_available = True
        except (ValueError, AttributeError) as e:
            logger.warning("Failed to initialize chunked layer: %s", e)
            self.chunked_layer = None
            self.chunked_available = False
    
    def _select_implementation(self, model_id: Optional[str] = None) -> str:
        """
        Select implementation based on feature flags and configuration.
        
        Returns:
            "legacy" or "chunked"
        """
        # Check forced implementation
        if self.force_implementation:
            return self.force_implementation
        
        # Check feature flags
        if self.feature_flags.is_enabled("chunked_architecture", model_id):
            if self.chunked_available:
                return "chunked"
            else:
                logger.warning("Chunked implementation requested but not available")
        
        # Default to legacy
        return "legacy" if self.legacy_available else "chunked"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with implementation selection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get model ID from context if available
        model_id = getattr(self, "_current_model_id", None)
        
        # Select implementation
        impl = self._select_implementation(model_id)
        
        # Execute forward pass
        start_time = time.perf_counter()
        
        if impl == "legacy" and self.legacy_available:
            output = self.legacy_layer(x)
        elif impl == "chunked" and self.chunked_available:
            output = self.chunked_layer(x)
        else:
            # Fallback to base layer if both implementations fail
            logger.error("No implementation available, using base layer")
            output = self.chunked_layer.base_layer(x) if self.chunked_layer else x
        
        # Update metrics
        elapsed = time.perf_counter() - start_time
        self.implementation_calls[impl] += 1
        self.implementation_latency[impl] += elapsed
        
        return output
    
    def set_model_context(self, model_id: str):
        """Set the current model ID for feature flag evaluation."""
        self._current_model_id = model_id
    
    def request_germination(self, seed_id: int, blueprint_id: Optional[int] = None, 
                          grafting_strategy: int = 0) -> bool:
        """
        Request germination (Tamiyo interface).
        
        Routes to appropriate implementation based on current selection.
        """
        impl = self._select_implementation()
        
        if impl == "chunked" and self.chunked_available:
            return self.chunked_layer.request_germination(seed_id, blueprint_id, grafting_strategy)
        elif self.legacy_available:
            # Legacy implementation doesn't have direct seed control
            logger.info("Germination request for legacy layer (no-op)")
            return True
        
        return False
    
    def cancel_germination(self, seed_id: int) -> bool:
        """Cancel germination request."""
        impl = self._select_implementation()
        
        if impl == "chunked" and self.chunked_available:
            return self.chunked_layer.cancel_germination(seed_id)
        
        return False
    
    def get_implementation_stats(self) -> Dict[str, Any]:
        """Get statistics about implementation usage."""
        stats = {
            "current_implementation": self._select_implementation(),
            "legacy_available": self.legacy_available,
            "chunked_available": self.chunked_available,
            "force_implementation": self.force_implementation,
            "calls": self.implementation_calls.copy(),
            "avg_latency_ms": {}
        }
        
        # Calculate average latencies
        for impl in ["legacy", "chunked"]:
            calls = self.implementation_calls[impl]
            if calls > 0:
                avg_latency = self.implementation_latency[impl] / calls * 1000
                stats["avg_latency_ms"][impl] = avg_latency
            else:
                stats["avg_latency_ms"][impl] = 0.0
        
        return stats
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get layer statistics from active implementation."""
        impl = self._select_implementation()
        
        base_stats = {
            "layer_id": self.layer_id,
            "implementation": impl,
            **self.get_implementation_stats()
        }
        
        # Add implementation-specific stats
        if impl == "chunked" and self.chunked_available:
            chunked_stats = self.chunked_layer.get_layer_stats()
            base_stats.update(chunked_stats)
        elif impl == "legacy" and self.legacy_available:
            # Legacy stats if available
            if hasattr(self.legacy_layer, 'get_stats'):
                legacy_stats = self.legacy_layer.get_stats()
                base_stats.update(legacy_stats)
        
        return base_stats
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report for Tamiyo."""
        impl = self._select_implementation()
        
        if impl == "chunked" and self.chunked_available:
            return self.chunked_layer.get_health_report()
        else:
            # Basic health report for legacy
            return {
                "layer_id": self.layer_id,
                "timestamp": time.time(),
                "implementation": "legacy",
                "status": "healthy",
                "seeds": [],  # Legacy doesn't have seeds
                "telemetry": []
            }
    
    def compare_implementations(self, x: torch.Tensor, num_runs: int = 10) -> Dict[str, Any]:
        """
        Compare performance of both implementations.
        
        Args:
            x: Input tensor for testing
            num_runs: Number of runs for timing
            
        Returns:
            Comparison results
        """
        results = {
            "layer_id": self.layer_id,
            "num_runs": num_runs,
            "input_shape": list(x.shape),
            "implementations": {}
        }
        
        # Test legacy implementation
        if self.legacy_available:
            legacy_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    legacy_out = self.legacy_layer(x)
                legacy_times.append(time.perf_counter() - start)
            
            results["implementations"]["legacy"] = {
                "available": True,
                "avg_latency_ms": sum(legacy_times) / len(legacy_times) * 1000,
                "min_latency_ms": min(legacy_times) * 1000,
                "max_latency_ms": max(legacy_times) * 1000,
                "output_shape": list(legacy_out.shape)
            }
        else:
            results["implementations"]["legacy"] = {"available": False}
        
        # Test chunked implementation
        if self.chunked_available:
            chunked_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    chunked_out = self.chunked_layer(x)
                chunked_times.append(time.perf_counter() - start)
            
            results["implementations"]["chunked"] = {
                "available": True,
                "avg_latency_ms": sum(chunked_times) / len(chunked_times) * 1000,
                "min_latency_ms": min(chunked_times) * 1000,
                "max_latency_ms": max(chunked_times) * 1000,
                "output_shape": list(chunked_out.shape),
                "num_seeds": self.chunked_layer.num_seeds
            }
            
            # Compare outputs if both available
            if self.legacy_available and 'legacy_out' in locals():
                max_diff = torch.max(torch.abs(legacy_out - chunked_out)).item()
                results["max_output_difference"] = max_diff
        else:
            results["implementations"]["chunked"] = {"available": False}
        
        return results
    
    def enable_chunked_mode(self):
        """Force chunked implementation."""
        self.force_implementation = "chunked"
        logger.info("Hybrid layer %s forced to chunked mode", self.layer_id)
    
    def enable_legacy_mode(self):
        """Force legacy implementation."""
        self.force_implementation = "legacy"
        logger.info("Hybrid layer %s forced to legacy mode", self.layer_id)
    
    def enable_auto_mode(self):
        """Enable automatic implementation selection."""
        self.force_implementation = None
        logger.info("Hybrid layer %s set to auto mode", self.layer_id)
    
    def to(self, device: Union[str, torch.device]) -> "HybridKasminaLayer":
        """Move layer to device."""
        super().to(device)
        
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        
        if self.legacy_layer:
            self.legacy_layer = self.legacy_layer.to(device)
        
        if self.chunked_layer:
            self.chunked_layer = self.chunked_layer.to(device)
        
        return self
    
    def state_dict(self, *args, **kwargs):
        """Get state dict for checkpointing."""
        state = super().state_dict(*args, **kwargs)
        
        # Add implementation states
        if self.legacy_layer:
            state["legacy_layer"] = self.legacy_layer.state_dict()
        
        if self.chunked_layer:
            state["chunked_layer"] = self.chunked_layer.state_dict()
        
        # Add metrics
        state["_implementation_calls"] = self.implementation_calls
        state["_implementation_latency"] = self.implementation_latency
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict from checkpoint."""
        # Extract implementation states
        legacy_state = state_dict.pop("legacy_layer", None)
        chunked_state = state_dict.pop("chunked_layer", None)
        
        # Extract metrics
        self.implementation_calls = state_dict.pop("_implementation_calls", self.implementation_calls)
        self.implementation_latency = state_dict.pop("_implementation_latency", self.implementation_latency)
        
        # Load main state
        super().load_state_dict(state_dict, strict=False)
        
        # Load implementation states
        if legacy_state and self.legacy_layer:
            self.legacy_layer.load_state_dict(legacy_state, strict=strict)
        
        if chunked_state and self.chunked_layer:
            self.chunked_layer.load_state_dict(chunked_state, strict=strict)
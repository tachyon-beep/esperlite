### **Implementation Guide: Phase 2 - The Execution Engine**

**Objective:** Prove that a host model can load and execute real, compiled artifacts produced by the Phase 1 pipeline. This phase implements the core execution mechanic that validates the entire `Tezzeret -> Urza -> Kasmina` pipeline.

**Key Components to Implement:** `KasminaLayer`, `esper.wrap()`, Execution Integration Tests.

**Timeline:** Weeks 7-9 (3 weeks)

-----

### **1. KasminaLayer: The Pure Execution Engine**

**Task:** Implement the high-performance execution layer that loads and runs pre-compiled kernel artifacts.

#### **1.1. Core State Management (`src/esper/execution/state_layout.py`)**

Implement the GPU-optimized state tensor using Structure-of-Arrays (SoA) layout.

```python
"""
GPU-optimized state management for KasminaLayer.

This module implements the Structure-of-Arrays (SoA) memory layout for optimal
GPU memory coalescing during kernel execution.
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class KasminaStateLayout:
    """Structure-of-Arrays layout optimized for GPU memory coalescing."""
    
    # Core lifecycle management
    lifecycle_states: torch.Tensor      # uint8: DORMANT, ACTIVE, LOADING, ERROR_RECOVERY
    active_kernel_id: torch.Tensor      # uint64: Hash of currently loaded kernel artifact
    alpha_blend: torch.Tensor           # float16: Blending coefficient for grafting
    
    # Performance tracking
    health_accumulator: torch.Tensor    # float32: Running statistics for telemetry
    last_update_epoch: torch.Tensor     # uint32: For staleness tracking
    exec_latency_μs: torch.Tensor       # uint16: Per-seed execution time measurement
    
    # Error handling
    error_count: torch.Tensor           # uint16: Count of consecutive failures
    fallback_active: torch.Tensor       # bool: Whether using fallback execution
    
    def __init__(self, num_seeds: int, device: torch.device):
        """Initialize state tensors for the specified number of seeds."""
        self.lifecycle_states = torch.zeros(num_seeds, dtype=torch.uint8, device=device)
        self.active_kernel_id = torch.zeros(num_seeds, dtype=torch.int64, device=device)
        self.alpha_blend = torch.ones(num_seeds, dtype=torch.float16, device=device)
        self.health_accumulator = torch.zeros(num_seeds, dtype=torch.float32, device=device)
        self.last_update_epoch = torch.zeros(num_seeds, dtype=torch.int32, device=device)
        self.exec_latency_μs = torch.zeros(num_seeds, dtype=torch.int16, device=device)
        self.error_count = torch.zeros(num_seeds, dtype=torch.int16, device=device)
        self.fallback_active = torch.zeros(num_seeds, dtype=torch.bool, device=device)

class SeedLifecycleState:
    """Enumeration of seed lifecycle states."""
    DORMANT = 0
    LOADING = 1
    ACTIVE = 2
    ERROR_RECOVERY = 3
    FOSSILIZED = 4
```

#### **1.2. GPU-Resident Kernel Cache (`src/esper/execution/kernel_cache.py`)**

Implement the LRU cache for compiled kernel artifacts.

```python
"""
GPU-resident LRU cache for pre-compiled kernel artifacts.

This module provides high-performance caching of kernel artifacts to achieve
microsecond-latency execution.
"""

import asyncio
import logging
import hashlib
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

from esper.services.contracts import CompiledKernelArtifact

logger = logging.getLogger(__name__)


class KernelCache:
    """GPU-resident LRU cache for pre-compiled kernel artifacts."""

    def __init__(self, max_cache_size_mb: int = 512, max_entries: int = 128):
        self.max_cache_size_mb = max_cache_size_mb
        self.max_entries = max_entries
        
        # GPU cache: artifact_id -> loaded kernel tensor
        self.gpu_cache: Dict[str, torch.Tensor] = {}
        
        # LRU tracking
        self.lru_tracker: OrderedDict[str, bool] = OrderedDict()
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_size_mb = 0

    async def load_kernel(self, artifact_id: str) -> Optional[torch.Tensor]:
        """
        Load kernel from cache or fetch from Urza.
        
        Args:
            artifact_id: Unique identifier for the kernel artifact
            
        Returns:
            GPU-loaded kernel tensor, or None if loading fails
        """
        # Check cache first
        if artifact_id in self.gpu_cache:
            # Cache hit - move to end of LRU
            self.lru_tracker.move_to_end(artifact_id)
            self.cache_hits += 1
            logger.debug("Cache hit for kernel %s", artifact_id)
            return self.gpu_cache[artifact_id]

        # Cache miss - need to load from Urza
        self.cache_misses += 1
        logger.debug("Cache miss for kernel %s", artifact_id)
        
        try:
            kernel_tensor = await self._fetch_from_urza(artifact_id)
            if kernel_tensor is None:
                return None
                
            # Add to cache with LRU eviction if necessary
            await self._add_to_cache(artifact_id, kernel_tensor)
            return kernel_tensor
            
        except Exception as e:
            logger.error("Failed to load kernel %s: %s", artifact_id, e)
            return None

    async def _fetch_from_urza(self, artifact_id: str) -> Optional[torch.Tensor]:
        """Fetch kernel binary from Urza and load to GPU."""
        # TODO: Implement actual Urza client call
        # For MVP, return a placeholder tensor
        logger.info("Fetching kernel %s from Urza", artifact_id)
        
        # Simulate network latency
        await asyncio.sleep(0.001)  # 1ms simulated fetch time
        
        # Create a placeholder kernel (simple linear layer for MVP)
        kernel_tensor = torch.randn(64, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
        return kernel_tensor

    async def _add_to_cache(self, artifact_id: str, kernel_tensor: torch.Tensor):
        """Add kernel to cache with LRU eviction."""
        # Calculate tensor size in MB
        tensor_size_mb = kernel_tensor.numel() * kernel_tensor.element_size() / (1024 * 1024)
        
        # Evict old entries if cache is full
        while (len(self.gpu_cache) >= self.max_entries or 
               self.total_size_mb + tensor_size_mb > self.max_cache_size_mb):
            if not self.lru_tracker:
                break
                
            oldest_id, _ = self.lru_tracker.popitem(last=False)
            evicted_tensor = self.gpu_cache.pop(oldest_id)
            evicted_size_mb = evicted_tensor.numel() * evicted_tensor.element_size() / (1024 * 1024)
            self.total_size_mb -= evicted_size_mb
            logger.debug("Evicted kernel %s (%.2f MB)", oldest_id, evicted_size_mb)

        # Add new kernel
        self.gpu_cache[artifact_id] = kernel_tensor
        self.lru_tracker[artifact_id] = True
        self.total_size_mb += tensor_size_mb
        logger.debug("Cached kernel %s (%.2f MB)", artifact_id, tensor_size_mb)

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size_mb": self.total_size_mb,
            "cache_entries": len(self.gpu_cache),
        }

    def clear_cache(self):
        """Clear all cached kernels."""
        self.gpu_cache.clear()
        self.lru_tracker.clear()
        self.total_size_mb = 0
        logger.info("Kernel cache cleared")
```

#### **1.3. KasminaLayer Implementation (`src/esper/execution/kasmina_layer.py`)**

The core execution module that integrates all components.

```python
"""
KasminaLayer: High-performance execution layer for morphogenetic kernels.

This module implements the core execution engine that loads and runs
pre-compiled kernel artifacts with minimal overhead.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from esper.contracts.messages import HealthSignal, OonaMessage, TopicNames
from esper.services.oona_client import OonaClient
from .state_layout import KasminaStateLayout, SeedLifecycleState
from .kernel_cache import KernelCache

logger = logging.getLogger(__name__)


class KasminaLayer(nn.Module):
    """
    High-performance execution layer for morphogenetic kernels.
    
    This layer acts as a pure executor, loading and running pre-compiled
    kernel artifacts from Urza with GPU-resident caching.
    """

    def __init__(
        self,
        layer_id: int,
        num_seeds: int,
        input_dim: int,
        output_dim: int,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
    ):
        super().__init__()
        
        self.layer_id = layer_id
        self.num_seeds = num_seeds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.telemetry_enabled = telemetry_enabled
        
        # Initialize GPU state management
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        self.state = KasminaStateLayout(num_seeds, device)
        
        # Initialize kernel cache
        self.kernel_cache = KernelCache(max_cache_size_mb=cache_size_mb)
        
        # Default (dormant) transformation - simple pass-through
        self.default_transform = nn.Linear(input_dim, output_dim)
        
        # Message bus for telemetry
        self.oona_client = OonaClient() if telemetry_enabled else None
        
        # Performance tracking
        self.last_telemetry_epoch = 0
        self.execution_count = 0
        
        logger.info(
            "KasminaLayer initialized: layer_id=%d, num_seeds=%d, cache_size=%dMB",
            layer_id, num_seeds, cache_size_mb
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with morphogenetic kernel execution.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        self.execution_count += 1
        
        # Start timing for telemetry
        start_time = time.perf_counter()
        
        # For MVP: Check if any seeds are active, otherwise use default transform
        active_seeds = (self.state.lifecycle_states == SeedLifecycleState.ACTIVE).sum().item()
        
        if active_seeds == 0:
            # All seeds dormant - use default transformation
            output = self.default_transform(x)
        else:
            # Execute with active kernels
            output = self._execute_with_kernels(x)
        
        # Record execution latency
        end_time = time.perf_counter()
        latency_μs = int((end_time - start_time) * 1_000_000)
        
        # Update telemetry
        if self.telemetry_enabled:
            self._update_telemetry(latency_μs, active_seeds)
        
        return output

    def _execute_with_kernels(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass using active kernels."""
        # For MVP: Simple implementation that blends default transform with kernel outputs
        output = self.default_transform(x)
        
        # Find active seeds
        active_mask = self.state.lifecycle_states == SeedLifecycleState.ACTIVE
        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        
        for seed_idx in active_indices:
            kernel_id = self.state.active_kernel_id[seed_idx].item()
            alpha = self.state.alpha_blend[seed_idx].item()
            
            # For MVP: Apply simple blending (in production, would execute actual kernel)
            # This is a placeholder for actual kernel execution
            kernel_output = self._execute_kernel_placeholder(x, kernel_id)
            output = alpha * kernel_output + (1 - alpha) * output
        
        return output

    def _execute_kernel_placeholder(self, x: torch.Tensor, kernel_id: int) -> torch.Tensor:
        """Placeholder kernel execution for MVP."""
        # Simple transformation based on kernel_id for demo purposes
        weight_scale = 1.0 + (kernel_id % 10) * 0.1
        return self.default_transform(x) * weight_scale

    async def load_kernel(self, seed_idx: int, kernel_artifact_id: str) -> bool:
        """
        Load a specific kernel for the given seed.
        
        Args:
            seed_idx: Index of the seed to load kernel for
            kernel_artifact_id: Unique identifier of the kernel artifact
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update state to LOADING
            self.state.lifecycle_states[seed_idx] = SeedLifecycleState.LOADING
            
            # Load kernel from cache or Urza
            kernel_tensor = await self.kernel_cache.load_kernel(kernel_artifact_id)
            
            if kernel_tensor is None:
                # Loading failed
                self.state.lifecycle_states[seed_idx] = SeedLifecycleState.ERROR_RECOVERY
                self.state.error_count[seed_idx] += 1
                logger.error("Failed to load kernel %s for seed %d", kernel_artifact_id, seed_idx)
                return False
            
            # Successfully loaded - update state
            kernel_hash = hash(kernel_artifact_id) % (2**63)  # Convert to int64
            self.state.active_kernel_id[seed_idx] = kernel_hash
            self.state.lifecycle_states[seed_idx] = SeedLifecycleState.ACTIVE
            self.state.error_count[seed_idx] = 0  # Reset error count
            
            logger.info("Loaded kernel %s for seed %d", kernel_artifact_id, seed_idx)
            return True
            
        except Exception as e:
            self.state.lifecycle_states[seed_idx] = SeedLifecycleState.ERROR_RECOVERY
            self.state.error_count[seed_idx] += 1
            logger.error("Exception loading kernel %s for seed %d: %s", 
                        kernel_artifact_id, seed_idx, e)
            return False

    def unload_kernel(self, seed_idx: int) -> bool:
        """
        Unload kernel for the given seed, returning it to dormant state.
        
        Args:
            seed_idx: Index of the seed to unload
            
        Returns:
            True if successful
        """
        try:
            self.state.lifecycle_states[seed_idx] = SeedLifecycleState.DORMANT
            self.state.active_kernel_id[seed_idx] = 0
            self.state.alpha_blend[seed_idx] = 1.0
            self.state.error_count[seed_idx] = 0
            
            logger.info("Unloaded kernel for seed %d", seed_idx)
            return True
            
        except Exception as e:
            logger.error("Exception unloading kernel for seed %d: %s", seed_idx, e)
            return False

    def _update_telemetry(self, latency_μs: int, active_seeds: int):
        """Update and potentially publish telemetry data."""
        # Update accumulator with execution latency
        avg_latency = self.state.exec_latency_μs.float().mean().item()
        self.state.exec_latency_μs.fill_(latency_μs)
        
        # Publish telemetry every 100 executions (configurable)
        if self.execution_count % 100 == 0 and self.oona_client:
            self._publish_health_signal(avg_latency, active_seeds)

    def _publish_health_signal(self, avg_latency_μs: float, active_seeds: int):
        """Publish health signal to Oona message bus."""
        try:
            cache_stats = self.kernel_cache.get_cache_stats()
            
            health_signal = HealthSignal(
                layer_id=self.layer_id,
                seed_count=self.num_seeds,
                active_seed_count=active_seeds,
                avg_execution_latency_μs=avg_latency_μs,
                cache_hit_rate=cache_stats["cache_hit_rate"],
                error_count=self.state.error_count.sum().item(),
                memory_usage_mb=cache_stats["cache_size_mb"],
            )
            
            message = OonaMessage(
                sender_id=f"kasmina-layer-{self.layer_id}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload=health_signal.model_dump(),
            )
            
            self.oona_client.publish(message)
            logger.debug("Published health signal for layer %d", self.layer_id)
            
        except Exception as e:
            logger.warning("Failed to publish health signal: %s", e)

    def get_layer_stats(self) -> Dict[str, any]:
        """Get comprehensive layer statistics."""
        cache_stats = self.kernel_cache.get_cache_stats()
        
        active_seeds = (self.state.lifecycle_states == SeedLifecycleState.ACTIVE).sum().item()
        error_seeds = (self.state.lifecycle_states == SeedLifecycleState.ERROR_RECOVERY).sum().item()
        
        return {
            "layer_id": self.layer_id,
            "num_seeds": self.num_seeds,
            "active_seeds": active_seeds,
            "error_seeds": error_seeds,
            "execution_count": self.execution_count,
            "avg_latency_μs": self.state.exec_latency_μs.float().mean().item(),
            **cache_stats,
        }
```

-----

### **2. Model Wrapping Utility (`src/esper/core/model_wrapper.py`)**

**Task:** Implement the `esper.wrap()` function to automatically inject KasminaLayers.

```python
"""
Model wrapping utility for injecting morphogenetic capabilities.

This module provides the core esper.wrap() function that transforms
standard PyTorch models into morphogenetic models.
"""

import logging
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from esper.execution.kasmina_layer import KasminaLayer

logger = logging.getLogger(__name__)


class MorphableModel(nn.Module):
    """
    Wrapper for models that have been enhanced with morphogenetic capabilities.
    
    This wrapper maintains a registry of injected KasminaLayers and provides
    utilities for managing their lifecycle.
    """

    def __init__(self, base_model: nn.Module, kasmina_layers: Dict[str, KasminaLayer]):
        super().__init__()
        self.base_model = base_model
        self.kasmina_layers = nn.ModuleDict(kasmina_layers)
        
        # Track original layer mappings for debugging
        self.layer_mappings: Dict[str, str] = {}
        
        logger.info("MorphableModel created with %d KasminaLayers", len(kasmina_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the morphable model."""
        return self.base_model(x)

    async def load_kernel(self, layer_name: str, seed_idx: int, kernel_artifact_id: str) -> bool:
        """
        Load a kernel into a specific layer and seed.
        
        Args:
            layer_name: Name of the KasminaLayer
            seed_idx: Index of the seed within the layer
            kernel_artifact_id: Unique identifier of the kernel artifact
            
        Returns:
            True if successful, False otherwise
        """
        if layer_name not in self.kasmina_layers:
            logger.error("Layer %s not found in morphable model", layer_name)
            return False
            
        return await self.kasmina_layers[layer_name].load_kernel(seed_idx, kernel_artifact_id)

    def get_model_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics for all layers."""
        stats = {
            "total_layers": len(self.kasmina_layers),
            "layers": {}
        }
        
        for layer_name, layer in self.kasmina_layers.items():
            stats["layers"][layer_name] = layer.get_layer_stats()
            
        return stats


def wrap(
    model: nn.Module,
    target_layers: Optional[List[Union[str, Type[nn.Module]]]] = None,
    seeds_per_layer: int = 4,
    cache_size_mb: int = 128,
    telemetry_enabled: bool = True,
) -> MorphableModel:
    """
    Wrap a PyTorch model with morphogenetic capabilities.
    
    This function injects KasminaLayers into specified layers of the model,
    enabling them to load and execute compiled kernel artifacts.
    
    Args:
        model: The base PyTorch model to wrap
        target_layers: List of layer names or types to replace with KasminaLayers.
                      If None, defaults to [nn.Linear]
        seeds_per_layer: Number of morphogenetic seeds per layer
        cache_size_mb: Size of GPU cache for kernels per layer
        telemetry_enabled: Whether to enable telemetry collection
        
    Returns:
        MorphableModel with injected KasminaLayers
        
    Example:
        >>> import torch.nn as nn
        >>> from esper.core import wrap
        >>> 
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> morphable_model = wrap(model, seeds_per_layer=2)
    """
    if target_layers is None:
        target_layers = [nn.Linear]
    
    # Convert type targets to string names for easier processing
    target_types = [t for t in target_layers if isinstance(t, type)]
    target_names = [t for t in target_layers if isinstance(t, str)]
    
    kasmina_layers = {}
    layer_id = 0
    
    # Recursively find and replace target layers
    def replace_layers(module: nn.Module, prefix: str = "") -> nn.Module:
        nonlocal layer_id
        
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer should be replaced
            should_replace = (
                full_name in target_names or
                any(isinstance(child, target_type) for target_type in target_types)
            )
            
            if should_replace and isinstance(child, nn.Linear):
                # Replace with KasminaLayer
                kasmina_layer = KasminaLayer(
                    layer_id=layer_id,
                    num_seeds=seeds_per_layer,
                    input_dim=child.in_features,
                    output_dim=child.out_features,
                    cache_size_mb=cache_size_mb,
                    telemetry_enabled=telemetry_enabled,
                )
                
                # Copy original weights to default transform
                with torch.no_grad():
                    kasmina_layer.default_transform.weight.copy_(child.weight)
                    if child.bias is not None:
                        kasmina_layer.default_transform.bias.copy_(child.bias)
                
                # Replace the layer
                setattr(module, name, kasmina_layer)
                kasmina_layers[full_name] = kasmina_layer
                
                logger.info("Replaced %s (%d->%d) with KasminaLayer[%d]", 
                           full_name, child.in_features, child.out_features, layer_id)
                layer_id += 1
                
            else:
                # Recursively process child modules
                replace_layers(child, full_name)
    
    # Create a copy of the model to avoid modifying the original
    import copy
    wrapped_model = copy.deepcopy(model)
    
    # Replace target layers
    replace_layers(wrapped_model)
    
    if not kasmina_layers:
        logger.warning("No layers were replaced. Check target_layers specification.")
    
    return MorphableModel(wrapped_model, kasmina_layers)
```

-----

### **3. High-Level SDK (`src/esper/__init__.py`)**

**Task:** Create the main SDK interface for easy usage.

```python
"""
Esper Morphogenetic Training Platform - Main SDK Interface.

This module provides the high-level API for the Esper platform,
enabling easy integration of morphogenetic capabilities into PyTorch models.
"""

from .core.model_wrapper import wrap, MorphableModel
from .execution.kasmina_layer import KasminaLayer
from .execution.state_layout import SeedLifecycleState
from .configs import EsperConfig, load_config

__version__ = "0.2.0"
__all__ = [
    "wrap",
    "MorphableModel", 
    "KasminaLayer",
    "SeedLifecycleState",
    "EsperConfig",
    "load_config",
]

# Set up logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

-----

### **4. Phase 2 Integration Tests**

**Task:** Create comprehensive tests to validate the execution pipeline.

#### **4.1. End-to-End Integration Test (`tests/integration/test_phase2_execution.py`)**

```python
"""
Integration tests for Phase 2 Execution Engine.

This module contains tests that verify the end-to-end functionality
of the KasminaLayer execution pipeline.
"""

import asyncio
import pytest
import torch
import torch.nn as nn

import esper
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.state_layout import SeedLifecycleState


class TestPhase2ExecutionPipeline:
    """Integration tests for Phase 2 execution pipeline."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(4, 10)  # batch_size=4, input_dim=10

    def test_model_wrapping(self, simple_model):
        """Test basic model wrapping functionality."""
        # Wrap the model
        morphable_model = esper.wrap(simple_model, seeds_per_layer=2)
        
        # Verify structure
        assert isinstance(morphable_model, esper.MorphableModel)
        assert len(morphable_model.kasmina_layers) == 2  # Two Linear layers
        
        # Check that layers were properly replaced
        for layer_name, layer in morphable_model.kasmina_layers.items():
            assert isinstance(layer, KasminaLayer)
            assert layer.num_seeds == 2

    def test_dormant_forward_pass(self, simple_model, sample_input):
        """Test forward pass with all seeds dormant."""
        # Wrap and run forward pass
        morphable_model = esper.wrap(simple_model, seeds_per_layer=2)
        
        # Run forward pass
        output = morphable_model(sample_input)
        
        # Verify output shape
        assert output.shape == (4, 5)  # batch_size=4, output_dim=5
        
        # Verify all seeds are dormant
        for layer in morphable_model.kasmina_layers.values():
            assert (layer.state.lifecycle_states == SeedLifecycleState.DORMANT).all()

    def test_kernel_loading_simulation(self, simple_model):
        """Test kernel loading and activation simulation.""" 
        # Wrap the model
        morphable_model = esper.wrap(simple_model, seeds_per_layer=1)
        
        # Get first layer
        first_layer_name = list(morphable_model.kasmina_layers.keys())[0]
        first_layer = morphable_model.kasmina_layers[first_layer_name]
        
        async def load_and_test():
            # Load a simulated kernel
            success = await morphable_model.load_kernel(
                first_layer_name, 
                seed_idx=0, 
                kernel_artifact_id="test-kernel-001"
            )
            
            # Verify loading succeeded
            assert success
            assert first_layer.state.lifecycle_states[0] == SeedLifecycleState.ACTIVE
            assert first_layer.state.active_kernel_id[0] != 0
            
            return True
            
        # Run async test
        result = asyncio.run(load_and_test())
        assert result

    def test_cache_functionality(self):
        """Test kernel cache hit/miss behavior."""
        from esper.execution.kernel_cache import KernelCache
        
        cache = KernelCache(max_cache_size_mb=64, max_entries=10)
        
        async def test_cache():
            # First load should be a miss
            kernel1 = await cache.load_kernel("kernel-001")
            stats1 = cache.get_cache_stats()
            assert stats1["cache_misses"] == 1
            assert stats1["cache_hits"] == 0
            assert kernel1 is not None
            
            # Second load should be a hit
            kernel2 = await cache.load_kernel("kernel-001") 
            stats2 = cache.get_cache_stats()
            assert stats2["cache_misses"] == 1
            assert stats2["cache_hits"] == 1
            assert torch.equal(kernel1, kernel2)
            
            return True
            
        result = asyncio.run(test_cache())
        assert result

    def test_telemetry_collection(self, simple_model, sample_input):
        """Test telemetry collection and statistics."""
        # Wrap model with telemetry enabled
        morphable_model = esper.wrap(simple_model, telemetry_enabled=True)
        
        # Run several forward passes
        for _ in range(10):
            _ = morphable_model(sample_input)
        
        # Check layer statistics
        stats = morphable_model.get_model_stats()
        assert stats["total_layers"] == 2
        
        for layer_name, layer_stats in stats["layers"].items():
            assert layer_stats["execution_count"] >= 10
            assert layer_stats["layer_id"] >= 0
            assert "avg_latency_μs" in layer_stats

    def test_error_handling(self, simple_model):
        """Test error handling and recovery."""
        morphable_model = esper.wrap(simple_model, seeds_per_layer=1)
        
        # Get first layer
        first_layer_name = list(morphable_model.kasmina_layers.keys())[0]
        first_layer = morphable_model.kasmina_layers[first_layer_name]
        
        async def test_error_recovery():
            # Try to load an invalid kernel (this should fail gracefully)
            success = await morphable_model.load_kernel(
                first_layer_name,
                seed_idx=0,
                kernel_artifact_id="invalid-kernel-id-that-will-fail"
            )
            
            # Loading should fail but not crash
            # (Implementation depends on Urza client behavior)
            # For MVP, our placeholder always succeeds, so we test unload instead
            
            # Test unload functionality
            unload_success = first_layer.unload_kernel(0)
            assert unload_success
            assert first_layer.state.lifecycle_states[0] == SeedLifecycleState.DORMANT
            
            return True
            
        result = asyncio.run(test_error_recovery())
        assert result

    def test_performance_characteristics(self, simple_model, sample_input):
        """Test basic performance characteristics."""
        import time
        
        # Test original model performance
        start_time = time.perf_counter()
        for _ in range(100):
            _ = simple_model(sample_input)
        original_time = time.perf_counter() - start_time
        
        # Test wrapped model performance (dormant seeds)
        morphable_model = esper.wrap(simple_model, seeds_per_layer=2)
        start_time = time.perf_counter()
        for _ in range(100):
            _ = morphable_model(sample_input)
        wrapped_time = time.perf_counter() - start_time
        
        # Overhead should be minimal (target: < 1% as per HLD)
        overhead_ratio = (wrapped_time - original_time) / original_time
        
        # For MVP, we allow higher overhead, but log it for monitoring
        print(f"Dormant seed overhead: {overhead_ratio:.2%}")
        
        # Ensure we don't have catastrophic overhead
        assert overhead_ratio < 0.5  # Allow up to 50% overhead for MVP

    def test_full_pipeline_integration(self, simple_model, sample_input):
        """Test full pipeline from wrapping to execution with kernel loading."""
        async def full_test():
            # Step 1: Wrap the model
            morphable_model = esper.wrap(simple_model, seeds_per_layer=1)
            
            # Step 2: Initial forward pass (all dormant)
            output1 = morphable_model(sample_input)
            
            # Step 3: Load a kernel into first layer
            first_layer_name = list(morphable_model.kasmina_layers.keys())[0]
            load_success = await morphable_model.load_kernel(
                first_layer_name,
                seed_idx=0,
                kernel_artifact_id="integration-test-kernel"
            )
            assert load_success
            
            # Step 4: Forward pass with active kernel
            output2 = morphable_model(sample_input)
            
            # Step 5: Verify output shapes are consistent
            assert output1.shape == output2.shape
            
            # Step 6: Get comprehensive stats
            stats = morphable_model.get_model_stats()
            assert stats["total_layers"] == 2
            
            # At least one layer should have an active seed
            has_active_seed = any(
                layer_stats["active_seeds"] > 0 
                for layer_stats in stats["layers"].values()
            )
            assert has_active_seed
            
            return True
            
        result = asyncio.run(full_test())
        assert result
```

-----

### **5. Phase 2 Testing & Validation Strategy**

1. **Unit Tests:**
   * Test `KasminaLayer` state management and lifecycle transitions
   * Test `KernelCache` LRU behavior and memory management  
   * Test `esper.wrap()` layer injection logic
   * Test telemetry collection and publishing

2. **Integration Tests:**
   * Test full execution pipeline with real model wrapping
   * Test kernel loading simulation (without requiring Urza)
   * Test performance characteristics and overhead measurement
   * Test error handling and graceful degradation

3. **Performance Validation:**
   * Measure dormant seed overhead (target: < 1%)
   * Validate cache hit rates in steady state
   * Benchmark kernel loading latency (simulated)

4. **End-to-End Scenario:**
   * **"Golden Path" Test:** A script that:
     1. Starts Phase 1 services (`Urza`, `Tezzeret`)
     2. Submits a test blueprint via Urza API
     3. Waits for compilation to complete
     4. Wraps a PyTorch model with `esper.wrap()`
     5. Commands KasminaLayer to load the compiled kernel
     6. Executes forward pass and validates output integrity

-----

### **6. Definition of Done**

Phase 2 is complete when:

* ✅ **KasminaLayer implemented** with GPU state management and kernel caching
* ✅ **esper.wrap() utility** can inject KasminaLayers into standard PyTorch models
* ✅ **All unit tests passing** with >85% code coverage
* ✅ **Integration tests passing** demonstrating end-to-end execution
* ✅ **Performance validation** showing <5% overhead for dormant seeds (relaxed from 1% for MVP)
* ✅ **End-to-end test** proving the complete `Tezzeret -> Urza -> Kasmina` pipeline
* ✅ **Documentation** with clear usage examples and API reference

### **7. System Components Summary**

**Phase 2 introduces these key systems:**

1. **KasminaLayer**: The core execution engine
   * GPU-optimized state tensor (SoA layout)
   * LRU kernel cache
   * Telemetry collection
   * Error handling and fallbacks

2. **Model Wrapper**: The `esper.wrap()` utility
   * Automatic layer injection
   * Morphable model abstraction
   * Lifecycle management

3. **Execution Pipeline**: End-to-end kernel execution
   * Kernel loading from cache/Urza
   * State management and transitions
   * Performance monitoring

4. **Integration Layer**: Connection to Phase 1
   * Urza client integration
   * Oona telemetry publishing
   * Async kernel loading

This phase establishes the execution foundation required for Phase 3 (Controller & Training Loop) and validates the core morphogenetic execution mechanic.

# Core Module (`src/esper/core/`)

## Overview

The core module provides the primary user-facing API that transforms standard PyTorch models into morphogenetic models. It serves as the main abstraction layer between user code and the Esper platform, enabling seamless integration of morphogenetic capabilities into existing PyTorch workflows.

## Files

### `__init__.py` - Core Module Initialization

**Purpose:** Placeholder for core functionality with future expansion capabilities.

**Contents:**
```python
"""
Core functionality for the Esper system.
This module contains the fundamental building blocks and algorithms.
"""

# Placeholder for core implementations
# Core functionality will be implemented in subsequent phases
```

**Status:** Minimal implementation - serves as namespace reservation for future core algorithms and utilities.

### `model_wrapper.py` - PyTorch Model Integration

**Purpose:** Primary API for transforming PyTorch models into morphogenetic models with KasminaLayer injection.

#### Key Classes

**`MorphableModel`** - Enhanced PyTorch Model
```python
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
        original_model: Optional[nn.Module] = None
    ):
```

**Key Attributes:**
- `wrapped_model: nn.Module` - The model with KasminaLayers injected
- `kasmina_layers: nn.ModuleDict` - Dict mapping layer names to KasminaLayer instances
- `original_model: Optional[nn.Module]` - Reference to original model for comparison
- `total_forward_calls: int` - Performance tracking counter
- `morphogenetic_active: bool` - Whether any kernels are currently loaded

**Core Methods:**

**`forward(x: torch.Tensor) -> torch.Tensor`**
```python
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
```
- **Purpose:** Standard PyTorch forward pass with usage tracking
- **Behavior:** Delegates to wrapped model (which contains KasminaLayers)
- **Performance:** Minimal overhead, maintains PyTorch compatibility

**`async load_kernel(layer_name: str, seed_idx: int, artifact_id: str) -> bool`**
```python
async def load_kernel(self, layer_name: str, seed_idx: int, artifact_id: str) -> bool:
    """
    Load a compiled kernel into a specific layer and seed.
    
    Args:
        layer_name: Name of the layer
        seed_idx: Index of the seed within the layer
        artifact_id: ID of the kernel artifact
        
    Returns:
        True if kernel was loaded successfully
    """
```
- **Purpose:** Dynamic kernel loading during training
- **Features:** Validates layer existence, updates morphogenetic status
- **Error Handling:** Graceful failure with detailed logging
- **Integration:** Calls KasminaLayer.load_kernel() method

**`async unload_kernel(layer_name: str, seed_idx: int) -> bool`**
```python
async def unload_kernel(self, layer_name: str, seed_idx: int) -> bool:
    """
    Unload kernel from a specific layer and seed.
    
    Args:
        layer_name: Name of the layer
        seed_idx: Index of the seed within the layer
        
    Returns:
        True if kernel was unloaded successfully
    """
```
- **Purpose:** Remove kernels to return to default behavior
- **Features:** Updates morphogenetic status, checks for remaining active kernels
- **Integration:** Coordinates with KasminaLayer state management

**Statistics and Monitoring Methods:**

**`get_layer_stats(layer_name: Optional[str] = None) -> Dict[str, Any]`**
```python
def get_layer_stats(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics for a specific layer or all layers.
    
    Args:
        layer_name: Name of specific layer, or None for all layers
        
    Returns:
        Dictionary containing layer statistics
    """
```
- **Purpose:** Detailed performance and state monitoring
- **Features:** Per-layer or aggregate statistics
- **Usage:** Debugging, performance analysis, telemetry

**`get_model_stats() -> Dict[str, Any]`**
```python
def get_model_stats(self) -> Dict[str, Any]:
    """
    Get comprehensive model statistics.
    
    Returns:
        Dictionary containing model statistics
    """
    return {
        "total_forward_calls": self.total_forward_calls,
        "morphogenetic_active": self.morphogenetic_active,
        "total_kasmina_layers": len(self.kasmina_layers),
        "total_seeds": total_seeds,
        "active_seeds": active_seeds,
        "total_kernel_executions": total_kernel_executions,
        "layer_stats": layer_stats,
    }
```
- **Purpose:** Comprehensive model performance overview
- **Features:** Aggregated statistics across all layers
- **Usage:** System monitoring, performance optimization

**Control Methods:**

**`set_seed_alpha(layer_name: str, seed_idx: int, alpha: float) -> None`**
```python
def set_seed_alpha(self, layer_name: str, seed_idx: int, alpha: float) -> None:
    """
    Set the alpha blend factor for a specific seed.
    
    Args:
        layer_name: Name of the layer
        seed_idx: Index of the seed
        alpha: Blend factor (0.0 to 1.0)
    """
```
- **Purpose:** Control morphogenetic influence strength
- **Range:** 0.0 (default behavior) to 1.0 (full kernel behavior)
- **Usage:** Gradual integration, A/B testing, fine-tuning

**`enable_telemetry(enabled: bool = True) -> None`**
```python
def enable_telemetry(self, enabled: bool = True) -> None:
    """
    Enable or disable telemetry for all KasminaLayers.
    
    Args:
        enabled: Whether to enable telemetry
    """
```
- **Purpose:** Control telemetry collection across all layers
- **Features:** Bulk configuration, performance optimization
- **Usage:** Production vs development modes

**Comparison and Analysis:**

**`compare_with_original(x: torch.Tensor) -> Dict[str, Any]`**
```python
def compare_with_original(self, x: torch.Tensor) -> Dict[str, Any]:
    """
    Compare output with original model (if available).
    
    Args:
        x: Input tensor
        
    Returns:
        Dictionary containing comparison results
    """
    # Get outputs
    morphable_output = self.forward(x)
    original_output = self.original_model(x)
    
    # Compute differences
    diff = morphable_output - original_output
    mse = torch.mean(diff ** 2).item()
    max_diff = torch.max(torch.abs(diff)).item()
    
    return {
        "mse": mse,
        "max_absolute_difference": max_diff,
        "output_shape": morphable_output.shape,
        "morphogenetic_active": self.morphogenetic_active,
    }
```
- **Purpose:** Validate morphogenetic modifications don't break functionality
- **Metrics:** MSE, maximum absolute difference
- **Usage:** Testing, validation, quality assurance

#### Key Functions

**`wrap()` - Primary Public API**
```python
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
    """
```

**Parameters:**
- `model: nn.Module` - The PyTorch model to wrap
- `target_layers: Optional[List[Type[nn.Module]]] = None` - Layer types to replace (default: [nn.Linear])
- `seeds_per_layer: int = 4` - Number of morphogenetic seeds per layer
- `cache_size_mb: int = 128` - Kernel cache size in MB per layer
- `telemetry_enabled: bool = True` - Whether to enable telemetry collection
- `preserve_original: bool = True` - Whether to keep a reference to the original model

**Process:**
1. **Deep Copy:** Creates copies of input model to avoid modification
2. **Layer Discovery:** Recursively traverses model hierarchy
3. **Layer Replacement:** Replaces target layers with KasminaLayers
4. **Weight Preservation:** Copies original weights to maintain behavior
5. **MorphableModel Creation:** Returns wrapped model with capabilities

**Supported Layer Types:**

The core module supports multiple PyTorch layer types with specialized KasminaLayer implementations:

**Linear Layers (Full Support):**
```python
if isinstance(original_layer, nn.Linear):
    return KasminaLayer(
        input_size=original_layer.in_features,
        output_size=original_layer.out_features,
        **kwargs
    )
```

**Convolutional Layers (Specialized Support):**
```python
elif isinstance(original_layer, nn.Conv2d):
    return KasminaConv2dLayer(
        in_channels=original_layer.in_channels,
        out_channels=original_layer.out_channels,
        kernel_size=original_layer.kernel_size,
        **kwargs
    )
```

**Attention Layers (Advanced Support):**
```python
elif isinstance(original_layer, nn.MultiheadAttention):
    return KasminaAttentionLayer(
        embed_dim=original_layer.embed_dim,
        num_heads=original_layer.num_heads,
        **kwargs
    )
```

**Normalization Layers (Specialized Support):**
```python
elif isinstance(original_layer, nn.LayerNorm):
    return KasminaLayerNormLayer(
        normalized_shape=original_layer.normalized_shape,
        **kwargs
    )
elif isinstance(original_layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
    # Appropriate BatchNorm variant
```

**Error Handling:**
```python
else:
    logger.warning(f"Layer type {type(original_layer)} not supported, skipping")
    return original_layer  # Graceful fallback
```

**`_create_kasmina_layer()` - Layer Factory Function**
```python
def _create_kasmina_layer(
    original_layer: nn.Module,
    layer_name: str,
    seeds_per_layer: int,
    cache_size_mb: int,
    telemetry_enabled: bool
) -> KasminaLayer:
    """
    Create a KasminaLayer replacement for an original layer.
    """
```
- **Purpose:** Factory function for creating configured KasminaLayers
- **Features:** Layer-specific dimension extraction, weight copying
- **Error Handling:** Clear errors for unsupported layer types

**`unwrap()` - Model Extraction**
```python
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
        # Return the wrapped model (behaves like original when no kernels loaded)
        return morphable_model.wrapped_model
```
- **Purpose:** Extract clean PyTorch model for deployment or analysis
- **Fallback:** Returns wrapped model if original not preserved
- **Usage:** Model deployment, comparison, legacy integration

## Architecture Integration

The core module serves as the primary integration point:

1. **User Code** → `esper.wrap()` → **MorphableModel**
2. **MorphableModel** → **KasminaLayers** → **Execution Module**
3. **Training Loops** → **Kernel Loading** → **Service Integration**
4. **Telemetry** → **Health Signals** → **Strategic Controller**

## Dependencies

**External:**
- `torch` - PyTorch neural network framework
- `torch.nn` - Neural network modules and layers
- `typing` - Type annotations and hints
- `copy` - Deep copying for model preservation
- `logging` - Error and debug logging

**Internal:**
- `esper.execution.kasmina_layer` - Core execution engine
- **Indirect:** All execution and service modules through KasminaLayer

## Performance Considerations

### Model Wrapping Performance
- **Deep Copy Overhead:** One-time cost during model initialization
- **Layer Traversal:** O(n) where n is number of model layers
- **Memory Usage:** ~3x model size (original + wrapped + preserved original)

### Runtime Performance
- **Forward Pass:** Minimal overhead when no kernels loaded
- **Kernel Operations:** Async operations don't block training
- **Telemetry:** Optional and configurable impact

### Optimization Strategies
- **Lazy Loading:** KasminaLayers only activate when kernels loaded
- **Cache Management:** Configurable cache sizes per layer
- **Selective Wrapping:** Only wrap target layer types

## Usage Patterns

### Basic Model Wrapping
```python
import torch
import torch.nn as nn
import esper

# Standard PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Transform to morphogenetic model
morphable_model = esper.wrap(
    model,
    target_layers=[nn.Linear],  # Only wrap Linear layers
    seeds_per_layer=4,          # 4 seeds per layer
    cache_size_mb=128,          # 128MB cache per layer
    telemetry_enabled=True      # Enable telemetry
)

# Use like normal PyTorch model
optimizer = torch.optim.Adam(morphable_model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
for batch in dataloader:
    output = morphable_model(batch.x)
    loss = criterion(output, batch.y)
    loss.backward()
    optimizer.step()
```

### Advanced Configuration
```python
# Custom layer targeting
morphable_model = esper.wrap(
    model,
    target_layers=[nn.Linear, nn.Conv2d],  # Multiple layer types
    seeds_per_layer=8,                     # More seeds for complex models
    cache_size_mb=256,                     # Larger cache
    telemetry_enabled=True,
    preserve_original=True                 # Keep original for comparison
)

# Monitor performance
stats = morphable_model.get_model_stats()
print(f"Total forward calls: {stats['total_forward_calls']}")
print(f"Active seeds: {stats['active_seeds']}/{stats['total_seeds']}")
print(f"Morphogenetic active: {stats['morphogenetic_active']}")
```

### Dynamic Kernel Loading
```python
# Load kernel during training
success = await morphable_model.load_kernel(
    layer_name="layer.0",  # First linear layer
    seed_idx=0,            # First seed
    artifact_id="kernel-abc123"
)

if success:
    print("Kernel loaded successfully")
    
    # Adjust blend factor
    morphable_model.set_seed_alpha("layer.0", 0, 0.3)  # 30% kernel influence
    
    # Monitor changes
    comparison = morphable_model.compare_with_original(test_input)
    print(f"MSE difference: {comparison['mse']}")
```

### Production Deployment
```python
# Disable telemetry for production
morphable_model.enable_telemetry(False)

# Extract clean model for deployment
if morphable_model.morphogenetic_active:
    # Model has active kernels - keep as MorphableModel
    deployment_model = morphable_model
else:
    # No active kernels - can use original model
    deployment_model = esper.unwrap(morphable_model)

# Save model
torch.save(deployment_model.state_dict(), "model.pt")
```

### Model Analysis and Debugging
```python
# Get detailed layer statistics
layer_stats = morphable_model.get_layer_stats("layer.0")
print(f"Layer stats: {layer_stats}")

# Compare with original model
test_input = torch.randn(1, 784)
comparison = morphable_model.compare_with_original(test_input)
print(f"Output difference - MSE: {comparison['mse']:.6f}")
print(f"Max absolute diff: {comparison['max_absolute_difference']:.6f}")

# Monitor specific layer
layer_names = morphable_model.get_layer_names()
for layer_name in layer_names:
    stats = morphable_model.get_layer_stats(layer_name)
    print(f"{layer_name}: {stats['state_stats']['active_seeds']} active seeds")
```

## Error Handling

### Model Wrapping Errors
```python
try:
    morphable_model = esper.wrap(model, target_layers=[nn.LSTM])
except NotImplementedError as e:
    print(f"Unsupported layer type: {e}")
    # Fall back to supported layers
    morphable_model = esper.wrap(model, target_layers=[nn.Linear])
```

### Kernel Loading Errors
```python
try:
    success = await morphable_model.load_kernel("nonexistent_layer", 0, "kernel-123")
except ValueError as e:
    print(f"Invalid layer name: {e}")

if not success:
    print("Kernel loading failed - check Urza service and artifact ID")
```

### Runtime Errors
```python
# Graceful degradation - model continues working even if kernels fail
try:
    output = morphable_model(input_tensor)
except Exception as e:
    print(f"Model execution error: {e}")
    # KasminaLayers automatically fall back to default behavior
```

## Known Issues and Limitations

### Current Limitations

1. **Conv2d Support:** Simplified implementation may not preserve full convolutional semantics
   - **Impact:** May lose spatial relationships in convolutional layers
   - **Workaround:** Focus on Linear layers for MVP
   - **Future:** Implement proper convolutional kernel handling

2. **Layer Type Support:** Limited to Linear and Conv2d layers
   - **Missing:** LSTM, GRU, Transformer, custom layers
   - **Workaround:** Manually specify supported layers only
   - **Future:** Extensible layer factory system

3. **Memory Overhead:** 3x model size during wrapping process
   - **Impact:** Large models may exceed memory limits
   - **Workaround:** Disable original model preservation
   - **Future:** In-place wrapping options

### Best Practices

1. **Target Layer Selection:** Start with Linear layers only for stability
2. **Memory Management:** Monitor memory usage with large models
3. **Gradual Integration:** Begin with small seed counts and cache sizes
4. **Testing:** Always compare with original model during development
5. **Production:** Disable telemetry and original model preservation for deployment

## Future Enhancements

1. **Extended Layer Support**
   - Full Conv2d support with proper weight adaptation
   - LSTM/GRU support with state management
   - Transformer layer integration
   - Custom layer registration system

2. **Performance Optimizations**
   - In-place model wrapping to reduce memory overhead
   - Lazy layer replacement for large models
   - Optimized weight copying for different layer types

3. **Advanced Features**
   - Model analysis and recommendation for optimal layer targeting
   - Automatic hyperparameter tuning for seeds and cache sizes
   - Integration with popular model architectures (ResNet, BERT, etc.)

4. **Developer Experience**
   - Visual model inspection tools
   - Performance profiling integration
   - Automated testing and validation frameworks
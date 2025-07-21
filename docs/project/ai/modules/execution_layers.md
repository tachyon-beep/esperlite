# Execution Layers Module (`src/esper/execution/`)

## Additional Specialized Execution Layers

Beyond the core KasminaLayer, the execution module includes specialized layer implementations for different PyTorch layer types, each optimized for specific morphogenetic use cases.

## Specialized Layer Files

### `kasmina_conv2d_layer.py` - Convolutional Layer Wrapper

**Purpose:** Specialized KasminaLayer for 2D convolutional operations with spatial awareness.

#### Key Components

**`KasminaConv2dLayer`** - Convolutional Morphogenetic Layer
```python
class KasminaConv2dLayer(KasminaLayer):
    """
    Specialized KasminaLayer for Conv2d operations.
    
    Handles spatial relationships and channel-aware kernel loading
    while maintaining convolutional semantics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        **kwargs
    ):
```

**Features:**
- **Spatial Awareness:** Maintains convolutional spatial relationships
- **Channel Handling:** Proper in/out channel management for kernel compatibility
- **Kernel Adaptation:** Converts loaded kernels to proper convolutional format
- **Performance Optimization:** Specialized fast paths for common conv2d patterns

**Integration Points:**
- Used by `wrap()` when targeting `nn.Conv2d` layers
- Preserves original conv2d parameters (stride, padding, dilation)
- Maintains weight and bias shapes for compatibility

### `kasmina_attention_layer.py` - Multi-Head Attention Wrapper

**Purpose:** Specialized wrapper for `nn.MultiheadAttention` with morphogenetic capabilities.

#### Key Components

**`KasminaAttentionLayer`** - Attention Morphogenetic Layer
```python
class KasminaAttentionLayer(KasminaLayer):
    """
    Specialized KasminaLayer for MultiheadAttention operations.
    
    Handles attention-specific morphogenetic adaptations including
    head-wise kernel loading and attention pattern modifications.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
```

**Features:**
- **Head-Aware Kernels:** Can load kernels targeting specific attention heads
- **Attention Pattern Adaptation:** Modify attention computation patterns
- **Dropout Preservation:** Maintains original dropout behavior
- **Query/Key/Value Handling:** Proper handling of attention weight matrices

**Special Considerations:**
- More complex than simple linear transformations
- Requires understanding of attention mechanism internals
- Kernel shapes must match attention head dimensions

### `kasmina_layernorm_layer.py` - Layer Normalization Wrapper

**Purpose:** Specialized wrapper for `nn.LayerNorm` with morphogenetic adaptations.

#### Key Components

**`KasminaLayerNormLayer`** - LayerNorm Morphogenetic Layer
```python
class KasminaLayerNormLayer(KasminaLayer):
    """
    Specialized KasminaLayer for LayerNorm operations.
    
    Enables morphogenetic adaptations to normalization behavior
    while preserving numerical stability.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs
    ):
```

**Features:**
- **Normalization Preservation:** Maintains numerical stability requirements
- **Scale/Bias Adaptation:** Can adapt learnable scale and bias parameters
- **Shape Handling:** Proper handling of multi-dimensional normalized shapes
- **Epsilon Preservation:** Maintains numerical stability epsilon

**Use Cases:**
- Adaptive normalization strategies
- Dynamic scale/bias adjustment
- Normalization pattern learning

### `kasmina_batchnorm_layer.py` - Batch Normalization Wrapper

**Purpose:** Specialized wrapper for `nn.BatchNorm1d` and `nn.BatchNorm2d` with morphogenetic capabilities.

#### Key Components

**`Kasmina1dBatchNormLayer`** - 1D BatchNorm Wrapper
```python
class Kasmina1dBatchNormLayer(KasminaLayer):
    """
    Specialized KasminaLayer for BatchNorm1d operations.
    
    Handles running statistics and morphogenetic adaptations
    for 1D batch normalization.
    """
```

**`Kasmina2dBatchNormLayer`** - 2D BatchNorm Wrapper
```python
class Kasmina2dBatchNormLayer(KasminaLayer):
    """
    Specialized KasminaLayer for BatchNorm2d operations.
    
    Handles running statistics and morphogenetic adaptations
    for 2D batch normalization (typical for CNNs).
    """
```

**Features:**
- **Running Statistics:** Proper handling of running mean/variance
- **Training/Eval Modes:** Correct behavior in training vs evaluation
- **Momentum Handling:** Preserves batch norm momentum settings
- **Affine Transformation:** Maintains scale/bias parameter handling

**Critical Considerations:**
- Must preserve running statistics for model consistency
- Training vs evaluation mode behavior is crucial
- Momentum and epsilon parameters affect stability

## Layer Creation and Integration

### Automatic Layer Selection

The `wrap()` function in `model_wrapper.py` automatically selects the appropriate specialized layer:

```python
def _create_kasmina_layer(original_layer, layer_name, **kwargs):
    """Create appropriate KasminaLayer variant based on layer type."""
    
    if isinstance(original_layer, nn.Linear):
        return KasminaLayer(**kwargs)
    elif isinstance(original_layer, nn.Conv2d):
        return KasminaConv2dLayer(**kwargs)
    elif isinstance(original_layer, nn.MultiheadAttention):
        return KasminaAttentionLayer(**kwargs)
    elif isinstance(original_layer, nn.LayerNorm):
        return KasminaLayerNormLayer(**kwargs)
    elif isinstance(original_layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if isinstance(original_layer, nn.BatchNorm1d):
            return Kasmina1dBatchNormLayer(**kwargs)
        else:
            return Kasmina2dBatchNormLayer(**kwargs)
    else:
        raise NotImplementedError(f"Layer type {type(original_layer)} not supported")
```

### Weight Preservation

Each specialized layer implements proper weight copying:

```python
# Example for Conv2d
def _copy_conv2d_weights(kasmina_layer, original_layer):
    """Copy weights preserving convolutional structure."""
    with torch.no_grad():
        # Reshape conv weights to work with linear transformation
        weight_reshaped = original_layer.weight.view(
            original_layer.out_channels, -1
        )
        kasmina_layer.default_transform.weight.copy_(weight_reshaped)
        
        if original_layer.bias is not None:
            kasmina_layer.default_transform.bias.copy_(original_layer.bias)
```

## Performance Characteristics

### Layer-Specific Optimizations

1. **Conv2d Layers:**
   - Spatial locality preservation
   - Channel-wise kernel adaptation
   - Optimized for CNN workloads

2. **Attention Layers:**
   - Head-specific kernel loading
   - Attention pattern adaptation
   - Transformer optimization

3. **Normalization Layers:**
   - Minimal overhead for stable operations
   - Statistics preservation
   - Numerical stability maintenance

### Memory Usage

- Each layer type has optimized memory layouts
- State tensors sized appropriately for layer semantics
- Kernel caches tuned for layer-specific patterns

## Best Practices

### Layer Selection
1. **Automatic Selection:** Let `wrap()` choose the appropriate layer type
2. **Manual Override:** Specify layer types only when needed
3. **Performance Testing:** Benchmark different layer combinations

### Configuration
1. **Seeds Per Layer:** Adjust based on layer complexity
2. **Cache Sizing:** Size caches based on layer memory requirements
3. **Telemetry:** Enable for performance monitoring

### Maintenance
1. **Weight Preservation:** Always verify original behavior is maintained
2. **Shape Compatibility:** Ensure kernel shapes match layer requirements
3. **State Management:** Proper handling of layer-specific state (running stats, etc.)

## Known Limitations

1. **Complex Layers:** Some PyTorch layers may not have specialized implementations
2. **Custom Layers:** User-defined layers require manual adaptation
3. **Shape Constraints:** Kernel shapes must be compatible with layer semantics
4. **Performance Overhead:** Some specialized layers may have higher overhead than basic KasminaLayer

## Future Enhancements

1. **Additional Layer Types:**
   - RNN/LSTM/GRU support
   - Transformer-specific layers
   - Custom activation functions

2. **Advanced Adaptations:**
   - Dynamic shape modification
   - Architecture search integration
   - Automated layer optimization

3. **Performance Optimizations:**
   - Layer-specific kernel formats
   - Optimized memory layouts
   - Hardware-specific accelerations
# Current Morphogenetic System API Documentation

## Overview

This document provides comprehensive API documentation for the current morphogenetic system implementation. It serves as the reference for deprecation planning during the migration to v2.

## KasminaLayer API

### Class: `esper.execution.kasmina_layer.KasminaLayer`

The core execution layer for morphogenetic kernels. Inherits from `torch.nn.Module`.

#### Constructor

```python
KasminaLayer(
    input_size: int,
    output_size: int,
    num_seeds: int = 4,
    cache_size_mb: int = 128,
    telemetry_enabled: bool = True,
    layer_name: str = "kasmina_layer"
)
```

**Parameters:**
- `input_size`: Input tensor dimensions
- `output_size`: Output tensor dimensions  
- `num_seeds`: Number of morphogenetic seeds (default: 4)
- `cache_size_mb`: Kernel cache size in MB (default: 128)
- `telemetry_enabled`: Enable telemetry collection (default: True)
- `layer_name`: Layer identifier for telemetry

**Migration Note:** Constructor will change significantly in v2 to support chunked architecture.

#### Public Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`

Standard PyTorch forward pass.

**Parameters:**
- `x`: Input tensor of shape `(batch_size, input_size)`

**Returns:**
- Output tensor of shape `(batch_size, output_size)`

**Migration Note:** Will remain compatible but internal implementation will use Triton kernels.

##### `get_layer_stats() -> Dict[str, Any]`

Get current layer statistics including health metrics and performance data.

**Returns:**
```python
{
    "active_seeds": int,
    "health_scores": List[float],
    "execution_stats": {
        "total_executions": int,
        "kernel_executions": int,
        "errors": int,
        "avg_latency_ms": float
    },
    "cache_stats": Dict[str, Any]
}
```

**Migration Note:** Will be extended with chunk-level statistics in v2.

##### `find_compatible_kernels(target_shape: Tuple[int, ...]) -> List[str]`

Find kernels compatible with given input shape.

**Parameters:**
- `target_shape`: Target tensor shape

**Returns:**
- List of compatible kernel IDs

**Deprecation:** Will be replaced by blueprint selection in v2.

##### `set_seed_alpha(seed_idx: int, alpha: float) -> None`

Set blending factor for a specific seed.

**Parameters:**
- `seed_idx`: Seed index (0 to num_seeds-1)
- `alpha`: Blending factor (0.0 to 1.0)

**Migration Note:** Will be handled by grafting strategies in v2.

##### `to(device: torch.device) -> "KasminaLayer"`

Move layer to specified device (standard PyTorch).

**Parameters:**
- `device`: Target device

**Returns:**
- Self reference

**Migration Note:** No changes expected.

### State Management

#### Class: `esper.execution.state_layout.KasminaStateLayout`

Manages seed states on GPU.

```python
class KasminaStateLayout:
    def __init__(self, num_seeds: int, device: torch.device)
    
    # Properties
    lifecycle_states: torch.Tensor  # Shape: (num_seeds,)
    active_kernel_id: torch.Tensor  # Shape: (num_seeds,)
    alpha_blend: torch.Tensor       # Shape: (num_seeds,)
    health_accumulator: torch.Tensor # Shape: (num_seeds,)
```

#### Enum: `SeedLifecycleState`

```python
class SeedLifecycleState(Enum):
    DORMANT = 0
    LOADING = 1
    ACTIVE = 2
    ERROR_RECOVERY = 3
    FOSSILIZED = 4
```

**Migration Note:** Will expand to 11 states in v2:
- DORMANT (0)
- GERMINATED (1) - NEW
- TRAINING (2) - NEW
- GRAFTING (3) - RENAMED from LOADING
- STABILIZATION (4) - NEW
- EVALUATING (5) - NEW  
- FINE_TUNING (6) - NEW
- FOSSILIZED (7)
- CULLED (8) - NEW
- CANCELLED (9) - NEW
- ROLLED_BACK (10) - NEW

## Tamiyo API

### Class: `esper.services.tamiyo.analyzer.ModelGraphAnalyzer`

Analyzes model state and constructs graph representations.

#### Constructor

```python
ModelGraphAnalyzer(health_history_window: int = 10)
```

**Parameters:**
- `health_history_window`: Number of historical health samples to track

#### Public Methods

##### `analyze_model_state(health_signals: Dict[str, HealthSignal], model_topology: Optional[Dict[str, Any]] = None) -> ModelGraphState`

Analyze current model state from health signals.

**Parameters:**
- `health_signals`: Dictionary mapping layer names to HealthSignal objects
- `model_topology`: Optional topology information

**Returns:**
- `ModelGraphState` object containing graph representation

**Migration Note:** Will accept telemetry via message bus in v2.

### Class: `esper.services.tamiyo.client.TamiyoClient`

Client interface for Tamiyo service.

#### Constructor

```python
TamiyoClient(
    base_url: str = "http://localhost:8100",
    timeout: float = 30.0,
    max_retries: int = 3
)
```

#### Public Methods

##### `analyze_model_state(health_signals: List[HealthSignal]) -> AnalysisResponse`

Submit health signals for analysis.

**Parameters:**
- `health_signals`: List of health signals from layers

**Returns:**
```python
class AnalysisResponse:
    decisions: List[AdaptationDecision]
    metadata: Dict[str, Any]
```

**Migration Note:** Will use async message bus in v2.

##### `submit_adaptation_feedback(decision_id: str, performance_impact: Dict[str, float]) -> None`

Submit feedback on adaptation effectiveness.

**Parameters:**
- `decision_id`: ID of the adaptation decision
- `performance_impact`: Performance metrics post-adaptation

**Migration Note:** Will send to Karn via field reports in v2.

### Data Classes

#### `HealthSignal`

```python
@dataclass
class HealthSignal:
    layer_id: str
    health_score: float
    gradient_norm: float
    gradient_variance: float
    gradient_sign_stability: float
    param_norm_ratio: float
    activation_sparsity: float
    dead_neuron_ratio: float
    error_count: int
    execution_latency: float
    cache_hit_rate: float
    total_executions: int
```

**Migration Note:** Will be batched in `LayerHealthReport` in v2.

#### `AdaptationDecision`

```python
@dataclass  
class AdaptationDecision:
    adaptation_type: str  # "add_seed", "remove_seed", etc.
    layer_name: str
    confidence: float
    urgency: float
    metadata: Dict[str, Any]
```

**Migration Note:** Will include blueprint_id and grafting_strategy in v2.

## Deprecation Timeline

### Phase 1 (Months 2-3.5)
- Current APIs remain fully functional
- New chunked APIs introduced alongside
- Deprecation warnings added to documentation

### Phase 2 (Months 3.5-5.5)
- Legacy lifecycle states mapped to new states
- Compatibility layer for 5→11 state transitions
- Runtime warnings for deprecated methods

### Phase 3 (Months 5.5-8)
- Performance overhead for legacy paths
- Strong encouragement to migrate

### Phase 4 (Months 8-9.5)
- Legacy APIs moved to `legacy` namespace
- Required opt-in for legacy behavior

### Phase 5 (Months 9.5-11.5)
- Final deprecation warnings
- Legacy removal in v3.0

## Migration Guide

### For KasminaLayer Users

#### Current Usage
```python
layer = KasminaLayer(768, 768, num_seeds=4)
output = layer(input_tensor)
stats = layer.get_layer_stats()
```

#### Future Usage (v2)
```python
# Automatic migration via compatibility layer
layer = KasminaLayer(768, 768, num_seeds=4)  # Works but suboptimal

# Recommended migration
from esper.morphogenetic_v2 import ChunkedKasminaLayer
layer = ChunkedKasminaLayer(768, 768, chunks_per_layer=1000)
output = layer(input_tensor)  # 10x faster
```

### For Tamiyo Users

#### Current Usage
```python
client = TamiyoClient()
response = client.analyze_model_state(health_signals)
for decision in response.decisions:
    apply_adaptation(decision)
```

#### Future Usage (v2)
```python
# Via message bus (recommended)
await oona.publish("kasmina.health", health_report)
# Tamiyo responds via command topics

# Direct API still available but deprecated
analyzer = TamiyoAnalyzer()  # Local analysis
decisions = analyzer.analyze(health_report)
```

## Breaking Changes

### Version 2.0
1. **Seed Management**: Single seed → thousands of chunks
2. **Lifecycle States**: 5 states → 11 states
3. **Synchronous → Asynchronous**: Most APIs become async
4. **Direct calls → Message bus**: Decoupled architecture

### Version 3.0
1. **Legacy API Removal**: All v1 APIs removed
2. **Mandatory chunking**: No single-seed mode
3. **GPU-only**: CPU fallback removed

## Support Resources

- Migration guide: `docs/migration/v1_to_v2.md`
- Example code: `examples/morphogenetic_migration/`
- Support channel: #morphogenetic-migration
- Office hours: Thursdays 2-4 PM

---

*Generated: 2024-01-24*
*Version: 1.0*
*Status: Current Production API*
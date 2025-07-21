# Contracts Module (`src/esper/contracts/`)

## Overview

The contracts module defines the foundational data models and interfaces that govern all inter-service communication in the Esper platform. Following the principle "Contracts are Law," these definitions serve as the authoritative source of truth for data structures, state machines, and message formats across the entire system.

## Files

### `__init__.py` - Contracts Initialization

**Purpose:** Module initialization with API versioning.

**Contents:**
```python
API_VERSION = "v1"
```

**Features:**
- Establishes contract versioning for backward compatibility
- Provides central version reference for all services

### `assets.py` - ✅ **ENHANCED IN PHASE 1** - Core Asset Models

**Purpose:** Defines the primary data entities with enhanced kernel metadata support.

#### ✅ **NEW: Kernel Metadata Models**

**`KernelMetadata`** - Compiled Kernel Metadata
```python
class KernelMetadata(BaseModel):
    """Metadata for compiled kernel artifacts."""
    
    kernel_id: str
    blueprint_id: str
    name: str = Field(min_length=1, max_length=255)
    input_shape: List[int]  # Expected input tensor shape (excluding batch)
    output_shape: List[int]  # Expected output tensor shape (excluding batch)
    parameter_count: int = Field(ge=0)
    device_requirements: List[str] = Field(default_factory=list)  # ["cuda", "cpu"]
    memory_footprint_mb: float = Field(ge=0.0)
    compilation_target: str = Field(default="torchscript")  # "torchscript" or "pickle"
    optimization_flags: Dict[str, Any] = Field(default_factory=dict)
    performance_profile: Dict[str, float] = Field(default_factory=dict)
    compatibility_version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checksum: Optional[str] = None  # SHA256 of kernel binary
    
    def is_compatible_with_shape(self, input_shape: List[int]) -> bool:
        """Check if kernel is compatible with given input shape."""
        if len(input_shape) != len(self.input_shape):
            return False
        
        # Allow flexible batch dimension (index 0), check others
        for i in range(1, len(input_shape)):
            if input_shape[i] != self.input_shape[i]:
                return False
        
        return True
    
    def get_memory_estimate(self, batch_size: int) -> float:
        """Estimate memory usage in MB for given batch size."""
        base_memory = self.memory_footprint_mb
        # Rough estimate: linear scaling with batch size
        return base_memory * (batch_size / 32.0)  # Assume baseline is batch=32
```

**`CompiledKernel`** - Complete Kernel Artifact
```python
class CompiledKernel(BaseModel):
    """A compiled kernel artifact ready for execution."""
    
    kernel_id: str = Field(default_factory=lambda: str(uuid4()))
    blueprint_id: str
    binary_ref: str  # S3 reference to kernel binary
    metadata: KernelMetadata
    status: str = Field(
        default="compiled",
        pattern=r"^(compiled|validated|deployed|deprecated)$"
    )
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    deployment_stats: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: Optional[datetime] = None
    
    def is_ready_for_deployment(self) -> bool:
        """Check if kernel is ready for deployment."""
        return self.status in {"validated", "deployed"}
    
    def update_usage_stats(self):
        """Update usage statistics."""
        self.last_used_at = datetime.now(UTC)
```

**Key Features:**
- **Shape Validation:** Built-in compatibility checking for tensor shapes
- **Memory Estimation:** Intelligent memory usage prediction
- **Checksum Verification:** SHA256 validation for kernel integrity
- **Performance Profiling:** Metadata for execution performance tracking
- **Device Requirements:** GPU/CPU compatibility specification

### `enums.py` - System-Wide Enumerations

**Purpose:** Defines controlled vocabulary for state machines and system components.

#### Key Enumerations

**`SeedState`** - Seed Lifecycle States
```python
class SeedState(str, Enum):
    DORMANT = "dormant"        # Initial monitoring state
    GERMINATED = "germinated"  # Activated for learning
    TRAINING = "training"      # Actively learning
    GRAFTING = "grafting"      # Being incorporated into model
    FOSSILIZED = "fossilized"  # Permanently integrated
    CULLED = "culled"          # Removed from system
```
- **Purpose:** 6-state lifecycle for morphogenetic seeds
- **Pattern:** String-based enum for JSON serialization
- **Usage:** State machine validation in KasminaLayer

**`BlueprintState`** - Blueprint Lifecycle
```python
class BlueprintState(str, Enum):
    PROPOSED = "proposed"          # Initial submission
    COMPILING = "compiling"        # Being compiled by Tezzeret
    VALIDATING = "validating"      # Under validation
    CHARACTERIZED = "characterized" # Performance characterized
    DEPLOYED = "deployed"          # Active in system
    FAILED = "failed"              # Compilation/validation failed
    ARCHIVED = "archived"          # Stored for future use
```
- **Purpose:** 7-state compilation and deployment pipeline
- **Integration:** Used by Urza, Tezzeret, and Urabrask services

**`SystemHealth`** - Overall Health Status
```python
class SystemHealth(str, Enum):
    HEALTHY = "healthy"      # Normal operation
    DEGRADED = "degraded"    # Some issues present
    CRITICAL = "critical"    # Severe problems
    OFFLINE = "offline"      # System unavailable
```
- **Purpose:** System-wide health monitoring
- **Usage:** Tamiyo strategic decision making

**`ComponentType`** - Service Component Registry
```python
class ComponentType(str, Enum):
    TAMIYO = "tamiyo"        # Strategic Controller
    KARN = "karn"            # Generative Architect
    KASMINA = "kasmina"      # Execution Layer
    TEZZERET = "tezzeret"    # Compilation Forge
    URABRASK = "urabrask"    # Evaluation Engine
    TOLARIA = "tolaria"      # Training Orchestrator
    URZA = "urza"            # Central Library
    OONA = "oona"            # Message Bus
    NISSA = "nissa"          # Observability
    SIMIC = "simic"          # Policy Training Environment
    EMRAKUL = "emrakul"      # Architectural Sculptor
```
- **Purpose:** Service discovery and identification
- **Features:** 11 specialized subsystems across 3 functional planes

**`BlueprintStatus` & `KernelStatus`** - Artifact States
```python
class BlueprintStatus(str, Enum):
    UNVALIDATED = "unvalidated"
    COMPILING = "compiling"
    VALIDATED = "validated"
    INVALID = "invalid"

class KernelStatus(str, Enum):
    VALIDATED = "validated"
    INVALID = "invalid"
    TESTING = "testing"
    DEPLOYED = "deployed"
```
- **Purpose:** Compilation pipeline state tracking
- **Integration:** Urza database models, Tezzeret worker

### `assets.py` - Core Business Entities

**Purpose:** Primary data models for system entities with comprehensive validation and optimization.

#### Key Classes

**`Seed`** - Morphogenetic Seed Entity
```python
class Seed(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Performance optimization
    )
    
    seed_id: str = Field(default_factory=lambda: str(uuid4()))
    layer_id: int = Field(ge=0)  # Non-negative layer ID
    position: int = Field(ge=0)  # Non-negative position
    state: SeedState = SeedState.DORMANT
    blueprint_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = Field(default_factory=_empty_dict)
```

**Key Methods:**
```python
def state_display(self) -> str:
    """Display-friendly state name."""
    return self.state.title()

def is_active(self) -> bool:
    """Check if seed is in an active training state."""
    return self.state in {SeedState.TRAINING, SeedState.GRAFTING}
```

**Features:**
- UUID-based identification
- Audit trail with timestamps
- Performance-optimized Pydantic configuration
- State machine helpers

**`Blueprint`** - Architectural Design Specification
```python
class Blueprint(BaseModel):
    blueprint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=1000)
    state: BlueprintState = BlueprintState.PROPOSED
    architecture: Dict[str, Any]  # Architecture definition
    hyperparameters: Dict[str, Any] = Field(default_factory=_empty_dict)
    performance_metrics: Dict[str, float] = Field(default_factory=_empty_float_dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = Field(min_length=1, max_length=100)
```

**Key Methods:**
```python
def is_ready_for_deployment(self) -> bool:
    """Check if blueprint is ready for deployment."""
    return self.state == BlueprintState.CHARACTERIZED

def get_performance_summary(self) -> str:
    """Get a summary of performance metrics."""
    if not self.performance_metrics:
        return "No metrics available"
    
    metrics: Dict[str, float] = self.performance_metrics
    metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
    return f"Performance: {metrics_str}"
```

**Features:**
- Bounded string validation
- Performance metrics tracking
- State-based deployment logic
- Creator attribution

**`TrainingSession`** - Complete Training Orchestration
```python
class TrainingSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=1000)
    training_model_config: Dict[str, Any]  # Renamed from model_config
    training_config: Dict[str, Any]
    seeds: List[Seed] = Field(default_factory=list)
    blueprints: List[Blueprint] = Field(default_factory=list)
    status: str = Field(
        default="initialized",
        pattern=r"^(initialized|running|paused|completed|failed)$"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**Key Methods:**
```python
def get_active_seed_count(self) -> int:
    """Get count of active seeds."""
    return sum(1 for seed in self.seeds if seed.is_active())

def get_session_summary(self) -> str:
    """Get a summary of the training session."""
    total_seeds = len(self.seeds)
    active_seeds = self.get_active_seed_count()
    total_blueprints = len(self.blueprints)
    
    return (f"Session '{self.name}': {total_seeds} seeds "
            f"({active_seeds} active), {total_blueprints} blueprints, "
            f"status: {self.status}")
```

**Features:**
- Regex pattern validation for status
- Aggregation methods for monitoring
- Complete training context tracking

#### Helper Functions

```python
def _empty_dict() -> Dict[str, Any]:
    """Factory function for empty dictionaries to help pylint understand types."""
    return {}

def _empty_float_dict() -> Dict[str, float]:
    """Factory function for empty float dictionaries to help pylint understand types."""
    return {}
```
- **Purpose:** Type-safe default factory functions
- **Benefit:** Avoids mutable default arguments while maintaining type clarity

### `operational.py` - Runtime Monitoring and Control

**Purpose:** High-frequency telemetry data models for runtime system monitoring and strategic control.

#### Key Classes

**`HealthSignal`** - High-Frequency Health Telemetry
```python
class HealthSignal(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Performance optimization
    )
    
    layer_id: int = Field(ge=0)  # Non-negative layer ID
    seed_id: int = Field(ge=0)   # Non-negative seed ID
    chunk_id: int = Field(ge=0)  # Non-negative chunk ID
    epoch: int = Field(ge=0)     # Non-negative epoch
    activation_variance: float = Field(ge=0.0)  # Non-negative variance
    dead_neuron_ratio: float = Field(ge=0.0, le=1.0)  # Ratio 0-1
    avg_correlation: float = Field(ge=-1.0, le=1.0)   # Correlation -1 to 1
    is_ready_for_transition: bool = False  # State machine sync
    
    # Additional Tamiyo analysis fields
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_latency: float = Field(default=0.0, ge=0.0)
    error_count: int = Field(default=0, ge=0)
    active_seeds: int = Field(default=1, ge=0)
    total_seeds: int = Field(default=1, ge=1)
    timestamp: float = Field(default_factory=time.time)
```

**Validation Methods:**
```python
@field_validator("seed_id", "layer_id", "chunk_id", "epoch")
@classmethod
def _validate_ids(cls, v):
    """Validate that IDs are non-negative integers."""
    if v < 0:
        raise ValueError("IDs must be non-negative")
    return v
```

**Analysis Methods:**
```python
def health_status(self) -> str:
    """Get health status as human-readable string."""
    if self.health_score >= 0.8:
        return "Healthy"
    elif self.health_score >= 0.6:
        return "Warning"
    else:
        return "Critical"

def is_healthy(self) -> bool:
    """Check if seed is in healthy state."""
    return self.health_score >= 0.7 and self.error_count < 5
```

**Features:**
- Comprehensive numeric validation with bounds
- Performance monitoring fields
- State machine synchronization support
- High-frequency telemetry optimization

**`ModelGraphState`** - GNN Analysis Context
```python
@dataclass
class ModelGraphState:
    """Represents the current state of the model graph for Tamiyo analysis."""
    topology: Any  # GraphTopology from analyzer
    health_signals: Dict[str, HealthSignal]
    health_trends: Dict[str, float]
    problematic_layers: Set[str]
    overall_health: float
    analysis_timestamp: float
```
- **Purpose:** Aggregated state for strategic decision making
- **Usage:** Input to Tamiyo policy network
- **Features:** Graph topology, trend analysis, problem identification

**`SystemStatePacket`** - Global System Monitoring
```python
class SystemStatePacket(BaseModel):
    epoch: int = Field(ge=0)  # Non-negative epoch
    total_seeds: int = Field(ge=0)  # Non-negative seed count
    active_seeds: int = Field(ge=0)  # Non-negative active seed count
    training_loss: float = Field(ge=0.0)  # Non-negative loss
    validation_loss: float = Field(ge=0.0)  # Non-negative validation loss
    system_load: float = Field(ge=0.0, le=1.0)  # Load percentage 0-100%
    memory_usage: float = Field(ge=0.0, le=1.0)  # Memory usage percentage 0-100%
```

**Analysis Methods:**
```python
def system_health(self) -> str:
    """Get overall system health status."""
    if self.system_load > 0.9 or self.memory_usage > 0.9:
        return "Critical"
    elif self.system_load > 0.7 or self.memory_usage > 0.7:
        return "Warning"
    else:
        return "Normal"

def is_overloaded(self) -> bool:
    """Check if system is overloaded."""
    return self.system_load > 0.8 or self.memory_usage > 0.8
```

**`AdaptationDecision`** - Strategic Controller Output
```python
class AdaptationDecision(BaseModel):
    layer_name: str = Field(min_length=1, max_length=255)
    adaptation_type: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^(add_seed|remove_seed|modify_architecture|optimize_parameters)$"
    )
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence level 0-1
    urgency: float = Field(ge=0.0, le=1.0)     # Urgency level 0-1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
```

**Decision Logic:**
```python
def decision_priority(self) -> str:
    """Get decision priority based on confidence and urgency."""
    priority_score = (self.confidence + self.urgency) / 2
    if priority_score >= 0.8:
        return "High"
    elif priority_score >= 0.5:
        return "Medium"
    else:
        return "Low"

def should_execute_immediately(self) -> bool:
    """Check if decision should be executed immediately."""
    return self.urgency > 0.7 and self.confidence > 0.6
```

**Features:**
- Controlled adaptation types via regex validation
- Confidence and urgency scoring
- Priority calculation algorithms
- Immediate execution logic

### `messages.py` - Message Bus Communication

**Purpose:** Oona message bus protocol definitions with topic management and event tracing.

#### Key Components

**`TopicNames`** - Centralized Topic Registry
```python
class TopicNames(str, Enum):
    TELEMETRY_SEED_HEALTH = "telemetry.seed.health"
    CONTROL_KASMINA_COMMANDS = "control.kasmina.commands"
    COMPILATION_BLUEPRINT_SUBMITTED = "compilation.blueprint.submitted"
    COMPILATION_KERNEL_READY = "compilation.kernel.ready"
    VALIDATION_KERNEL_CHARACTERIZED = "validation.kernel.characterized"
    SYSTEM_EVENTS_EPOCH = "system.events.epoch"
    INNOVATION_FIELD_REPORTS = "innovation.field_reports"
```
- **Purpose:** Centralized topic definitions prevent typos and ensure consistency
- **Pattern:** Hierarchical naming convention (service.category.event)
- **Usage:** All Oona message publishing and subscription

**`OonaMessage`** - Universal Message Envelope
```python
class OonaMessage(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str  # e.g., 'Tezzeret-Worker-5', 'Tamiyo-Controller'
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str  # To trace a request across multiple services
    topic: TopicNames
    payload: Dict[str, Any]
```

**Features:**
- UUID-based event identification
- Distributed tracing support via trace_id
- Automatic timestamping
- Type-safe topic enforcement
- Flexible payload structure

**`BlueprintSubmitted`** - Event Payload Example
```python
class BlueprintSubmitted(BaseModel):
    blueprint_id: str
    submitted_by: str
```
- **Purpose:** Demonstrates specific event payload structure
- **Pattern:** Type-safe payload definitions for critical events

### `validators.py` - Custom Validation Functions

**Purpose:** High-performance validation functions for complex data types used across contracts.

#### Key Validators

**`validate_seed_id`** - Seed ID Tuple Validation
```python
def validate_seed_id(v: Tuple[int, int]) -> Tuple[int, int]:
    """
    Validates a seed_id tuple ensuring proper format and values.
    
    A seed_id is a unique identifier tuple consisting of (layer_id, seed_idx)
    used to reference specific neural network seeds within layers.
    
    Args:
        v: Input value to validate as a seed ID tuple
    
    Returns:
        Tuple[int, int]: The validated seed ID tuple (layer_id, seed_idx)
    
    Raises:
        ValueError: If input is not a 2-element tuple
        TypeError: If tuple elements are not integers
        ValueError: If tuple elements are negative
    """
    # Fast path: check if it's a tuple with exactly 2 elements
    if not isinstance(v, tuple) or len(v) != 2:
        raise ValueError("Seed ID must be a tuple of (layer_id, seed_idx)")
    
    layer_id, seed_idx = v
    
    # Type validation with early exit
    if not isinstance(layer_id, int) or not isinstance(seed_idx, int):
        raise TypeError("Seed ID elements must be integers")
    
    # Value validation
    if layer_id < 0 or seed_idx < 0:
        raise ValueError("Seed ID elements must be non-negative")
    
    return v
```

**Features:**
- Fast path optimization with early exit
- Comprehensive error messages
- Type and value validation
- Performance-oriented implementation

**Usage Pattern:**
```python
from pydantic import BaseModel, field_validator
from esper.contracts.validators import validate_seed_id

class SeedReference(BaseModel):
    seed_id: Tuple[int, int]
    
    @field_validator('seed_id')
    @classmethod
    def validate_seed_id_field(cls, v):
        return validate_seed_id(v)
```

## Architecture Integration

The contracts module serves as the foundation for:

1. **Inter-Service Communication** - All services use these models for data exchange
2. **State Machine Validation** - Enums enforce valid state transitions
3. **Message Bus Protocol** - Standardized message formats across Oona
4. **Database Models** - SQLAlchemy models inherit validation from contracts
5. **API Schemas** - FastAPI automatically generates OpenAPI specs from contracts

## Dependencies

**External:**
- `pydantic` - Data validation and serialization
- `enum` - Enumeration support
- `datetime` - Timestamp handling
- `uuid` - Unique identifier generation
- `typing` - Type annotations

**Internal:**
- No internal dependencies (foundational module)

## Performance Considerations

### Pydantic Optimization
```python
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    use_enum_values=True,
    validate_assignment=True,
    str_to_lower=False,
    str_strip_whitespace=True,
    extra="forbid"  # Prevents extra fields for better performance
)
```

### Validation Strategy
- **Bounded Validation:** All numeric fields have appropriate ge/le constraints
- **String Length Limits:** Prevent unbounded memory usage
- **Early Exit Patterns:** Fast validation with minimal overhead
- **Type Caching:** Pydantic model compilation for repeated validation

## Best Practices

### Contract Design
1. **Immutable Contracts** - Once published, contracts should not break backward compatibility
2. **Comprehensive Validation** - All fields should have appropriate constraints
3. **Clear Documentation** - Each field should have purpose and usage documentation
4. **Performance Optimization** - Use appropriate Pydantic configuration for high-frequency models

### State Machine Design
1. **Explicit States** - All possible states should be enumerated
2. **Valid Transitions** - Document allowed state transitions
3. **Error States** - Include states for error handling and recovery
4. **Audit Trail** - Track state changes with timestamps

### Message Protocol
1. **Topic Hierarchy** - Use consistent naming conventions
2. **Trace Support** - Include trace IDs for distributed debugging
3. **Payload Typing** - Define specific payload models for critical events
4. **Backward Compatibility** - Support message versioning

## Common Usage Patterns

### Health Signal Processing
```python
from esper.contracts.operational import HealthSignal

# Create health signal
health_signal = HealthSignal(
    layer_id=1,
    seed_id=0,
    chunk_id=0,
    epoch=100,
    activation_variance=0.05,
    dead_neuron_ratio=0.02,
    avg_correlation=0.85,
    health_score=0.9
)

# Check health status
if health_signal.is_healthy():
    print(f"Layer {health_signal.layer_id} is healthy")
else:
    print(f"Layer {health_signal.layer_id} needs attention: {health_signal.health_status()}")
```

### Blueprint Lifecycle Management
```python
from esper.contracts.assets import Blueprint, BlueprintState

# Create new blueprint
blueprint = Blueprint(
    name="ResNet Attention Module",
    description="Attention mechanism for ResNet architectures",
    architecture={"type": "attention", "heads": 8},
    created_by="Karn-Architect-1"
)

# Check deployment readiness
if blueprint.is_ready_for_deployment():
    print("Blueprint ready for deployment")
    print(blueprint.get_performance_summary())
```

### Message Bus Communication
```python
from esper.contracts.messages import OonaMessage, TopicNames

# Create message
message = OonaMessage(
    sender_id="Tamiyo-Controller-1",
    trace_id="trace-12345",
    topic=TopicNames.CONTROL_KASMINA_COMMANDS,
    payload={
        "command": "load_kernel",
        "layer_name": "layer1",
        "seed_idx": 0,
        "artifact_id": "kernel-abc123"
    }
)

# Publish via Oona client
await oona_client.publish(message)
```

## Error Handling

### Validation Errors
- **Field Validation:** Clear error messages for out-of-range values
- **Type Errors:** Explicit type checking with helpful suggestions
- **Constraint Violations:** Detailed information about validation failures

### State Machine Errors
- **Invalid Transitions:** Prevention of illegal state changes
- **State Consistency:** Validation of state-dependent fields
- **Recovery Patterns:** Well-defined error and recovery states

## Future Enhancements

1. **Schema Evolution** - Support for contract versioning and migration
2. **Performance Profiling** - Detailed performance metrics for high-frequency models
3. **Advanced Validation** - Cross-field validation and business rule enforcement
4. **Code Generation** - Automatic client library generation from contracts
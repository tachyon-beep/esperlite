# LLM_CODEBASE_GUIDE.md

# Esper Morphogenetic Training Platform - LLM Codebase Guide

**Optimized for AI Agent Understanding and Development**

## Executive Summary

The Esper Morphogenetic Training Platform is a revolutionary neural network training system that enables **autonomous architectural evolution** during training. Unlike traditional static neural networks, Esper allows models to modify their own structure dynamically by integrating specialized sub-networks called "kernels" based on real-time performance analysis.

### Core Innovation
- **Morphogenetic Architecture**: Neural networks that evolve their structure during training
- **Zero Training Disruption**: Asynchronous compilation ensures training never stops
- **Strategic Intelligence**: GNN-based policy networks decide when/where to adapt
- **Production-Ready**: Distributed system with comprehensive observability and safety

### Three-Plane Architecture
1. **Training Plane**: Tolaria (orchestrator) + Kasmina (execution engine)
2. **Control Plane**: Tamiyo (strategic controller) + Urza (artifact storage) 
3. **Innovation Plane**: Tezzeret (compilation forge) + Oona (message bus)

---

## Module-by-Module Breakdown

### 1. Contracts Module (`src/esper/contracts/`)

**Purpose**: Foundational data models and interfaces that govern all inter-service communication. "Contracts are Law" - these definitions are the authoritative source of truth.

#### Core Components

**`enums.py`** - System-wide controlled vocabulary
```python
class SeedState(str, Enum):
    DORMANT = "dormant"      # Initial state
    GERMINATED = "germinated" # Activated for learning  
    TRAINING = "training"    # Actively learning
    GRAFTING = "grafting"    # Being incorporated
    FOSSILIZED = "fossilized" # Preserved/archived
    CULLED = "culled"        # Removed from system
```

**`assets.py`** - Core business entities
```python
class Seed(BaseModel):
    seed_id: str = Field(default_factory=lambda: str(uuid4()))
    layer_id: int
    position: int
    state: SeedState = SeedState.DORMANT
    blueprint_id: Optional[str] = None
    # ... timestamps, metadata

class Blueprint(BaseModel):
    blueprint_id: str = Field(default_factory=lambda: str(uuid4()))
    architecture: Dict[str, Any]    # Architecture definition
    state: BlueprintState = BlueprintState.PROPOSED
    # ... hyperparameters, performance metrics
```

**`operational.py`** - Runtime monitoring and health
```python
class HealthSignal(BaseModel):
    layer_id: int
    seed_id: int
    activation_variance: float
    dead_neuron_ratio: float = Field(..., ge=0.0, le=1.0)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_latency: float = Field(default=0.0, ge=0.0)
```

**`messages.py`** - Message bus communication contracts
```python
class OonaMessage(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    topic: TopicNames
    payload: Dict[str, Any]
    trace_id: str  # For distributed tracing
```

#### Key Patterns
- **UUID-based identification** for all entities
- **Bounded numeric validation** using Pydantic Field constraints
- **State machine enforcement** through enum constraints
- **Audit trails** via comprehensive timestamping

---

### 2. Core Module (`src/esper/core/`)

**Purpose**: User-facing API that transforms standard PyTorch models into morphogenetic models. Primary abstraction layer between user code and Esper platform.

#### Architecture

**`model_wrapper.py`** - Main public API
```python
def wrap(model: nn.Module, 
         target_layers: Optional[List[Type[nn.Module]]] = None,
         seeds_per_layer: int = 4,
         cache_size_mb: int = 128) -> MorphableModel:
    """Wraps PyTorch model to enable morphogenetic capabilities"""

class MorphableModel(nn.Module):
    def __init__(self, wrapped_model, kasmina_layers, original_model=None):
        self.wrapped_model = wrapped_model      # Modified model with KasminaLayers
        self.kasmina_layers = nn.ModuleDict()  # References to injected layers
        self.original_model = original_model   # Optional backup for comparison
```

#### Integration Points
- **Layer Replacement**: Targets `nn.Linear` by default, extensible to other layer types
- **Weight Preservation**: Original weights copied to `default_transform` property
- **State Management**: Multi-level tracking (model/layer/seed levels)
- **Kernel Management**: Async kernel loading/unloading with validation

#### Critical Methods
```python
async def load_kernel(self, layer_name: str, seed_idx: int, artifact_id: str) -> bool
async def unload_kernel(self, layer_name: str, seed_idx: int) -> bool
def get_model_stats(self) -> Dict[str, Any]
def compare_with_original(self, x: torch.Tensor) -> Dict[str, Any]
```

#### Design Patterns
- **Wrapper Pattern**: Non-invasive enhancement of existing models
- **Composition**: KasminaLayers injected into model hierarchy
- **Strategy Pattern**: Configurable layer targeting and seed management

---

### 3. Execution Module (`src/esper/execution/`)

**Purpose**: Performance-critical morphogenetic execution engine. Handles real-time kernel loading, seed lifecycle management, and GPU-optimized state tracking.

#### Components

**`kasmina_layer.py`** - Main execution engine
```python
class KasminaLayer(nn.Module):
    def __init__(self, original_layer, num_seeds=4, cache_size_mb=128):
        self.default_transform = original_layer  # Fallback transformation
        self.state_layout = KasminaStateLayout(num_seeds)
        self.kernel_cache = KernelCache(cache_size_mb)
        self.oona_client = OonaClient()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path for dormant seeds
        if not self.state_layout.has_active_seeds():
            return self.default_transform(x)
        
        # Hybrid execution with kernel blending
        return self._execute_with_kernels(x)
```

**`state_layout.py`** - GPU-optimized state management
```python
@dataclass
class KasminaStateLayout:
    # Structure-of-Arrays for memory coalescing
    lifecycle_states: torch.Tensor      # uint8 - SeedLifecycleState
    active_kernel_id: torch.Tensor      # int64 - Hash of loaded kernel
    alpha_blend: torch.Tensor           # float16 - Blending coefficients
    health_accumulator: torch.Tensor    # float32 - EMA health scores
    exec_latency_us: torch.Tensor       # uint16 - Per-seed latency
```

**`kernel_cache.py`** - High-performance caching
```python
class KernelCache:
    _cache: OrderedDict[str, torch.Tensor]       # LRU-ordered kernels
    _cache_info: OrderedDict[str, Dict[str, Any]] # Metadata
    _lock: asyncio.Lock                           # Thread safety
    
    async def load_kernel(self, artifact_id: str) -> Optional[torch.Tensor]:
        # LRU cache with Urza API integration
        # Automatic GPU migration and size management
```

#### Performance Optimizations
- **Fast Path Detection**: O(1) CPU check avoids GPU operations when dormant
- **Memory Coalescing**: Structure-of-Arrays layout for optimal GPU bandwidth
- **LRU Caching**: Microsecond-latency kernel access with automatic eviction
- **Alpha Blending**: Weighted combination of default and kernel outputs

#### Error Handling
- **Circuit Breaker**: 3-strike rule for problematic seeds
- **Graceful Fallback**: Automatic default transformation on kernel failures
- **Health Monitoring**: Real-time telemetry with async publishing

---

### 4. Services Module (`src/esper/services/`)

**Purpose**: Distributed system components implementing the three-plane architecture. Each service is independently deployable with well-defined APIs.

#### Oona Client (`oona_client.py`) - Message Bus
```python
class OonaClient:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.Redis.from_url(redis_url)
    
    async def publish(self, message: OonaMessage) -> None:
        # Publishes to Redis Streams with topic routing
    
    async def consume(self, streams: List[str], consumer_group: str) -> List[OonaMessage]:
        # Consumer group-based message consumption
```

#### Tamiyo Service (`services/tamiyo/`) - Strategic Controller

**Core Architecture**: Graph Neural Network-based policy system
```python
class TamiyoPolicyGNN(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        # 3-layer Graph Convolutional Network
        self.node_encoder = nn.Sequential(...)  # 64 → 128 dim encoding
        self.gcn_layers = nn.ModuleList([...])  # GCN with residual connections
        self.decision_head = nn.Linear(hidden_dim, 3)  # adaptation, layer_priority, urgency
        self.value_head = nn.Linear(hidden_dim, 1)     # RL value estimation

class ModelGraphAnalyzer:
    def analyze_health_signals(self, signals: List[HealthSignal]) -> ModelGraphState:
        # Converts health signals to graph representation
        # Identifies problematic layers and trends
```

**Decision Pipeline**:
1. Collect health signals from KasminaLayers via Oona
2. Construct graph representation of model topology
3. Run GNN inference for adaptation decisions
4. Apply confidence thresholds and cooldown logic
5. Publish adaptation decisions back to execution layer

#### Tolaria Service (`services/tolaria/`) - Training Orchestrator

**Primary Workflow**:
```python
class TolariaTrainer:
    async def run_training(self) -> None:
        # 1. Initialize morphogenetic model wrapping
        model = self._create_model()
        morphable_model = wrap(model, target_layers=self.config.model.target_layers)
        
        # 2. Training loop with adaptation integration
        for epoch in range(self.config.max_epochs):
            train_loss = await self._train_epoch(morphable_model)
            val_loss = await self._validate_epoch(morphable_model)
            
            # 3. Consult Tamiyo for adaptation decisions
            if epoch % self.config.adaptation_frequency == 0:
                await self._apply_adaptations(morphable_model)
```

#### Tezzeret Service (`services/tezzeret/`) - Compilation Forge

**Compilation Pipeline**:
```python
class TezzeretWorker:
    async def process_blueprints(self) -> None:
        while True:
            # 1. Poll Urza for unvalidated blueprints
            blueprints = await self._fetch_unvalidated_blueprints()
            
            for blueprint in blueprints:
                # 2. Convert IR to PyTorch module
                module = self._ir_to_module(blueprint.architecture)
                
                # 3. Compile with torch.compile
                compiled_kernel = torch.compile(module)
                
                # 4. Upload to S3 and update Urza
                artifact_id = await self._upload_artifact(compiled_kernel)
                await self._submit_kernel_to_urza(blueprint.id, artifact_id)
```

#### Urza Service (`services/urza/`) - Central Library

**Database Models**:
```python
class Blueprint(Base):
    __tablename__ = "blueprints"
    id = Column(String, primary_key=True)
    status = Column(String)  # UNVALIDATED/COMPILING/VALIDATED/INVALID
    architecture_ir = Column(Text)  # JSON IR
    
class CompiledKernel(Base):
    __tablename__ = "compiled_kernels"
    id = Column(String, primary_key=True)
    blueprint_id = Column(String, ForeignKey("blueprints.id"))
    kernel_binary_ref = Column(Text)  # S3 URI
    validation_report = Column(JSON)
```

**REST API Endpoints**:
- `POST /blueprints/` - Submit new blueprint
- `GET /kernels/{kernel_id}` - Retrieve compiled kernel
- `GET /internal/unvalidated-blueprints` - Worker polling endpoint

#### Service Integration Patterns
- **Message Bus Coordination**: Health signals → Tamiyo → Adaptation decisions
- **REST API Communication**: Heavy data transfer and artifact management
- **Event-Driven Architecture**: Pub/sub for real-time system coordination
- **Service Discovery**: Configuration-based endpoint resolution

---

### 5. Utils Module (`src/esper/utils/`)

**Purpose**: Shared utilities for logging, storage, and common operations across all services.

#### Components

**`logging.py`** - Centralized logging configuration
```python
def setup_logging(service_name: str, level: int = logging.INFO) -> logging.Logger:
    # Structured logging with service identification
    # Consistent format across all services
    # Duplicate handler prevention
```

**`s3_client.py`** - Object storage abstraction
```python
def get_s3_client():
    # MinIO/S3 configuration from environment
    # Boto3 client with proper authentication

def upload_bytes(client, data: bytes, bucket_name: str, object_key: str) -> str:
    # Direct bytes upload for compiled kernels
    
def download_bytes(client, bucket_name: str, object_key: str) -> bytes:
    # Direct bytes download for kernel loading
```

#### Integration Patterns
- **Environment-based Configuration**: Credentials via environment variables
- **Error Handling**: Comprehensive exception handling with logging
- **URI Parsing**: S3 URI parsing for kernel artifact references

---

### 6. Configuration System (`src/esper/configs.py`)

**Purpose**: Type-safe, hierarchical configuration system using Pydantic models.

#### Configuration Hierarchy
```python
class EsperConfig(BaseModel):
    name: str
    version: str = "0.1.0"
    environment: str = "development"
    
    database: DatabaseConfig      # PostgreSQL connection
    redis: RedisConfig           # Message bus and caching
    storage: StorageConfig       # S3-compatible object storage
    
    components: Dict[str, ComponentConfig]  # Service-specific configs
    training: Dict[str, Any]                # Training parameters
```

#### Configuration Files
- **`development.yaml`**: Minimal config for testing (5 epochs, 2 seeds, layer1 only)
- **`phase1_mvp.yaml`**: Production-ready config for morphogenetic training
- **Experiment configs**: CIFAR-10/100 specific configurations

#### Entry Points

**`train.py`** - Single command interface
```python
# Quick start with predefined configs
python train.py --quick-start cifar10 --output ./results

# Custom configuration
python train.py --config configs/my_experiment.yaml

# Environment validation and dry run
python train.py --config configs/test.yaml --dry-run
```

---

## Cross-Reference Dependency Matrix

### Module Dependencies (Hierarchical)

```
Level 1 (Foundation):
├── contracts/ (enums, assets, operational, messages, validators)
├── utils/ (logging, s3_client)
└── configs.py

Level 2 (Execution):
├── execution/ → contracts/, utils/, torch
│   ├── state_layout.py → contracts/enums
│   ├── kernel_cache.py → services/contracts, utils/s3_client
│   └── kasmina_layer.py → execution/{state_layout,kernel_cache}, contracts/{operational,messages}, services/oona_client

Level 3 (Core API):
└── core/ → execution/, torch
    └── model_wrapper.py → execution/kasmina_layer

Level 4 (Services):
├── services/oona_client.py → contracts/messages, redis
├── services/urza/ → contracts/, fastapi, sqlalchemy
├── services/tezzeret/ → contracts/, utils/s3_client, torch
├── services/tamiyo/ → contracts/, services/oona_client, torch_geometric
└── services/tolaria/ → core/, services/{oona_client,tamiyo}, contracts/

Level 5 (Application):
└── train.py → services/tolaria/, utils/logging
```

### External Dependencies by Category

**Core ML/AI**: `torch`, `torchvision`, `torch-geometric`, `numpy`, `networkx`
**Web/API**: `fastapi`, `uvicorn`, `requests`, `httpx`
**Data/Storage**: `pydantic`, `sqlalchemy`, `psycopg2-binary`, `redis`, `boto3`, `pyyaml`
**Development**: `black`, `ruff`, `pytype`, `pytest`, `pytest-cov`

---

## Key Architectural Decisions & Rationale

### 1. Contract-First Design
**Decision**: All inter-service communication governed by Pydantic data contracts
**Rationale**: Ensures type safety, API compatibility, and clear service boundaries in distributed system
**Impact**: Prevents runtime errors, enables independent service development, supports API versioning

### 2. Three-Plane Architecture
**Decision**: Separate Training, Control, and Innovation planes
**Rationale**: Clean separation of concerns enables independent scaling and development
**Impact**: Training can continue while innovation services compile new architectures

### 3. Wrapper-Based Integration  
**Decision**: Non-invasive wrapping of existing PyTorch models
**Rationale**: Preserves compatibility with existing codebases and training workflows
**Impact**: Easy adoption, minimal code changes required, gradual migration path

### 4. Message Bus Architecture
**Decision**: Redis Streams for inter-service communication
**Rationale**: Enables real-time coordination while maintaining loose coupling
**Impact**: Scalable pub/sub, consumer groups for fault tolerance, event sourcing capability

### 5. GPU-Optimized State Management
**Decision**: Structure-of-Arrays layout for seed state
**Rationale**: Optimal GPU memory bandwidth utilization for high-frequency operations
**Impact**: Microsecond-latency state updates, efficient memory coalescing

### 6. LRU Kernel Caching
**Decision**: In-memory cache with size-based eviction
**Rationale**: Balance between memory usage and kernel loading performance
**Impact**: Sub-millisecond kernel access, automatic memory management

---

## Common Patterns & Conventions

### Design Patterns

#### 1. Wrapper Pattern
- **Usage**: `MorphableModel` wraps `nn.Module`, `KasminaLayer` wraps individual layers
- **Benefit**: Non-invasive enhancement, maintains original API compatibility
- **Implementation**: Composition over inheritance, delegation to wrapped objects

#### 2. Observer Pattern  
- **Usage**: Health signal generation and consumption, event-driven adaptations
- **Benefit**: Loose coupling between health monitoring and strategic decisions
- **Implementation**: Message bus pub/sub, async event handlers

#### 3. Strategy Pattern
- **Usage**: Multiple adaptation strategies, configurable training policies
- **Benefit**: Runtime behavior modification, extensible decision logic
- **Implementation**: Policy networks, configuration-driven strategy selection

#### 4. State Machine Pattern
- **Usage**: Seed lifecycle management, blueprint status transitions
- **Benefit**: Clear state transitions, validation of allowed operations
- **Implementation**: Enum-based states, transition validation

#### 5. Factory Pattern
- **Usage**: Dynamic kernel instantiation, blueprint compilation
- **Benefit**: Runtime object creation, type-safe instantiation
- **Implementation**: IR-to-module conversion, kernel artifact loading

### Coding Conventions

#### Type Safety
```python
# Full type annotations required
def process_health_signal(signal: HealthSignal) -> AdaptationDecision:
    return AdaptationDecision(...)

# Avoid typing.Any in public APIs
def flexible_config(params: Dict[str, Union[int, float, str]]) -> None:
    # Specific union types preferred over Any
```

#### Import Style
```python
# Force single-line imports (ruff enforced)
import logging
import typing
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

# Firm imports only - no conditional try/except
from esper.contracts.operational import HealthSignal  # Not: try/except ImportError
```

#### Validation Patterns
```python
# Pydantic field validation
class HealthSignal(BaseModel):
    health_score: float = Field(..., ge=0.0, le=1.0)  # Bounded validation
    seed_id: int = Field(..., ge=0)                   # Non-negative IDs
    
    @field_validator("seed_id")
    @classmethod
    def validate_seed_id(cls, v):
        if v < 0:
            raise ValueError("Seed ID must be non-negative")
        return v
```

### Testing Strategies

#### Layered Testing
```python
# Unit tests mirror source structure
tests/core/test_model_wrapper.py     # Unit: individual functions
tests/execution/test_kasmina_layer.py # Unit: component isolation
tests/integration/test_phase1_pipeline.py # Integration: service interactions
tests/integration/test_phase4_full_system.py # E2E: complete workflows
```

#### Mock Patterns
```python
# Service boundary mocking
@pytest.fixture
def mock_oona_client():
    with patch('esper.services.oona_client.OonaClient') as mock:
        yield mock

# Async testing patterns
@pytest.mark.asyncio
async def test_kernel_loading():
    result = await kasmina_layer.load_kernel("layer1", 0, "artifact_123")
    assert result is True
```

#### Coverage Requirements
- **Target**: >90% line coverage for new code
- **Reporting**: HTML coverage reports in `htmlcov/`
- **CI Integration**: Coverage checks in automated pipeline
- **Exclusions**: `pragma: no cover` for defensive error handling

### Error Handling Approaches

#### Graceful Degradation
```python
# Continue operation with reduced functionality
def forward(self, x: torch.Tensor) -> torch.Tensor:
    try:
        if self.has_active_seeds():
            return self._execute_with_kernels(x)
    except Exception as e:
        logger.warning("Kernel execution failed, falling back to default: %s", e)
    
    return self.default_transform(x)
```

#### Comprehensive Logging
```python
# Structured logging with context
logger.info("Kernel loaded", extra={
    "layer_name": layer_name,
    "seed_idx": seed_idx,
    "artifact_id": artifact_id,
    "cache_hit": cache_hit,
    "load_time_ms": load_time_ms
})
```

#### Circuit Breaker Pattern
```python
# Progressive error handling
if error_count >= MAX_CONSECUTIVE_FAILURES:
    self.transition_seed_state(seed_idx, SeedLifecycleState.ERROR_RECOVERY)
    logger.warning("Seed %d entered error recovery after %d failures", seed_idx, error_count)
```

---

## Performance Characteristics

### Critical Performance Paths

#### 1. Forward Pass Execution
- **Hot Path**: Dormant seed check (O(1) CPU operation)
- **Warm Path**: Single active kernel execution (1-2ms overhead)
- **Cold Path**: Multiple kernel blending (2-5ms overhead)

#### 2. Kernel Loading
- **Cache Hit**: <1ms (in-memory LRU access)
- **Cache Miss**: 10-100ms (network + deserialization)
- **First Load**: 100-500ms (compilation + caching)

#### 3. Health Signal Processing
- **Generation**: <0.1ms per forward pass
- **Transport**: 1-5ms via Redis pub/sub
- **Analysis**: 10-50ms for GNN inference

### Scaling Characteristics
- **Seed Count**: Linear memory scaling, O(1) execution time per active seed
- **Layer Count**: Linear scaling in model size, independent layer processing
- **Service Count**: Horizontal scaling via consumer groups and load balancing

### Memory Usage Patterns
- **Kernel Cache**: Configurable size limit (default 128MB per layer)
- **State Layout**: ~100 bytes per seed (GPU-resident tensors)
- **Message Bus**: Bounded by Redis memory configuration

---

## Entry Points for Understanding Data Flow

### Primary User Journey
1. **Model Wrapping**: `esper.wrap(pytorch_model)` → `MorphableModel`
2. **Training Start**: `train.py --config experiment.yaml` → `TolariaService`
3. **Forward Pass**: `model(input)` → `KasminaLayer.forward()` → health signals
4. **Strategic Analysis**: Health signals → `TamiyoPolicyGNN` → adaptation decisions
5. **Architecture Evolution**: Decisions → `TezzeretWorker` → compiled kernels
6. **Runtime Adaptation**: Kernels → `KernelCache` → `KasminaLayer` execution

### Message Flow Patterns
```
HealthSignal: KasminaLayer → Oona → Tamiyo
AdaptationDecision: Tamiyo → Oona → KasminaLayer  
BlueprintRequest: Tamiyo → Urza API → Tezzeret polling
CompiledKernel: Tezzeret → S3 upload → Urza → KernelCache
```

### Configuration Flow
```
YAML Config → TolariaConfig → Service initialization
Environment vars → Service-specific configs → Runtime behavior
Quick-start templates → Generated configs → Training execution
```

This guide provides the foundational understanding needed for an LLM to effectively work with the Esper codebase. The architecture enables autonomous neural network evolution while maintaining production-ready reliability and extensive observability throughout the morphogenetic training process.
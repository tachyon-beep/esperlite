# Urza Service (`src/esper/services/urza/`)

## Overview

Urza is the compilation and kernel management service for the Esper platform. Named after the legendary artificer, it transforms BlueprintIR into optimized TorchScript kernels and manages their lifecycle. Urza serves as the bridge between abstract blueprint specifications and executable kernels, providing compilation, caching, optimization, and serving capabilities. The service achieves <100ms compilation times for standard kernels while maintaining compatibility with PyTorch's execution model.

## Architecture

```
Blueprint (Tezzeret) → BlueprintIR → Urza → TorchScript Kernel → KasminaLayer Execution
                                        ↓
                                  Kernel Cache (Storage)
```

## Core Components

### Service Configuration
```python
@dataclass
class UrzaConfig:
    """Configuration for Urza service."""
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Compilation settings
    compile_timeout_seconds: int = 300
    max_concurrent_compilations: int = 10
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive
    
    # Cache settings
    enable_kernel_cache: bool = True
    cache_ttl_hours: int = 168  # 1 week
    max_cache_size_gb: int = 100
    
    # Performance settings
    enable_profiling: bool = True
    enable_metrics: bool = True
    batch_compilation: bool = True
```

### Main Service (`service.py`)

**Purpose:** FastAPI-based REST service for kernel compilation and management.

#### Key Endpoints

**`POST /compile` - Compile Blueprint to Kernel**
```python
@app.post("/compile", response_model=CompilationResponse)
async def compile_blueprint(request: CompilationRequest) -> CompilationResponse:
    """
    Compile BlueprintIR to executable kernel.
    
    Request:
        {
            "blueprint_ir": {...},  # BlueprintIR structure
            "optimization_hints": {
                "target_hardware": "cuda",
                "expected_batch_size": 32,
                "precision": "fp16"
            },
            "cache_key": "optional-cache-key"
        }
    
    Response:
        {
            "kernel_id": "kernel-abc123",
            "compilation_time_ms": 85.3,
            "optimizations_applied": ["fusion", "quantization"],
            "cache_hit": false,
            "metadata": {...}
        }
    """
```

**`GET /kernel/{kernel_id}` - Retrieve Compiled Kernel**
```python
@app.get("/kernel/{kernel_id}")
async def get_kernel(kernel_id: str) -> KernelResponse:
    """
    Retrieve compiled kernel by ID.
    
    Returns serialized TorchScript kernel ready for loading.
    """
```

**`POST /optimize` - Optimize Existing Kernel**
```python
@app.post("/optimize/{kernel_id}")
async def optimize_kernel(
    kernel_id: str,
    optimization_request: OptimizationRequest
) -> OptimizationResponse:
    """
    Apply additional optimizations to existing kernel.
    
    Optimizations:
    - Operator fusion
    - Quantization
    - Memory layout optimization
    - Hardware-specific tuning
    """
```

**`GET /status` - Service Health Status**
```python
@app.get("/status")
async def get_status() -> StatusResponse:
    """
    Get service health and statistics.
    
    Response:
        {
            "status": "healthy",
            "compilations_total": 1532,
            "compilations_active": 3,
            "cache_hit_rate": 0.87,
            "avg_compilation_time_ms": 92.5,
            "kernel_cache_size_mb": 4567
        }
    """
```

### Kernel Manager (`kernel_manager.py`)

**Purpose:** Manages kernel lifecycle, caching, and optimization strategies.

#### Key Classes

**`KernelManager`** - Central Kernel Management
```python
class KernelManager:
    """
    Manages compiled kernels with caching and optimization.
    
    Responsibilities:
    - Kernel compilation orchestration
    - Cache management
    - Optimization pipeline
    - Metrics collection
    """
    
    def __init__(
        self,
        compiler: KernelCompiler,
        cache: PersistentKernelCache,
        optimizer: KernelOptimizer,
        config: UrzaConfig,
    ):
        self.compiler = compiler
        self.cache = cache
        self.optimizer = optimizer
        self.config = config
        
        # Compilation tracking
        self.active_compilations = {}
        self.compilation_history = deque(maxlen=1000)
```

**Key Methods:**

**`async compile_kernel(blueprint_ir: BlueprintIR, hints: Dict[str, Any]) -> CompiledKernel`**
```python
async def compile_kernel(
    self,
    blueprint_ir: BlueprintIR,
    hints: Optional[Dict[str, Any]] = None
) -> CompiledKernel:
    """
    Compile blueprint to kernel with caching.
    
    Process:
    1. Generate cache key from blueprint
    2. Check cache for existing kernel
    3. If miss, compile new kernel
    4. Apply optimizations based on hints
    5. Store in cache
    6. Return compiled kernel
    """
    
    # Cache lookup
    cache_key = self._generate_cache_key(blueprint_ir, hints)
    cached_kernel = await self.cache.get(cache_key)
    
    if cached_kernel:
        self._record_cache_hit(cache_key)
        return self._deserialize_kernel(cached_kernel)
    
    # Compilation
    self._record_cache_miss(cache_key)
    kernel = await self.compiler.compile(blueprint_ir, hints)
    
    # Optimization
    if self.config.optimization_level > 0:
        kernel = await self.optimizer.optimize(kernel, hints)
    
    # Cache storage
    await self.cache.put(cache_key, self._serialize_kernel(kernel))
    
    return kernel
```

**`async batch_compile(requests: List[CompilationRequest]) -> List[CompiledKernel]`**
```python
async def batch_compile(
    self,
    requests: List[CompilationRequest]
) -> List[CompiledKernel]:
    """
    Compile multiple kernels efficiently.
    
    Features:
    - Deduplication of identical requests
    - Parallel compilation
    - Shared optimization passes
    - Batch cache operations
    """
```

### Kernel Compiler (`kernel_compiler.py`)

**Purpose:** Transforms BlueprintIR into executable TorchScript kernels.

#### Key Classes

**`KernelCompiler`** - Blueprint to TorchScript Compiler
```python
class KernelCompiler:
    """
    Compiles BlueprintIR to TorchScript kernels.
    
    Features:
    - Multi-stage compilation pipeline
    - Error recovery
    - Performance profiling
    - Correctness verification
    """
    
    def __init__(
        self,
        enable_verification: bool = True,
        enable_profiling: bool = True,
    ):
        self.enable_verification = enable_verification
        self.enable_profiling = enable_profiling
        self.compilation_stats = defaultdict(list)
```

**Compilation Pipeline:**

**`async compile(blueprint_ir: BlueprintIR, hints: Dict[str, Any]) -> torch.jit.ScriptModule`**
```python
async def compile(
    self,
    blueprint_ir: BlueprintIR,
    hints: Optional[Dict[str, Any]] = None
) -> torch.jit.ScriptModule:
    """
    Compile BlueprintIR to TorchScript.
    
    Pipeline stages:
    1. Validation - Verify blueprint correctness
    2. Lowering - Convert to PyTorch operations
    3. Tracing/Scripting - Generate TorchScript
    4. Optimization - Apply TorchScript optimizations
    5. Verification - Ensure correctness
    """
    
    start_time = time.time()
    
    # Stage 1: Validation
    self._validate_blueprint(blueprint_ir)
    
    # Stage 2: Lowering
    pytorch_module = await self._lower_to_pytorch(blueprint_ir)
    
    # Stage 3: TorchScript generation
    if self._should_use_tracing(blueprint_ir):
        script_module = await self._trace_module(pytorch_module, blueprint_ir)
    else:
        script_module = await self._script_module(pytorch_module)
    
    # Stage 4: Optimization
    script_module = self._optimize_torchscript(script_module, hints)
    
    # Stage 5: Verification
    if self.enable_verification:
        self._verify_kernel(script_module, blueprint_ir)
    
    # Record stats
    compilation_time = time.time() - start_time
    self._record_compilation_stats(blueprint_ir, compilation_time)
    
    return script_module
```

**Lowering to PyTorch:**
```python
async def _lower_to_pytorch(
    self,
    blueprint_ir: BlueprintIR
) -> nn.Module:
    """
    Convert BlueprintIR to PyTorch module.
    
    Handles:
    - Node to operation mapping
    - Edge to tensor flow mapping
    - Parameter initialization
    - Custom operation registration
    """
    
    class GeneratedModule(nn.Module):
        def __init__(self, blueprint: BlueprintIR):
            super().__init__()
            self.blueprint = blueprint
            self._build_operations()
            
        def _build_operations(self):
            for node in self.blueprint.nodes:
                op = self._create_operation(node)
                self.add_module(f"op_{node.id}", op)
                
        def forward(self, x):
            # Execute operations following blueprint edges
            intermediates = {}
            
            for node in self._topological_sort(self.blueprint.nodes):
                inputs = [intermediates[e.source] for e in self._get_inputs(node)]
                output = getattr(self, f"op_{node.id}")(*inputs)
                intermediates[node.id] = output
                
            return output
    
    return GeneratedModule(blueprint_ir)
```

### Kernel Optimizer (`kernel_optimizer.py`)

**Purpose:** Applies hardware-specific and algorithmic optimizations to compiled kernels.

#### Optimization Strategies

**`OperatorFusion`** - Fuse Sequential Operations
```python
class OperatorFusion(OptimizationPass):
    """
    Fuse compatible operations to reduce memory traffic.
    
    Examples:
    - Conv + BatchNorm + ReLU → FusedConvBNReLU
    - Linear + Activation → FusedLinearAct
    - Attention components → FlashAttention
    """
    
    async def apply(
        self,
        kernel: torch.jit.ScriptModule,
        hints: Dict[str, Any]
    ) -> torch.jit.ScriptModule:
        # Identify fusion opportunities
        fusion_patterns = self._find_fusion_patterns(kernel.graph)
        
        # Apply fusions
        for pattern in fusion_patterns:
            kernel = self._apply_fusion(kernel, pattern)
            
        return kernel
```

**`Quantization`** - Precision Reduction
```python
class Quantization(OptimizationPass):
    """
    Apply quantization for faster inference.
    
    Modes:
    - INT8: 8-bit integer quantization
    - FP16: Half-precision floating point
    - Dynamic: Runtime quantization
    - QAT: Quantization-aware training
    """
    
    async def apply(
        self,
        kernel: torch.jit.ScriptModule,
        hints: Dict[str, Any]
    ) -> torch.jit.ScriptModule:
        precision = hints.get("precision", "fp32")
        
        if precision == "int8":
            return self._quantize_int8(kernel)
        elif precision == "fp16":
            return self._convert_fp16(kernel)
        else:
            return kernel
```

**`MemoryOptimization`** - Memory Layout Tuning
```python
class MemoryOptimization(OptimizationPass):
    """
    Optimize memory access patterns.
    
    Techniques:
    - Tensor layout optimization (NCHW vs NHWC)
    - Memory pooling
    - In-place operations
    - Gradient checkpointing hints
    """
```

### Performance Profiling (`profiler.py`)

**Purpose:** Detailed performance analysis of compiled kernels.

```python
class KernelProfiler:
    """
    Profile kernel execution characteristics.
    
    Metrics:
    - Execution time
    - Memory usage
    - FLOPs count
    - Memory bandwidth
    - Cache efficiency
    """
    
    async def profile_kernel(
        self,
        kernel: torch.jit.ScriptModule,
        sample_inputs: List[torch.Tensor]
    ) -> ProfilingReport:
        """
        Run comprehensive profiling on kernel.
        
        Returns detailed performance characteristics.
        """
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            # Warmup runs
            for _ in range(10):
                kernel(*sample_inputs)
                
            # Profiled runs
            for _ in range(100):
                kernel(*sample_inputs)
        
        return self._analyze_profile(prof)
```

## Integration with Other Services

### Tamiyo Integration
```python
# Tamiyo requests kernel compilation
async def handle_adaptation_decision(decision: AdaptationDecision):
    if decision.requires_new_kernel:
        # Generate blueprint
        blueprint = tezzeret.create_blueprint(decision.specifications)
        
        # Compile via Urza
        kernel = await urza_client.compile_kernel(
            blueprint_ir=blueprint.to_ir(),
            hints={
                "optimization_level": 2,
                "target_hardware": "cuda",
                "expected_workload": decision.workload_profile
            }
        )
        
        # Load into model
        await model.load_kernel(decision.layer_name, kernel.id)
```

### Storage Integration
```python
class UrzaKernelManager(KernelManager):
    """Extended manager with storage integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asset_repo = kwargs.get('asset_repository')
        
    async def compile_kernel(self, blueprint_ir, hints):
        kernel = await super().compile_kernel(blueprint_ir, hints)
        
        # Store in asset repository
        metadata = AssetMetadata(
            asset_id=kernel.id,
            asset_type="kernel",
            version=kernel.version,
            properties={
                "blueprint_hash": blueprint_ir.hash(),
                "optimization_level": self.config.optimization_level,
                "compilation_time_ms": kernel.compilation_time
            }
        )
        
        await self.asset_repo.create_asset(
            metadata=metadata,
            data=kernel.serialize()
        )
        
        return kernel
```

## Performance Characteristics

### Compilation Performance
- **Simple kernels:** <50ms compilation time
- **Complex kernels:** 100-500ms compilation time
- **Cache hit rate:** 85-95% in production
- **Parallel compilation:** Up to 10x speedup

### Runtime Performance
- **Kernel overhead:** <1% vs hand-written PyTorch
- **Memory efficiency:** 10-30% reduction with optimizations
- **Fusion benefits:** 20-40% speedup for compatible patterns
- **Quantization speedup:** 2-4x for INT8

## Configuration Examples

### Development Setup
```python
config = UrzaConfig(
    host="localhost",
    port=8000,
    workers=1,
    optimization_level=0,  # No optimization for debugging
    enable_profiling=True,
    enable_kernel_cache=False  # Fresh compilation each time
)

service = UrzaService(config)
```

### Production Setup
```python
config = UrzaConfig(
    host="0.0.0.0",
    port=8000,
    workers=8,
    optimization_level=2,
    max_concurrent_compilations=20,
    enable_kernel_cache=True,
    cache_ttl_hours=720,  # 30 days
    batch_compilation=True
)

# With storage backend
kernel_manager = KernelManager(
    compiler=KernelCompiler(),
    cache=PersistentKernelCache(cache_config),
    optimizer=KernelOptimizer(optimization_level=2),
    config=config
)
```

## API Usage Examples

### Basic Compilation
```python
# Client-side usage
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://urza:8000/compile",
        json={
            "blueprint_ir": blueprint.to_dict(),
            "optimization_hints": {
                "target_hardware": "cuda",
                "precision": "fp16"
            }
        }
    )
    
    result = response.json()
    kernel_id = result["kernel_id"]
    
    # Retrieve compiled kernel
    kernel_response = await client.get(f"http://urza:8000/kernel/{kernel_id}")
    kernel_bytes = kernel_response.content
```

### Batch Compilation
```python
# Compile multiple kernels efficiently
requests = [
    {"blueprint_ir": bp1.to_dict(), "hints": {"precision": "int8"}},
    {"blueprint_ir": bp2.to_dict(), "hints": {"precision": "fp16"}},
    {"blueprint_ir": bp3.to_dict(), "hints": {"optimize_memory": True}},
]

response = await client.post(
    "http://urza:8000/compile/batch",
    json={"requests": requests}
)

results = response.json()["results"]
kernel_ids = [r["kernel_id"] for r in results]
```

## Monitoring and Observability

### Prometheus Metrics
```python
# Compilation metrics
urza_compilations_total{status="success"} 1523
urza_compilations_total{status="failure"} 12
urza_compilation_duration_seconds{quantile="0.99"} 0.487
urza_cache_hit_rate 0.89
urza_active_compilations 3

# Optimization metrics
urza_optimizations_applied{type="fusion"} 892
urza_optimizations_applied{type="quantization"} 234
urza_optimization_speedup{type="fusion",quantile="0.5"} 1.35

# Resource metrics
urza_kernel_cache_size_bytes 4823159808
urza_compilation_memory_peak_bytes{quantile="0.99"} 536870912
```

### Health Endpoints
```python
@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe."""
    
    # Check compilation capacity
    if len(kernel_manager.active_compilations) >= config.max_concurrent_compilations:
        return JSONResponse(
            status_code=503,
            content={"status": "overloaded"}
        )
    
    return {"status": "ready"}
```

## Error Handling

### Compilation Errors
```python
class CompilationError(Exception):
    """Base class for compilation errors."""
    pass

class InvalidBlueprintError(CompilationError):
    """Blueprint validation failed."""
    pass

class TorchScriptError(CompilationError):
    """TorchScript generation failed."""
    pass

class OptimizationError(CompilationError):
    """Optimization pass failed."""
    pass

# Error handling in service
@app.exception_handler(CompilationError)
async def compilation_error_handler(request: Request, exc: CompilationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "blueprint_id": getattr(exc, "blueprint_id", None)
        }
    )
```

### Graceful Degradation
```python
async def compile_with_fallback(blueprint_ir, hints):
    try:
        # Try with full optimization
        return await kernel_manager.compile_kernel(blueprint_ir, hints)
    except OptimizationError:
        # Fall back to no optimization
        logger.warning("Optimization failed, compiling without optimization")
        hints["optimization_level"] = 0
        return await kernel_manager.compile_kernel(blueprint_ir, hints)
    except Exception as e:
        # Last resort: return reference kernel
        logger.error(f"Compilation failed: {e}")
        return get_reference_kernel(blueprint_ir)
```

## Best Practices

1. **Cache Key Design**
   - Include blueprint hash, optimization hints, and target hardware
   - Version cache keys when compiler changes

2. **Resource Management**
   - Limit concurrent compilations to prevent OOM
   - Monitor compilation memory usage
   - Use compilation timeouts

3. **Optimization Strategy**
   - Profile before optimizing
   - Test optimizations on representative workloads
   - Maintain correctness verification

4. **Error Recovery**
   - Always have fallback kernels
   - Log detailed compilation errors
   - Monitor compilation success rates

## Future Enhancements

1. **Advanced Optimizations**
   - Custom CUDA kernel generation
   - MLIR integration
   - Polyhedral optimization

2. **Distributed Compilation**
   - Compilation farm for large workloads
   - Cross-compilation for different targets
   - Incremental compilation

3. **Intelligence Features**
   - ML-based optimization selection
   - Compilation time prediction
   - Automatic performance regression detection

4. **Developer Experience**
   - Compilation debugging tools
   - Performance comparison dashboard
   - Blueprint visualization
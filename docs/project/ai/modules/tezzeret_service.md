# Tezzeret Service (`src/esper/services/tezzeret/`)

## Overview

Tezzeret is the blueprint synthesis and architecture generation service for the Esper platform. Named after the master artificer, it creates optimized neural network architectures as BlueprintIR representations based on adaptation decisions from Tamiyo. The service combines neural architecture search (NAS), performance prediction, and domain-specific optimizations to generate blueprints that can be compiled into executable kernels by Urza.

## Architecture

```
Tamiyo (Decision) → Tezzeret (Blueprint Synthesis) → BlueprintIR → Urza (Compilation)
                          ↓
                    Blueprint Registry
```

## Core Components

### Service Configuration
```python
@dataclass
class TezzeretConfig:
    """Configuration for Tezzeret service."""
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 4
    
    # Synthesis settings
    synthesis_timeout_seconds: int = 60
    max_blueprint_complexity: int = 1000  # Max nodes in blueprint
    enable_performance_prediction: bool = True
    
    # Search settings
    search_algorithm: str = "evolutionary"  # evolutionary, random, bayesian
    population_size: int = 100
    generations: int = 50
    
    # Registry settings
    enable_blueprint_cache: bool = True
    blueprint_ttl_days: int = 30
    max_cached_blueprints: int = 10000
```

### Main Service (`service.py`)

**Purpose:** FastAPI-based REST service for blueprint synthesis and management.

#### Key Endpoints

**`POST /synthesize` - Generate Blueprint from Requirements**
```python
@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_blueprint(request: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize blueprint based on requirements.
    
    Request:
        {
            "requirements": {
                "input_shape": [32, 128],
                "output_shape": [32, 64],
                "operation_type": "transformer_layer",
                "constraints": {
                    "max_latency_ms": 10,
                    "max_memory_mb": 100,
                    "target_accuracy": 0.95
                }
            },
            "optimization_target": "latency",  # latency, memory, accuracy
            "hardware_target": "cuda",
            "existing_architecture": {...}  # Optional, for modifications
        }
    
    Response:
        {
            "blueprint_id": "bp-xyz789",
            "blueprint_ir": {...},
            "predicted_performance": {
                "latency_ms": 8.5,
                "memory_mb": 85,
                "accuracy_estimate": 0.96
            },
            "synthesis_time_ms": 2340,
            "algorithm_used": "evolutionary"
        }
    """
```

**`POST /modify` - Modify Existing Blueprint**
```python
@app.post("/modify/{blueprint_id}", response_model=ModificationResponse)
async def modify_blueprint(
    blueprint_id: str,
    request: ModificationRequest
) -> ModificationResponse:
    """
    Apply modifications to existing blueprint.
    
    Modifications:
    - Add/remove layers
    - Change dimensions
    - Alter connections
    - Apply constraints
    """
```

**`POST /analyze` - Analyze Blueprint Performance**
```python
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_blueprint(request: AnalysisRequest) -> AnalysisResponse:
    """
    Predict performance characteristics of blueprint.
    
    Returns detailed analysis including:
    - Computational complexity
    - Memory requirements
    - Expected latency
    - Bottleneck identification
    """
```

**`GET /templates` - List Blueprint Templates**
```python
@app.get("/templates", response_model=List[TemplateInfo])
async def list_templates() -> List[TemplateInfo]:
    """
    Get available blueprint templates.
    
    Templates include:
    - Transformer blocks
    - CNN architectures
    - RNN variants
    - Custom patterns
    """
```

### Blueprint Synthesizer (`synthesizer.py`)

**Purpose:** Core synthesis engine that generates optimal blueprints.

#### Key Classes

**`BlueprintSynthesizer`** - Main Synthesis Engine
```python
class BlueprintSynthesizer:
    """
    Synthesizes optimal blueprints based on requirements.
    
    Combines multiple strategies:
    - Template-based generation
    - Evolutionary search
    - Constraint satisfaction
    - Performance prediction
    """
    
    def __init__(
        self,
        registry: BlueprintRegistry,
        predictor: PerformancePredictor,
        config: TezzeretConfig,
    ):
        self.registry = registry
        self.predictor = predictor
        self.config = config
        
        # Search algorithms
        self.search_algorithms = {
            "evolutionary": EvolutionarySearch(),
            "random": RandomSearch(),
            "bayesian": BayesianOptimization(),
        }
```

**Key Methods:**

**`async synthesize(requirements: Requirements) -> BlueprintIR`**
```python
async def synthesize(
    self,
    requirements: Requirements
) -> BlueprintIR:
    """
    Synthesize blueprint matching requirements.
    
    Process:
    1. Analyze requirements
    2. Select synthesis strategy
    3. Generate candidate architectures
    4. Evaluate candidates
    5. Optimize best candidate
    6. Return blueprint
    """
    
    # Check cache for similar requirements
    cached = await self._check_cache(requirements)
    if cached:
        return cached
    
    # Select strategy based on requirements
    strategy = self._select_strategy(requirements)
    
    # Generate candidates
    if strategy == "template":
        candidates = await self._generate_from_templates(requirements)
    elif strategy == "search":
        candidates = await self._run_architecture_search(requirements)
    else:
        candidates = await self._hybrid_generation(requirements)
    
    # Evaluate and select best
    best_candidate = await self._evaluate_candidates(candidates, requirements)
    
    # Final optimization
    optimized = await self._optimize_blueprint(best_candidate, requirements)
    
    # Cache result
    await self._cache_blueprint(optimized, requirements)
    
    return optimized
```

**Architecture Search Methods:**

**`async _run_architecture_search(requirements: Requirements) -> List[BlueprintIR]`**
```python
async def _run_architecture_search(
    self,
    requirements: Requirements
) -> List[BlueprintIR]:
    """
    Run neural architecture search.
    
    Uses configured search algorithm to explore
    architecture space efficiently.
    """
    
    algorithm = self.search_algorithms[self.config.search_algorithm]
    
    # Define search space
    search_space = self._define_search_space(requirements)
    
    # Run search
    population = algorithm.initialize_population(
        search_space,
        self.config.population_size
    )
    
    for generation in range(self.config.generations):
        # Evaluate population
        scores = await self._evaluate_population(population, requirements)
        
        # Evolve population
        population = algorithm.evolve(population, scores)
        
        # Early stopping
        if self._should_stop_early(scores):
            break
    
    # Return top candidates
    return algorithm.get_best_candidates(population, n=10)
```

### Template Manager (`template_manager.py`)

**Purpose:** Manages pre-built blueprint templates for common patterns.

```python
class TemplateManager:
    """
    Manages blueprint templates for rapid synthesis.
    
    Templates are parameterized blueprints that can be
    quickly adapted to specific requirements.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, BlueprintTemplate]:
        """Load built-in templates."""
        return {
            "transformer_block": TransformerBlockTemplate(),
            "resnet_block": ResNetBlockTemplate(),
            "mobilenet_block": MobileNetBlockTemplate(),
            "attention_layer": AttentionLayerTemplate(),
            "conv_block": ConvBlockTemplate(),
            "mlp_block": MLPBlockTemplate(),
        }
```

**Template Example - Transformer Block:**
```python
class TransformerBlockTemplate(BlueprintTemplate):
    """Template for transformer blocks."""
    
    def instantiate(self, params: Dict[str, Any]) -> BlueprintIR:
        """Create transformer block blueprint."""
        
        d_model = params["d_model"]
        n_heads = params["n_heads"]
        d_ff = params.get("d_ff", 4 * d_model)
        
        nodes = []
        edges = []
        
        # Multi-head attention
        nodes.append(NodeSpec(
            id="mha",
            op_type="multi_head_attention",
            attributes={
                "d_model": d_model,
                "n_heads": n_heads,
                "dropout": params.get("dropout", 0.1)
            }
        ))
        
        # Add & Norm
        nodes.append(NodeSpec(
            id="add_norm_1",
            op_type="add_and_norm",
            attributes={"d_model": d_model}
        ))
        
        # Feed-forward
        nodes.append(NodeSpec(
            id="ff",
            op_type="feed_forward",
            attributes={
                "d_model": d_model,
                "d_ff": d_ff,
                "activation": "gelu"
            }
        ))
        
        # Add & Norm
        nodes.append(NodeSpec(
            id="add_norm_2",
            op_type="add_and_norm",
            attributes={"d_model": d_model}
        ))
        
        # Connect nodes
        edges = [
            EdgeSpec("input", "mha", "x"),
            EdgeSpec("mha", "add_norm_1", "attn_out"),
            EdgeSpec("input", "add_norm_1", "residual"),
            EdgeSpec("add_norm_1", "ff", "x"),
            EdgeSpec("ff", "add_norm_2", "ff_out"),
            EdgeSpec("add_norm_1", "add_norm_2", "residual"),
            EdgeSpec("add_norm_2", "output", "x"),
        ]
        
        return BlueprintIR(
            nodes=nodes,
            edges=edges,
            metadata=BlueprintMetadata(
                name=f"transformer_block_{d_model}_{n_heads}",
                version="1.0.0",
                framework="pytorch",
                properties=params
            )
        )
```

### Performance Predictor (`performance_predictor.py`)

**Purpose:** Predicts blueprint performance without compilation.

```python
class PerformancePredictor:
    """
    ML-based performance prediction for blueprints.
    
    Uses trained models to estimate:
    - Latency
    - Memory usage
    - Accuracy potential
    - Power consumption
    """
    
    def __init__(
        self,
        latency_model_path: str,
        memory_model_path: str,
        accuracy_model_path: str,
    ):
        self.latency_predictor = self._load_model(latency_model_path)
        self.memory_predictor = self._load_model(memory_model_path)
        self.accuracy_predictor = self._load_model(accuracy_model_path)
```

**Prediction Methods:**
```python
async def predict_performance(
    self,
    blueprint: BlueprintIR,
    hardware: str = "cuda"
) -> PerformancePrediction:
    """
    Predict blueprint performance characteristics.
    """
    
    # Extract features from blueprint
    features = self._extract_features(blueprint)
    
    # Run predictions in parallel
    latency_task = self._predict_latency(features, hardware)
    memory_task = self._predict_memory(features, hardware)
    accuracy_task = self._predict_accuracy(features)
    
    latency, memory, accuracy = await asyncio.gather(
        latency_task, memory_task, accuracy_task
    )
    
    return PerformancePrediction(
        latency_ms=latency,
        memory_mb=memory,
        accuracy_estimate=accuracy,
        confidence=self._calculate_confidence(features)
    )

def _extract_features(self, blueprint: BlueprintIR) -> np.ndarray:
    """
    Extract numerical features from blueprint.
    
    Features include:
    - Node count by type
    - Edge connectivity metrics
    - Depth and width
    - Parameter count estimates
    - Operation complexity
    """
    
    features = []
    
    # Structural features
    features.append(len(blueprint.nodes))
    features.append(len(blueprint.edges))
    features.append(self._calculate_depth(blueprint))
    features.append(self._calculate_width(blueprint))
    
    # Operation features
    op_counts = defaultdict(int)
    for node in blueprint.nodes:
        op_counts[node.op_type] += 1
    
    for op_type in KNOWN_OP_TYPES:
        features.append(op_counts.get(op_type, 0))
    
    # Complexity features
    features.append(self._estimate_flops(blueprint))
    features.append(self._estimate_parameters(blueprint))
    
    return np.array(features)
```

### Constraint Solver (`constraint_solver.py`)

**Purpose:** Ensures blueprints satisfy specified constraints.

```python
class ConstraintSolver:
    """
    Validates and enforces constraints on blueprints.
    
    Constraint types:
    - Performance (latency, memory)
    - Structural (depth, width)
    - Hardware (operations support)
    - Numerical (precision, stability)
    """
    
    def validate_constraints(
        self,
        blueprint: BlueprintIR,
        constraints: Constraints
    ) -> ValidationResult:
        """Check if blueprint satisfies constraints."""
        
        violations = []
        
        # Check each constraint
        for constraint in constraints:
            if not self._check_constraint(blueprint, constraint):
                violations.append(constraint)
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations
        )
    
    def enforce_constraints(
        self,
        blueprint: BlueprintIR,
        constraints: Constraints
    ) -> BlueprintIR:
        """Modify blueprint to satisfy constraints."""
        
        modified = blueprint.copy()
        
        for constraint in constraints:
            if not self._check_constraint(modified, constraint):
                modified = self._apply_constraint(modified, constraint)
        
        return modified
```

### Blueprint Optimizer (`blueprint_optimizer.py`)

**Purpose:** Optimizes blueprints for specific objectives.

```python
class BlueprintOptimizer:
    """
    Optimizes blueprints for various objectives.
    
    Optimization strategies:
    - Latency minimization
    - Memory reduction
    - Accuracy maximization
    - Multi-objective optimization
    """
    
    async def optimize(
        self,
        blueprint: BlueprintIR,
        objective: str,
        constraints: Optional[Constraints] = None
    ) -> BlueprintIR:
        """
        Optimize blueprint for given objective.
        """
        
        if objective == "latency":
            return await self._optimize_latency(blueprint, constraints)
        elif objective == "memory":
            return await self._optimize_memory(blueprint, constraints)
        elif objective == "accuracy":
            return await self._optimize_accuracy(blueprint, constraints)
        elif objective == "balanced":
            return await self._multi_objective_optimize(blueprint, constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")
```

**Optimization Techniques:**
```python
async def _optimize_latency(
    self,
    blueprint: BlueprintIR,
    constraints: Optional[Constraints] = None
) -> BlueprintIR:
    """
    Optimize for minimal latency.
    
    Techniques:
    - Operation fusion
    - Parallel path identification
    - Bottleneck elimination
    - Precision reduction
    """
    
    optimized = blueprint.copy()
    
    # Identify fusion opportunities
    fusion_groups = self._find_fusable_operations(optimized)
    for group in fusion_groups:
        optimized = self._fuse_operations(optimized, group)
    
    # Find parallel paths
    parallel_paths = self._find_parallel_paths(optimized)
    optimized = self._optimize_parallel_execution(optimized, parallel_paths)
    
    # Reduce precision where possible
    if self._allows_reduced_precision(constraints):
        optimized = self._apply_mixed_precision(optimized)
    
    return optimized
```

## Integration Examples

### With Tamiyo
```python
# Tamiyo requests blueprint for adaptation
async def handle_adaptation_request(decision: AdaptationDecision):
    # Extract requirements from decision
    requirements = Requirements(
        input_shape=decision.layer_info["input_shape"],
        output_shape=decision.layer_info["output_shape"],
        operation_type=decision.adaptation_type,
        constraints=Constraints(
            max_latency_ms=decision.performance_target["latency"],
            max_memory_mb=decision.performance_target["memory"]
        )
    )
    
    # Request blueprint from Tezzeret
    response = await tezzeret_client.synthesize_blueprint(
        requirements=requirements,
        optimization_target="latency",
        hardware_target="cuda"
    )
    
    # Send to Urza for compilation
    kernel_id = await urza_client.compile_kernel(
        blueprint_ir=response.blueprint_ir,
        hints={"source": "tezzeret", "priority": "high"}
    )
    
    return kernel_id
```

### With Blueprint Registry
```python
class TezzeretBlueprintManager:
    """Manages blueprint storage and retrieval."""
    
    async def store_synthesized_blueprint(
        self,
        blueprint: BlueprintIR,
        synthesis_metadata: Dict[str, Any]
    ):
        # Store in registry
        blueprint_id = await self.registry.register_blueprint(
            blueprint=blueprint,
            metadata={
                "synthesizer": "tezzeret",
                "synthesis_time": synthesis_metadata["time_ms"],
                "algorithm": synthesis_metadata["algorithm"],
                "requirements": synthesis_metadata["requirements"]
            }
        )
        
        # Cache for future similar requests
        await self.cache.set(
            key=self._requirements_hash(synthesis_metadata["requirements"]),
            value=blueprint_id,
            ttl=86400  # 24 hours
        )
```

## Performance Characteristics

### Synthesis Performance
- **Template instantiation:** <10ms
- **Simple search:** 100-500ms
- **Complex search:** 1-10s
- **Performance prediction:** <50ms per blueprint

### Quality Metrics
- **Constraint satisfaction:** 99%+
- **Performance prediction accuracy:** ±15%
- **Search efficiency:** 10x faster than random
- **Template coverage:** 80% of common patterns

## Monitoring and Observability

### Prometheus Metrics
```python
# Synthesis metrics
tezzeret_synthesis_requests_total{algorithm="evolutionary"} 523
tezzeret_synthesis_duration_seconds{algorithm="evolutionary",quantile="0.99"} 8.7
tezzeret_synthesis_success_rate 0.98

# Blueprint metrics
tezzeret_blueprints_generated_total 1847
tezzeret_blueprint_complexity{quantile="0.95"} 487
tezzeret_template_usage{template="transformer_block"} 234

# Prediction metrics
tezzeret_prediction_accuracy{metric="latency"} 0.87
tezzeret_prediction_duration_milliseconds{quantile="0.99"} 48
```

### Logging
```python
logger.info(
    "Blueprint synthesized",
    extra={
        "blueprint_id": blueprint.id,
        "requirements": requirements.to_dict(),
        "synthesis_time_ms": synthesis_time,
        "algorithm": algorithm_used,
        "predicted_performance": predictions.to_dict()
    }
)
```

## Best Practices

1. **Requirements Specification**
   - Be specific about constraints
   - Provide hardware target information
   - Include workload characteristics

2. **Template Usage**
   - Use templates for common patterns
   - Customize templates before search
   - Cache template instances

3. **Search Configuration**
   - Start with small populations for quick results
   - Increase generations for better quality
   - Use early stopping for efficiency

4. **Performance Prediction**
   - Validate predictions periodically
   - Update models with real measurements
   - Consider prediction confidence

## Future Enhancements

1. **Advanced Synthesis**
   - Learned synthesis policies
   - Cross-domain blueprint transfer
   - Automated template discovery

2. **Better Search**
   - Gradient-based architecture search
   - Hardware-aware search spaces
   - Multi-fidelity optimization

3. **Integration Features**
   - Real-time performance feedback
   - Collaborative blueprint editing
   - Version control for blueprints

4. **Intelligence**
   - Blueprint recommendation system
   - Automated constraint relaxation
   - Performance regression detection
# Phase 1: Logical/Physical Separation Implementation Plan

## Overview
Phase 1 introduces the chunked architecture that separates the logical view (thousands of seeds) from the physical implementation (single optimized layer). This phase maintains full backward compatibility while laying the foundation for massive parallelism.

## Timeline: 6 Weeks (Weeks 5-10)

### Week 1-2: Core Architecture
- Implement ChunkManager for tensor splitting/concatenation
- Create LogicalSeed abstraction layer
- Design state tensor schema

### Week 3-4: Implementation
- Build ChunkedKasminaLayer with multi-seed support
- Create HybridKasminaLayer for compatibility
- Implement state management system

### Week 5-6: Testing & Integration
- Comprehensive testing of chunk operations
- Performance validation
- A/B testing setup

## Key Components

### 1. ChunkManager
**Purpose**: Manages the splitting of activation tensors into chunks and their reconstruction.

**Key Features**:
- Efficient tensor splitting without copies
- Dimension-preserving concatenation
- Support for variable chunk sizes
- Edge case handling (non-divisible dimensions)

### 2. LogicalSeed
**Purpose**: Provides the abstraction of independent seed agents while being backed by efficient tensor operations.

**Key Features**:
- Logical seed ID to physical chunk mapping
- State isolation between seeds
- Lifecycle management interface
- Telemetry generation per seed

### 3. State Tensor
**Purpose**: GPU-resident state management for all seeds in a layer.

**Schema**:
```
Shape: (num_seeds, 4)
Columns:
  [0]: lifecycle_state (int8) - Current lifecycle stage
  [1]: blueprint_id (int32) - Active blueprint identifier
  [2]: epochs_in_state (int16) - Time in current state
  [3]: grafting_strategy (int8) - Active grafting strategy
```

### 4. HybridKasminaLayer
**Purpose**: Provides seamless switching between legacy and chunked implementations.

**Features**:
- Feature flag integration
- Runtime implementation selection
- Performance monitoring
- Gradual rollout support

## Technical Requirements

### Performance Targets
- **Overhead**: <5% vs current implementation
- **Memory**: <10% increase
- **Scalability**: Support up to 10,000 seeds per layer

### Compatibility Requirements
- 100% API compatibility with existing KasminaLayer
- No changes required in calling code
- Seamless feature flag switching
- Full regression test passage

## Implementation Steps

### Step 1: Create Base Infrastructure
1. Set up kasmina v2 module structure
2. Implement dimension calculations
3. Create tensor manipulation utilities

### Step 2: Build Core Components
1. ChunkManager with splitting logic
2. LogicalSeed interface definition
3. State tensor initialization

### Step 3: Integration Layer
1. ChunkedKasminaLayer implementation
2. HybridKasminaLayer wrapper
3. Feature flag integration

### Step 4: Testing Suite
1. Unit tests for each component
2. Integration tests for full flow
3. Performance benchmarks
4. A/B testing setup

### Step 5: Documentation
1. API documentation
2. Migration guide
3. Performance tuning guide

## Risk Mitigation

### Technical Risks
1. **Tensor alignment issues**
   - Mitigation: Extensive edge case testing
   - Fallback: Padding strategies

2. **Memory fragmentation**
   - Mitigation: Contiguous tensor allocation
   - Monitoring: GPU memory profiling

3. **Performance regression**
   - Mitigation: Continuous benchmarking
   - Fallback: Feature flag disable

### Operational Risks
1. **Breaking changes**
   - Mitigation: Comprehensive regression tests
   - Process: Gradual rollout

2. **Debugging complexity**
   - Mitigation: Enhanced logging
   - Tools: Chunk visualization utilities

## Success Criteria

### Functional
- ✓ All regression tests pass
- ✓ Chunk operations produce identical outputs
- ✓ State updates are atomic
- ✓ Feature flags control implementation

### Performance
- ✓ <5% overhead in single-seed mode
- ✓ Linear scaling with seed count
- ✓ Memory usage within bounds
- ✓ No GPU synchronization issues

### Quality
- ✓ >95% test coverage
- ✓ Zero Codacy issues
- ✓ Documentation complete
- ✓ A/B tests show parity

## Deliverables

1. **Code**
   - `src/esper/morphogenetic_v2/kasmina/chunk_manager.py`
   - `src/esper/morphogenetic_v2/kasmina/logical_seed.py`
   - `src/esper/morphogenetic_v2/kasmina/state_tensor.py`
   - `src/esper/morphogenetic_v2/kasmina/chunked_layer.py`
   - `src/esper/morphogenetic_v2/kasmina/hybrid_layer.py`

2. **Tests**
   - Unit tests for all components
   - Integration tests
   - Performance benchmarks

3. **Documentation**
   - Technical design document
   - API reference
   - Migration guide

4. **Metrics**
   - Performance baseline comparison
   - A/B test results
   - Memory usage analysis

## Next Phase Preview
Phase 2 will build on this foundation to implement the extended 11-stage lifecycle, adding the missing states (GERMINATED, TRAINING, STABILIZATION, EVALUATING, FINE_TUNING, CULLED, CANCELLED, ROLLED_BACK) and their transition logic.
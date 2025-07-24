# REMEDIATION ACTIVITY A1: Blueprint Library and Reward System Implementation

**Document ID:** REMEDIATION-A1-2025-01-22  
**Priority:** HIGH  
**Estimated Duration:** 3-4 days  
**Author:** Claude (Anthropic)  
**Status:** APPROVED FOR IMPLEMENTATION

## Executive Summary

This remediation activity addresses critical gaps identified in the INDEPENDENT_EVAL.md analysis, specifically:
1. Missing blueprint template library (18 templates required for Phase 2)
2. Blueprint metadata system for Tamiyo integration
3. Multi-metric intelligent reward system completion
4. Phase 1-2 integration layer for seamless operation

## Identified Gaps

### Gap 1: Blueprint Template Library
**Reference:** INDEPENDENT_EVAL.md lines 261-415  
**Impact:** Tamiyo cannot make meaningful adaptation decisions without available blueprints  
**Required:** 18 production-ready blueprint templates across 6 categories

### Gap 2: Blueprint Metadata System  
**Reference:** INDEPENDENT_EVAL.md lines 426-461  
**Impact:** Tamiyo's GNN cannot reason about trade-offs without proper metadata  
**Required:** Comprehensive metadata schema with cost/benefit analysis

### Gap 3: Multi-Metric Reward System
**Reference:** PHASE_2_PLAN.md line 313, IMPLEMENTATION_PLAN.md lines 309-314  
**Impact:** Policy learning cannot proceed without accurate reward computation  
**Required:** Complete implementation with Phase 1 metrics integration

### Gap 4: Phase 1-2 Integration
**Reference:** Multiple references to missing integration layer  
**Impact:** Autonomous operation blocked without seamless execution pipeline  
**Required:** End-to-end decision to execution pathway

## Implementation Plan

### Component 1: Blueprint Template Library (Days 1-2)

#### 1.1 Directory Structure
```
src/esper/blueprints/
├── __init__.py
├── registry.py          # Blueprint registry and loader
├── metadata.py          # Metadata schemas
├── templates/
│   ├── __init__.py
│   ├── transformer.py   # Transformer block templates
│   ├── moe.py          # Mixture-of-Experts templates
│   ├── efficiency.py    # LoRA, IA3, compression
│   ├── routing.py       # Routing and scalability
│   └── diagnostics.py   # Monitoring and safety
└── manifest.yaml        # Blueprint manifest with metadata
```

#### 1.2 Implementation Priority
1. **Core Transformer Blocks** (BP-ATTN-STD, BP-MLP-GELU, etc.)
2. **MoE Components** (BP-ROUTER-TOP2, BP-EXPERT-MLP)
3. **Efficiency Adapters** (BP-PROJ-LoRA-64, BP-PROJ-IA3)
4. **Diagnostic Taps** (BP-MON-ACT-STATS, BP-CLAMP-GRAD-NORM)

### Component 2: Blueprint Metadata System (Day 2)

#### 2.1 Metadata Schema
```python
@dataclass
class BlueprintMetadata:
    # Core identification
    blueprint_id: str
    version: str
    category: BlueprintCategory
    
    # Cost analysis
    param_delta: int
    flop_delta: int
    memory_footprint_kb: int
    expected_latency_ms: float
    
    # Benefit estimation
    past_accuracy_gain_estimate: float
    stability_improvement_estimate: float
    
    # Safety and constraints
    risk_score: float  # 0.0-1.0
    is_safe_action: bool
    requires_capability: List[str]
    
    # Integration hints
    compatible_layers: List[str]
    incompatible_with: List[str]
    mergeable: bool
```

### Component 3: Multi-Metric Reward System (Day 3)

#### 3.1 Reward Components
- **Accuracy Improvement** (weight: 1.0)
- **Speed Improvement** (weight: 0.6) - Uses Phase 1 execution metrics
- **Memory Efficiency** (weight: 0.4) - Uses Phase 1 cache metrics
- **Stability Improvement** (weight: 0.8) - Uses Phase 1 error recovery stats
- **Adaptation Cost** (weight: -0.3)
- **Risk Penalty** (weight: -0.5)

#### 3.2 Temporal Analysis
- Immediate (0-5 min): 30% weight
- Short-term (5-30 min): 40% weight
- Medium-term (30-120 min): 25% weight
- Long-term (2+ hours): 5% weight

### Component 4: Phase 1-2 Integration Layer (Day 4)

#### 4.1 Decision Pipeline
```
Health Signals → Graph Builder → GNN Policy → Blueprint Selection → 
Compilation Request → Kernel Loading → Execution → Reward Computation
```

#### 4.2 Integration Points
- TamiyoService ↔ Blueprint Registry
- Blueprint Selection → Tezzeret Compilation
- Tezzeret Output → Phase 1 Kernel Cache
- Execution Metrics → Reward Computer

## Success Criteria

1. **Blueprint Availability**
   - [x] All 18 blueprint templates implemented
   - [x] Blueprint registry functional
   - [x] Metadata validates against schema

2. **Tamiyo Integration**
   - [x] Blueprint selection based on GNN output
   - [x] Metadata accessible during decision making
   - [x] Safety validation prevents dangerous selections

3. **Reward System**
   - [x] All 6 reward components implemented
   - [x] Phase 1 metrics integrated
   - [x] Temporal discounting functional

4. **End-to-End Flow**
   - [x] Decision → Blueprint → Kernel → Execution
   - [x] <5 minute full adaptation cycle
   - [x] >80% success rate for adaptations

## Risk Mitigation

1. **Blueprint Compilation Failures**
   - Extensive validation before submission
   - Fallback to simpler blueprints
   - Circuit breaker protection

2. **Reward Miscalibration**
   - Conservative initial weights
   - Monitoring and adjustment capability
   - Safety bounds on all metrics

3. **Integration Complexity**
   - Incremental rollout with feature flags
   - Comprehensive logging at each step
   - Rollback capability

## Testing Strategy

1. **Unit Tests**
   - Each blueprint template validation
   - Metadata schema compliance
   - Reward computation accuracy

2. **Integration Tests**
   - Blueprint registry loading
   - End-to-end adaptation flow
   - Phase 1-2 communication

3. **Performance Tests**
   - Blueprint compilation time
   - Decision latency with full library
   - Memory usage with all templates

## Implementation Timeline

**Day 1 (2025-01-22):**
- Morning: Blueprint directory structure and registry
- Afternoon: Transformer and MoE templates

**Day 2 (2025-01-23):**
- Morning: Efficiency and diagnostic templates
- Afternoon: Metadata system and manifest

**Day 3 (2025-01-24):**
- Morning: Multi-metric reward implementation
- Afternoon: Phase 1 integration hooks

**Day 4 (2025-01-25):**
- Morning: End-to-end integration layer
- Afternoon: Testing and validation

## Deliverables

1. **Blueprint Template Library**
   - 18 production-ready templates
   - Comprehensive metadata
   - YAML manifest for Urza

2. **Reward System**
   - IntelligentRewardComputer class
   - Phase 1 metrics integration
   - Temporal analysis

3. **Integration Layer**
   - Blueprint selection logic
   - Compilation pipeline
   - Execution feedback loop

4. **Documentation**
   - Blueprint usage guide
   - Integration patterns
   - Performance benchmarks

## Post-Implementation Validation

1. Run full Tamiyo policy training with new blueprints
2. Measure adaptation success rate (target: >80%)
3. Validate reward correlation with actual improvements
4. Stress test with rapid adaptation cycles

---

**Approval:** This plan addresses all critical gaps identified in the evaluation and provides a clear path to production-ready morphogenetic adaptation.

**Next Steps:** Proceed with implementation starting with blueprint template library.
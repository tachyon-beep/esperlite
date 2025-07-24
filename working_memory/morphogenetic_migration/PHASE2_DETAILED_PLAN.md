# Phase 2: Extended Lifecycle - Detailed Implementation Plan

*Created: 2025-01-24*

## Executive Summary

Phase 2 extends the morphogenetic system from the simplified 5-state lifecycle to the full 11-state design, adding sophisticated state management, checkpoint/restore capabilities, and enhanced grafting strategies. This phase is critical for enabling advanced adaptation behaviors while maintaining system stability.

## Timeline and Milestones

**Duration**: 8 weeks (Weeks 11-18 of migration plan)
**Start Date**: After Phase 1 validation (estimated Week 11)
**End Date**: Week 18

### Week-by-Week Breakdown

#### Weeks 11-12: State Machine Foundation
- Implement ExtendedLifecycle enum with 11 states
- Create LifecycleManager with transition validation
- Update StateTensor to support extended state tracking
- Build comprehensive state transition test suite

#### Weeks 13-14: Checkpoint/Restore System
- Design checkpoint format for seed states
- Implement serialization/deserialization logic
- Add version migration support
- Create recovery mechanisms for corrupted states

#### Weeks 15-16: Advanced Grafting Strategies
- Implement drift-controlled grafting
- Add momentum-based grafting
- Create adaptive ramp duration
- Build strategy selection logic

#### Weeks 17-18: Integration and Testing
- Update ChunkedKasminaLayer for extended lifecycle
- Integrate with existing telemetry system
- Performance optimization
- Comprehensive integration testing

## Technical Design

### 1. Extended Lifecycle States

```python
class ExtendedLifecycle(IntEnum):
    """Full 11-state lifecycle for morphogenetic seeds"""
    DORMANT = 0          # Monitoring only
    GERMINATED = 1       # Queued for training
    TRAINING = 2         # Self-supervised learning
    GRAFTING = 3         # Blending into network
    STABILIZATION = 4    # Network settling
    EVALUATING = 5       # Performance validation
    FINE_TUNING = 6      # Task-specific training
    FOSSILIZED = 7       # Permanently integrated
    CULLED = 8           # Failed adaptation
    CANCELLED = 9        # User/system cancelled
    ROLLED_BACK = 10     # Emergency revert
```

### 2. State Transition Matrix

```python
class StateTransition:
    """Defines valid state transitions and validation rules"""
    
    VALID_TRANSITIONS = {
        ExtendedLifecycle.DORMANT: [
            ExtendedLifecycle.GERMINATED
        ],
        ExtendedLifecycle.GERMINATED: [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.CANCELLED
        ],
        ExtendedLifecycle.TRAINING: [
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.CANCELLED
        ],
        ExtendedLifecycle.GRAFTING: [
            ExtendedLifecycle.STABILIZATION,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.STABILIZATION: [
            ExtendedLifecycle.EVALUATING,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.EVALUATING: [
            ExtendedLifecycle.FINE_TUNING,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.ROLLED_BACK
        ],
        ExtendedLifecycle.FINE_TUNING: [
            ExtendedLifecycle.FOSSILIZED,
            ExtendedLifecycle.CULLED,
            ExtendedLifecycle.ROLLED_BACK
        ],
        # Terminal states have no outgoing transitions
        ExtendedLifecycle.FOSSILIZED: [],
        ExtendedLifecycle.CULLED: [],
        ExtendedLifecycle.CANCELLED: [],
        ExtendedLifecycle.ROLLED_BACK: []
    }
    
    @staticmethod
    def validate_transition(
        from_state: ExtendedLifecycle,
        to_state: ExtendedLifecycle,
        context: TransitionContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate if a state transition is allowed"""
        if to_state not in StateTransition.VALID_TRANSITIONS[from_state]:
            return False, f"Invalid transition: {from_state.name} -> {to_state.name}"
        
        # State-specific validation
        validators = {
            (ExtendedLifecycle.TRAINING, ExtendedLifecycle.GRAFTING): 
                StateTransition._validate_reconstruction,
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.FINE_TUNING):
                StateTransition._validate_positive_evaluation,
            (ExtendedLifecycle.EVALUATING, ExtendedLifecycle.CULLED):
                StateTransition._validate_negative_evaluation
        }
        
        validator = validators.get((from_state, to_state))
        if validator:
            return validator(context)
        
        return True, None
```

### 3. Enhanced State Management

```python
class ExtendedStateTensor:
    """GPU-resident state management for extended lifecycle"""
    
    def __init__(self, num_seeds: int, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_seeds = num_seeds
        
        # Extended state variables
        self.state_tensor = torch.zeros(
            (num_seeds, 8),  # Increased from 4 to 8 columns
            dtype=torch.int32,
            device=self.device
        )
        
        # Column definitions
        self.LIFECYCLE_STATE = 0
        self.BLUEPRINT_ID = 1
        self.EPOCHS_IN_STATE = 2
        self.GRAFTING_STRATEGY = 3
        self.PARENT_STATE = 4      # For rollback
        self.CHECKPOINT_ID = 5     # For recovery
        self.EVALUATION_SCORE = 6  # Performance metric
        self.ERROR_COUNT = 7       # Failure tracking
        
        # Additional tracking tensors
        self.transition_history = torch.zeros(
            (num_seeds, 10, 2),  # Last 10 transitions
            dtype=torch.int32,
            device=self.device
        )
        
        self.performance_metrics = torch.zeros(
            (num_seeds, 4),  # Loss, accuracy, stability, efficiency
            dtype=torch.float32,
            device=self.device
        )
```

### 4. Checkpoint/Restore System

```python
class CheckpointManager:
    """Manages seed state persistence and recovery"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(
        self,
        layer_id: str,
        seed_id: int,
        state_data: Dict[str, Any],
        blueprint_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> str:
        """Save seed state to disk"""
        checkpoint_id = f"{layer_id}_{seed_id}_{int(time.time())}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        checkpoint = {
            'version': 2,  # Checkpoint format version
            'layer_id': layer_id,
            'seed_id': seed_id,
            'timestamp': time.time(),
            'state_data': state_data,
            'blueprint_state': blueprint_state,
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_id
    
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        target_device: torch.device
    ) -> Dict[str, Any]:
        """Restore seed state from checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        
        # Version migration if needed
        if checkpoint['version'] < 2:
            checkpoint = self._migrate_checkpoint_v1_to_v2(checkpoint)
        
        return checkpoint
```

### 5. Advanced Grafting Strategies

```python
class GraftingStrategyBase(ABC):
    """Base class for grafting strategies"""
    
    @abstractmethod
    def compute_alpha(
        self,
        current_epoch: int,
        total_epochs: int,
        metrics: Dict[str, float]
    ) -> float:
        """Compute blending alpha for current epoch"""
        pass

class DriftControlledGrafting(GraftingStrategyBase):
    """Pauses grafting if weight drift exceeds threshold"""
    
    def __init__(self, drift_threshold: float = 0.01, pause_duration: int = 10):
        self.drift_threshold = drift_threshold
        self.pause_duration = pause_duration
        self.pause_counter = 0
        self.weight_history = []
        
    def compute_alpha(self, current_epoch: int, total_epochs: int, metrics: Dict[str, float]) -> float:
        drift = metrics.get('weight_drift', 0.0)
        
        if drift > self.drift_threshold:
            self.pause_counter = self.pause_duration
        
        if self.pause_counter > 0:
            self.pause_counter -= 1
            return metrics.get('current_alpha', 0.0)  # Hold current value
        
        # Normal linear ramp
        return min(1.0, current_epoch / total_epochs)

class MomentumGrafting(GraftingStrategyBase):
    """Accelerates grafting based on positive performance trends"""
    
    def __init__(self, momentum_factor: float = 0.1):
        self.momentum_factor = momentum_factor
        self.velocity = 0.0
        
    def compute_alpha(self, current_epoch: int, total_epochs: int, metrics: Dict[str, float]) -> float:
        performance_delta = metrics.get('performance_delta', 0.0)
        
        # Update velocity based on performance
        self.velocity = self.momentum_factor * self.velocity + performance_delta
        
        # Base ramp with momentum adjustment
        base_alpha = current_epoch / total_epochs
        adjusted_alpha = base_alpha + self.velocity
        
        return np.clip(adjusted_alpha, 0.0, 1.0)
```

## Implementation Components

### Core Files to Create

1. **`src/esper/morphogenetic_v2/lifecycle/extended_lifecycle.py`**
   - ExtendedLifecycle enum
   - StateTransition validation logic
   - Transition context definitions

2. **`src/esper/morphogenetic_v2/lifecycle/state_manager.py`**
   - ExtendedStateTensor implementation
   - State query and update methods
   - Transition history tracking

3. **`src/esper/morphogenetic_v2/lifecycle/checkpoint_manager.py`**
   - Checkpoint save/restore logic
   - Version migration support
   - Corruption recovery

4. **`src/esper/morphogenetic_v2/grafting/strategies.py`**
   - Base grafting strategy class
   - Drift-controlled strategy
   - Momentum-based strategy
   - Adaptive strategies

5. **`src/esper/morphogenetic_v2/grafting/strategy_selector.py`**
   - Strategy selection logic
   - Performance-based switching
   - Strategy configuration

### Updated Files

1. **`src/esper/morphogenetic_v2/kasmina/chunked_layer.py`**
   - Integrate ExtendedLifecycle
   - Add checkpoint support
   - Enhanced telemetry for new states

2. **`src/esper/morphogenetic_v2/kasmina/state_tensor.py`**
   - Extend to support 8 state variables
   - Add performance metrics tracking
   - Transition history management

## Testing Strategy

### Unit Tests

1. **State Transition Tests**
   - Valid transition paths
   - Invalid transition rejection
   - Context validation
   - Edge cases

2. **Checkpoint Tests**
   - Save/restore correctness
   - Version migration
   - Corruption handling
   - Performance impact

3. **Grafting Strategy Tests**
   - Alpha computation accuracy
   - Strategy switching
   - Performance metrics
   - Edge conditions

### Integration Tests

1. **Full Lifecycle Tests**
   - Complete state progression
   - Rollback scenarios
   - Concurrent state changes
   - Performance under load

2. **Recovery Tests**
   - Checkpoint recovery
   - State corruption
   - Partial failures
   - System restart

## Performance Targets

### Latency
- State transition: <0.1ms
- Checkpoint save: <10ms
- Checkpoint restore: <20ms
- Strategy computation: <0.5ms

### Memory
- Extended state tensor: 32 bytes/seed
- Transition history: 80 bytes/seed
- Performance metrics: 16 bytes/seed
- Total overhead: <150 bytes/seed

### Scalability
- Support 10,000+ seeds
- Concurrent transitions: 1000+
- Checkpoint throughput: 100/sec

## Risk Mitigation

### Technical Risks

1. **State Explosion**
   - Risk: Complex state space leads to bugs
   - Mitigation: Formal state machine validation
   - Mitigation: Comprehensive test coverage

2. **Checkpoint Corruption**
   - Risk: Lost work due to bad checkpoints
   - Mitigation: Redundant checkpoints
   - Mitigation: Integrity validation

3. **Performance Degradation**
   - Risk: Extended tracking slows system
   - Mitigation: GPU-optimized operations
   - Mitigation: Lazy checkpoint writing

### Operational Risks

1. **Migration Complexity**
   - Risk: Difficult upgrade from Phase 1
   - Mitigation: Compatibility layer
   - Mitigation: Gradual rollout

2. **Strategy Selection**
   - Risk: Poor strategy choices
   - Mitigation: Default conservative strategy
   - Mitigation: A/B testing framework

## Success Criteria

### Functional
- ✓ All 11 states implemented
- ✓ Valid transitions enforced
- ✓ Checkpoint/restore working
- ✓ Multiple grafting strategies
- ✓ Backward compatibility

### Performance
- ✓ Meeting latency targets
- ✓ Memory usage acceptable
- ✓ Scaling to 10K seeds
- ✓ No regression vs Phase 1

### Quality
- ✓ 95%+ test coverage
- ✓ Codacy compliance
- ✓ Documentation complete
- ✓ Integration tests passing

## Deliverables

### Week 12
- Extended lifecycle implementation
- State transition validation
- Basic unit tests

### Week 14
- Checkpoint system complete
- Recovery mechanisms
- Integration with StateTensor

### Week 16
- All grafting strategies
- Strategy selection logic
- Performance optimization

### Week 18
- Full integration
- Documentation
- Deployment ready

## Dependencies

### Phase 1 Components
- ChunkManager
- LogicalSeed abstraction
- StateTensor (to be extended)
- ChunkedKasminaLayer (to be updated)

### External Dependencies
- PyTorch 2.0+ (checkpoint format)
- Python 3.8+ (typing features)
- NumPy (strategy computations)

## Rollout Plan

### Stage 1: Internal Testing (Week 18)
- Deploy to development
- Run regression tests
- Performance validation

### Stage 2: Limited Rollout (Week 19)
- Enable for test models
- Monitor state transitions
- Validate checkpoints

### Stage 3: Gradual Expansion (Week 20-21)
- 10% → 25% → 50% rollout
- A/B testing vs Phase 1
- Performance monitoring

### Stage 4: Full Deployment (Week 22)
- 100% rollout
- Phase 1 deprecated
- Phase 3 planning begins

## Future Considerations

### Phase 3 Integration
- Triton kernels for state ops
- Fused transition validation
- GPU-accelerated checkpoints

### Phase 4 Preparation
- Message bus for state events
- Distributed state management
- Cross-node checkpoint sync

### Long-term Evolution
- Neural state controllers
- Learned transition policies
- Adaptive lifecycle tuning
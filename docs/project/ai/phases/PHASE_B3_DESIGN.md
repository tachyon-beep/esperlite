# Phase B3: Intelligent Seed Selection - Detailed Design

## Overview

Phase B3 replaces the hardcoded `seed_idx=0` approach with a dynamic, performance-based seed selection system. This enables the morphogenetic platform to intelligently choose which seeds to activate based on their historical performance and exploration needs.

## Core Problem

Currently, the system always uses seed index 0 when loading kernels:
- `blueprint_integration.py:496`: `seed_idx=0  # TODO: Smarter seed selection`
- This wastes the multi-seed architecture's potential
- No exploration of alternative architectures
- No performance-based optimization

## Architecture Design

### 1. Seed Performance Tracking

```python
class SeedPerformanceMetrics:
    """Tracks performance history for a single seed."""
    seed_idx: int
    layer_name: str
    kernel_id: Optional[str]
    
    # Performance metrics
    accuracy_improvements: List[float]  # Delta accuracy per epoch
    loss_reductions: List[float]        # Delta loss per epoch
    execution_latencies: List[float]    # Forward pass latency
    memory_usage: List[float]           # GPU memory consumption
    
    # Statistical tracking
    total_activations: int              # Times this seed was active
    successful_grafts: int              # Successful integrations
    failed_attempts: int                # Error count
    last_activation_epoch: int          # Most recent use
    
    # Computed scores
    ucb_score: float                    # Upper Confidence Bound
    thompson_score: float               # Thompson sampling score
    performance_score: float            # Direct performance metric
```

### 2. Selection Algorithms

#### 2.1 Upper Confidence Bound (UCB)
Balances exploration vs exploitation using confidence intervals:
```python
ucb_score = mean_performance + sqrt(2 * log(total_trials) / seed_trials)
```

#### 2.2 Thompson Sampling
Probabilistic selection based on Beta distribution:
```python
# Model each seed as Beta(successes + 1, failures + 1)
sample = np.random.beta(successes + 1, failures + 1)
```

#### 2.3 Epsilon-Greedy
Simple baseline with random exploration:
```python
if random.random() < epsilon:
    return random_seed()
else:
    return best_performing_seed()
```

#### 2.4 Performance-Weighted
Direct optimization based on recent performance:
```python
weight = exponential_decay * performance_delta
```

### 3. Integration Points

#### 3.1 With KasminaLayer
- Query seed states before selection
- Respect seed lifecycle states
- Avoid overloading active seeds

#### 3.2 With Tamiyo Service
- Receive health signals for performance tracking
- Update metrics based on execution feedback
- Coordinate with GNN policy decisions

#### 3.3 With Blueprint Integration
- Replace hardcoded seed_idx=0
- Provide selection reasoning for logging
- Support fallback to index 0 if needed

### 4. Selection Framework

```python
class SeedSelector:
    """Main seed selection framework."""
    
    def __init__(self, strategy: SelectionStrategy):
        self.strategy = strategy
        self.performance_tracker = PerformanceTracker()
        self.layer_seed_states = {}  # Track per-layer seed availability
        
    async def select_seed(
        self,
        layer_name: str,
        available_seeds: List[int],
        context: SelectionContext
    ) -> Tuple[int, SelectionReason]:
        """
        Select optimal seed for kernel loading.
        
        Returns:
            (seed_idx, reason) tuple
        """
        # 1. Get performance history
        metrics = self.performance_tracker.get_layer_metrics(layer_name)
        
        # 2. Filter by availability and state
        viable_seeds = self._filter_viable_seeds(available_seeds, layer_name)
        
        # 3. Apply selection strategy
        seed_idx = self.strategy.select(viable_seeds, metrics, context)
        
        # 4. Update tracking
        self.performance_tracker.record_selection(layer_name, seed_idx)
        
        return seed_idx, self.strategy.get_reason()
```

### 5. Performance Tracking System

```python
class PerformanceTracker:
    """Centralized performance tracking for all seeds."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.metrics: Dict[str, SeedPerformanceMetrics] = {}
        self.redis_client = redis_client  # For persistence
        
    async def update_metrics(
        self,
        layer_name: str,
        seed_idx: int,
        performance_delta: PerformanceDelta
    ):
        """Update seed performance based on execution results."""
        key = f"{layer_name}:{seed_idx}"
        
        if key not in self.metrics:
            self.metrics[key] = SeedPerformanceMetrics(
                seed_idx=seed_idx,
                layer_name=layer_name
            )
            
        metric = self.metrics[key]
        
        # Update performance arrays with sliding window
        metric.accuracy_improvements.append(performance_delta.accuracy_delta)
        metric.loss_reductions.append(performance_delta.loss_delta)
        metric.execution_latencies.append(performance_delta.latency_ms)
        
        # Maintain window size
        if len(metric.accuracy_improvements) > WINDOW_SIZE:
            metric.accuracy_improvements.pop(0)
            # ... same for other arrays
            
        # Recompute scores
        self._update_scores(metric)
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_metrics(key, metric)
```

### 6. Configuration

```yaml
seed_selection:
  strategy: "ucb"  # ucb, thompson, epsilon_greedy, performance_weighted
  
  ucb:
    exploration_constant: 2.0
    
  thompson:
    prior_alpha: 1.0
    prior_beta: 1.0
    
  epsilon_greedy:
    epsilon: 0.1
    decay_rate: 0.995
    
  performance:
    window_size: 20
    decay_factor: 0.9
    
  constraints:
    min_dormant_epochs: 5      # Cooldown between activations
    max_concurrent_seeds: 2    # Per layer limit
    fallback_to_zero: true     # Use seed 0 if selection fails
```

## Implementation Plan

### Phase 1: Core Framework (Day 1-2)
1. Create `src/esper/services/tamiyo/seed_selector.py`
2. Create `src/esper/services/tamiyo/performance_tracker.py`
3. Define interfaces and data structures
4. Implement basic selection strategies

### Phase 2: Integration (Day 3-4)
1. Update `blueprint_integration.py` to use SeedSelector
2. Connect to health signal processing
3. Add Redis persistence for metrics
4. Implement fallback mechanisms

### Phase 3: Testing & Tuning (Day 5)
1. Unit tests for all strategies
2. Integration tests with mock data
3. Performance benchmarks
4. Parameter tuning

## Success Criteria

1. **Dynamic Selection**: No more hardcoded seed_idx=0
2. **Performance Improvement**: 15%+ better adaptation effectiveness
3. **Exploration**: All seeds get tried over time
4. **No Starvation**: High-performing seeds get appropriate usage
5. **Robustness**: Graceful fallback on failures

## Risk Mitigation

1. **Poor Initial Selection**: Start with epsilon-greedy for exploration
2. **Metric Staleness**: Use exponential decay for old measurements
3. **State Conflicts**: Coordinate with KasminaLayer state management
4. **Performance Overhead**: Cache selection decisions

## Testing Strategy

### Unit Tests
- Each selection algorithm independently
- Performance tracker with mock data
- Edge cases (no viable seeds, all equal performance)

### Integration Tests
- Full pipeline with BlueprintIntegration
- Concurrent selections for multiple layers
- Failure scenarios and recovery

### Performance Tests
- Selection latency < 1ms
- Memory usage with large history
- Redis persistence overhead

## Future Enhancements

1. **Multi-Armed Bandit Variants**: Contextual bandits, adversarial bandits
2. **Neural Selection**: Train a small network for seed selection
3. **Hierarchical Selection**: Layer-type specific strategies
4. **A/B Testing**: Compare strategies in production
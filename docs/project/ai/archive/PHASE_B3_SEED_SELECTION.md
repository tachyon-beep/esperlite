# Phase B3: Intelligent Seed Selection Strategy - Detailed Implementation

## Overview

Phase B3 replaces the hardcoded `seed_idx=0` with an intelligent, performance-based seed selection strategy. This phase implements multiple selection algorithms that balance exploration of new seeds with exploitation of high-performing ones.

## Current State Analysis

### Problem Statement
- Current implementation always uses `seed_idx=0` in blueprint integration
- No tracking of seed performance history
- No exploration of potentially better seeds
- Missing opportunity for optimization based on runtime characteristics

### Code to Replace
```python
# Current problematic code in various places
seed_idx = 0  # Always using first seed!
active_seed = blueprint.seeds[seed_idx]
```

## Detailed Implementation Plan

### 1. Seed Performance Tracker

**File**: `src/esper/services/tamiyo/performance_tracker.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Deque
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta

@dataclass
class SeedPerformanceMetrics:
    """Metrics tracked for each seed."""
    seed_id: str
    blueprint_id: str
    
    # Performance metrics
    accuracy_improvements: Deque[float]  # Rolling window
    latency_measurements: Deque[float]   # In milliseconds
    memory_usage: Deque[float]           # In MB
    throughput: Deque[float]             # Samples/second
    
    # Usage statistics
    selection_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_selected: Optional[datetime] = None
    
    # Computed statistics
    mean_accuracy_gain: float = 0.0
    mean_latency: float = 0.0
    success_rate: float = 0.0
    confidence_interval: float = 0.0

class PerformanceTracker:
    """Tracks and analyzes seed performance over time."""
    
    def __init__(self, 
                 window_size: int = 100,
                 decay_factor: float = 0.95,
                 min_samples: int = 5):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.min_samples = min_samples
        self.metrics: Dict[str, SeedPerformanceMetrics] = {}
        self.global_stats = GlobalPerformanceStats()
        
    def record_performance(self,
                         seed_id: str,
                         blueprint_id: str,
                         accuracy_delta: float,
                         latency: float,
                         memory: float,
                         throughput: float,
                         success: bool) -> None:
        """Record performance metrics for a seed execution."""
        if seed_id not in self.metrics:
            self.metrics[seed_id] = SeedPerformanceMetrics(
                seed_id=seed_id,
                blueprint_id=blueprint_id,
                accuracy_improvements=deque(maxlen=self.window_size),
                latency_measurements=deque(maxlen=self.window_size),
                memory_usage=deque(maxlen=self.window_size),
                throughput=deque(maxlen=self.window_size)
            )
            
        metrics = self.metrics[seed_id]
        
        # Update rolling windows
        metrics.accuracy_improvements.append(accuracy_delta)
        metrics.latency_measurements.append(latency)
        metrics.memory_usage.append(memory)
        metrics.throughput.append(throughput)
        
        # Update counters
        metrics.selection_count += 1
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            
        metrics.last_selected = datetime.now()
        
        # Recompute statistics
        self._update_statistics(metrics)
        
    def _update_statistics(self, metrics: SeedPerformanceMetrics) -> None:
        """Update computed statistics with decay."""
        if len(metrics.accuracy_improvements) >= self.min_samples:
            # Apply exponential decay to older measurements
            weights = np.array([
                self.decay_factor ** i 
                for i in range(len(metrics.accuracy_improvements))
            ])
            weights = weights / weights.sum()
            
            # Weighted statistics
            metrics.mean_accuracy_gain = np.average(
                list(metrics.accuracy_improvements), 
                weights=weights
            )
            metrics.mean_latency = np.average(
                list(metrics.latency_measurements),
                weights=weights
            )
            
            # Success rate
            metrics.success_rate = (
                metrics.success_count / metrics.selection_count
                if metrics.selection_count > 0 else 0
            )
            
            # Confidence interval (using standard error)
            if len(metrics.accuracy_improvements) > 1:
                std_error = np.std(list(metrics.accuracy_improvements)) / \
                           np.sqrt(len(metrics.accuracy_improvements))
                metrics.confidence_interval = 1.96 * std_error  # 95% CI
                
    def get_seed_stats(self, seed_id: str) -> Optional[SeedPerformanceMetrics]:
        """Get performance statistics for a specific seed."""
        return self.metrics.get(seed_id)
        
    def get_top_performers(self, n: int = 5) -> List[str]:
        """Get top N performing seeds by accuracy improvement."""
        sorted_seeds = sorted(
            self.metrics.items(),
            key=lambda x: x[1].mean_accuracy_gain,
            reverse=True
        )
        return [seed_id for seed_id, _ in sorted_seeds[:n]]
```

### 2. Seed Selection Strategies

**File**: `src/esper/services/tamiyo/selection_strategies.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional

class SelectionStrategy(ABC):
    """Base class for seed selection strategies."""
    
    @abstractmethod
    def select_seed(self, 
                   available_seeds: List[SeedData],
                   performance_tracker: PerformanceTracker,
                   context: Dict[str, Any]) -> int:
        """Select a seed index from available seeds."""
        pass

class UCBSelectionStrategy(SelectionStrategy):
    """Upper Confidence Bound selection - balances exploration and exploitation."""
    
    def __init__(self, exploration_constant: float = 2.0):
        self.exploration_constant = exploration_constant
        self.total_selections = 0
        
    def select_seed(self, 
                   available_seeds: List[SeedData],
                   performance_tracker: PerformanceTracker,
                   context: Dict[str, Any]) -> int:
        """Select seed using UCB algorithm."""
        self.total_selections += 1
        
        ucb_scores = []
        for i, seed in enumerate(available_seeds):
            stats = performance_tracker.get_seed_stats(seed.id)
            
            if stats is None or stats.selection_count < 3:
                # Explore unvisited or rarely visited seeds
                ucb_scores.append(float('inf'))
            else:
                # UCB formula: mean + c * sqrt(ln(N) / n)
                exploitation = stats.mean_accuracy_gain
                exploration = self.exploration_constant * np.sqrt(
                    np.log(self.total_selections) / stats.selection_count
                )
                ucb_scores.append(exploitation + exploration)
                
        return int(np.argmax(ucb_scores))

class ThompsonSamplingStrategy(SelectionStrategy):
    """Thompson Sampling - probabilistic selection based on posterior distributions."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Prior success
        self.beta = beta    # Prior failure
        
    def select_seed(self,
                   available_seeds: List[SeedData],
                   performance_tracker: PerformanceTracker,
                   context: Dict[str, Any]) -> int:
        """Select seed using Thompson Sampling."""
        samples = []
        
        for seed in available_seeds:
            stats = performance_tracker.get_seed_stats(seed.id)
            
            if stats is None:
                # Use prior for unknown seeds
                sample = np.random.beta(self.alpha, self.beta)
            else:
                # Update posterior with observed data
                alpha = self.alpha + stats.success_count
                beta = self.beta + stats.failure_count
                sample = np.random.beta(alpha, beta)
                
            samples.append(sample)
            
        return int(np.argmax(samples))

class EpsilonGreedyStrategy(SelectionStrategy):
    """Epsilon-Greedy - simple exploration/exploitation trade-off."""
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = 0.01
        
    def select_seed(self,
                   available_seeds: List[SeedData],
                   performance_tracker: PerformanceTracker,
                   context: Dict[str, Any]) -> int:
        """Select seed using epsilon-greedy algorithm."""
        # Decay epsilon over time
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.randint(0, len(available_seeds))
        else:
            # Exploit: select best performing
            best_score = -float('inf')
            best_idx = 0
            
            for i, seed in enumerate(available_seeds):
                stats = performance_tracker.get_seed_stats(seed.id)
                score = stats.mean_accuracy_gain if stats else 0
                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            return best_idx

class ContextualBanditStrategy(SelectionStrategy):
    """Contextual bandit - considers current model state in selection."""
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.weights = {}  # Per-seed linear models
        
    def select_seed(self,
                   available_seeds: List[SeedData],
                   performance_tracker: PerformanceTracker,
                   context: Dict[str, Any]) -> int:
        """Select seed based on contextual features."""
        # Extract context features
        features = self._extract_features(context)
        
        predicted_rewards = []
        for seed in available_seeds:
            if seed.id not in self.weights:
                # Initialize weights for new seed
                self.weights[seed.id] = np.random.randn(self.feature_dim) * 0.01
                
            # Predict reward for this context
            reward = np.dot(self.weights[seed.id], features)
            predicted_rewards.append(reward)
            
        # Add exploration noise
        noise = np.random.randn(len(predicted_rewards)) * 0.1
        scores = np.array(predicted_rewards) + noise
        
        return int(np.argmax(scores))
        
    def _extract_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from context."""
        features = []
        
        # Model state features
        features.append(context.get('current_loss', 0))
        features.append(context.get('current_accuracy', 0))
        features.append(context.get('learning_rate', 0))
        features.append(context.get('epoch', 0) / 100)  # Normalize
        
        # Resource features
        features.append(context.get('gpu_memory_used', 0) / 100)
        features.append(context.get('batch_size', 32) / 128)
        
        # Training dynamics
        features.append(context.get('loss_momentum', 0))
        features.append(context.get('gradient_norm', 1.0))
        
        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features.extend([0] * (self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
            
        return np.array(features)
```

### 3. Intelligent Seed Selector

**File**: `src/esper/services/tamiyo/seed_selector.py`

```python
class IntelligentSeedSelector:
    """Main seed selection interface with strategy management."""
    
    def __init__(self,
                 default_strategy: str = "ucb",
                 performance_tracker: Optional[PerformanceTracker] = None):
        self.performance_tracker = performance_tracker or PerformanceTracker()
        
        # Initialize available strategies
        self.strategies = {
            "ucb": UCBSelectionStrategy(),
            "thompson": ThompsonSamplingStrategy(),
            "epsilon_greedy": EpsilonGreedyStrategy(),
            "contextual": ContextualBanditStrategy()
        }
        
        self.current_strategy = self.strategies[default_strategy]
        self.selection_history = []
        
    def select_seed(self,
                   blueprint: Blueprint,
                   context: Optional[Dict[str, Any]] = None) -> Tuple[int, SeedData]:
        """
        Select the best seed for current context.
        
        Returns:
            Tuple of (seed_index, seed_data)
        """
        if not blueprint.seeds:
            raise ValueError("Blueprint has no seeds available")
            
        context = context or self._get_default_context()
        
        # Let strategy select seed index
        seed_idx = self.current_strategy.select_seed(
            blueprint.seeds,
            self.performance_tracker,
            context
        )
        
        # Record selection
        selected_seed = blueprint.seeds[seed_idx]
        self.selection_history.append({
            'timestamp': datetime.now(),
            'blueprint_id': blueprint.id,
            'seed_id': selected_seed.id,
            'seed_idx': seed_idx,
            'strategy': type(self.current_strategy).__name__,
            'context': context.copy()
        })
        
        logger.info(
            f"Selected seed {seed_idx} ({selected_seed.id}) "
            f"for blueprint {blueprint.id} using {type(self.current_strategy).__name__}"
        )
        
        return seed_idx, selected_seed
        
    def record_outcome(self,
                      seed_id: str,
                      blueprint_id: str,
                      performance_metrics: Dict[str, float],
                      success: bool = True) -> None:
        """Record the outcome of using a selected seed."""
        self.performance_tracker.record_performance(
            seed_id=seed_id,
            blueprint_id=blueprint_id,
            accuracy_delta=performance_metrics.get('accuracy_delta', 0),
            latency=performance_metrics.get('latency_ms', 0),
            memory=performance_metrics.get('memory_mb', 0),
            throughput=performance_metrics.get('throughput', 0),
            success=success
        )
        
    def switch_strategy(self, strategy_name: str) -> None:
        """Switch to a different selection strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        self.current_strategy = self.strategies[strategy_name]
        logger.info(f"Switched to {strategy_name} selection strategy")
        
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about seed selection."""
        stats = {
            'total_selections': len(self.selection_history),
            'unique_seeds_selected': len(set(
                h['seed_id'] for h in self.selection_history
            )),
            'strategy_usage': defaultdict(int),
            'top_performing_seeds': self.performance_tracker.get_top_performers()
        }
        
        for history in self.selection_history:
            stats['strategy_usage'][history['strategy']] += 1
            
        return stats
        
    def _get_default_context(self) -> Dict[str, Any]:
        """Get default context when none provided."""
        return {
            'current_loss': 0.0,
            'current_accuracy': 0.0,
            'learning_rate': 0.001,
            'epoch': 0,
            'gpu_memory_used': 0.0,
            'batch_size': 32,
            'loss_momentum': 0.0,
            'gradient_norm': 1.0
        }
```

### 4. Integration with Tamiyo Service

**File**: `src/esper/services/tamiyo/service.py` (updates)

```python
class TamiyoService:
    def __init__(self, config: TamiyoConfig):
        super().__init__(config)
        # Initialize intelligent seed selector
        self.seed_selector = IntelligentSeedSelector(
            default_strategy=config.selection_strategy,
            performance_tracker=PerformanceTracker(
                window_size=config.performance_window_size,
                decay_factor=config.performance_decay_factor
            )
        )
        
    async def make_decision(self, state: ModelState) -> AdaptationDecision:
        """Enhanced decision making with intelligent seed selection."""
        # Existing decision logic...
        
        if decision.action == "GRAFT":
            # Get blueprint for grafting
            blueprint = await self._get_blueprint(decision.blueprint_id)
            
            # Intelligently select seed
            context = self._extract_context_from_state(state)
            seed_idx, selected_seed = self.seed_selector.select_seed(
                blueprint, context
            )
            
            # Update decision with selected seed
            decision.metadata['seed_idx'] = seed_idx
            decision.metadata['seed_id'] = selected_seed.id
            decision.metadata['selection_reason'] = (
                f"Selected by {type(self.seed_selector.current_strategy).__name__}"
            )
            
        return decision
        
    async def record_adaptation_outcome(self,
                                      adaptation_id: str,
                                      outcome: AdaptationOutcome) -> None:
        """Record outcome for learning."""
        # Existing outcome recording...
        
        # Update seed selector with performance data
        if outcome.seed_id:
            self.seed_selector.record_outcome(
                seed_id=outcome.seed_id,
                blueprint_id=outcome.blueprint_id,
                performance_metrics={
                    'accuracy_delta': outcome.accuracy_improvement,
                    'latency_ms': outcome.execution_latency,
                    'memory_mb': outcome.memory_usage,
                    'throughput': outcome.throughput
                },
                success=outcome.success
            )
```

### 5. Testing Framework

**File**: `tests/services/tamiyo/test_seed_selection.py`

```python
class TestSeedSelection:
    def test_ucb_exploration(self):
        """Test UCB explores unvisited seeds."""
        selector = IntelligentSeedSelector(default_strategy="ucb")
        blueprint = create_test_blueprint(num_seeds=5)
        
        selected_seeds = []
        for _ in range(20):
            idx, seed = selector.select_seed(blueprint)
            selected_seeds.append(idx)
            
            # Simulate performance
            selector.record_outcome(
                seed.id, blueprint.id,
                {'accuracy_delta': np.random.randn() * 0.1},
                success=True
            )
            
        # Verify all seeds were explored
        assert len(set(selected_seeds)) == 5
        
    def test_performance_based_selection(self):
        """Test that high-performing seeds are selected more often."""
        selector = IntelligentSeedSelector(default_strategy="epsilon_greedy")
        blueprint = create_test_blueprint(num_seeds=3)
        
        # Assign different performance levels
        performance_levels = [0.1, 0.5, 0.9]  # Seed 2 is best
        
        for _ in range(100):
            idx, seed = selector.select_seed(blueprint)
            
            # Record performance based on seed
            selector.record_outcome(
                seed.id, blueprint.id,
                {'accuracy_delta': performance_levels[idx] + np.random.randn() * 0.01},
                success=True
            )
            
        # Check selection statistics
        stats = selector.get_selection_stats()
        selection_counts = defaultdict(int)
        
        for history in selector.selection_history:
            selection_counts[history['seed_idx']] += 1
            
        # Best seed should be selected most often
        assert selection_counts[2] > selection_counts[0]
        assert selection_counts[2] > selection_counts[1]
```

## Migration Plan

### 1. Update Existing Code
```python
# Before:
seed_idx = 0
active_seed = blueprint.seeds[seed_idx]

# After:
seed_idx, active_seed = self.tamiyo_client.seed_selector.select_seed(
    blueprint, 
    context=self._get_current_context()
)
```

### 2. Configuration Updates
```yaml
# config/tamiyo.yaml
seed_selection:
  default_strategy: "ucb"  # Options: ucb, thompson, epsilon_greedy, contextual
  performance_window_size: 100
  performance_decay_factor: 0.95
  min_samples_required: 5
  
  # Strategy-specific settings
  ucb:
    exploration_constant: 2.0
  epsilon_greedy:
    initial_epsilon: 0.1
    decay_rate: 0.995
  thompson:
    alpha_prior: 1.0
    beta_prior: 1.0
```

## Success Criteria

### Performance Metrics
- 15%+ improvement in adaptation effectiveness
- < 1ms selection latency
- Proper exploration of all seeds

### Selection Quality
- High-performing seeds selected 70%+ of time (after exploration)
- No seed starvation (all seeds tried at least once)
- Adaptation to changing performance characteristics

### System Integration
- Zero failures in seed selection
- Proper context extraction from model state
- Accurate performance tracking

## Risks and Mitigations

### Risk 1: Poor Initial Selection
**Mitigation**: High initial exploration rate, minimum sample requirements

### Risk 2: Context Feature Engineering
**Mitigation**: Start with simple features, iterate based on results

### Risk 3: Strategy Overfitting
**Mitigation**: Multiple strategies available, A/B testing capability

### Risk 4: Performance Tracking Overhead
**Mitigation**: Efficient data structures, async recording
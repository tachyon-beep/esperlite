"""Unit tests for grafting strategies."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from esper.morphogenetic_v2.grafting import (
    GraftingConfig,
    GraftingContext,
    LinearGrafting,
    DriftControlledGrafting,
    MomentumGrafting,
    AdaptiveGrafting,
    StabilityGrafting,
    create_grafting_strategy
)


class TestGraftingConfig:
    """Test grafting configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GraftingConfig()
        assert config.ramp_duration == 50
        assert config.drift_threshold == 0.01
        assert config.pause_duration == 10
        assert config.momentum_factor == 0.1
        assert config.stability_window == 5
        assert config.min_alpha == 0.0
        assert config.max_alpha == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GraftingConfig(
            ramp_duration=100,
            drift_threshold=0.05,
            min_alpha=0.1,
            max_alpha=0.9
        )
        assert config.ramp_duration == 100
        assert config.drift_threshold == 0.05
        assert config.min_alpha == 0.1
        assert config.max_alpha == 0.9


class TestGraftingContext:
    """Test grafting context."""
    
    def test_context_creation(self):
        """Test context creation."""
        context = GraftingContext(
            seed_id=42,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.5,
            metrics={'loss': 0.1}
        )
        assert context.seed_id == 42
        assert context.current_epoch == 10
        assert context.total_epochs == 100
        assert context.current_alpha == 0.5
        assert context.metrics['loss'] == 0.1
        assert context.model_weights is None
        assert context.blueprint_weights is None


class TestLinearGrafting:
    """Test LinearGrafting strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create linear strategy."""
        config = GraftingConfig(ramp_duration=100)
        return LinearGrafting(config)
    
    def test_linear_progression(self, strategy):
        """Test linear alpha progression."""
        # Test various epochs
        test_cases = [
            (0, 0.0),    # Start
            (25, 0.25),  # Quarter way
            (50, 0.5),   # Half way
            (75, 0.75),  # Three quarters
            (100, 1.0),  # Complete
            (150, 1.0)   # Beyond duration
        ]
        
        for epoch, expected_alpha in test_cases:
            context = GraftingContext(
                seed_id=1,
                current_epoch=epoch,
                total_epochs=200,
                current_alpha=0.0,
                metrics={}
            )
            alpha = strategy.compute_alpha(context)
            assert alpha == pytest.approx(expected_alpha)
    
    def test_alpha_clipping(self, strategy):
        """Test alpha value clipping."""
        strategy.config.min_alpha = 0.2
        strategy.config.max_alpha = 0.8
        
        # Test clipping at boundaries
        context_min = GraftingContext(
            seed_id=1,
            current_epoch=0,
            total_epochs=100,
            current_alpha=0.0,
            metrics={}
        )
        assert strategy.compute_alpha(context_min) == 0.2
        
        context_max = GraftingContext(
            seed_id=1,
            current_epoch=200,
            total_epochs=100,
            current_alpha=0.0,
            metrics={}
        )
        assert strategy.compute_alpha(context_max) == 0.8
    
    def test_reset(self, strategy):
        """Test reset does nothing for linear strategy."""
        strategy.reset()  # Should not raise


class TestDriftControlledGrafting:
    """Test DriftControlledGrafting strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create drift-controlled strategy."""
        config = GraftingConfig(
            ramp_duration=100,
            drift_threshold=0.01,
            pause_duration=5
        )
        return DriftControlledGrafting(config)
    
    def test_normal_progression(self, strategy):
        """Test progression without drift."""
        context = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.0,
            metrics={},
            model_weights=None  # No weights = no drift check
        )
        alpha = strategy.compute_alpha(context)
        assert alpha == 0.5
    
    def test_drift_detection_and_pause(self, strategy):
        """Test drift detection triggers pause."""
        # First call initializes EMA
        weights1 = torch.randn(10, 10)
        context1 = GraftingContext(
            seed_id=1,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.1,
            metrics={},
            model_weights=weights1
        )
        alpha1 = strategy.compute_alpha(context1)
        assert alpha1 == 0.1  # Normal progression
        
        # Large weight change triggers drift
        weights2 = weights1 + torch.randn(10, 10) * 5  # Large change
        context2 = GraftingContext(
            seed_id=1,
            current_epoch=20,
            total_epochs=100,
            current_alpha=0.2,
            metrics={},
            model_weights=weights2
        )
        alpha2 = strategy.compute_alpha(context2)
        assert alpha2 == 0.2  # Paused, returns current alpha
        assert strategy.pause_counter > 0
    
    def test_pause_countdown(self, strategy):
        """Test pause counter decreases."""
        strategy.pause_counter = 3
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.5,
            metrics={}
        )
        
        # First call
        alpha1 = strategy.compute_alpha(context)
        assert alpha1 == 0.5  # Paused
        assert strategy.pause_counter == 2
        
        # Second call
        alpha2 = strategy.compute_alpha(context)
        assert alpha2 == 0.5  # Still paused
        assert strategy.pause_counter == 1
        
        # Third call
        alpha3 = strategy.compute_alpha(context)
        assert alpha3 == 0.5  # Still paused
        assert strategy.pause_counter == 0
        
        # Fourth call - pause ended
        context.current_epoch = 20
        alpha4 = strategy.compute_alpha(context)
        assert alpha4 == 0.2  # Normal progression resumes
    
    def test_weight_history_tracking(self, strategy):
        """Test weight history is maintained."""
        weights = torch.randn(5, 5)
        context = GraftingContext(
            seed_id=1,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.1,
            metrics={},
            model_weights=weights
        )
        
        # Track weights over multiple epochs
        for i in range(10):
            context.model_weights = weights + torch.randn(5, 5) * 0.001
            strategy.on_epoch_end(context)
        
        # History should be limited to stability window
        assert len(strategy.weight_history) <= strategy.config.stability_window
    
    def test_reset_state(self, strategy):
        """Test reset clears all state."""
        strategy.pause_counter = 5
        strategy.weight_history = [torch.randn(5, 5)]
        strategy.weight_ema = torch.randn(5, 5)
        
        strategy.reset()
        
        assert strategy.pause_counter == 0
        assert len(strategy.weight_history) == 0
        assert strategy.weight_ema is None


class TestMomentumGrafting:
    """Test MomentumGrafting strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create momentum strategy."""
        config = GraftingConfig(
            ramp_duration=100,
            momentum_factor=0.5
        )
        return MomentumGrafting(config)
    
    def test_positive_momentum(self, strategy):
        """Test acceleration with positive performance."""
        context = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.4,
            metrics={'performance_delta': 0.1}  # Positive trend
        )
        
        alpha = strategy.compute_alpha(context)
        assert alpha > 0.5  # Accelerated beyond linear
        assert strategy.velocity > 0
    
    def test_negative_momentum(self, strategy):
        """Test deceleration with negative performance."""
        # Build up positive velocity first
        strategy.velocity = 0.05
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.5,
            metrics={'performance_delta': -0.2}  # Negative trend
        )
        
        alpha = strategy.compute_alpha(context)
        assert alpha < 0.55  # Slowed down
        assert strategy.velocity < 0.05  # Velocity reduced
    
    def test_alpha_bounds(self, strategy):
        """Test alpha stays within bounds."""
        # Large positive momentum
        strategy.velocity = 1.0
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=90,
            total_epochs=100,
            current_alpha=0.8,
            metrics={'performance_delta': 0.5}
        )
        
        alpha = strategy.compute_alpha(context)
        assert alpha <= 1.0  # Capped at max
        assert alpha >= 0.8  # Never goes backward
    
    def test_performance_history(self, strategy):
        """Test performance history tracking."""
        context = GraftingContext(
            seed_id=1,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.1,
            metrics={'performance_score': 0.8}
        )
        
        for i in range(10):
            context.metrics['performance_score'] = 0.8 + i * 0.01
            strategy.on_epoch_end(context)
        
        assert len(strategy.performance_history) <= strategy.config.stability_window


class TestAdaptiveGrafting:
    """Test AdaptiveGrafting strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create adaptive strategy."""
        config = GraftingConfig(ramp_duration=100)
        return AdaptiveGrafting(config)
    
    def test_strategy_switching(self, strategy):
        """Test strategy selection based on conditions."""
        # High drift -> drift strategy
        weights = torch.randn(10, 10)
        context_drift = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.5,
            metrics={},
            model_weights=weights
        )
        
        # Initialize drift strategy EMA
        strategy.strategies['drift'].weight_ema = weights
        
        # Large weight change
        context_drift.model_weights = weights + torch.randn(10, 10) * 5
        strategy._select_strategy(context_drift)
        assert strategy.current_strategy == 'drift'
        
        # Good performance -> momentum strategy
        context_momentum = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.5,
            metrics={'performance_delta': 0.02},
            model_weights=None  # No weights, so no drift check
        )
        strategy._select_strategy(context_momentum)
        assert strategy.current_strategy == 'momentum'
        
        # Default -> linear strategy
        # Reset drift strategy EMA to ensure no drift detected
        strategy.strategies['drift'].weight_ema = None
        context_linear = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.5,
            metrics={'performance_delta': 0.0},
            model_weights=None  # No weights = no drift check
        )
        strategy._select_strategy(context_linear)
        assert strategy.current_strategy == 'linear'
    
    def test_strategy_scoring(self, strategy):
        """Test strategy effectiveness tracking."""
        context = GraftingContext(
            seed_id=1,
            current_epoch=50,
            total_epochs=100,
            current_alpha=0.5,
            metrics={'performance_score': 0.8}
        )
        
        # Good performance increases score
        strategy.current_strategy = 'linear'
        initial_score = strategy.strategy_scores['linear']
        strategy._update_scores(context)
        assert strategy.strategy_scores['linear'] > initial_score
        
        # Bad performance decreases score
        # After normalization, score relationship might change
        context.metrics['performance_score'] = -0.5
        before_update = strategy.strategy_scores['linear']
        strategy._update_scores(context)
        # After normalization, linear's relative score should be lower
        assert strategy.strategy_scores['linear'] < before_update
    
    def test_reset_all_strategies(self, strategy):
        """Test reset resets all sub-strategies."""
        # Set some state in sub-strategies
        strategy.strategies['drift'].pause_counter = 5
        strategy.strategies['momentum'].velocity = 0.5
        strategy.current_strategy = 'momentum'
        
        strategy.reset()
        
        assert strategy.strategies['drift'].pause_counter == 0
        assert strategy.strategies['momentum'].velocity == 0.0
        assert strategy.current_strategy == 'linear'
        assert sum(strategy.strategy_scores.values()) == pytest.approx(1.0)


class TestStabilityGrafting:
    """Test StabilityGrafting strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create stability strategy."""
        config = GraftingConfig(
            ramp_duration=100,
            stability_window=5
        )
        return StabilityGrafting(config)
    
    def test_stability_checkpoints(self, strategy):
        """Test stability checkpoint creation."""
        # Stable context
        stable_context = GraftingContext(
            seed_id=1,
            current_epoch=5,  # Checkpoint epoch
            total_epochs=100,
            current_alpha=0.05,
            metrics={
                'loss_variance': 0.05,
                'gradient_norm': 5.0,
                'accuracy_variance': 0.02
            }
        )
        
        alpha = strategy.compute_alpha(stable_context)
        assert strategy.last_stable_alpha == stable_context.current_alpha
        assert strategy.instability_counter == 0
    
    def test_instability_detection(self, strategy):
        """Test instability detection and response."""
        # Unstable context
        unstable_context = GraftingContext(
            seed_id=1,
            current_epoch=10,  # Checkpoint epoch
            total_epochs=100,
            current_alpha=0.1,
            metrics={
                'loss_variance': 0.2,  # High variance
                'gradient_norm': 50.0,  # Exploding gradients
                'accuracy_variance': 0.1
            }
        )
        
        alpha = strategy.compute_alpha(unstable_context)
        assert strategy.instability_counter == 1
        
        # Multiple instabilities trigger reversion
        for i in range(2):
            unstable_context.current_epoch += 5
            strategy.compute_alpha(unstable_context)
        
        assert strategy.instability_counter >= 3
        
        # Next call should revert
        unstable_context.current_epoch += 5
        strategy.last_stable_alpha = 0.05  # Set a stable point
        alpha = strategy.compute_alpha(unstable_context)
        assert alpha == 0.05  # Reverted to stable
    
    def test_progression_slowdown(self, strategy):
        """Test progression slows after instability."""
        strategy.instability_counter = 1  # Some instability
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=52,  # Not a stability check epoch
            total_epochs=100,
            current_alpha=0.4,
            metrics={}
        )
        
        alpha = strategy.compute_alpha(context)
        assert alpha == 0.26  # Half speed (52 / 100 * 0.5)
    
    def test_stability_criteria(self, strategy):
        """Test various stability criteria."""
        context = GraftingContext(
            seed_id=1,
            current_epoch=5,
            total_epochs=100,
            current_alpha=0.05,
            metrics={}
        )
        
        # All criteria missing = stable (defaults)
        assert strategy._check_stability(context) == True
        
        # Loss unstable
        context.metrics['loss_variance'] = 0.15
        assert strategy._check_stability(context) == False
        
        # Gradient unstable
        context.metrics['loss_variance'] = 0.05
        context.metrics['gradient_norm'] = 15.0
        assert strategy._check_stability(context) == False
        
        # Accuracy unstable
        context.metrics['gradient_norm'] = 5.0
        context.metrics['accuracy_variance'] = 0.1
        assert strategy._check_stability(context) == False


class TestGraftingStrategyFactory:
    """Test strategy factory function."""
    
    def test_create_all_strategies(self):
        """Test creating all available strategies."""
        strategies = [
            'linear',
            'drift_controlled',
            'momentum',
            'adaptive',
            'stability'
        ]
        
        for name in strategies:
            strategy = create_grafting_strategy(name)
            assert strategy is not None
            assert hasattr(strategy, 'compute_alpha')
            assert hasattr(strategy, 'reset')
    
    def test_custom_config(self):
        """Test creating strategy with custom config."""
        config = GraftingConfig(ramp_duration=200)
        strategy = create_grafting_strategy('linear', config)
        assert strategy.config.ramp_duration == 200
    
    def test_invalid_strategy_name(self):
        """Test error on invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown grafting strategy"):
            create_grafting_strategy('invalid_strategy')


class TestGraftingScenarios:
    """Test complete grafting scenarios."""
    
    def test_full_grafting_lifecycle(self):
        """Test strategy through complete grafting process."""
        config = GraftingConfig(ramp_duration=20)
        strategy = LinearGrafting(config)
        
        # Simulate full grafting
        alphas = []
        for epoch in range(25):
            context = GraftingContext(
                seed_id=1,
                current_epoch=epoch,
                total_epochs=25,
                current_alpha=alphas[-1] if alphas else 0.0,
                metrics={'performance': 0.8}
            )
            
            alpha = strategy.compute_alpha(context)
            alphas.append(alpha)
            
            # Check abort condition
            assert not strategy.should_abort(context)
        
        # Verify progression
        assert alphas[0] == 0.0
        assert alphas[10] == 0.5
        assert alphas[20] == 1.0
        assert alphas[24] == 1.0
    
    def test_abort_on_high_errors(self):
        """Test grafting aborts on high error rate."""
        strategy = LinearGrafting(GraftingConfig())
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=10,
            total_epochs=100,
            current_alpha=0.1,
            metrics={'error_rate': 0.6}  # High error rate
        )
        
        assert strategy.should_abort(context) == True
    
    def test_adaptive_strategy_evolution(self):
        """Test adaptive strategy changes over time."""
        config = GraftingConfig(ramp_duration=50)
        strategy = AdaptiveGrafting(config)
        
        # Phase 1: Linear progression
        for epoch in range(10):
            context = GraftingContext(
                seed_id=1,
                current_epoch=epoch,
                total_epochs=100,
                current_alpha=epoch * 0.02,
                metrics={'performance_delta': 0.0}
            )
            strategy.compute_alpha(context)
            strategy.on_epoch_end(context)
        
        assert strategy.current_strategy == 'linear'
        
        # Phase 2: Good performance -> momentum
        for epoch in range(10, 20):
            context = GraftingContext(
                seed_id=1,
                current_epoch=epoch,
                total_epochs=100,
                current_alpha=epoch * 0.02,
                metrics={
                    'performance_delta': 0.05,
                    'performance_score': 0.9
                }
            )
            strategy.compute_alpha(context)
            strategy.on_epoch_end(context)
        
        assert strategy.current_strategy == 'momentum'
        
        # Phase 3: Weight drift -> drift control
        weights = torch.randn(10, 10)
        strategy.strategies['drift'].weight_ema = weights
        
        context = GraftingContext(
            seed_id=1,
            current_epoch=25,
            total_epochs=100,
            current_alpha=0.5,
            metrics={},
            model_weights=weights + torch.randn(10, 10) * 2
        )
        alpha = strategy.compute_alpha(context)
        assert strategy.current_strategy == 'drift'
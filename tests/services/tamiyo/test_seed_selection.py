"""
Comprehensive tests for intelligent seed selection system.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from esper.services.tamiyo.performance_tracker import PerformanceDelta
from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.services.tamiyo.performance_tracker import SeedPerformanceMetrics
from esper.services.tamiyo.seed_selector import EpsilonGreedyStrategy
from esper.services.tamiyo.seed_selector import PerformanceWeightedStrategy
from esper.services.tamiyo.seed_selector import SeedSelector
from esper.services.tamiyo.seed_selector import SelectionContext
from esper.services.tamiyo.seed_selector import SelectionStrategy
from esper.services.tamiyo.seed_selector import ThompsonSamplingStrategy
from esper.services.tamiyo.seed_selector import UCBStrategy


class TestPerformanceTracker:
    """Test performance tracking system."""

    @pytest.mark.asyncio
    async def test_update_metrics(self):
        """Test updating seed performance metrics."""
        tracker = PerformanceTracker(window_size=10)

        # Add some performance data
        delta = PerformanceDelta(
            accuracy_delta=0.05, loss_delta=-0.1, latency_ms=2.5, memory_mb=100
        )

        await tracker.update_metrics(
            layer_name="layer1",
            seed_idx=0,
            performance_delta=delta,
            kernel_id="kernel_123",
        )

        # Check metrics were recorded
        metrics = await tracker.get_layer_metrics("layer1")
        assert 0 in metrics
        assert metrics[0].total_activations == 1
        assert metrics[0].successful_grafts == 1
        assert metrics[0].accuracy_improvements == [0.05]
        assert metrics[0].kernel_id == "kernel_123"

    @pytest.mark.asyncio
    async def test_window_size_limit(self):
        """Test that window size is respected."""
        tracker = PerformanceTracker(window_size=5)

        # Add more data than window size
        for i in range(10):
            delta = PerformanceDelta(
                accuracy_delta=i * 0.01,
                loss_delta=-i * 0.01,
                latency_ms=i,
                memory_mb=100 + i,
            )
            await tracker.update_metrics(
                layer_name="layer1", seed_idx=0, performance_delta=delta
            )

        metrics = await tracker.get_layer_metrics("layer1")
        assert len(metrics[0].accuracy_improvements) == 5
        # Should keep the last 5 values
        assert metrics[0].accuracy_improvements == [0.05, 0.06, 0.07, 0.08, 0.09]

    @pytest.mark.asyncio
    async def test_score_computation(self):
        """Test UCB and Thompson score computation."""
        tracker = PerformanceTracker()

        # Add data for multiple seeds
        for seed_idx in range(3):
            for _ in range(seed_idx + 1):  # Different amounts of data
                delta = PerformanceDelta(
                    accuracy_delta=0.1 if seed_idx == 1 else 0.05,
                    loss_delta=-0.05,
                    latency_ms=1.0,
                    memory_mb=100,
                )
                await tracker.update_metrics(
                    layer_name="layer1", seed_idx=seed_idx, performance_delta=delta
                )

        metrics = await tracker.get_layer_metrics("layer1")

        # Seed 0: Less data, should have higher UCB score (exploration)
        # Seed 1: Better performance
        # Seed 2: More data but worse performance
        assert metrics[0].ucb_score > metrics[2].ucb_score  # Exploration bonus
        assert metrics[1].performance_score > metrics[0].performance_score


class TestSeedSelectionStrategies:
    """Test individual selection strategies."""

    def _create_test_metrics(self):
        """Create test metrics for 3 seeds."""
        metrics = {}

        # Seed 0: Good performance, well explored
        metrics[0] = SeedPerformanceMetrics(
            seed_idx=0,
            layer_name="test_layer",
            accuracy_improvements=[0.05, 0.06, 0.07, 0.05, 0.06],
            loss_reductions=[-0.1, -0.12, -0.08, -0.11, -0.09],
            total_activations=20,
            successful_grafts=15,
            failed_attempts=5,
        )
        metrics[0].ucb_score = 0.8
        metrics[0].thompson_score = 0.75
        metrics[0].performance_score = 0.7

        # Seed 1: Unexplored
        metrics[1] = SeedPerformanceMetrics(
            seed_idx=1, layer_name="test_layer", total_activations=0
        )
        metrics[1].ucb_score = float("inf")
        metrics[1].thompson_score = 0.5
        metrics[1].performance_score = 0.0

        # Seed 2: Poor performance
        metrics[2] = SeedPerformanceMetrics(
            seed_idx=2,
            layer_name="test_layer",
            accuracy_improvements=[0.01, 0.02, 0.01, 0.00, 0.01],
            loss_reductions=[-0.02, -0.01, -0.03, -0.01, -0.02],
            total_activations=10,
            successful_grafts=3,
            failed_attempts=7,
        )
        metrics[2].ucb_score = 0.4
        metrics[2].thompson_score = 0.3
        metrics[2].performance_score = 0.2

        return metrics

    def test_ucb_strategy(self):
        """Test UCB selection strategy."""
        strategy = UCBStrategy(exploration_constant=2.0)
        metrics = self._create_test_metrics()
        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # Should select unexplored seed (infinite UCB score)
        selected = strategy.select([0, 1, 2], metrics, context)
        assert selected == 1

        # Without unexplored seed, should select best UCB
        selected = strategy.select([0, 2], metrics, context)
        assert selected == 0

    def test_thompson_sampling(self):
        """Test Thompson sampling strategy."""
        strategy = ThompsonSamplingStrategy()
        metrics = self._create_test_metrics()
        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # Run multiple times to check probabilistic behavior
        selections = []
        for _ in range(100):
            selected = strategy.select([0, 1, 2], metrics, context)
            selections.append(selected)

        # Should select all seeds at least once (probabilistic)
        assert len(set(selections)) > 1

    def test_epsilon_greedy(self):
        """Test epsilon-greedy strategy."""
        strategy = EpsilonGreedyStrategy(epsilon=0.2, decay_rate=0.99)
        metrics = self._create_test_metrics()
        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # Run multiple times
        explorations = 0
        selections = []
        for _ in range(100):
            selected = strategy.select([0, 1, 2], metrics, context)
            selections.append(selected)
            if strategy.last_choice == "explore":
                explorations += 1

        # Should have some exploration
        assert explorations > 0
        assert explorations < 100  # Not all exploration

        # Best seed (0) should be selected most when exploiting
        exploitation_selections = [
            s for i, s in enumerate(selections) if i >= explorations
        ]
        if exploitation_selections:
            assert (
                max(set(exploitation_selections), key=exploitation_selections.count)
                == 0
            )

    def test_performance_weighted(self):
        """Test performance-weighted strategy."""
        strategy = PerformanceWeightedStrategy(temperature=1.0)
        metrics = self._create_test_metrics()
        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
            urgency=0.8,
        )

        # With temperature, should have probabilistic selection
        selections = []
        for _ in range(100):
            selected = strategy.select([0, 1, 2], metrics, context)
            selections.append(selected)

        # Seed 0 should be selected most often (best performance)
        assert selections.count(0) > selections.count(1)
        assert selections.count(0) > selections.count(2)


class TestSeedSelector:
    """Test main seed selector framework."""

    @pytest.mark.asyncio
    async def test_basic_selection(self):
        """Test basic seed selection flow."""
        selector = SeedSelector(
            strategy=SelectionStrategy.UCB, config={"fallback_to_zero": True}
        )

        context = SelectionContext(
            current_epoch=5,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # First selection should explore
        seed_idx, reason = await selector.select_seed(
            layer_name="layer1", available_seeds=[0, 1, 2, 3], context=context
        )

        assert seed_idx in [0, 1, 2, 3]
        assert reason.strategy == "UCBStrategy"
        assert reason.alternatives_considered == 4

    @pytest.mark.asyncio
    async def test_viability_filtering(self):
        """Test seed viability filtering."""
        tracker = PerformanceTracker()
        selector = SeedSelector(
            strategy=SelectionStrategy.UCB,
            performance_tracker=tracker,
            config={"min_dormant_epochs": 5, "max_concurrent_seeds": 2},
        )

        # Add some history
        for i in range(3):
            delta = PerformanceDelta(
                accuracy_delta=0.05, loss_delta=-0.05, latency_ms=1.0, memory_mb=100
            )
            await tracker.update_metrics(
                layer_name="layer1", seed_idx=i, performance_delta=delta
            )
            await tracker.record_selection("layer1", i, epoch=i)

        context = SelectionContext(
            current_epoch=4,  # Too soon for reactivation
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # Seeds 0-2 were recently active, only 3 should be viable
        seed_idx, _ = await selector.select_seed(
            layer_name="layer1",
            available_seeds=[0, 1, 2, 3],
            context=context,
            active_seeds=[],
        )

        assert seed_idx == 3  # Only viable option

    @pytest.mark.asyncio
    async def test_memory_constraints(self):
        """Test memory-based filtering."""
        tracker = PerformanceTracker()

        # Add high memory usage for seed 0
        for _ in range(5):
            delta = PerformanceDelta(
                accuracy_delta=0.1,
                loss_delta=-0.1,
                latency_ms=1.0,
                memory_mb=900,  # High memory usage
            )
            await tracker.update_metrics(
                layer_name="layer1", seed_idx=0, performance_delta=delta
            )

        selector = SeedSelector(
            strategy=SelectionStrategy.PERFORMANCE_WEIGHTED, performance_tracker=tracker
        )

        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,  # Limited memory
        )

        # Seed 0 should be filtered out due to memory
        seed_idx, _ = await selector.select_seed(
            layer_name="layer1", available_seeds=[0, 1], context=context
        )

        assert seed_idx == 1  # Should avoid high-memory seed

    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test fallback to seed 0."""
        selector = SeedSelector(
            strategy=SelectionStrategy.UCB,
            config={"fallback_to_zero": True, "max_concurrent_seeds": 1},
        )

        context = SelectionContext(
            current_epoch=10,
            total_epochs=100,
            current_loss=0.5,
            learning_rate=0.001,
            layer_type="Linear",
            available_memory_mb=1000,
        )

        # All seeds active except 0
        seed_idx, reason = await selector.select_seed(
            layer_name="layer1",
            available_seeds=[0, 1, 2],
            context=context,
            active_seeds=[1, 2],  # Max concurrent seeds reached
        )

        assert seed_idx == 0
        assert "falling back" in reason.reason


class TestIntegration:
    """Test integration with blueprint integration system."""

    @pytest.mark.asyncio
    async def test_performance_update_flow(self):
        """Test full flow of selection and performance update."""
        from esper.services.tamiyo.blueprint_integration import (
            ExecutionSystemIntegrator,
        )

        # Mock dependencies
        oona_client = MagicMock()
        oona_client.publish = MagicMock(return_value=asyncio.Future())
        oona_client.publish.return_value.set_result(None)

        integrator = ExecutionSystemIntegrator(
            oona_client=oona_client, urza_url="http://test"
        )

        # Select a seed
        seed_idx, _ = await integrator.select_seed_for_layer(
            layer_name="layer1", available_seeds=[0, 1, 2, 3]
        )

        assert seed_idx in [0, 1, 2, 3]

        # Update performance
        await integrator.update_performance_metrics(
            layer_name="layer1",
            seed_idx=seed_idx,
            accuracy_delta=0.05,
            loss_delta=-0.1,
            latency_ms=2.0,
            memory_mb=150,
        )

        # Check metrics were recorded
        metrics = await integrator.performance_tracker.get_layer_metrics("layer1")
        assert seed_idx in metrics
        assert metrics[seed_idx].total_activations == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

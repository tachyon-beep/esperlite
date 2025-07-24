"""
Tests for Seed Orchestrator - Phase B4 Dynamic Architecture Modification.

REFACTORED VERSION: Uses real components and tests actual functionality,
not implementation details.
"""

import pytest
import torch.nn as nn
from unittest.mock import AsyncMock
import time

from esper.core.seed_orchestrator import SeedOrchestrator, SeedStrategy
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.state_layout import SeedLifecycleState
from tests.helpers import create_valid_adaptation_decision
from tests.fixtures.test_infrastructure import (
    real_performance_tracker,
    real_blueprint_registry,
    real_seed_orchestrator_components,
)
from tests.fixtures.real_components import TestKernelFactory


class _TestModel(nn.Module):
    """Test model with Kasmina layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = KasminaLayer(128, 256, num_seeds=4, telemetry_enabled=False)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = KasminaLayer(512, 128, num_seeds=4, telemetry_enabled=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


@pytest.mark.real_components
class TestSeedOrchestratorWithRealComponents:
    """Test suite for seed orchestrator using real components."""

    @pytest.fixture
    def orchestrator(self, real_seed_orchestrator_components):
        """Create seed orchestrator instance with real components."""
        return SeedOrchestrator(**real_seed_orchestrator_components)

    @pytest.fixture
    def test_model(self):
        """Create test model with Kasmina layers."""
        return _TestModel()

    @pytest.fixture
    def kernel_factory(self):
        """Factory for creating real test kernels."""
        return TestKernelFactory()

    @pytest.mark.asyncio
    async def test_seed_performance_affects_modification_strategy(
        self, orchestrator, test_model, real_performance_tracker
    ):
        """Test that real performance metrics drive modification decisions."""
        # Set up diverse performance scenarios
        await real_performance_tracker.record_seed_metrics(
            "layer1", 0, {"accuracy_trend": 0.9, "loss_trend": 0.1, "efficiency": 0.95}
        )
        await real_performance_tracker.record_seed_metrics(
            "layer1", 1, {"accuracy_trend": 0.3, "loss_trend": 0.7, "efficiency": 0.4}
        )

        # Create decision to add seeds
        decision = create_valid_adaptation_decision(
            layer_name="layer1",
            adaptation_type="add_seed",
            confidence=0.85,
            urgency=0.7,
            parameters={"num_seeds": 2},
        )

        # Set up layer state - mark first 2 seeds as active
        layer = test_model.layer1
        layer.state_layout.transition_seed_state(
            0, SeedLifecycleState.ACTIVE, kernel_id=1
        )
        layer.state_layout.transition_seed_state(
            1, SeedLifecycleState.ACTIVE, kernel_id=2
        )
        layer.state_layout.alpha_blend[0] = 0.6
        layer.state_layout.alpha_blend[1] = 0.4

        seed_analysis = await orchestrator._analyze_seed_performance("layer1", layer)

        # Verify analysis reflects real metrics
        assert seed_analysis[0]["accuracy_score"] == 0.9
        assert seed_analysis[0]["composite_score"] > 0.8  # High performer
        assert seed_analysis[1]["accuracy_score"] == 0.3
        assert seed_analysis[1]["composite_score"] < 0.5  # Low performer

        # Create modification plan based on real analysis
        plan = orchestrator._create_modification_plan(decision, layer, seed_analysis)

        # Should target inactive seeds for diversification
        assert plan.strategy == SeedStrategy.DIVERSIFY
        assert 2 in plan.seed_modifications
        assert 3 in plan.seed_modifications
        # Seed 1 has composite score ~0.33 which is above threshold of 0.3, so not replaced

    @pytest.mark.asyncio
    async def test_modification_history_tracks_real_changes(
        self, orchestrator, test_model, real_performance_tracker
    ):
        """Test that modification history accurately reflects what happened."""
        # Set up underperforming layer
        for i in range(4):
            await real_performance_tracker.record_seed_metrics(
                "layer3",
                i,
                {"accuracy_trend": 0.4, "loss_trend": 0.6, "efficiency": 0.5},
            )

        decision = create_valid_adaptation_decision(
            layer_name="layer3",
            adaptation_type="optimize_parameters",
            confidence=0.9,
            urgency=0.8,
            reasoning="All seeds underperforming",
        )

        # Mock execution but let real planning happen
        orchestrator._execute_modification_plan = AsyncMock(return_value=True)

        success, details = await orchestrator.apply_architecture_modification(
            test_model, decision
        )

        assert success
        assert details["strategy"] == "replace"  # Should replace underperformers

        # Check real history
        assert len(orchestrator.modification_history) == 1
        record = orchestrator.modification_history[0]
        assert record["layer_name"] == "layer3"
        assert record["adaptation_type"] == "optimize_parameters"
        assert record["confidence"] == 0.9
        assert record["urgency"] == 0.8
        assert "reasoning" in record["metadata"]

    def test_different_adaptation_types_produce_different_strategies(
        self, orchestrator, test_model
    ):
        """Test that adaptation types correctly map to strategies."""
        layer = test_model.layer1

        # Mock seed analysis for consistent testing
        seed_analysis = {
            0: {"is_active": True, "composite_score": 0.7, "blend_factor": 0.5},
            1: {"is_active": True, "composite_score": 0.6, "blend_factor": 0.3},
            2: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            3: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
        }

        # Test each adaptation type
        adaptations = [
            ("add_seed", SeedStrategy.DIVERSIFY),
            ("remove_seed", SeedStrategy.SPECIALIZE),
            ("modify_architecture", SeedStrategy.ENSEMBLE),
            ("optimize_parameters", SeedStrategy.REPLACE),
        ]

        for adaptation_type, expected_strategy in adaptations:
            decision = create_valid_adaptation_decision(
                layer_name="layer1", adaptation_type=adaptation_type, confidence=0.8
            )

            plan = orchestrator._create_modification_plan(
                decision, layer, seed_analysis
            )
            assert plan.strategy == expected_strategy, f"Failed for {adaptation_type}"

    def test_cooldown_prevents_rapid_modifications(self, orchestrator):
        """Test that cooldown period is enforced."""
        # No previous modifications
        assert orchestrator.can_modify_layer("layer1", 10)

        # Simulate a modification at epoch 5
        orchestrator.last_modification_epoch["layer1"] = 5

        # Too soon (default cooldown is 5 epochs)
        assert not orchestrator.can_modify_layer("layer1", 8)
        assert not orchestrator.can_modify_layer("layer1", 9)

        # Exactly at cooldown
        assert orchestrator.can_modify_layer("layer1", 10)

        # After cooldown
        assert orchestrator.can_modify_layer("layer1", 11)

    def test_modification_stats_aggregate_correctly(self, orchestrator):
        """Test that stats are calculated from real history."""
        # Add real modification history
        orchestrator.modification_history = [
            {
                "timestamp": time.time(),
                "layer_name": "layer1",
                "strategy": "diversify",
                "success": True,
                "duration_ms": 120,
            },
            {
                "timestamp": time.time(),
                "layer_name": "layer2",
                "strategy": "specialize",
                "success": True,
                "duration_ms": 80,
            },
            {
                "timestamp": time.time(),
                "layer_name": "layer3",
                "strategy": "diversify",
                "success": False,
                "duration_ms": 50,
            },
        ]

        stats = orchestrator.get_modification_stats()

        assert stats["total_modifications"] == 3
        assert stats["success_rate"] == pytest.approx(2 / 3)
        assert stats["strategies_used"]["diversify"] == 2
        assert stats["strategies_used"]["specialize"] == 1
        assert stats["avg_duration_ms"] == pytest.approx(250 / 3)

    @pytest.mark.asyncio
    async def test_ensemble_strategy_activates_all_seeds(
        self, orchestrator, test_model
    ):
        """Test that ensemble strategy properly plans to activate all seeds."""
        layer = test_model.layer1

        # Set up mixed active/inactive seeds
        seed_analysis = {
            0: {"is_active": True, "composite_score": 0.7, "blend_factor": 0.6},
            1: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            2: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            3: {"is_active": True, "composite_score": 0.6, "blend_factor": 0.4},
        }

        decision = create_valid_adaptation_decision(
            layer_name="layer1",
            adaptation_type="modify_architecture",  # Maps to ensemble
            confidence=0.8,
        )

        plan = orchestrator._create_ensemble_plan(layer, seed_analysis, decision)

        # Should plan to activate all seeds
        assert len(plan.seed_modifications) == 4

        # Inactive seeds should be loaded
        assert plan.seed_modifications[1]["action"] == "load_ensemble_kernel"
        assert plan.seed_modifications[2]["action"] == "load_ensemble_kernel"

        # Active seeds should have blend adjusted
        assert plan.seed_modifications[0]["action"] == "adjust_blend"
        assert plan.seed_modifications[3]["action"] == "adjust_blend"

        # All should have equal blend
        target_blend = 0.25
        for i in [0, 3]:
            assert plan.seed_modifications[i]["new_blend"] == pytest.approx(
                target_blend
            )


@pytest.mark.integration
class TestSeedOrchestratorIntegration:
    """Integration tests with multiple real components."""

    @pytest.mark.asyncio
    async def test_performance_driven_adaptation_cycle(
        self, real_seed_orchestrator_components, real_performance_tracker
    ):
        """Test complete adaptation cycle with real performance feedback."""
        orchestrator = SeedOrchestrator(**real_seed_orchestrator_components)
        model = _TestModel()

        # Simulate performance degradation
        for epoch in range(5):
            # Record worsening metrics
            for seed_idx in range(2):
                await real_performance_tracker.record_seed_metrics(
                    "layer1",
                    seed_idx,
                    {
                        "accuracy_trend": 0.7 - epoch * 0.1,
                        "loss_trend": 0.3 + epoch * 0.1,
                        "efficiency": 0.8 - epoch * 0.05,
                    },
                )

        # Trigger adaptation based on poor performance
        decision = create_valid_adaptation_decision(
            layer_name="layer1",
            adaptation_type="add_seed",
            confidence=0.9,
            urgency=0.85,
            reasoning="Performance degradation detected",
        )

        # Mock only external calls
        orchestrator._execute_modification_plan = AsyncMock(return_value=True)

        success, details = await orchestrator.apply_architecture_modification(
            model, decision
        )

        assert success
        assert details["strategy"] == "diversify"
        assert details["expected_improvement"] > 0

        # Verify history reflects the cycle
        history = real_performance_tracker.get_history()
        assert len(history) == 10  # 5 epochs * 2 seeds

        # Check performance degradation was captured
        first_metric = history[0]["metrics"]["accuracy_trend"]
        last_metric = history[-1]["metrics"]["accuracy_trend"]
        assert last_metric < first_metric  # Performance got worse

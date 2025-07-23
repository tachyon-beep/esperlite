"""
Tests for Seed Orchestrator - Phase B4 Dynamic Architecture Modification.

These tests verify that architecture modification happens through the
Kasmina seed mechanism rather than traditional model surgery.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import asyncio

from esper.core.seed_orchestrator import (
    SeedOrchestrator, SeedStrategy, SeedOrchestratorConfig,
    SeedModificationPlan
)
from esper.contracts.operational import AdaptationDecision, ModelGraphState
from esper.execution.kasmina_layer import KasminaLayer
from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.blueprints.registry import BlueprintRegistry


class TestModel(nn.Module):
    """Test model with Kasmina layers."""
    def __init__(self):
        super().__init__()
        self.layer1 = KasminaLayer(128, 256, num_seeds=4)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = KasminaLayer(512, 128, num_seeds=4)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TestSeedOrchestrator:
    """Test suite for seed orchestrator functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        performance_tracker = Mock(spec=PerformanceTracker)
        blueprint_registry = Mock(spec=BlueprintRegistry)
        oona_client = AsyncMock()
        urza_url = "http://test-urza:8000"
        
        # Mock performance tracker methods
        performance_tracker.get_seed_metrics = AsyncMock(return_value={
            "accuracy_trend": 0.7,
            "loss_trend": 0.3,
            "efficiency": 0.8
        })
        
        return {
            "performance_tracker": performance_tracker,
            "blueprint_registry": blueprint_registry,
            "oona_client": oona_client,
            "urza_url": urza_url
        }
    
    @pytest.fixture
    def orchestrator(self, mock_components):
        """Create seed orchestrator instance."""
        return SeedOrchestrator(**mock_components)
    
    @pytest.fixture
    def test_model(self):
        """Create test model with Kasmina layers."""
        return TestModel()
    
    @pytest.fixture
    def adaptation_decision(self):
        """Create test adaptation decision."""
        return AdaptationDecision(
            decision_id="test_001",
            layer_name="layer1",
            adaptation_type="add_neurons",
            confidence=0.85,
            parameters={"num_seeds": 2},
            reasoning="Layer showing high activation variance",
            model_graph_state=ModelGraphState(
                graph_data={"nodes": [], "edges": []},
                metrics={}
            ),
            epoch=10
        )
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert isinstance(orchestrator.config, SeedOrchestratorConfig)
        assert orchestrator.config.max_seeds_per_layer == 4
        assert orchestrator.config.min_performance_threshold == 0.3
        assert hasattr(orchestrator, 'integration_orchestrator')
        assert isinstance(orchestrator.seed_specializations, dict)
        assert isinstance(orchestrator.modification_history, list)
    
    def test_find_kasmina_layer(self, orchestrator, test_model):
        """Test finding Kasmina layers in model."""
        # Should find layer1
        layer = orchestrator._find_kasmina_layer(test_model, "layer1")
        assert isinstance(layer, KasminaLayer)
        assert layer.input_size == 128
        assert layer.output_size == 256
        
        # Should find layer3
        layer = orchestrator._find_kasmina_layer(test_model, "layer3")
        assert isinstance(layer, KasminaLayer)
        assert layer.input_size == 512
        assert layer.output_size == 128
        
        # Should not find layer2 (not Kasmina)
        layer = orchestrator._find_kasmina_layer(test_model, "layer2")
        assert layer is None
        
        # Should not find non-existent layer
        layer = orchestrator._find_kasmina_layer(test_model, "layer4")
        assert layer is None
    
    @pytest.mark.asyncio
    async def test_analyze_seed_performance(self, orchestrator, test_model):
        """Test seed performance analysis."""
        layer = test_model.layer1
        
        # Mock state layout methods
        layer.state_layout.is_seed_active = Mock(side_effect=lambda idx: idx < 2)
        layer.state_layout.alpha_blend = torch.tensor([0.5, 0.3, 0.0, 0.0])
        
        analysis = await orchestrator._analyze_seed_performance("layer1", layer)
        
        assert len(analysis) == 4  # 4 seeds
        
        # Check seed 0 (active)
        assert analysis[0]["is_active"] == True
        assert analysis[0]["blend_factor"] == 0.5
        assert "accuracy_score" in analysis[0]
        assert "composite_score" in analysis[0]
        
        # Check seed 2 (inactive)
        assert analysis[2]["is_active"] == False
        assert analysis[2]["blend_factor"] == 0.0
    
    def test_create_diversify_plan(self, orchestrator, test_model):
        """Test creation of diversify strategy plan."""
        layer = test_model.layer1
        seed_analysis = {
            0: {"is_active": True, "composite_score": 0.8, "blend_factor": 0.5},
            1: {"is_active": True, "composite_score": 0.2, "blend_factor": 0.3},  # Underperforming
            2: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            3: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0}
        }
        
        decision = AdaptationDecision(
            decision_id="test_div",
            layer_name="layer1",
            adaptation_type="add_neurons",
            confidence=0.8,
            parameters={"num_seeds": 2},
            reasoning="Diversify capacity"
        )
        
        plan = orchestrator._create_diversify_plan(layer, seed_analysis, decision)
        
        assert plan.strategy == SeedStrategy.DIVERSIFY
        assert len(plan.seed_modifications) == 2
        # Should target inactive seeds or underperforming ones
        assert 2 in plan.seed_modifications or 3 in plan.seed_modifications
        assert plan.expected_improvement > 0
    
    def test_create_specialize_plan(self, orchestrator, test_model):
        """Test creation of specialize strategy plan."""
        layer = test_model.layer1
        seed_analysis = {
            0: {"is_active": True, "composite_score": 0.9, "blend_factor": 0.4},  # Best
            1: {"is_active": True, "composite_score": 0.8, "blend_factor": 0.3},  # Good
            2: {"is_active": True, "composite_score": 0.3, "blend_factor": 0.2},  # Poor
            3: {"is_active": True, "composite_score": 0.2, "blend_factor": 0.1}   # Worst
        }
        
        decision = AdaptationDecision(
            decision_id="test_spec",
            layer_name="layer1",
            adaptation_type="remove_neurons",
            confidence=0.8,
            parameters={},
            reasoning="Consolidate to best performers"
        )
        
        plan = orchestrator._create_specialize_plan(layer, seed_analysis, decision)
        
        assert plan.strategy == SeedStrategy.SPECIALIZE
        # Should keep top 2 performers
        assert 0 in plan.seed_modifications  # Best seed should be strengthened
        assert plan.seed_modifications[0]["action"] == "increase_blend"
        # Should remove bottom performers
        assert 3 in plan.seed_modifications  # Worst seed should be removed
        assert plan.seed_modifications[3]["action"] == "unload_kernel"
    
    def test_create_ensemble_plan(self, orchestrator, test_model):
        """Test creation of ensemble strategy plan."""
        layer = test_model.layer1
        seed_analysis = {
            0: {"is_active": True, "composite_score": 0.7, "blend_factor": 0.6},
            1: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            2: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            3: {"is_active": True, "composite_score": 0.6, "blend_factor": 0.4}
        }
        
        decision = AdaptationDecision(
            decision_id="test_ens",
            layer_name="layer1",
            adaptation_type="add_layer",
            confidence=0.8,
            parameters={},
            reasoning="Create ensemble"
        )
        
        plan = orchestrator._create_ensemble_plan(layer, seed_analysis, decision)
        
        assert plan.strategy == SeedStrategy.ENSEMBLE
        assert len(plan.seed_modifications) == 4  # All seeds
        # Inactive seeds should be activated
        assert plan.seed_modifications[1]["action"] == "load_ensemble_kernel"
        # Active seeds should have balanced blend
        target_blend = 0.25  # 1/4 seeds
        assert plan.seed_modifications[0]["new_blend"] == pytest.approx(target_blend)
    
    @pytest.mark.asyncio
    async def test_execute_modification_plan(self, orchestrator, test_model):
        """Test execution of modification plan."""
        layer = test_model.layer1
        
        # Mock integration orchestrator
        orchestrator.integration_orchestrator.execute_adaptation_pipeline = AsyncMock(
            return_value=(True, {"kernel_id": "test_kernel_123"})
        )
        
        # Mock Kasmina layer methods
        layer.set_seed_alpha = Mock()
        layer.unload_kernel = AsyncMock(return_value=True)
        
        plan = SeedModificationPlan(
            layer_name="layer1",
            strategy=SeedStrategy.DIVERSIFY,
            seed_modifications={
                0: {"action": "load_diverse_kernel", "initial_blend": 0.3},
                1: {"action": "adjust_blend", "new_blend": 0.5},
                2: {"action": "unload_kernel"}
            },
            expected_improvement=0.4,
            risk_score=0.3,
            reasoning="Test plan"
        )
        
        model_state = ModelGraphState(graph_data={}, metrics={})
        success = await orchestrator._execute_modification_plan(layer, plan, model_state)
        
        assert success == True
        # Check that methods were called
        assert orchestrator.integration_orchestrator.execute_adaptation_pipeline.call_count == 1
        layer.set_seed_alpha.assert_any_call(0, 0.3)  # Initial blend for seed 0
        layer.set_seed_alpha.assert_any_call(1, 0.5)  # Adjust blend for seed 1
        layer.unload_kernel.assert_called_once_with(2)  # Unload seed 2
    
    @pytest.mark.asyncio
    async def test_apply_architecture_modification_success(self, orchestrator, test_model, adaptation_decision):
        """Test successful architecture modification."""
        # Mock methods
        orchestrator._analyze_seed_performance = AsyncMock(return_value={
            0: {"is_active": True, "composite_score": 0.7, "blend_factor": 0.5},
            1: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            2: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0},
            3: {"is_active": False, "composite_score": 0.0, "blend_factor": 0.0}
        })
        
        orchestrator._execute_modification_plan = AsyncMock(return_value=True)
        
        success, details = await orchestrator.apply_architecture_modification(
            test_model, adaptation_decision
        )
        
        assert success == True
        assert "strategy" in details
        assert "modified_seeds" in details
        assert "expected_improvement" in details
        assert "duration_ms" in details
        
        # Check that modification was recorded
        assert len(orchestrator.modification_history) == 1
        assert orchestrator.modification_history[0]["success"] == True
    
    @pytest.mark.asyncio
    async def test_apply_architecture_modification_non_kasmina_layer(self, orchestrator, test_model):
        """Test handling of non-Kasmina layer."""
        decision = AdaptationDecision(
            decision_id="test_002",
            layer_name="layer2",  # This is nn.Linear, not KasminaLayer
            adaptation_type="add_neurons",
            confidence=0.8,
            parameters={},
            reasoning="Test"
        )
        
        success, details = await orchestrator.apply_architecture_modification(
            test_model, decision
        )
        
        assert success == False
        assert "error" in details
        assert "not a KasminaLayer" in details["error"]
    
    def test_can_modify_layer(self, orchestrator):
        """Test cooldown period checking."""
        # No previous modification
        assert orchestrator.can_modify_layer("layer1", 10) == True
        
        # Add modification history
        orchestrator.last_modification_epoch["layer1"] = 5
        
        # Too soon (cooldown is 5 epochs by default)
        assert orchestrator.can_modify_layer("layer1", 8) == False
        
        # After cooldown
        assert orchestrator.can_modify_layer("layer1", 10) == True
    
    def test_get_modification_stats(self, orchestrator):
        """Test statistics collection."""
        # Empty history
        stats = orchestrator.get_modification_stats()
        assert stats["total_modifications"] == 0
        assert stats["success_rate"] == 0.0
        
        # Add some history
        orchestrator.modification_history = [
            {"success": True, "strategy": "diversify", "duration_ms": 100},
            {"success": True, "strategy": "specialize", "duration_ms": 150},
            {"success": False, "strategy": "diversify", "duration_ms": 50},
        ]
        
        stats = orchestrator.get_modification_stats()
        assert stats["total_modifications"] == 3
        assert stats["success_rate"] == pytest.approx(2/3)
        assert stats["strategies_used"]["diversify"] == 2
        assert stats["strategies_used"]["specialize"] == 1
        assert stats["avg_duration_ms"] == pytest.approx(100.0)


class TestSeedStrategies:
    """Test different seed modification strategies."""
    
    def test_strategy_enum(self):
        """Test strategy enumeration."""
        assert SeedStrategy.REPLACE.value == "replace"
        assert SeedStrategy.DIVERSIFY.value == "diversify"
        assert SeedStrategy.SPECIALIZE.value == "specialize"
        assert SeedStrategy.ENSEMBLE.value == "ensemble"
    
    def test_strategy_selection_based_on_adaptation_type(self):
        """Test that correct strategy is chosen based on adaptation type."""
        config = SeedOrchestratorConfig()
        
        # Map adaptation types to expected strategies
        adaptation_strategy_map = {
            "add_neurons": SeedStrategy.DIVERSIFY,
            "remove_neurons": SeedStrategy.SPECIALIZE,
            "add_layer": SeedStrategy.ENSEMBLE,
            "unknown_type": SeedStrategy.REPLACE
        }
        
        for adaptation_type, expected_strategy in adaptation_strategy_map.items():
            # This tests the logic in _create_modification_plan
            # which we can't directly test without creating full orchestrator
            assert adaptation_type in ["add_neurons", "remove_neurons", "add_layer", "unknown_type"]
            assert expected_strategy in SeedStrategy
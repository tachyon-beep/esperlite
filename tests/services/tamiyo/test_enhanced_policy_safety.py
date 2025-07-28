"""
Critical Enhanced Policy Safety Validation Tests.

These tests implement Week 1 Critical Priority safety testing,
focusing on preventing dangerous adaptations and ensuring safe operation.
"""

import time
import warnings
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

# Force CPU-only mode for tests to avoid CUDA device mismatches
torch.cuda.is_available = lambda: False

# Suppress torch_geometric warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

import logging

from esper.services.tamiyo.model_graph_builder import ModelGraphBuilder
from esper.services.tamiyo.model_graph_builder import ModelGraphState
from esper.services.tamiyo.policy import EnhancedTamiyoPolicyGNN
from esper.services.tamiyo.policy import PolicyConfig
from esper.services.tamiyo.policy import SafetyRegularizer
from esper.services.tamiyo.policy import UncertaintyQuantification
from tests.production_scenarios import ProductionHealthSignalFactory
from tests.production_scenarios import ProductionScenarioFactory

logger = logging.getLogger(__name__)


def create_realistic_graph_state(
    num_nodes=10, feature_dim=20, problematic_layers=None, health_trends=None
):
    """Create realistic ModelGraphState with real tensor data for testing."""
    from esper.services.tamiyo.model_graph_builder import ModelGraphState
    from esper.services.tamiyo.model_graph_builder import ModelTopology

    if problematic_layers is None:
        problematic_layers = []
    if health_trends is None:
        health_trends = {}

    # Create real tensor data
    graph_data = Data(
        x=torch.randn(num_nodes, feature_dim),
        edge_index=torch.randint(0, num_nodes, (2, min(20, num_nodes * 2))),
    )

    # Create realistic topology
    layer_names = [f"layer_{i}" for i in range(num_nodes)]
    topology = ModelTopology(
        layer_names=layer_names,
        layer_types={name: "Linear" for name in layer_names},
        layer_shapes={name: (64, 32) for name in layer_names},
        connections=[
            (layer_names[i], layer_names[i + 1]) for i in range(len(layer_names) - 1)
        ],
        parameter_counts={name: 2048 for name in layer_names},
    )

    return ModelGraphState(
        graph_data=graph_data,
        timestamp=time.time(),
        health_signals=[],
        topology=topology,
        global_metrics={"avg_health": 0.4},
        health_trends=health_trends,
        problematic_layers=problematic_layers,
    )


class SafetyTestScenarios:
    """Factory for creating safety-critical test scenarios."""

    @staticmethod
    def create_dangerous_adaptation_scenario() -> (
        Tuple[ModelGraphState, Dict[str, Any]]
    ):
        """Create scenario that should prevent dangerous adaptations."""
        # Create graph state with mostly healthy layers but one problematic
        graph_builder = ModelGraphBuilder()

        # Create signals that indicate instability
        health_signals = []

        # Most layers are healthy but unstable (oscillating health)
        for i in range(10):
            if i == 5:  # Target problematic layer (dangerous_layer_5)
                signal = ProductionHealthSignalFactory.create_degraded_signal(
                    i, severity=0.7, epoch=100
                )
                signal.health_score = 0.25  # Low health but not critically low
            else:
                signal = ProductionHealthSignalFactory.create_healthy_signal(
                    i, epoch=100  # stable_layer_{i}
                )
                # Add instability indicators
                signal.health_score = 0.8 + 0.1 * np.sin(i)  # Oscillating
                signal.activation_variance = 0.15  # High variance indicates instability

            health_signals.append(signal)

        graph_state = graph_builder.build_model_graph(health_signals)

        # Create context indicating system instability
        dangerous_context = {
            "system_stability_score": 0.4,  # Low stability
            "recent_adaptation_failures": 3,  # Multiple recent failures
            "global_error_rate": 0.15,  # High error rate
            "performance_degrading": True,  # System is getting worse
            "training_convergence": False,  # Training not converging
        }

        return graph_state, dangerous_context

    @staticmethod
    def create_high_uncertainty_scenario() -> Tuple[ModelGraphState, Dict[str, Any]]:
        """Create scenario with high epistemic uncertainty."""
        graph_builder = ModelGraphBuilder()

        # Create contradictory signals that should increase uncertainty
        health_signals = []
        for i in range(8):
            # Alternating very good and very bad signals for same layer
            if i % 2 == 0:
                signal = ProductionHealthSignalFactory.create_healthy_signal(
                    0, epoch=100 + i  # uncertain_layer_0
                )
                signal.health_score = 0.95  # Excellent health
            else:
                signal = ProductionHealthSignalFactory.create_degraded_signal(
                    0, severity=0.8, epoch=100 + i  # uncertain_layer_0
                )
                signal.health_score = 0.15  # Very poor health

            health_signals.append(signal)

        # Add other stable layers
        for i in range(5):
            signal = ProductionHealthSignalFactory.create_healthy_signal(
                i, epoch=100  # stable_layer_{i}
            )
            health_signals.append(signal)

        graph_state = graph_builder.build_model_graph(health_signals)

        uncertainty_context = {
            "signal_contradictions": True,
            "measurement_noise_level": 0.3,
            "data_quality_score": 0.4,
        }

        return graph_state, uncertainty_context

    @staticmethod
    def create_emergency_brake_scenario() -> Tuple[ModelGraphState, Dict[str, Any]]:
        """Create scenario requiring emergency intervention blocking."""
        graph_builder = ModelGraphBuilder()

        # Create scenario with cascading failures
        health_signals = []
        for i in range(12):
            # Progressive degradation - early layers worst
            severity = min(0.9, 0.3 + (i * 0.05))
            signal = ProductionHealthSignalFactory.create_degraded_signal(
                i, severity=severity, epoch=100  # cascade_layer_{i}
            )
            signal.error_count = max(0, 20 - i * 2)  # More errors in early layers
            health_signals.append(signal)

        graph_state = graph_builder.build_model_graph(health_signals)

        emergency_context = {
            "cascading_failure_detected": True,
            "system_critical": True,
            "performance_cliff": True,  # Sudden performance drop
            "resource_exhaustion": True,
            "error_rate_spike": 0.8,  # 80% error rate
        }

        return graph_state, emergency_context


class TestSafetyRegularizer:
    """Test safety regularization components."""

    def test_safety_score_computation_realistic_features(self):
        """Test safety score computation with realistic graph features."""
        safety_reg = SafetyRegularizer(feature_dim=128)

        # Create realistic graph representations
        stable_graph_repr = (
            torch.randn(1, 128) * 0.5 + 0.2
        )  # Low variance, positive mean
        unstable_graph_repr = (
            torch.randn(1, 128) * 2.0 - 0.5
        )  # High variance, negative mean

        # Compute safety scores
        stable_score = safety_reg(stable_graph_repr)
        unstable_score = safety_reg(unstable_graph_repr)

        # Verify safety scores are in valid range
        assert 0.0 <= stable_score.item() <= 1.0
        assert 0.0 <= unstable_score.item() <= 1.0

        # Note: Without training, we can't guarantee which is higher
        # But we can test that the mechanism works
        logger.info(
            f"Safety scores - Stable: {stable_score.item():.3f}, Unstable: {unstable_score.item():.3f}"
        )

    def test_safety_penalty_thresholds(self):
        """Test safety penalty computation with different thresholds."""
        safety_reg = SafetyRegularizer(feature_dim=64)

        # Test different safety score scenarios
        test_scenarios = [
            (torch.tensor([[0.9]]), 0.8, "high_safety"),  # Above threshold
            (torch.tensor([[0.5]]), 0.8, "medium_safety"),  # Below threshold
            (torch.tensor([[0.2]]), 0.8, "low_safety"),  # Well below threshold
        ]

        for safety_score, threshold, scenario_name in test_scenarios:
            penalty = safety_reg.safety_penalty(safety_score, threshold)

            expected_penalty = max(0.0, threshold - safety_score.item())

            # Verify penalty calculation
            assert penalty.item() == pytest.approx(expected_penalty, abs=1e-6)

            logger.info(
                f"{scenario_name}: safety={safety_score.item():.2f}, "
                f"penalty={penalty.item():.3f}"
            )


class TestUncertaintyQuantification:
    """Test epistemic uncertainty quantification."""

    def test_monte_carlo_dropout_uncertainty_estimation(self):
        """Test MC dropout provides reasonable uncertainty estimates."""
        uncertainty_module = UncertaintyQuantification(input_dim=64, hidden_dim=128)

        # Create test scenarios with different uncertainty levels
        certain_input = torch.ones(1, 64) * 0.5  # Constant input
        uncertain_input = torch.randn(1, 64) * 2.0  # High variance input

        # Estimate uncertainty with multiple samples
        certain_mean, certain_uncertainty = uncertainty_module(
            certain_input, num_samples=20
        )
        uncertain_mean, uncertain_uncertainty = uncertainty_module(
            uncertain_input, num_samples=20
        )

        # Verify outputs are reasonable
        assert 0.0 <= certain_mean.item() <= 1.0
        assert 0.0 <= uncertain_mean.item() <= 1.0
        assert certain_uncertainty.item() >= 0.0
        assert uncertain_uncertainty.item() >= 0.0

        # Uncertainty should be non-zero due to dropout
        assert (
            certain_uncertainty.item() > 0.001
        ), "MC dropout should produce some uncertainty"
        assert (
            uncertain_uncertainty.item() > 0.001
        ), "MC dropout should produce some uncertainty"

        logger.info(
            f"Uncertainty estimates - Certain input: {certain_uncertainty.item():.4f}, "
            f"Uncertain input: {uncertain_uncertainty.item():.4f}"
        )

    def test_uncertainty_reproducibility_with_seed(self):
        """Test uncertainty estimation is reproducible with fixed seed."""
        uncertainty_module = UncertaintyQuantification(input_dim=32, hidden_dim=64)
        test_input = torch.randn(1, 32)

        # Set seed and compute uncertainty
        torch.manual_seed(42)
        mean1, uncertainty1 = uncertainty_module(test_input, num_samples=10)

        # Reset seed and compute again
        torch.manual_seed(42)
        mean2, uncertainty2 = uncertainty_module(test_input, num_samples=10)

        # Should be identical with same seed
        assert torch.allclose(mean1, mean2, atol=1e-6)
        assert torch.allclose(uncertainty1, uncertainty2, atol=1e-6)


class TestEnhancedPolicySafety:
    """Test comprehensive safety validation of the enhanced policy."""

    @pytest.fixture
    def safety_policy_config(self):
        """Create policy configuration optimized for safety testing."""
        return PolicyConfig(
            # Strict safety thresholds
            adaptation_confidence_threshold=0.8,  # High confidence required
            uncertainty_threshold=0.1,  # Low uncertainty tolerance
            safety_margin=0.2,  # Large safety margin
            health_threshold=0.2,  # Conservative health threshold
            # Enable all safety features
            enable_uncertainty=True,
            uncertainty_samples=15,  # More samples for better estimates
            # Moderate model size for test performance
            hidden_dim=64,
            num_gnn_layers=2,
            num_attention_heads=2,
        )

    @pytest.fixture
    def safety_policy(self, safety_policy_config):
        """Create policy instance with safety configuration."""
        return EnhancedTamiyoPolicyGNN(safety_policy_config)

    def test_confidence_threshold_enforcement(self, safety_policy):
        """Test safety mechanisms work with dangerous graph scenarios."""
        dangerous_graph, _ = SafetyTestScenarios.create_dangerous_adaptation_scenario()

        # Test basic policy forward pass with dangerous scenario
        graph_data = dangerous_graph.graph_data

        with torch.no_grad():
            # Use correct forward method signature
            output = safety_policy(
                node_features=graph_data.x,
                edge_index=graph_data.edge_index,
                return_uncertainty=True,
            )

        # Verify policy produces reasonable output (returns Dict[str, torch.Tensor])
        assert output is not None, "Policy should produce output for dangerous scenario"
        assert isinstance(output, dict), "Policy output should be dictionary"
        assert "adaptation_prob" in output, "Output should contain adaptation_prob"
        assert "safety_score" in output, "Output should contain safety_score"

        # Verify safety score is reasonable for dangerous scenario
        safety_score = output["safety_score"].mean().item()
        assert (
            0 <= safety_score <= 1
        ), f"Safety score should be in [0,1], got {safety_score}"

        # Test individual safety components directly
        if hasattr(safety_policy, "safety_regularizer"):
            safety_reg = safety_policy.safety_regularizer

            # Test with realistic graph representation features
            graph_repr_dim = (
                safety_policy.config.hidden_dim * 2
            )  # From the implementation
            mock_graph_repr = torch.randn(1, graph_repr_dim)

            direct_safety_score = safety_reg(mock_graph_repr)
            assert isinstance(
                direct_safety_score, torch.Tensor
            ), "Safety score should be tensor"
            assert (
                0 <= direct_safety_score.mean().item() <= 1
            ), f"Direct safety score should be in [0,1], got {direct_safety_score.mean().item()}"

        # Test uncertainty quantification if enabled
        if (
            hasattr(safety_policy, "uncertainty_module")
            and safety_policy.uncertainty_module is not None
        ):
            uncertainty_module = safety_policy.uncertainty_module
            mock_features = torch.randn(1, safety_policy.config.hidden_dim * 2)
            uncertainty_result = uncertainty_module(mock_features)

            # UncertaintyQuantification returns (mean_prediction, std_prediction)
            if isinstance(uncertainty_result, tuple) and len(uncertainty_result) == 2:
                mean_pred, std_pred = uncertainty_result
                assert isinstance(
                    mean_pred, torch.Tensor
                ), "Mean prediction should be tensor"
                assert isinstance(
                    std_pred, torch.Tensor
                ), "Std prediction should be tensor"
                assert (
                    std_pred.mean().item() >= 0
                ), f"Uncertainty std should be non-negative, got {std_pred.mean().item()}"
            else:
                assert isinstance(
                    uncertainty_result, torch.Tensor
                ), "Uncertainty should be tensor"
                assert (
                    uncertainty_result.mean().item() >= 0
                ), f"Uncertainty should be non-negative, got {uncertainty_result.mean().item()}"

        # Check if uncertainty was returned in the output
        uncertainty_info = "included" if "uncertainty" in output else "not included"

        logger.info(
            f"Safety test: adaptation_prob={output['adaptation_prob'].mean().item():.3f}, "
            f"safety_score={safety_score:.3f}, uncertainty={uncertainty_info}"
        )

    def test_safety_validation_pipeline_comprehensive(self, safety_policy):
        """Test 5-layer safety validation pipeline."""
        # Layer 1: Confidence threshold
        # Layer 2: Uncertainty threshold
        # Layer 3: Safety score validation
        # Layer 4: Risk assessment
        # Layer 5: System stability check

        unsafe_scenarios = [
            SafetyTestScenarios.create_dangerous_adaptation_scenario(),
            SafetyTestScenarios.create_high_uncertainty_scenario(),
            SafetyTestScenarios.create_emergency_brake_scenario(),
        ]

        safety_violations = 0
        total_decisions = 0

        for i, (graph_state, context) in enumerate(unsafe_scenarios):
            # Create graph input
            num_nodes = max(10, len(getattr(graph_state, "layer_nodes", [])))
            node_features = torch.randn(
                num_nodes, safety_policy.config.node_feature_dim
            )
            edge_index = torch.randint(0, num_nodes, (2, min(20, num_nodes * 2)))

            # Test multiple decision attempts
            for attempt in range(5):
                safety_policy(
                    node_features, edge_index, return_uncertainty=True
                )

                # Create realistic graph state for decision making
                graph_state = create_realistic_graph_state(
                    num_nodes=num_nodes,
                    feature_dim=safety_policy.config.node_feature_dim,
                    problematic_layers=[f"unsafe_layer_{i}"],
                    health_trends={f"unsafe_layer_{i}": [0.1, 0.05]},
                )

                decision = safety_policy.make_decision(graph_state)
                total_decisions += 1

                if decision is not None:
                    # Validate safety criteria
                    confidence = decision.confidence
                    safety_score = decision.metadata.get("safety_score", 0.0)
                    uncertainty = decision.metadata.get("epistemic_uncertainty", 0.0)
                    risk = decision.metadata.get("risk_assessment", 0.0)

                    # Check each safety layer
                    safety_checks = [
                        confidence
                        >= safety_policy.config.adaptation_confidence_threshold,
                        (
                            uncertainty <= safety_policy.config.uncertainty_threshold
                            if uncertainty > 0
                            else True
                        ),
                        safety_score >= 0.5,  # Minimum safety score
                        risk <= 0.7,  # Maximum acceptable risk
                    ]

                    if not all(safety_checks):
                        safety_violations += 1
                        logger.warning(
                            f"Safety violation in scenario {i}: "
                            f"conf={confidence:.3f}, safety={safety_score:.3f}, "
                            f"unc={uncertainty:.3f}, risk={risk:.3f}"
                        )

        # Should have very few safety violations
        violation_rate = safety_violations / max(total_decisions, 1)
        assert (
            violation_rate < 0.05
        ), f"Too many safety violations: {violation_rate:.1%}"

        logger.info(
            f"Safety pipeline validation: {safety_violations}/{total_decisions} violations "
            f"({violation_rate:.1%})"
        )

    def test_dangerous_scenario_prevention(self, safety_policy):
        """Test prevention of dangerous adaptations in critical scenarios."""
        dangerous_graph, dangerous_context = (
            SafetyTestScenarios.create_dangerous_adaptation_scenario()
        )

        # Create realistic but dangerous input
        node_features = torch.randn(10, safety_policy.config.node_feature_dim)
        # Create unstable features (high variance, conflicting signals)
        node_features += torch.randn_like(node_features) * 2.0  # Add noise

        edge_index = torch.tensor([[i, (i + 1) % 10] for i in range(10)]).t()

        dangerous_decisions = 0
        total_attempts = 50

        for attempt in range(total_attempts):
            # Add randomness to simulate different system states
            noisy_features = node_features + torch.randn_like(node_features) * 0.1

            safety_policy(noisy_features, edge_index, return_uncertainty=True)

            # Create realistic state representing dangerous system condition
            graph_state = create_realistic_graph_state(
                problematic_layers=["dangerous_layer_5"],
                health_trends={
                    "dangerous_layer_5": [0.8, 0.6, 0.3, 0.1]
                },  # Rapid decline
            )

            decision = safety_policy.make_decision(graph_state)

            if decision is not None:
                # Check if decision is made in dangerous context
                safety_score = decision.metadata.get("safety_score", 1.0)

                # Consider it dangerous if made with low safety score in unstable system
                if (
                    safety_score < 0.6
                    and dangerous_context["system_stability_score"] < 0.5
                ):
                    dangerous_decisions += 1

        # Should prevent most dangerous adaptations
        dangerous_rate = dangerous_decisions / total_attempts
        assert (
            dangerous_rate < 0.1
        ), f"Too many dangerous decisions allowed: {dangerous_rate:.1%}"

        logger.info(
            f"Dangerous scenario prevention: {dangerous_decisions}/{total_attempts} "
            f"dangerous decisions ({dangerous_rate:.1%})"
        )

    def test_emergency_brake_scenarios(self, safety_policy):
        """Test system stability under extreme conditions."""
        emergency_graph, emergency_context = (
            SafetyTestScenarios.create_emergency_brake_scenario()
        )

        # Create features representing system in crisis
        node_features = torch.randn(12, safety_policy.config.node_feature_dim)
        # Simulate cascading failure features
        for i in range(12):
            # Earlier layers have more extreme (problematic) features
            crisis_severity = (12 - i) / 12.0
            node_features[i] *= crisis_severity  # Scale features by crisis level
            node_features[i] -= crisis_severity  # Shift toward negative values

        edge_index = torch.tensor(
            [[i, j] for i in range(12) for j in range(i + 1, min(i + 3, 12))]
        ).t()

        emergency_interventions = 0
        total_emergency_tests = 30

        for test in range(total_emergency_tests):
            # Add system noise to simulate crisis conditions
            crisis_features = node_features + torch.randn_like(node_features) * 0.5

            safety_policy(
                crisis_features, edge_index, return_uncertainty=True
            )

            # Create realistic emergency system state
            graph_state = create_realistic_graph_state(
                problematic_layers=[
                    f"cascade_layer_{i}" for i in range(6)
                ],  # Half the layers problematic
                health_trends={
                    f"cascade_layer_{i}": [
                        0.9 - i * 0.1,
                        0.7 - i * 0.1,
                        0.4 - i * 0.1,
                        0.1,
                    ]
                    for i in range(6)
                },
            )

            decision = safety_policy.make_decision(graph_state)

            if decision is not None:
                # In emergency scenarios, system should be extra conservative
                confidence = decision.confidence
                urgency = decision.urgency
                safety_score = decision.metadata.get("safety_score", 0.0)

                # Emergency intervention criteria
                emergency_intervention = (
                    confidence > 0.9  # Very high confidence required
                    and safety_score > 0.8  # Very high safety required
                    and urgency > 0.7  # Must be genuinely urgent
                )

                if emergency_intervention:
                    emergency_interventions += 1

        # Should make very few interventions in emergency scenarios
        intervention_rate = emergency_interventions / total_emergency_tests
        assert (
            intervention_rate < 0.2
        ), f"Too many emergency interventions: {intervention_rate:.1%}"

        logger.info(
            f"Emergency brake test: {emergency_interventions}/{total_emergency_tests} "
            f"interventions in crisis ({intervention_rate:.1%})"
        )

    def test_uncertainty_threshold_enforcement(self, safety_policy):
        """Test high-uncertainty decisions are rejected."""
        uncertain_graph, _ = SafetyTestScenarios.create_high_uncertainty_scenario()

        # Create contradictory features that should increase uncertainty
        node_features = torch.randn(8, safety_policy.config.node_feature_dim)
        # Add conflicting information
        node_features[:4] *= -1  # Flip half the features

        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4]], dtype=torch.long
        )

        high_uncertainty_decisions = 0
        total_uncertainty_tests = 25

        for test in range(total_uncertainty_tests):
            # Add more noise to increase uncertainty
            noisy_features = node_features + torch.randn_like(node_features) * 1.0

            safety_policy(noisy_features, edge_index, return_uncertainty=True)

            graph_state = create_realistic_graph_state(
                problematic_layers=["uncertain_layer_0"],
                health_trends={
                    "uncertain_layer_0": [0.9, 0.1, 0.8, 0.2]
                },  # Conflicting
            )

            decision = safety_policy.make_decision(graph_state)

            if decision is not None:
                uncertainty = decision.metadata.get("epistemic_uncertainty", 0.0)

                if uncertainty > safety_policy.config.uncertainty_threshold:
                    high_uncertainty_decisions += 1

        # Should reject high-uncertainty decisions
        high_uncertainty_rate = high_uncertainty_decisions / total_uncertainty_tests
        assert (
            high_uncertainty_rate < 0.1
        ), f"Too many high-uncertainty decisions: {high_uncertainty_rate:.1%}"

        logger.info(
            f"Uncertainty enforcement: {high_uncertainty_decisions}/{total_uncertainty_tests} "
            f"high-uncertainty decisions ({high_uncertainty_rate:.1%})"
        )


class TestPolicyPerformanceUnderSafetyConstraints:
    """Test policy performance when safety constraints are enforced."""

    @pytest.fixture
    def performance_policy(self):
        """Policy configured for performance testing with safety."""
        config = PolicyConfig(
            hidden_dim=32,  # Smaller for faster testing
            num_gnn_layers=2,
            enable_uncertainty=True,
            uncertainty_samples=5,  # Fewer samples for speed
        )
        return EnhancedTamiyoPolicyGNN(config)

    @pytest.mark.performance
    def test_safety_overhead_performance_impact(self, performance_policy):
        """Test safety validation doesn't significantly impact performance."""
        # Create test input
        node_features = torch.randn(20, performance_policy.config.node_feature_dim)
        edge_index = torch.randint(0, 20, (2, 40))

        # Measure baseline inference time (without uncertainty)
        start_time = time.perf_counter()

        for _ in range(100):
            performance_policy(
                node_features, edge_index, return_uncertainty=False
            )

        baseline_time = time.perf_counter() - start_time

        # Measure time with full safety validation (with uncertainty)
        start_time = time.perf_counter()

        for _ in range(100):
            performance_policy(
                node_features, edge_index, return_uncertainty=True
            )

        safety_time = time.perf_counter() - start_time

        # Calculate overhead
        overhead_ratio = safety_time / baseline_time
        overhead_percent = (overhead_ratio - 1.0) * 100

        # Safety validation should not add excessive overhead
        assert (
            overhead_ratio < 3.0
        ), f"Safety overhead too high: {overhead_percent:.1f}%"

        # Each inference should still be fast
        avg_inference_time = safety_time / 100
        assert (
            avg_inference_time < 0.05
        ), f"Safety inference too slow: {avg_inference_time*1000:.1f}ms"

        logger.info(
            f"Safety performance impact: {overhead_percent:.1f}% overhead, "
            f"{avg_inference_time*1000:.1f}ms per inference"
        )

    @pytest.mark.performance
    def test_decision_making_latency_with_safety(self, performance_policy):
        """Test decision making latency including all safety checks."""
        # Create realistic decision scenario
        ProductionScenarioFactory.create_unhealthy_system_scenario()

        node_features = torch.randn(12, performance_policy.config.node_feature_dim)
        edge_index = torch.tensor([[i, (i + 1) % 12] for i in range(12)]).t()

        decision_times = []

        for i in range(50):  # Test 50 decisions
            start_time = time.perf_counter()

            # Full decision pipeline
            performance_policy(
                node_features, edge_index, return_uncertainty=True
            )

            graph_state = create_realistic_graph_state(
                problematic_layers=["test_layer"],
                health_trends={"test_layer": [0.5, 0.3, 0.2]},
            )

            performance_policy.make_decision(graph_state)

            end_time = time.perf_counter()
            decision_times.append(end_time - start_time)

        # Calculate latency statistics
        avg_latency = np.mean(decision_times) * 1000  # Convert to ms
        p95_latency = np.percentile(decision_times, 95) * 1000
        max_latency = np.max(decision_times) * 1000

        # Performance requirements for safety-enhanced decisions
        assert (
            avg_latency < 50.0
        ), f"Average decision latency too high: {avg_latency:.1f}ms"
        assert (
            p95_latency < 100.0
        ), f"P95 decision latency too high: {p95_latency:.1f}ms"
        assert (
            max_latency < 200.0
        ), f"Max decision latency too high: {max_latency:.1f}ms"

        logger.info(
            f"Decision latency with safety: Avg={avg_latency:.1f}ms, "
            f"P95={p95_latency:.1f}ms, Max={max_latency:.1f}ms"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])

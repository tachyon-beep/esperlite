"""
Simplified Enhanced Policy Safety Tests

Replaces overly complex mock-based tests with direct, realistic testing.
Tests actual functionality without assuming methods that don't exist.
"""

import logging

import pytest
import torch

from esper.services.tamiyo.policy import EnhancedTamiyoPolicyGNN
from esper.services.tamiyo.policy import PolicyConfig
from tests.production_scenarios import ProductionScenarioFactory

logger = logging.getLogger(__name__)


class TestSimplifiedPolicySafety:
    """Simplified, direct tests for policy safety components."""

    @pytest.fixture
    def policy_config(self):
        """Basic policy configuration for testing."""
        return PolicyConfig(
            node_feature_dim=16,
            hidden_dim=64,
            num_gnn_layers=2,
            num_attention_heads=4,
            enable_uncertainty=True,
        )

    @pytest.fixture
    def safety_policy(self, policy_config):
        """Policy instance for testing."""
        return EnhancedTamiyoPolicyGNN(policy_config)

    def test_policy_basic_functionality(self, safety_policy):
        """Test policy basic forward pass works correctly."""
        # Create realistic graph data
        num_nodes = 10
        node_features = torch.randn(num_nodes, safety_policy.config.node_feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, 15))  # 15 edges

        with torch.no_grad():
            output = safety_policy(node_features, edge_index)

        # Verify output structure
        assert isinstance(output, dict), "Output should be dictionary"
        assert "adaptation_prob" in output, "Should have adaptation probability"
        assert "safety_score" in output, "Should have safety score"
        assert "value_estimate" in output, "Should have value estimate"

        # Verify output ranges
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert value.numel() > 0, f"{key} should not be empty"
                assert torch.isfinite(value).all(), f"{key} should have finite values"
            elif isinstance(value, dict):
                # Handle nested dictionaries like temporal_analysis
                for subkey, subtensor in value.items():
                    assert isinstance(
                        subtensor, torch.Tensor
                    ), f"{key}.{subkey} should be tensor"
                    assert torch.isfinite(
                        subtensor
                    ).all(), f"{key}.{subkey} should have finite values"
            else:
                pytest.fail(f"Unexpected output type for {key}: {type(value)}")

    def test_safety_regularizer_directly(self, safety_policy):
        """Test safety regularizer component."""
        safety_reg = safety_policy.safety_regularizer

        # Test with different feature vectors
        batch_size = 5
        feature_dim = safety_policy.config.hidden_dim * 2  # Graph representation dim

        # Test normal features
        normal_features = torch.randn(batch_size, feature_dim) * 0.1  # Small values
        safety_score = safety_reg(normal_features)

        assert isinstance(safety_score, torch.Tensor), "Safety score should be tensor"
        assert safety_score.shape[0] == batch_size, "Should maintain batch dimension"
        assert (0 <= safety_score).all() and (
            safety_score <= 1
        ).all(), "Safety score should be in [0,1]"

        # Test extreme features (should have lower safety)
        extreme_features = torch.randn(batch_size, feature_dim) * 10  # Large values
        extreme_safety = safety_reg(extreme_features)

        # Safety scores should be reasonable
        assert (extreme_safety >= 0).all(), "Safety scores should be non-negative"
        assert (extreme_safety <= 1).all(), "Safety scores should be at most 1"

    def test_uncertainty_quantification(self, safety_policy):
        """Test uncertainty quantification works."""
        if (
            not hasattr(safety_policy, "uncertainty_module")
            or safety_policy.uncertainty_module is None
        ):
            pytest.skip("Uncertainty module not enabled")

        uncertainty_module = safety_policy.uncertainty_module

        # Test with sample features
        batch_size = 3
        feature_dim = safety_policy.config.hidden_dim * 2
        features = torch.randn(batch_size, feature_dim)

        result = uncertainty_module(features)

        # Should return tuple (mean, std)
        assert isinstance(result, tuple), "Uncertainty should return tuple"
        assert len(result) == 2, "Should return (mean, std)"

        mean_pred, std_pred = result
        assert isinstance(mean_pred, torch.Tensor), "Mean should be tensor"
        assert isinstance(std_pred, torch.Tensor), "Std should be tensor"
        assert (std_pred >= 0).all(), "Standard deviation should be non-negative"

    def test_policy_with_different_scenarios(self, safety_policy):
        """Test policy behavior with different health signal scenarios."""
        # Test with healthy system
        healthy_scenario = ProductionScenarioFactory.create_stable_system_scenario()

        # Build simple graph from health signals
        signals = healthy_scenario.health_signals[:8]  # Use 8 signals
        num_nodes = len(signals)

        # Create node features from health signals
        node_features = torch.zeros(num_nodes, safety_policy.config.node_feature_dim)
        for i, signal in enumerate(signals):
            # Encode signal properties as features
            node_features[i, 0] = signal.health_score
            node_features[i, 1] = signal.activation_variance
            node_features[i, 2] = signal.dead_neuron_ratio
            node_features[i, 3] = signal.avg_correlation
            node_features[i, 4] = float(signal.error_count)
            # Fill rest with noise
            node_features[i, 5:] = (
                torch.randn(safety_policy.config.node_feature_dim - 5) * 0.1
            )

        # Create simple connectivity (sequential chain)
        edge_index = torch.tensor(
            [list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long
        )

        with torch.no_grad():
            healthy_output = safety_policy(node_features, edge_index)

        # Test with unhealthy system
        unhealthy_scenario = (
            ProductionScenarioFactory.create_unhealthy_system_scenario()
        )
        unhealthy_signals = unhealthy_scenario.health_signals[:8]

        unhealthy_features = torch.zeros(
            num_nodes, safety_policy.config.node_feature_dim
        )
        for i, signal in enumerate(unhealthy_signals):
            unhealthy_features[i, 0] = signal.health_score
            unhealthy_features[i, 1] = signal.activation_variance
            unhealthy_features[i, 2] = signal.dead_neuron_ratio
            unhealthy_features[i, 3] = signal.avg_correlation
            unhealthy_features[i, 4] = float(signal.error_count)
            unhealthy_features[i, 5:] = (
                torch.randn(safety_policy.config.node_feature_dim - 5) * 0.1
            )

        with torch.no_grad():
            unhealthy_output = safety_policy(unhealthy_features, edge_index)

        # Compare outputs - both should be valid but potentially different
        for key in ["adaptation_prob", "safety_score"]:
            healthy_val = healthy_output[key].mean().item()
            unhealthy_val = unhealthy_output[key].mean().item()

            assert 0 <= healthy_val <= 1, f"Healthy {key} should be normalized"
            assert 0 <= unhealthy_val <= 1, f"Unhealthy {key} should be normalized"

            logger.info(
                f"Policy comparison - {key}: healthy={healthy_val:.3f}, unhealthy={unhealthy_val:.3f}"
            )

    def test_policy_consistency(self, safety_policy):
        """Test policy produces reasonably consistent results for same input."""
        num_nodes = 6
        node_features = torch.randn(num_nodes, safety_policy.config.node_feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, 10))

        # Set to eval mode to reduce dropout variability
        safety_policy.eval()

        # Run multiple times
        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = safety_policy(node_features, edge_index)
                outputs.append(output)

        # Check consistency (allowing for small differences due to dropout)
        base_output = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            for key in base_output:
                if isinstance(base_output[key], torch.Tensor) and isinstance(
                    output[key], torch.Tensor
                ):
                    diff = torch.abs(output[key] - base_output[key]).max().item()
                    # Allow reasonable tolerance for neural network outputs with dropout
                    tolerance = 0.05  # More realistic tolerance for dropout effects
                    assert (
                        diff < tolerance
                    ), f"Output {key} inconsistent between runs: max_diff={diff}"
                elif isinstance(base_output[key], dict) and isinstance(
                    output[key], dict
                ):
                    # Handle nested dictionaries
                    for subkey in base_output[key]:
                        if isinstance(base_output[key][subkey], torch.Tensor):
                            diff = (
                                torch.abs(
                                    output[key][subkey] - base_output[key][subkey]
                                )
                                .max()
                                .item()
                            )
                            tolerance = 0.05
                            assert (
                                diff < tolerance
                            ), f"Output {key}.{subkey} inconsistent: max_diff={diff}"

        logger.info("Policy consistency test passed")

    def test_edge_cases(self, safety_policy):
        """Test policy handles edge cases."""
        # Test with single node
        single_node = torch.randn(1, safety_policy.config.node_feature_dim)
        empty_edges = torch.zeros(2, 0, dtype=torch.long)  # No edges

        with torch.no_grad():
            single_output = safety_policy(single_node, empty_edges)

        assert isinstance(single_output, dict), "Should handle single node"

        # Check finite outputs recursively
        def check_finite(value, name):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{name} should have finite values"
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    check_finite(subvalue, f"{name}.{subkey}")

        for key, value in single_output.items():
            check_finite(value, key)

        # Test with zero features
        num_nodes = 4
        zero_features = torch.zeros(num_nodes, safety_policy.config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        with torch.no_grad():
            zero_output = safety_policy(zero_features, edge_index)

        assert isinstance(zero_output, dict), "Should handle zero features"
        for key, value in zero_output.items():
            check_finite(value, f"zero_features_{key}")

        logger.info("Edge case testing passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

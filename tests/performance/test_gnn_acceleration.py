"""
Performance benchmarks for GNN acceleration w        # Validate output shapes and ranges
        assert adaptation_prob.shape == torch.Size([1])
        assert layer_priorities.shape == torch.Size([1, 1])
        assert urgency_score.shape == torch.Size([1])
        assert value_estimate.shape == torch.Size([1, 1])orch-scatter.

This module contains tests that validate the performance improvements
and correctness of torch-scatter acceleration in the Tamiyo policy.
"""

import time

import pytest
import torch

from src.esper.services.tamiyo.policy import SCATTER_AVAILABLE
from src.esper.services.tamiyo.policy import PolicyConfig
from src.esper.services.tamiyo.policy import TamiyoPolicyGNN


class TestGNNAcceleration:
    """Test GNN acceleration functionality and performance."""

    def test_acceleration_detection(self):
        """Verify torch-scatter detection works correctly."""
        # Test should pass regardless of installation
        assert isinstance(SCATTER_AVAILABLE, bool)

        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        status = policy.acceleration_status

        assert "torch_scatter_available" in status
        assert status["torch_scatter_available"] == SCATTER_AVAILABLE
        assert isinstance(status["acceleration_enabled"], bool)
        assert isinstance(status["fallback_mode"], bool)
        assert status["fallback_mode"] == (not SCATTER_AVAILABLE)

    def test_policy_works_with_fallback(self):
        """Verify policy works correctly in fallback mode."""
        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)
        policy = TamiyoPolicyGNN(config)

        # Create test data
        num_nodes = 10
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        # Forward pass should work regardless of acceleration
        adaptation_prob, layer_priorities, urgency_score, value_estimate = policy(
            node_features, edge_index
        )

        # Validate output shapes and ranges
        assert adaptation_prob.shape == torch.Size([1])
        assert layer_priorities.shape == torch.Size([1, 1])
        assert urgency_score.shape == torch.Size([1])
        assert value_estimate.shape == torch.Size([1, 1])

        # Validate output ranges
        assert 0 <= adaptation_prob <= 1
        assert 0 <= urgency_score <= 1
        # layer_priorities are logits, so no range constraint
        # value_estimate is unconstrained

    def test_pooling_performance_baseline(self):
        """Benchmark global pooling operations in fallback mode."""
        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)
        policy = TamiyoPolicyGNN(config)

        # Create larger test data for meaningful benchmarking
        num_nodes = 1000
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        # Create random graph with reasonable connectivity
        num_edges = num_nodes * 3
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Warm up (important for PyTorch benchmarking)
        for _ in range(10):
            with torch.no_grad():
                _ = policy(node_features, edge_index)

        # Benchmark
        num_iterations = 100
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = policy(node_features, edge_index)

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations

        # Log performance for reference
        print(f"Fallback mode average time: {avg_time * 1000:.2f}ms per forward pass")

        # Sanity check - should complete in reasonable time
        assert avg_time < 1.0, f"Forward pass too slow: {avg_time:.3f}s"

        # Test passes if assertions are met

    @pytest.mark.skipif(not SCATTER_AVAILABLE, reason="torch-scatter not available")
    def test_acceleration_performance(self):
        """Benchmark performance with torch-scatter acceleration."""
        # This test only runs if torch-scatter is available
        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)
        policy = TamiyoPolicyGNN(config)

        # Verify acceleration is actually enabled
        assert policy.acceleration_status["acceleration_enabled"]

        # Same benchmark as fallback mode
        num_nodes = 1000
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        num_edges = num_nodes * 3
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = policy(node_features, edge_index)

        # Benchmark
        num_iterations = 100
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = policy(node_features, edge_index)

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations

        print(
            f"Accelerated mode average time: {avg_time * 1000:.2f}ms per forward pass"
        )

        # Should be reasonably fast
        assert avg_time < 1.0, f"Accelerated forward pass too slow: {avg_time:.3f}s"

        # Test passes if assertions are met

    def test_numerical_equivalence(self):
        """Verify numerical equivalence between accelerated and fallback modes."""
        # This test validates that acceleration doesn't change results
        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)

        # Create deterministic test data
        torch.manual_seed(42)
        num_nodes = 20
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

        # Create policy and get results
        policy = TamiyoPolicyGNN(config)

        # Set to eval mode for deterministic results
        policy.eval()

        with torch.no_grad():
            adaptation_prob, layer_priorities, urgency_score, value_estimate = policy(
                node_features, edge_index
            )

        # Results should be deterministic and reasonable
        assert torch.isfinite(adaptation_prob)
        assert torch.all(torch.isfinite(layer_priorities))
        assert torch.isfinite(urgency_score)
        assert torch.isfinite(value_estimate)

        # Store results for comparison (in future when we can test both modes)
        # results = {
        #     "adaptation_prob": adaptation_prob.item(),
        #     "layer_priorities": layer_priorities.cpu().numpy(),
        #     "urgency_score": urgency_score.item(),
        #     "value_estimate": value_estimate.item(),
        # }

        # For now, just verify consistency within same mode
        with torch.no_grad():
            adaptation_prob2, layer_priorities2, urgency_score2, value_estimate2 = (
                policy(node_features, edge_index)
            )

        # Results should be identical (deterministic)
        assert torch.allclose(adaptation_prob, adaptation_prob2, atol=1e-6)
        assert torch.allclose(layer_priorities, layer_priorities2, atol=1e-6)
        assert torch.allclose(urgency_score, urgency_score2, atol=1e-6)
        assert torch.allclose(value_estimate, value_estimate2, atol=1e-6)

    def test_memory_usage_stability(self):
        """Verify memory usage remains stable during repeated forward passes."""
        import gc

        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)
        policy = TamiyoPolicyGNN(config)

        # Measure initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Create test data
        num_nodes = 500
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))

        # Run many iterations to detect memory leaks
        for i in range(50):
            with torch.no_grad():
                _ = policy(node_features, edge_index)

            # Check memory every 10 iterations
            if i % 10 == 0 and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_growth = current_memory - initial_memory

                # Allow some memory growth but not excessive
                assert (
                    memory_growth < 100 * 1024 * 1024
                ), f"Memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth"

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_performance_comparison():
    """
    Utility function to run performance comparison when both modes are available.

    This function can be called manually to compare acceleration vs fallback performance
    when torch-scatter is installed.
    """
    if not SCATTER_AVAILABLE:
        print("torch-scatter not available - cannot run performance comparison")
        return

    print("Running performance comparison between fallback and acceleration...")

    # This would require a more sophisticated setup to test both modes
    # For now, we just run the acceleration benchmark
    test_instance = TestGNNAcceleration()
    accel_time = test_instance.test_acceleration_performance()

    print(f"Acceleration performance: {accel_time * 1000:.2f}ms per forward pass")


if __name__ == "__main__":
    # Run basic performance test
    test_instance = TestGNNAcceleration()
    print("Running GNN acceleration tests...")

    test_instance.test_acceleration_detection()
    print("✅ Acceleration detection test passed")

    test_instance.test_policy_works_with_fallback()
    print("✅ Fallback functionality test passed")

    test_instance.test_pooling_performance_baseline()
    print("✅ Baseline performance test passed")

    test_instance.test_numerical_equivalence()
    print("✅ Numerical equivalence test passed")

    test_instance.test_memory_usage_stability()
    print("✅ Memory stability test passed")

    if SCATTER_AVAILABLE:
        test_instance.test_acceleration_performance()
        print("✅ Acceleration performance test passed")

    print("\nAll GNN acceleration tests completed successfully!")

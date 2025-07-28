"""
Tests for Triton GPU kernels.

This module tests the custom Triton kernels developed for Phase 3
GPU optimization of the morphogenetic migration.
"""

import pytest
import torch

from esper.morphogenetic_v2.lifecycle.extended_lifecycle import ExtendedLifecycle
from esper.morphogenetic_v2.triton.forward_kernel import KasminaForwardKernel


class TestKasminaForwardKernel:
    """Test suite for the Kasmina forward kernel."""

    @pytest.fixture
    def device(self):
        """GPU device for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device('cuda:0')

    @pytest.fixture
    def kernel_config(self):
        """Standard kernel configuration."""
        return {
            'num_seeds': 100,
            'chunk_size': 64,
            'hidden_dim': 512,
            'batch_size': 32
        }

    @pytest.fixture
    def kernel(self, kernel_config):
        """Create kernel instance."""
        return KasminaForwardKernel(
            num_seeds=kernel_config['num_seeds'],
            chunk_size=kernel_config['chunk_size'],
            hidden_dim=kernel_config['hidden_dim']
        )

    @pytest.fixture
    def test_tensors(self, kernel_config, device):
        """Create test tensors."""
        batch_size = kernel_config['batch_size']
        hidden_dim = kernel_config['hidden_dim']
        num_seeds = kernel_config['num_seeds']

        # Input activations
        activations = torch.randn(batch_size, hidden_dim, device=device)

        # State tensor with 8 variables per seed
        state_tensor = torch.zeros((num_seeds, 8), dtype=torch.float32, device=device)

        # Set some seeds to active states
        for i in range(0, num_seeds, 3):  # Every 3rd seed
            state_tensor[i, 0] = ExtendedLifecycle.GRAFTING.value  # Lifecycle state
            state_tensor[i, 1] = i % 10  # Blueprint ID
            state_tensor[i, 2] = 5  # Epochs in state
            state_tensor[i, 3] = i % 3  # Grafting strategy (0, 1, or 2)
            if i % 3 == 0:  # Linear blending strategy
                state_tensor[i, 4] = 0.5  # Blend factor

        # Blueprint weights (10 blueprints)
        blueprint_weights = torch.randn(10, hidden_dim, device=device)

        # Pre-allocated output
        output = torch.zeros_like(activations)

        # Telemetry buffer
        telemetry = torch.zeros((num_seeds, 4), device=device)

        return {
            'activations': activations,
            'state_tensor': state_tensor,
            'blueprint_weights': blueprint_weights,
            'output': output,
            'telemetry': telemetry
        }

    def test_kernel_identity_for_dormant_seeds(self, kernel, test_tensors):
        """Test that dormant seeds produce identity output."""
        # Set all seeds to dormant
        test_tensors['state_tensor'][:, 0] = ExtendedLifecycle.DORMANT.value

        # Run kernel
        output = kernel(
            test_tensors['activations'],
            test_tensors['state_tensor'],
            test_tensors['blueprint_weights'],
            test_tensors['output'],
            test_tensors['telemetry']
        )

        # Output should match input for dormant seeds
        torch.testing.assert_close(output, test_tensors['activations'])

    def test_kernel_applies_blueprints(self, kernel, test_tensors):
        """Test that active seeds apply blueprints."""
        # Set one seed to active with additive strategy
        test_tensors['state_tensor'][0, 0] = ExtendedLifecycle.GRAFTING.value
        test_tensors['state_tensor'][0, 1] = 0  # Blueprint 0
        test_tensors['state_tensor'][0, 3] = 2  # Additive strategy

        # Run kernel
        output = kernel(
            test_tensors['activations'],
            test_tensors['state_tensor'],
            test_tensors['blueprint_weights'],
            test_tensors['output'],
            test_tensors['telemetry']
        )

        # Check that output differs from input (blueprint was applied)
        assert not torch.allclose(output, test_tensors['activations'])

    def test_kernel_telemetry_accumulation(self, kernel, test_tensors):
        """Test that telemetry is properly accumulated."""
        # Run kernel
        kernel(
            test_tensors['activations'],
            test_tensors['state_tensor'],
            test_tensors['blueprint_weights'],
            test_tensors['output'],
            test_tensors['telemetry']
        )

        # Check telemetry was accumulated
        assert test_tensors['telemetry'].sum() > 0

        # Check batch count (should be batch_size for each seed)
        batch_counts = test_tensors['telemetry'][:, 3]
        assert torch.all(batch_counts == test_tensors['activations'].shape[0])

    def test_kernel_different_strategies(self, kernel, test_tensors, device):
        """Test different grafting strategies."""
        batch_size = 1  # Simpler for testing
        hidden_dim = test_tensors['activations'].shape[1]

        # Create simple test case
        activations = torch.ones(batch_size, hidden_dim, device=device) * 2.0
        blueprint_weights = torch.ones(10, hidden_dim, device=device) * 3.0

        # Test linear blending (strategy 0)
        test_tensors['state_tensor'][0, 0] = ExtendedLifecycle.GRAFTING.value
        test_tensors['state_tensor'][0, 1] = 0  # Blueprint 0
        test_tensors['state_tensor'][0, 3] = 0  # Linear blending
        test_tensors['state_tensor'][0, 4] = 0.5  # Blend factor

        output = torch.zeros_like(activations)
        kernel(activations, test_tensors['state_tensor'], blueprint_weights, output, test_tensors['telemetry'])

        # Expected: 0.5 * 3 * 2 + 0.5 * 2 = 3 + 1 = 4
        chunk_size = kernel.chunk_size
        expected_chunk = torch.ones(chunk_size, device=device) * 4.0
        assert torch.allclose(output[0, :chunk_size], expected_chunk, atol=1e-5)

        # Test multiplicative (strategy 1)
        test_tensors['state_tensor'][1, 0] = ExtendedLifecycle.GRAFTING.value
        test_tensors['state_tensor'][1, 1] = 0
        test_tensors['state_tensor'][1, 3] = 1  # Multiplicative

        output.zero_()
        kernel(activations, test_tensors['state_tensor'], blueprint_weights, output, test_tensors['telemetry'])

        # Expected: 3 * 2 = 6
        expected_chunk = torch.ones(chunk_size, device=device) * 6.0
        assert torch.allclose(output[0, chunk_size:2*chunk_size], expected_chunk, atol=1e-5)

    def test_kernel_performance(self, kernel, test_tensors):
        """Basic performance test."""
        # Warmup
        for _ in range(10):
            kernel(
                test_tensors['activations'],
                test_tensors['state_tensor'],
                test_tensors['blueprint_weights'],
                test_tensors['output'],
                test_tensors['telemetry']
            )

        # Time the kernel
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(100):
            kernel(
                test_tensors['activations'],
                test_tensors['state_tensor'],
                test_tensors['blueprint_weights'],
                test_tensors['output'],
                test_tensors['telemetry']
            )
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 100

        print(f"\nKernel execution time: {elapsed_ms:.3f} ms")

        # Calculate throughput
        total_elements = test_tensors['activations'].numel()
        throughput_gb = (total_elements * 4 * 2) / (elapsed_ms * 1e6)  # Read + write, 4 bytes per float
        print(f"Memory throughput: {throughput_gb:.2f} GB/s")

        # Performance assertion (adjust based on GPU)
        assert elapsed_ms < 10.0  # Should be well under 10ms

    def test_kernel_edge_cases(self, kernel, device):
        """Test edge cases."""
        # Test with chunk_size > hidden_dim
        small_kernel = KasminaForwardKernel(
            num_seeds=10,
            chunk_size=128,
            hidden_dim=64
        )

        activations = torch.randn(8, 64, device=device)
        state_tensor = torch.zeros((10, 8), device=device)
        blueprint_weights = torch.randn(5, 64, device=device)
        output = torch.zeros_like(activations)
        telemetry = torch.zeros((10, 4), device=device)

        # Should handle gracefully
        result = small_kernel(activations, state_tensor, blueprint_weights, output, telemetry)
        assert result.shape == activations.shape

    @pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
    @pytest.mark.parametrize("num_seeds", [10, 100, 1000])
    def test_kernel_scalability(self, batch_size, num_seeds, device):
        """Test kernel with different configurations."""
        hidden_dim = 256
        chunk_size = 32

        kernel = KasminaForwardKernel(num_seeds, chunk_size, hidden_dim)

        activations = torch.randn(batch_size, hidden_dim, device=device)
        state_tensor = torch.zeros((num_seeds, 8), device=device)
        blueprint_weights = torch.randn(10, hidden_dim, device=device)
        output = torch.zeros_like(activations)
        telemetry = torch.zeros((num_seeds, 4), device=device)

        # Should work for all configurations
        result = kernel(activations, state_tensor, blueprint_weights, output, telemetry)
        assert result.shape == activations.shape

"""
Tests for the Triton-optimized Kasmina layer.
"""

import numpy as np
import pytest
import torch

from esper.morphogenetic_v2.lifecycle.extended_lifecycle import ExtendedLifecycle
from esper.morphogenetic_v2.triton.forward_kernel_v2 import TritonKasminaLayer


class TestTritonKasminaLayer:
    """Test suite for Triton-optimized Kasmina layer."""

    @pytest.fixture
    def device(self):
        """GPU device for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device('cuda:0')

    @pytest.fixture
    def layer(self, device):
        """Create a test layer."""
        return TritonKasminaLayer(
            hidden_dim=512,
            num_seeds=100,
            chunk_size=64
        ).to(device)

    def test_identity_for_dormant_seeds(self, layer, device):
        """Test that dormant seeds produce identity output."""
        # All seeds start as dormant (state 0)
        batch_size = 16
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        output = layer(x)

        # Should be identity transformation
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-5)

    def test_active_seeds_apply_transformation(self, layer, device):
        """Test that active seeds modify the output."""
        batch_size = 16
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        # Set some seeds to active
        for i in range(0, 10):
            layer.set_seed_state(
                seed_id=i,
                lifecycle_state=ExtendedLifecycle.GRAFTING.value,
                blueprint_id=0,
                grafting_strategy=2  # Additive
            )

        output = layer(x)

        # Output should be different from input
        assert not torch.allclose(output, x)

        # Check that first chunk is modified
        first_chunk = output[:, :layer.chunk_size]
        first_chunk_input = x[:, :layer.chunk_size]
        assert not torch.allclose(first_chunk, first_chunk_input)

    def test_different_grafting_strategies(self, layer, device):
        """Test different grafting strategies produce different outputs."""
        batch_size = 1
        x = torch.ones(batch_size, layer.hidden_dim, device=device) * 2.0

        # Set blueprint weights to known values
        layer.blueprint_weights[0] = torch.ones(layer.hidden_dim, device=device) * 3.0

        outputs = []

        # Test each strategy
        for strategy in [0, 1, 2]:  # Linear, Multiplicative, Additive
            # Reset states
            layer.lifecycle_states.zero_()

            # Set first seed with different strategy
            layer.set_seed_state(
                seed_id=0,
                lifecycle_state=ExtendedLifecycle.GRAFTING.value,
                blueprint_id=0,
                grafting_strategy=strategy,
                blend_factor=0.5 if strategy == 0 else 1.0
            )

            output = layer(x)
            outputs.append(output.clone())

        # Check expected values for first chunk
        chunk_size = layer.chunk_size

        # Linear blending: 0.5 * 3 * 2 + 0.5 * 2 = 4
        expected_linear = torch.ones(chunk_size, device=device) * 4.0
        torch.testing.assert_close(outputs[0][0, :chunk_size], expected_linear, rtol=1e-4)

        # Multiplicative: 3 * 2 = 6
        expected_mult = torch.ones(chunk_size, device=device) * 6.0
        torch.testing.assert_close(outputs[1][0, :chunk_size], expected_mult, rtol=1e-4)

        # Additive: 2 + 3 = 5
        expected_add = torch.ones(chunk_size, device=device) * 5.0
        torch.testing.assert_close(outputs[2][0, :chunk_size], expected_add, rtol=1e-4)

    def test_telemetry_collection(self, layer, device):
        """Test that telemetry is properly collected."""
        batch_size = 32
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        # Run forward pass
        _ = layer(x)

        # Get telemetry
        telemetry = layer.get_telemetry()

        # Check telemetry structure
        assert 'mean' in telemetry
        assert 'variance' in telemetry
        assert 'std' in telemetry
        assert 'count' in telemetry

        # Check that counts match expected
        expected_count = batch_size * layer.chunk_size
        assert telemetry['count'][0] == expected_count  # First seed processes first chunk

        # Check that statistics are reasonable
        assert np.all(telemetry['variance'] >= 0)
        assert np.all(telemetry['std'] >= 0)

    def test_multiple_active_seeds(self, layer, device):
        """Test with multiple active seeds."""
        batch_size = 8
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        # Activate every 3rd seed
        active_seeds = []
        for i in range(0, layer.num_seeds, 3):
            layer.set_seed_state(
                seed_id=i,
                lifecycle_state=ExtendedLifecycle.GRAFTING.value,
                blueprint_id=i % 10,
                grafting_strategy=i % 3
            )
            active_seeds.append(i)

        output = layer(x)

        # Output shape should be preserved
        assert output.shape == x.shape

        # Check that multiple chunks are modified
        num_modified_chunks = 0
        for i in range(layer.num_seeds):
            chunk_start = i * layer.chunk_size
            chunk_end = min(chunk_start + layer.chunk_size, layer.hidden_dim)

            if chunk_end > chunk_start:
                chunk_out = output[:, chunk_start:chunk_end]
                chunk_in = x[:, chunk_start:chunk_end]

                if not torch.allclose(chunk_out, chunk_in, rtol=1e-4):
                    num_modified_chunks += 1

        # Should have modified chunks for active seeds
        assert num_modified_chunks == len(active_seeds)

    def test_state_persistence(self, layer, device):
        """Test that states persist across forward passes."""
        batch_size = 4

        # Set some states
        layer.set_seed_state(0, lifecycle_state=3, blueprint_id=5)
        layer.set_seed_state(10, lifecycle_state=4, blueprint_id=7)

        # Multiple forward passes
        for _ in range(5):
            x = torch.randn(batch_size, layer.hidden_dim, device=device)
            _ = layer(x)

        # Check states are preserved
        assert layer.lifecycle_states[0] == 3
        assert layer.blueprint_ids[0] == 5
        assert layer.lifecycle_states[10] == 4
        assert layer.blueprint_ids[10] == 7

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    @pytest.mark.parametrize("hidden_dim", [256, 512, 1024])
    def test_various_configurations(self, batch_size, hidden_dim, device):
        """Test with various tensor sizes."""
        layer = TritonKasminaLayer(
            hidden_dim=hidden_dim,
            num_seeds=50,
            chunk_size=min(64, hidden_dim // 8)
        ).to(device)

        x = torch.randn(batch_size, hidden_dim, device=device)
        output = layer(x)

        assert output.shape == (batch_size, hidden_dim)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, layer, device):
        """Test that gradients flow through the layer."""
        layer.blueprint_weights.requires_grad_(True)

        # Set some seeds active
        for i in range(5):
            layer.set_seed_state(i, lifecycle_state=3, blueprint_id=0)

        x = torch.randn(8, layer.hidden_dim, device=device, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.blueprint_weights.grad is not None

        # Check gradients are non-zero for active blueprint
        assert layer.blueprint_weights.grad[0].abs().sum() > 0

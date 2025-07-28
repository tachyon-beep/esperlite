"""
Integration tests for Triton-optimized morphogenetic layers.
"""

import pytest
import torch
import torch.nn as nn

from esper.morphogenetic_v2.kasmina.triton_chunked_layer import (
    TritonChunkedKasminaLayer,
)
from esper.morphogenetic_v2.lifecycle.extended_lifecycle import ExtendedLifecycle


class TestTritonIntegration:
    """Test Phase 3 integration with Phase 1/2 components."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for Triton tests")
        return torch.device('cuda:0')

    @pytest.fixture
    def base_layer(self):
        """Simple base layer for testing."""
        return nn.Linear(512, 512)

    @pytest.fixture
    def layer(self, base_layer, device):
        """Create integrated layer."""
        # Enable Triton feature flag
        FeatureFlags._load_config({'triton_kernels': {'enabled': True}})

        return TritonChunkedKasminaLayer(
            base_layer=base_layer,
            chunks_per_layer=100,
            device=device,
            enable_triton=True
        ).to(device)

    def test_triton_forward_pass(self, layer, device):
        """Test Triton-accelerated forward pass."""
        batch_size = 16
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        # Forward pass
        output = layer(x)

        assert output.shape == x.shape
        assert output.device == x.device

        # Should use Triton path
        assert layer.enable_triton

    def test_fallback_to_pytorch(self, base_layer, device):
        """Test fallback when Triton is disabled."""
        # Disable Triton
        FeatureFlags._load_config({'triton_kernels': {'enabled': False}})

        layer = TritonChunkedKasminaLayer(
            base_layer=base_layer,
            chunks_per_layer=100,
            device=device,
            enable_triton=True  # Will be overridden by feature flag
        ).to(device)

        assert not layer.enable_triton

        x = torch.randn(8, layer.hidden_dim, device=device)
        output = layer(x)

        assert output.shape == x.shape

    def test_state_transitions(self, layer, device):
        """Test lifecycle state transitions work with Triton."""
        # Request germination
        success = layer.request_state_transition(
            seed_id=0,
            target_state=ExtendedLifecycle.GERMINATED
        )
        assert success

        # Check state is synced to Triton arrays
        assert layer.lifecycle_states[0] == ExtendedLifecycle.GERMINATED.value

        # Request training
        success = layer.request_state_transition(
            seed_id=0,
            target_state=ExtendedLifecycle.TRAINING
        )
        assert success
        assert layer.lifecycle_states[0] == ExtendedLifecycle.TRAINING.value

    def test_active_seeds_transform_output(self, layer, device):
        """Test that active seeds modify output."""
        batch_size = 8
        x = torch.randn(batch_size, layer.hidden_dim, device=device)

        # Initial forward (all dormant)
        output1 = layer(x)

        # Activate some seeds
        for i in range(10):
            layer.extended_state.update_state(i, {
                'lifecycle': ExtendedLifecycle.GRAFTING.value,
                'blueprint': i % 5,
                'grafting_strategy': i % 3
            })

        # Forward with active seeds
        output2 = layer(x)

        # Outputs should differ
        assert not torch.allclose(output1, output2)

        # Check affected chunks
        for i in range(10):
            start = i * layer.chunk_size
            end = min(start + layer.chunk_size, layer.hidden_dim)

            chunk1 = output1[:, start:end]
            chunk2 = output2[:, start:end]

            assert not torch.allclose(chunk1, chunk2)

    def test_performance_monitoring(self, layer, device):
        """Test performance monitoring integration."""
        # Run some forward passes
        for _ in range(5):
            x = torch.randn(16, layer.hidden_dim, device=device)
            _ = layer(x)

        # Get performance summary
        summary = layer.get_performance_summary()

        assert 'total_seeds' in summary
        assert 'active_seeds' in summary
        assert 'triton_enabled' in summary
        assert summary['triton_enabled'] == True
        assert summary['total_seeds'] == 100

    def test_checkpoint_save_load(self, layer, device, tmp_path):
        """Test checkpoint functionality."""
        # Set up checkpoint directory
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        layer.checkpoint_manager = None  # Reset
        from esper.morphogenetic_v2.lifecycle.checkpoint_manager_v2 import (
            CheckpointManager,
        )
        layer.checkpoint_manager = CheckpointManager(str(checkpoint_dir))

        # Activate some seeds
        for i in range(5):
            layer.extended_state.update_state(i, {
                'lifecycle': ExtendedLifecycle.GRAFTING.value,
                'blueprint': i
            })

        # Save checkpoint
        checkpoint_id = layer.save_checkpoint('test')

        # Modify state
        for i in range(5):
            layer.extended_state.update_state(i, {
                'lifecycle': ExtendedLifecycle.DORMANT.value
            })

        # Verify state changed
        assert layer.get_active_seed_count() == 0

        # Load checkpoint
        layer.load_checkpoint(checkpoint_id)

        # Verify state restored
        assert layer.get_active_seed_count() == 5
        assert layer.lifecycle_states[0] == ExtendedLifecycle.GRAFTING.value

    def test_mixed_precision_compatibility(self, layer, device):
        """Test compatibility with mixed precision."""
        with torch.cuda.amp.autocast():
            x = torch.randn(8, layer.hidden_dim, device=device)
            output = layer(x)

            assert output.dtype == torch.float16
            assert output.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    @pytest.mark.parametrize("num_seeds", [50, 100, 500])
    def test_various_configurations(self, base_layer, device, batch_size, num_seeds):
        """Test various batch and seed configurations."""
        layer = TritonChunkedKasminaLayer(
            base_layer=base_layer,
            chunks_per_layer=num_seeds,
            device=device,
            enable_triton=True
        ).to(device)

        x = torch.randn(batch_size, layer.hidden_dim, device=device)
        output = layer(x)

        assert output.shape == (batch_size, layer.hidden_dim)
        assert torch.isfinite(output).all()

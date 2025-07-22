"""
Unit tests for KasminaLayer.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.state_layout import SeedLifecycleState


class TestKasminaLayer:
    """Test cases for KasminaLayer."""

    def test_initialization(self):
        """Test basic initialization."""
        layer = KasminaLayer(
            input_size=10,
            output_size=5,
            num_seeds=4,
            cache_size_mb=64,
            telemetry_enabled=False,  # Disable for testing
            layer_name="test_layer",
        )

        assert layer.input_size == 10
        assert layer.output_size == 5
        assert layer.num_seeds == 4
        assert layer.layer_name == "test_layer"
        assert not layer.telemetry_enabled
        assert isinstance(layer.default_transform, nn.Linear)
        assert layer.default_transform.in_features == 10
        assert layer.default_transform.out_features == 5
        assert layer.state_layout.num_seeds == 4
        assert layer.total_forward_calls == 0
        assert layer.total_kernel_executions == 0

    def test_forward_pass_dormant_seeds(self):
        """Test forward pass with all seeds dormant."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Create test input
        x = torch.randn(3, 10)

        # Forward pass
        output = layer(x)

        # Should be same as default transform
        expected = layer.default_transform(x)
        assert torch.allclose(output, expected)
        assert layer.total_forward_calls == 1
        assert layer.total_kernel_executions == 0

    def test_forward_pass_active_seeds(self):
        """Test forward pass with active seeds."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Activate a seed
        layer.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE)
        layer.state_layout.alpha_blend[0] = 0.5

        # Create test input
        x = torch.randn(3, 10)

        # Forward pass
        output = layer(x)

        # Should be different from default transform due to kernel execution
        default_output = layer.default_transform(x)
        assert not torch.allclose(output, default_output)
        assert layer.total_forward_calls == 1
        assert layer.total_kernel_executions == 1

    def test_execute_kernel_placeholder(self):
        """Test placeholder kernel execution."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Create test input
        x = torch.randn(3, 10)

        # Execute kernel for different seeds
        output0 = layer._execute_kernel_placeholder(x, 0)
        output1 = layer._execute_kernel_placeholder(x, 1)

        # Different seeds should produce different outputs
        assert not torch.allclose(output0, output1)
        assert output0.shape == (3, 5)
        assert output1.shape == (3, 5)

    def test_blend_outputs(self):
        """Test output blending."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Create test tensors
        x = torch.randn(3, 10)
        default_output = layer.default_transform(x)
        kernel_output = default_output * 2.0  # Different output

        # Set up active seeds
        active_seeds = torch.tensor([True, False, True, False])
        layer.state_layout.alpha_blend[0] = 0.3
        layer.state_layout.alpha_blend[2] = 0.2

        # Blend outputs
        blended = layer._blend_outputs(default_output, kernel_output, active_seeds)

        # Should be blend of default and kernel outputs
        expected_alpha = 0.3 + 0.2  # Total alpha from active seeds
        expected = (
            1.0 - expected_alpha
        ) * default_output + expected_alpha * kernel_output
        assert torch.allclose(blended, expected)

    def test_blend_outputs_clamping(self):
        """Test alpha clamping in output blending."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Create test tensors
        x = torch.randn(3, 10)
        default_output = layer.default_transform(x)
        kernel_output = default_output * 2.0

        # Set up active seeds with total alpha > 1.0
        active_seeds = torch.tensor([True, True, True, False])
        layer.state_layout.alpha_blend[0] = 0.5
        layer.state_layout.alpha_blend[1] = 0.4
        layer.state_layout.alpha_blend[2] = 0.3

        # Blend outputs
        blended = layer._blend_outputs(default_output, kernel_output, active_seeds)

        # Alpha should be clamped to 1.0
        expected = 0.0 * default_output + 1.0 * kernel_output
        assert torch.allclose(blended, expected)

    def test_compute_health_score(self):
        """Test health score computation."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Test with small tensor
        small_tensor = torch.randn(3, 5) * 0.1
        health_score = layer._compute_health_score(small_tensor)
        assert 0.0 <= health_score <= 1.0

        # Test with large tensor
        large_tensor = torch.randn(3, 5) * 10.0
        health_score_large = layer._compute_health_score(large_tensor)
        assert 0.0 <= health_score_large <= 1.0

        # Smaller tensor should have higher health score
        assert health_score > health_score_large

    @pytest.mark.asyncio
    async def test_load_kernel_success(self):
        """Test kernel loading updates state correctly when kernel is available."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Initially seed should be dormant
        assert layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert layer.state_layout.alpha_blend[0].item() == 0.0

        # Load kernel - this will use the HTTP mocked response from conftest.py
        success = await layer.load_kernel(0, "test-kernel-123")

        # Verify kernel loading succeeded and updated layer state
        assert success, "Kernel loading should succeed with valid HTTP response"
        assert layer.state_layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
        assert layer.state_layout.active_kernel_id[0] == hash("test-kernel-123")
        assert (
            layer.state_layout.alpha_blend[0].item() > 0
        ), "Alpha should be set for active kernel"

    @pytest.mark.asyncio
    async def test_load_kernel_not_found(self):
        """Test kernel loading when kernel is not found."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Initially seed should be dormant
        assert layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT

        # Try to load kernel with invalid ID (triggers 404 in mock)
        success = await layer.load_kernel(0, "invalid-kernel-id")

        assert not success, "Kernel loading should fail when kernel not found"
        assert layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert (
            layer.state_layout.alpha_blend[0].item() == 0.0
        ), "Alpha should remain 0 for failed load"

    @pytest.mark.asyncio
    async def test_load_kernel_invalid_index(self):
        """Test kernel loading with invalid seed index."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        with pytest.raises(ValueError, match="Invalid seed index"):
            await layer.load_kernel(10, "test-kernel-123")

        with pytest.raises(ValueError, match="Invalid seed index"):
            await layer.load_kernel(-1, "test-kernel-123")

    @pytest.mark.asyncio
    async def test_unload_kernel_success(self):
        """Test successful kernel unloading."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Set up active seed
        layer.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE)
        layer.state_layout.alpha_blend[0] = 0.5

        # Unload kernel
        success = await layer.unload_kernel(0)

        assert success
        assert layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert abs(layer.state_layout.alpha_blend[0].item()) < 0.01

    @pytest.mark.asyncio
    async def test_unload_kernel_invalid_index(self):
        """Test kernel unloading with invalid seed index."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        with pytest.raises(ValueError, match="Invalid seed index"):
            await layer.unload_kernel(10)

    def test_get_layer_stats(self):
        """Test layer statistics collection."""
        layer = KasminaLayer(
            input_size=10,
            output_size=5,
            num_seeds=4,
            telemetry_enabled=False,
            layer_name="test_layer",
        )

        # Set up some state
        layer.total_forward_calls = 10
        layer.total_kernel_executions = 5

        stats = layer.get_layer_stats()

        assert stats["layer_name"] == "test_layer"
        assert stats["total_forward_calls"] == 10
        assert stats["total_kernel_executions"] == 5
        assert abs(stats["kernel_execution_ratio"] - 0.5) < 0.01
        assert not stats["telemetry_enabled"]
        assert "state_stats" in stats
        assert "cache_stats" in stats

    def test_set_seed_alpha(self):
        """Test setting seed alpha blend factor."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Set alpha
        layer.set_seed_alpha(0, 0.7)
        assert abs(layer.state_layout.alpha_blend[0].item() - 0.7) < 0.01

        # Test clamping
        layer.set_seed_alpha(1, 1.5)  # Should be clamped to 1.0
        assert abs(layer.state_layout.alpha_blend[1].item() - 1.0) < 0.01

        layer.set_seed_alpha(2, -0.3)  # Should be clamped to 0.0
        assert abs(layer.state_layout.alpha_blend[2].item() - 0.0) < 0.01

    def test_set_seed_alpha_invalid_index(self):
        """Test setting alpha with invalid seed index."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        with pytest.raises(ValueError, match="Invalid seed index"):
            layer.set_seed_alpha(10, 0.5)

    def test_to_device(self):
        """Test moving layer to different device."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Move to same device (should work)
        layer.to(torch.device("cpu"))
        assert layer.state_layout.device == torch.device("cpu")

        # Check that state tensors are on correct device
        assert layer.state_layout.lifecycle_states.device == torch.device("cpu")
        assert layer.state_layout.alpha_blend.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_to_cuda(self, mock_cuda_available):
        """Test moving layer to CUDA device."""
        layer = KasminaLayer(
            input_size=10, output_size=5, num_seeds=4, telemetry_enabled=False
        )

        # Move to CUDA
        cuda_device = torch.device("cuda")
        layer.to(cuda_device)

        assert layer.state_layout.device == cuda_device
        assert layer.state_layout.lifecycle_states.is_cuda
        assert layer.state_layout.alpha_blend.is_cuda

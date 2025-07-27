"""
Unit tests for model wrapper functionality.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from esper.core.model_wrapper import MorphableModel
from esper.core.model_wrapper import _create_kasmina_layer
from esper.core.model_wrapper import unwrap
from esper.core.model_wrapper import wrap
from esper.execution.kasmina_layer import KasminaLayer


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class TestMorphableModel:
    """Test cases for MorphableModel."""

    # Removed trivial test_initialization - only checked constructor parameters
    # without testing any actual behavior

    def test_forward_pass(self):
        """Test forward pass through MorphableModel."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        # Create test input
        x = torch.randn(3, 10)

        # Forward pass
        output = morphable(x)

        # Should be same as original model
        expected = model(x)
        assert torch.allclose(output, expected)
        assert morphable.total_forward_calls == 1

    @pytest.mark.asyncio
    async def test_load_kernel_success(self):
        """Test successful kernel loading."""
        model = SimpleModel()
        mock_layer = Mock(spec=KasminaLayer)
        mock_layer.load_kernel = AsyncMock(return_value=True)

        kasmina_layers = {"layer1": mock_layer}
        morphable = MorphableModel(model, kasmina_layers)

        # Load kernel
        success = await morphable.load_kernel("layer1", 0, "test-kernel-123")

        assert success
        assert morphable.morphogenetic_active
        mock_layer.load_kernel.assert_called_once_with(0, "test-kernel-123")

    @pytest.mark.asyncio
    async def test_load_kernel_failure(self):
        """Test kernel loading failure."""
        model = SimpleModel()
        mock_layer = Mock(spec=KasminaLayer)
        mock_layer.load_kernel = AsyncMock(return_value=False)

        kasmina_layers = {"layer1": mock_layer}
        morphable = MorphableModel(model, kasmina_layers)

        # Load kernel
        success = await morphable.load_kernel("layer1", 0, "test-kernel-123")

        assert not success
        mock_layer.load_kernel.assert_called_once_with(0, "test-kernel-123")

    @pytest.mark.asyncio
    async def test_load_kernel_invalid_layer(self):
        """Test kernel loading with invalid layer name."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            await morphable.load_kernel("nonexistent", 0, "test-kernel-123")

    @pytest.mark.asyncio
    async def test_unload_kernel_success(self):
        """Test successful kernel unloading."""
        model = SimpleModel()
        mock_layer = Mock(spec=KasminaLayer)
        mock_layer.unload_kernel = AsyncMock(return_value=True)
        mock_state_layout = Mock()
        mock_state_layout.get_active_seeds = Mock(
            return_value=torch.tensor([False, False, False, False])
        )
        mock_layer.state_layout = mock_state_layout

        kasmina_layers = {"layer1": mock_layer}
        morphable = MorphableModel(model, kasmina_layers)
        morphable.morphogenetic_active = True

        # Unload kernel
        success = await morphable.unload_kernel("layer1", 0)

        assert success
        assert not morphable.morphogenetic_active  # Should be False after unloading
        mock_layer.unload_kernel.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_unload_kernel_invalid_layer(self):
        """Test kernel unloading with invalid layer name."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            await morphable.unload_kernel("nonexistent", 0)

    def test_get_layer_names(self):
        """Test getting layer names."""
        model = SimpleModel()
        mock_layer1 = Mock(spec=KasminaLayer)
        mock_layer2 = Mock(spec=KasminaLayer)
        kasmina_layers = {"layer1": mock_layer1, "layer2": mock_layer2}
        morphable = MorphableModel(model, kasmina_layers)

        names = morphable.get_layer_names()
        assert set(names) == {"layer1", "layer2"}

    def test_get_layer_stats_specific(self):
        """Test getting stats for specific layer."""
        model = SimpleModel()
        mock_layer = Mock(spec=KasminaLayer)
        mock_layer.get_layer_stats = Mock(return_value={"test": "stats"})

        kasmina_layers = {"layer1": mock_layer}
        morphable = MorphableModel(model, kasmina_layers)

        stats = morphable.get_layer_stats("layer1")
        assert stats == {"test": "stats"}
        mock_layer.get_layer_stats.assert_called_once()

    def test_get_layer_stats_all(self):
        """Test getting stats for all layers."""
        model = SimpleModel()
        mock_layer1 = Mock(spec=KasminaLayer)
        mock_layer1.get_layer_stats = Mock(return_value={"test": "stats1"})
        mock_layer2 = Mock(spec=KasminaLayer)
        mock_layer2.get_layer_stats = Mock(return_value={"test": "stats2"})

        kasmina_layers = {"layer1": mock_layer1, "layer2": mock_layer2}
        morphable = MorphableModel(model, kasmina_layers)

        stats = morphable.get_layer_stats()
        assert stats == {"layer1": {"test": "stats1"}, "layer2": {"test": "stats2"}}

    def test_get_layer_stats_invalid_layer(self):
        """Test getting stats for invalid layer."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            morphable.get_layer_stats("nonexistent")

    def test_get_model_stats(self):
        """Test getting comprehensive model statistics."""
        model = SimpleModel()

        # Create mock layers
        mock_layer1 = Mock(spec=KasminaLayer)
        mock_layer1.num_seeds = 4
        mock_layer1.total_kernel_executions = 10
        mock_state_layout1 = Mock()
        mock_state_layout1.get_active_seeds = Mock(
            return_value=torch.tensor([True, False, True, False])
        )
        mock_layer1.state_layout = mock_state_layout1
        mock_layer1.get_layer_stats = Mock(return_value={"test": "stats1"})

        mock_layer2 = Mock(spec=KasminaLayer)
        mock_layer2.num_seeds = 4
        mock_layer2.total_kernel_executions = 5
        mock_state_layout2 = Mock()
        mock_state_layout2.get_active_seeds = Mock(
            return_value=torch.tensor([False, True, False, False])
        )
        mock_layer2.state_layout = mock_state_layout2
        mock_layer2.get_layer_stats = Mock(return_value={"test": "stats2"})

        kasmina_layers = {"layer1": mock_layer1, "layer2": mock_layer2}
        morphable = MorphableModel(model, kasmina_layers)
        morphable.total_forward_calls = 20
        morphable.morphogenetic_active = True

        stats = morphable.get_model_stats()

        assert stats["total_forward_calls"] == 20
        assert stats["morphogenetic_active"]
        assert stats["total_kasmina_layers"] == 2
        assert stats["total_seeds"] == 8
        assert stats["active_seeds"] == 3  # 2 + 1
        assert stats["total_kernel_executions"] == 15  # 10 + 5
        assert "layer_stats" in stats

    def test_set_seed_alpha(self):
        """Test setting seed alpha blend factor."""
        model = SimpleModel()
        mock_layer = Mock(spec=KasminaLayer)
        mock_layer.set_seed_alpha = Mock()

        kasmina_layers = {"layer1": mock_layer}
        morphable = MorphableModel(model, kasmina_layers)

        # Set alpha
        morphable.set_seed_alpha("layer1", 0, 0.7)
        mock_layer.set_seed_alpha.assert_called_once_with(0, 0.7)

    def test_set_seed_alpha_invalid_layer(self):
        """Test setting alpha with invalid layer name."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            morphable.set_seed_alpha("nonexistent", 0, 0.5)

    def test_enable_telemetry(self):
        """Test enabling/disabling telemetry."""
        model = SimpleModel()
        mock_layer1 = Mock(spec=KasminaLayer)
        mock_layer2 = Mock(spec=KasminaLayer)

        kasmina_layers = {"layer1": mock_layer1, "layer2": mock_layer2}
        morphable = MorphableModel(model, kasmina_layers)

        # Enable telemetry
        morphable.enable_telemetry(True)
        assert mock_layer1.telemetry_enabled
        assert mock_layer2.telemetry_enabled

        # Disable telemetry
        morphable.enable_telemetry(False)
        assert not mock_layer1.telemetry_enabled
        assert not mock_layer2.telemetry_enabled

    def test_compare_with_original(self):
        """Test comparison with original model."""
        model = SimpleModel()
        morphable = MorphableModel(model, {}, original_model=model)

        # Create test input
        x = torch.randn(3, 10)

        # Compare (should be identical since no morphogenetic modifications)
        comparison = morphable.compare_with_original(x)

        assert "mse" in comparison
        assert "max_absolute_difference" in comparison
        assert "output_shape" in comparison
        assert "morphogenetic_active" in comparison
        assert comparison["mse"] < 1e-6  # Should be nearly identical
        assert comparison["max_absolute_difference"] < 1e-6

    def test_compare_with_original_no_original(self):
        """Test comparison when no original model is available."""
        model = SimpleModel()
        morphable = MorphableModel(model, {})

        x = torch.randn(3, 10)

        with pytest.raises(ValueError, match="Original model not available"):
            morphable.compare_with_original(x)


class TestWrapFunction:
    """Test cases for the wrap function."""

    def test_wrap_basic(self):
        """Test basic model wrapping."""
        model = SimpleModel()

        # Wrap model
        morphable = wrap(model, telemetry_enabled=False)

        assert isinstance(morphable, MorphableModel)
        assert len(morphable.kasmina_layers) == 2  # Should replace 2 Linear layers
        assert morphable.original_model is not None

        # Check that layer names are correct
        layer_names = morphable.get_layer_names()
        assert "linear1" in layer_names
        assert "linear2" in layer_names

    def test_wrap_custom_target_layers(self):
        """Test wrapping with custom target layers."""
        model = SimpleModel()

        # Only wrap ReLU layers (should be none)
        morphable = wrap(model, target_layers=[nn.ReLU], telemetry_enabled=False)

        assert len(morphable.kasmina_layers) == 0  # No ReLU layers replaced

    def test_wrap_custom_parameters(self):
        """Test wrapping with custom parameters."""
        model = SimpleModel()

        morphable = wrap(
            model,
            seeds_per_layer=8,
            cache_size_mb=256,
            telemetry_enabled=True,
            preserve_original=False,
        )

        assert morphable.original_model is None  # Not preserved

        # Check that KasminaLayers have correct parameters
        for layer in morphable.kasmina_layers.values():
            assert layer.num_seeds == 8
            assert layer.telemetry_enabled

    def test_wrap_preserves_behavior(self):
        """Test that wrapped model preserves original behavior."""
        model = SimpleModel()
        morphable = wrap(model, telemetry_enabled=False)

        # Create test input
        x = torch.randn(3, 10)

        # Compare outputs
        original_output = model(x)
        wrapped_output = morphable(x)

        # Should be identical (no kernels loaded)
        assert torch.allclose(original_output, wrapped_output, atol=1e-6)

    def test_wrap_weights_copied(self):
        """Test that original weights are copied to KasminaLayers."""
        model = SimpleModel()
        morphable = wrap(model, telemetry_enabled=False)

        # Check that weights are copied
        original_linear1 = model.linear1
        kasmina_layer1 = morphable.kasmina_layers["linear1"]

        assert torch.allclose(
            original_linear1.weight, kasmina_layer1.default_transform.weight
        )
        assert torch.allclose(
            original_linear1.bias, kasmina_layer1.default_transform.bias
        )


class TestCreateKasminaLayer:
    """Test cases for _create_kasmina_layer function."""

    def test_create_from_linear(self):
        """Test creating KasminaLayer from Linear layer."""
        linear_layer = nn.Linear(10, 5)

        kasmina_layer = _create_kasmina_layer(
            linear_layer,
            "test_layer",
            seeds_per_layer=4,
            cache_size_mb=128,
            telemetry_enabled=False,
        )

        assert isinstance(kasmina_layer, KasminaLayer)
        assert kasmina_layer.input_size == 10
        assert kasmina_layer.output_size == 5
        assert kasmina_layer.num_seeds == 4
        assert kasmina_layer.layer_name == "test_layer"
        assert not kasmina_layer.telemetry_enabled

        # Check that weights are copied
        assert torch.allclose(
            linear_layer.weight, kasmina_layer.default_transform.weight
        )
        assert torch.allclose(linear_layer.bias, kasmina_layer.default_transform.bias)

    def test_create_from_unsupported_layer(self):
        """Test creating KasminaLayer from unsupported layer type."""
        relu_layer = nn.ReLU()

        with pytest.raises(NotImplementedError, match="Layer type.*not yet supported"):
            _create_kasmina_layer(
                relu_layer,
                "test_layer",
                seeds_per_layer=4,
                cache_size_mb=128,
                telemetry_enabled=False,
            )


class TestUnwrapFunction:
    """Test cases for the unwrap function."""

    def test_unwrap_with_original(self):
        """Test unwrapping when original model is available."""
        model = SimpleModel()
        morphable = wrap(model, preserve_original=True, telemetry_enabled=False)

        unwrapped = unwrap(morphable)

        # Should return the original model
        assert unwrapped is morphable.original_model

    def test_unwrap_without_original(self):
        """Test unwrapping when no original model is available."""
        model = SimpleModel()
        morphable = wrap(model, preserve_original=False, telemetry_enabled=False)

        unwrapped = unwrap(morphable)

        # Should return the wrapped model
        assert unwrapped is morphable.wrapped_model

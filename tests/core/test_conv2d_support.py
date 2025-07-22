"""
Tests for Conv2d layer support in morphogenetic model wrapping.

This module tests the Conv2d-specific KasminaLayer functionality
and ensures proper integration with common CNN architectures.
"""

import pytest
import torch
import torch.nn as nn

from esper.core.model_wrapper import wrap
from esper.execution.kasmina_conv2d_layer import KasminaConv2dLayer


class TestConv2dSupport:
    """Test Conv2d layer support functionality."""

    def test_simple_conv2d_wrapping(self):
        """Test wrapping a simple Conv2d layer."""
        # Create a simple Conv2d layer
        original_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        # Create a simple model
        model = nn.Sequential(
            original_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        # Wrap the model
        wrapped_model = wrap(
            model,
            target_layers=[nn.Conv2d, nn.Linear],
            seeds_per_layer=2,
            telemetry_enabled=False,
        )

        # Check that Conv2d was wrapped
        assert len(wrapped_model.kasmina_layers) == 2  # Conv2d + Linear
        assert "0" in wrapped_model.kasmina_layers  # Conv2d layer
        assert "4" in wrapped_model.kasmina_layers  # Linear layer

        # Check that the Conv2d layer is KasminaConv2dLayer
        conv_layer = wrapped_model.kasmina_layers["0"]
        assert isinstance(conv_layer, KasminaConv2dLayer)
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 16
        assert conv_layer.kernel_size == (3, 3)

    def test_conv2d_weight_preservation(self):
        """Test that Conv2d weights are preserved exactly."""
        # Create original Conv2d layer
        original_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)

        # Initialize with specific weights
        with torch.no_grad():
            original_conv.weight.fill_(0.5)
            original_conv.bias.fill_(0.1)

        # Create model and wrap
        model = nn.Sequential(original_conv)
        wrapped_model = wrap(model, target_layers=[nn.Conv2d], telemetry_enabled=False)

        # Get the wrapped layer
        conv_layer = wrapped_model.kasmina_layers["0"]

        # Check that weights were copied exactly
        assert torch.allclose(conv_layer.default_transform.weight, original_conv.weight)
        assert torch.allclose(conv_layer.default_transform.bias, original_conv.bias)

    def test_conv2d_forward_pass_dormant_seeds(self):
        """Test forward pass with dormant seeds maintains original behavior."""
        # Create original model
        original_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        original_model = nn.Sequential(original_conv, nn.ReLU())

        # Wrap the model
        wrapped_model = wrap(
            original_model, target_layers=[nn.Conv2d], telemetry_enabled=False
        )

        # Test input
        x = torch.randn(2, 3, 32, 32)

        # Forward pass through both models
        with torch.no_grad():
            original_output = original_model(x)
            wrapped_output = wrapped_model(x)

        # Should be identical when no seeds are active
        assert torch.allclose(original_output, wrapped_output, atol=1e-6)

    def test_conv2d_input_validation(self):
        """Test that Conv2d layers validate input shapes correctly."""
        # Create Conv2d layer
        conv_layer = KasminaConv2dLayer(
            in_channels=3, out_channels=16, kernel_size=3, telemetry_enabled=False
        )

        # Valid 4D input should work
        valid_input = torch.randn(2, 3, 32, 32)
        output = conv_layer(valid_input)
        assert output.shape == (2, 16, 30, 30)  # No padding, so size reduces

        # Invalid 3D input should raise error
        invalid_input = torch.randn(2, 3, 32)
        with pytest.raises(ValueError, match="Conv2d input must be 4D"):
            conv_layer(invalid_input)

        # Wrong number of channels should raise error
        wrong_channels = torch.randn(2, 5, 32, 32)  # 5 channels instead of 3
        with pytest.raises(ValueError, match="Input channels 5 != expected 3"):
            conv_layer(wrong_channels)

    def test_conv2d_parameter_preservation(self):
        """Test that all Conv2d parameters are preserved correctly."""
        original_conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(1, 2),
            dilation=(1, 2),
            groups=1,
            bias=True,
        )

        conv_layer = KasminaConv2dLayer(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(1, 2),
            dilation=(1, 2),
            groups=1,
            bias=True,
            telemetry_enabled=False,
        )

        # Copy weights
        conv_layer.copy_weights_from_conv2d(original_conv)

        # Test that parameters match
        assert conv_layer.in_channels == original_conv.in_channels
        assert conv_layer.out_channels == original_conv.out_channels
        assert conv_layer.kernel_size == original_conv.kernel_size
        assert conv_layer.stride == original_conv.stride
        assert conv_layer.padding == original_conv.padding
        assert conv_layer.dilation == original_conv.dilation
        assert conv_layer.groups == original_conv.groups

    def test_resnet_basic_block_wrapping(self):
        """Test wrapping a ResNet-style basic block."""

        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                return self.relu(out)

        # Create and wrap basic block
        block = BasicBlock(64, 64)
        wrapped_block = wrap(block, target_layers=[nn.Conv2d], telemetry_enabled=False)

        # Should have wrapped both conv layers
        assert len(wrapped_block.kasmina_layers) == 2
        assert "conv1" in wrapped_block.kasmina_layers
        assert "conv2" in wrapped_block.kasmina_layers

        # Test forward pass
        x = torch.randn(1, 64, 32, 32)
        output = wrapped_block(x)
        assert output.shape == (1, 64, 32, 32)

    def test_vgg_style_architecture(self):
        """Test wrapping a VGG-style architecture."""
        model = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Classifier
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

        # Wrap the model
        wrapped_model = wrap(
            model, target_layers=[nn.Conv2d, nn.Linear], telemetry_enabled=False
        )

        # Should have wrapped 4 conv layers + 2 linear layers
        assert len(wrapped_model.kasmina_layers) == 6

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = wrapped_model(x)
        assert output.shape == (1, 10)

    def test_mixed_layer_types(self):
        """Test wrapping models with both Linear and Conv2d layers."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        wrapped_model = wrap(
            model, target_layers=[nn.Conv2d, nn.Linear], telemetry_enabled=False
        )

        # Should wrap Conv2d + 2 Linear layers
        assert len(wrapped_model.kasmina_layers) == 3

        # Check layer types
        conv_layer = wrapped_model.kasmina_layers["0"]
        _ = wrapped_model.kasmina_layers["4"]
        _ = wrapped_model.kasmina_layers["6"]

        assert isinstance(conv_layer, KasminaConv2dLayer)
        assert conv_layer.in_channels == 3
        assert conv_layer.out_channels == 32


class TestConv2dLayerStats:
    """Test Conv2d layer statistics and monitoring."""

    def test_conv2d_layer_stats(self):
        """Test that Conv2d layers provide comprehensive statistics."""
        conv_layer = KasminaConv2dLayer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            num_seeds=4,
            telemetry_enabled=False,
            layer_name="test_conv",
        )

        stats = conv_layer.get_layer_stats()

        # Check that Conv2d-specific stats are included
        assert "conv_params" in stats
        assert "input_channels" in stats
        assert "output_channels" in stats
        assert "kernel_size" in stats
        assert "stride" in stats
        assert "padding" in stats

        # Check values
        assert stats["input_channels"] == 3
        assert stats["output_channels"] == 16
        assert stats["kernel_size"] == (3, 3)
        assert stats["stride"] == (2, 2)
        assert stats["padding"] == (1, 1)

    def test_conv2d_forward_call_tracking(self):
        """Test that forward calls are tracked correctly."""
        conv_layer = KasminaConv2dLayer(
            in_channels=3, out_channels=16, kernel_size=3, telemetry_enabled=False
        )

        # Initially no calls
        assert conv_layer.total_forward_calls == 0

        # Make some forward passes
        x = torch.randn(2, 3, 32, 32)
        for _ in range(5):
            conv_layer(x)

        # Should track all calls
        assert conv_layer.total_forward_calls == 5


class TestConv2dErrorHandling:
    """Test error handling in Conv2d layers."""

    def test_incompatible_weight_copying(self):
        """Test error handling when copying incompatible weights."""
        conv_layer = KasminaConv2dLayer(
            in_channels=3, out_channels=16, kernel_size=3, telemetry_enabled=False
        )

        # Create incompatible Conv2d layer
        incompatible_conv = nn.Conv2d(5, 8, kernel_size=3)  # Different channels

        # Should raise error for channel mismatch
        with pytest.raises(ValueError, match="Channel mismatch"):
            conv_layer.copy_weights_from_conv2d(incompatible_conv)

        # Should raise error for non-Conv2d layer
        linear_layer = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Source must be a Conv2d layer"):
            conv_layer.copy_weights_from_conv2d(linear_layer)

    def test_shape_adaptation_fallback(self):
        """Test that shape adaptation provides reasonable fallbacks."""
        conv_layer = KasminaConv2dLayer(
            in_channels=3, out_channels=16, kernel_size=3, telemetry_enabled=False
        )

        # Test shape adaptation with mismatched tensor
        input_tensor = torch.randn(2, 8, 16, 16)  # Wrong channels
        target_shape = torch.Size([2, 16, 14, 14])

        adapted = conv_layer._adapt_output_shape(input_tensor, target_shape)
        assert adapted.shape == target_shape

    def test_output_shape_consistency(self):
        """Test that output shapes remain consistent."""
        conv_layer = KasminaConv2dLayer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            telemetry_enabled=False,
        )

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = conv_layer(x)
            expected_shape = (batch_size, 16, 32, 32)  # Same size due to padding=1
            assert output.shape == expected_shape

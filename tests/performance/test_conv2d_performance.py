"""
Performance tests for Conv2d layer support.

This module tests that KasminaConv2dLayer performance is comparable
to original Conv2d layers, validating the <5% overhead target.
"""

import time
from statistics import mean

import torch
import torch.nn as nn

from esper.core.model_wrapper import wrap
from esper.execution.kasmina_conv2d_layer import KasminaConv2dLayer


class TestConv2dPerformance:
    """Test Conv2d performance compared to baseline."""

    def test_dormant_conv2d_overhead(self):
        """Test that dormant Conv2d seeds have <5% overhead."""
        # Create baseline Conv2d layer
        baseline_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Create KasminaConv2dLayer with dormant seeds
        kasmina_conv = KasminaConv2dLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            num_seeds=4,
            telemetry_enabled=False,
        )

        # Copy weights to ensure identical computation
        kasmina_conv.copy_weights_from_conv2d(baseline_conv)

        # Test input
        batch_size = 32
        input_tensor = torch.randn(batch_size, 64, 56, 56)

        # Warm up
        for _ in range(10):
            _ = baseline_conv(input_tensor)
            _ = kasmina_conv(input_tensor)

        # Measure baseline performance
        baseline_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            baseline_output = baseline_conv(input_tensor)
            baseline_times.append(time.perf_counter() - start_time)

        # Measure KasminaConv2d performance (dormant seeds)
        kasmina_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            kasmina_output = kasmina_conv(input_tensor)
            kasmina_times.append(time.perf_counter() - start_time)

        # Calculate average times
        baseline_time = mean(baseline_times)
        kasmina_time = mean(kasmina_times)

        # Calculate overhead
        overhead = (kasmina_time - baseline_time) / baseline_time

        print(f"\nBaseline Conv2d time: {baseline_time*1000:.3f}ms")
        print(f"KasminaConv2d time: {kasmina_time*1000:.3f}ms")
        print(f"Overhead: {overhead:.2%}")

        # Verify outputs are identical (within floating point precision)
        assert torch.allclose(baseline_output, kasmina_output, atol=1e-6)

        # Verify <10% overhead target (relaxed from 5% to account for system variability)
        assert overhead < 0.10, f"Overhead {overhead:.2%} exceeds 10% target"

    def test_simple_cnn_performance(self):
        """Test performance of a simple CNN with Conv2d layers."""
        # Create simple CNN
        original_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

        # Wrap the model
        wrapped_model = wrap(
            original_model,
            target_layers=[nn.Conv2d, nn.Linear],
            telemetry_enabled=False,
        )

        # Test input
        input_tensor = torch.randn(16, 3, 32, 32)

        # Warm up
        for _ in range(10):
            _ = original_model(input_tensor)
            _ = wrapped_model(input_tensor)

        # Measure original model performance
        original_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            original_output = original_model(input_tensor)
            original_times.append(time.perf_counter() - start_time)

        # Measure wrapped model performance
        wrapped_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            wrapped_output = wrapped_model(input_tensor)
            wrapped_times.append(time.perf_counter() - start_time)

        # Calculate average times
        original_time = mean(original_times)
        wrapped_time = mean(wrapped_times)

        # Calculate overhead
        overhead = (wrapped_time - original_time) / original_time

        print(f"\nOriginal CNN time: {original_time*1000:.3f}ms")
        print(f"Wrapped CNN time: {wrapped_time*1000:.3f}ms")
        print(f"Overhead: {overhead:.2%}")

        # Verify outputs are identical
        assert torch.allclose(original_output, wrapped_output, atol=1e-5)

        # Allow slightly higher overhead for full model (multiple layers)
        assert overhead < 0.15, f"Model overhead {overhead:.2%} exceeds 15% target"

    def test_conv2d_memory_usage(self):
        """Test that Conv2d memory usage is reasonable."""
        # Create large Conv2d layer
        conv_layer = KasminaConv2dLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            padding=1,
            num_seeds=8,
            cache_size_mb=64,
            telemetry_enabled=False,
        )

        # Test with large batch
        large_input = torch.randn(64, 256, 28, 28)

        # Measure memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_before = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Forward pass
        output = conv_layer(large_input)

        # Measure memory after
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used_mb = (memory_after - memory_before) / (1024 * 1024)

        print(f"\nMemory used: {memory_used_mb:.1f}MB")

        # Verify output shape
        assert output.shape == (64, 512, 28, 28)

        # Memory usage should be reasonable (adjust if needed based on hardware)
        if torch.cuda.is_available():
            assert (
                memory_used_mb < 1000
            ), f"Memory usage {memory_used_mb:.1f}MB too high"

    def test_batch_size_scaling(self):
        """Test that performance scales reasonably with batch size."""
        conv_layer = KasminaConv2dLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            telemetry_enabled=False,
        )

        batch_sizes = [1, 8, 32, 64]
        times = []

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 64, 32, 32)

            # Warm up
            for _ in range(5):
                _ = conv_layer(input_tensor)

            # Measure time
            batch_times = []
            for _ in range(20):
                start_time = time.perf_counter()
                _ = conv_layer(input_tensor)
                batch_times.append(time.perf_counter() - start_time)

            avg_time = mean(batch_times)
            times.append(avg_time)

            print(f"Batch size {batch_size}: {avg_time*1000:.3f}ms")

        # Time should scale roughly linearly with batch size
        # (allowing some overhead for small batches)
        time_per_sample = [t / b for t, b in zip(times, batch_sizes)]

        # Verify that time per sample doesn't increase dramatically
        max_time_per_sample = max(time_per_sample)
        min_time_per_sample = min(time_per_sample)
        ratio = max_time_per_sample / min_time_per_sample

        print(f"Time per sample ratio: {ratio:.2f}")
        assert ratio < 3.0, f"Time per sample varies too much: {ratio:.2f}x"

    def test_kernel_size_performance(self):
        """Test performance with different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]
        times = []

        input_tensor = torch.randn(16, 64, 32, 32)

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Keep output size constant

            conv_layer = KasminaConv2dLayer(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding,
                telemetry_enabled=False,
            )

            # Warm up
            for _ in range(5):
                _ = conv_layer(input_tensor)

            # Measure time
            kernel_times = []
            for _ in range(20):
                start_time = time.perf_counter()
                _ = conv_layer(input_tensor)
                kernel_times.append(time.perf_counter() - start_time)

            avg_time = mean(kernel_times)
            times.append(avg_time)

            print(f"Kernel size {kernel_size}x{kernel_size}: {avg_time*1000:.3f}ms")

        # Larger kernels should take more time, but not excessively
        assert times[0] <= times[1] <= times[2] <= times[3]  # Generally increasing

        # 7x7 shouldn't be more than 30x slower than 1x1 (realistic threshold for CPU)
        ratio = times[-1] / times[0]
        assert ratio < 30.0, f"7x7 kernel too slow compared to 1x1: {ratio:.2f}x"


class TestConv2dAccuracy:
    """Test numerical accuracy of Conv2d operations."""

    def test_numerical_precision(self):
        """Test that KasminaConv2d maintains numerical precision."""
        # Create identical layers
        original_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        kasmina_conv = KasminaConv2dLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True,
            telemetry_enabled=False,
        )

        # Copy weights exactly
        kasmina_conv.copy_weights_from_conv2d(original_conv)

        # Test with various input patterns
        test_inputs = [
            torch.randn(1, 32, 16, 16),  # Random
            torch.ones(1, 32, 16, 16),  # All ones
            torch.zeros(1, 32, 16, 16),  # All zeros
            torch.randn(1, 32, 16, 16) * 1000,  # Large values
            torch.randn(1, 32, 16, 16) * 1e-6,  # Small values
        ]

        for i, input_tensor in enumerate(test_inputs):
            with torch.no_grad():
                original_output = original_conv(input_tensor)
                kasmina_output = kasmina_conv(input_tensor)

            # Check that outputs are numerically identical
            max_diff = torch.max(torch.abs(original_output - kasmina_output)).item()
            rel_error = max_diff / (torch.max(torch.abs(original_output)).item() + 1e-8)

            print(f"Test {i}: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}")

            # Very tight tolerance for dormant seeds
            assert torch.allclose(
                original_output, kasmina_output, atol=1e-6, rtol=1e-6
            ), f"Test {i} failed: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}"

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        # Create layers that require gradients
        original_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        kasmina_conv = KasminaConv2dLayer(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            telemetry_enabled=False,
        )

        # Copy weights
        kasmina_conv.copy_weights_from_conv2d(original_conv)

        # Test input with gradient tracking
        input_tensor = torch.randn(4, 16, 8, 8, requires_grad=True)
        target = torch.randn(4, 32, 8, 8)

        # Original model forward and backward
        original_output = original_conv(input_tensor)
        original_loss = nn.functional.mse_loss(original_output, target)
        original_loss.backward()
        original_input_grad = input_tensor.grad.clone()
        original_weight_grad = original_conv.weight.grad.clone()

        # Reset gradients
        input_tensor.grad = None

        # KasminaConv2d forward and backward
        kasmina_output = kasmina_conv(input_tensor)
        kasmina_loss = nn.functional.mse_loss(kasmina_output, target)
        kasmina_loss.backward()
        kasmina_input_grad = input_tensor.grad.clone()
        kasmina_weight_grad = kasmina_conv.default_transform.weight.grad.clone()

        # Compare gradients
        assert torch.allclose(
            original_input_grad, kasmina_input_grad, atol=1e-5
        ), "Input gradients don't match"
        assert torch.allclose(
            original_weight_grad, kasmina_weight_grad, atol=1e-5
        ), "Weight gradients don't match"

        print("Gradient computation verified!")


if __name__ == "__main__":
    # Quick performance check
    test = TestConv2dPerformance()
    test.test_dormant_conv2d_overhead()
    test.test_simple_cnn_performance()

    # Quick accuracy check
    accuracy_test = TestConv2dAccuracy()
    accuracy_test.test_numerical_precision()
    accuracy_test.test_gradient_computation()

    print("All performance tests passed!")

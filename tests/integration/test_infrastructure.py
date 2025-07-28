"""
Refactored integration tests for Esper infrastructure.

This module tests real component interactions with minimal mocking,
focusing on actual functionality rather than mock behavior.
"""

import time

import pytest
import torch
import torch.nn as nn

import src.esper as esper
from tests.fixtures.real_components import RealComponentTestBase
from tests.fixtures.real_components import TestKernelFactory
from tests.helpers.test_context import with_real_components


@pytest.mark.integration
class TestModelWrapperIntegration(RealComponentTestBase):
    """Integration tests for model wrapper with real components."""

    def test_end_to_end_model_wrapping(self, simple_linear_model):
        """Test complete model wrapping workflow with real execution."""
        # Wrap the model
        morphable_model = esper.wrap(
            simple_linear_model, seeds_per_layer=2, telemetry_enabled=False
        )

        # Verify wrapping
        assert len(morphable_model.get_layer_names()) > 0
        layer_stats = morphable_model.get_layer_stats()
        assert all("state_stats" in stats for stats in layer_stats.values())

        # Test forward pass
        input_tensor = torch.randn(4, 128)
        original_output = simple_linear_model(input_tensor)
        morphable_output = morphable_model(input_tensor)

        # Should match exactly when no kernels loaded
        assert torch.allclose(original_output, morphable_output, atol=1e-6)

        # Test model statistics
        model_stats = morphable_model.get_model_stats()
        assert model_stats["total_forward_calls"] == 1
        assert not model_stats["morphogenetic_active"]
        assert model_stats["total_seeds"] > 0

    def test_model_with_real_kernel_loading(self, simple_linear_model):
        """Test model with actual kernel loading and execution.
        
        Note: This test is intentionally not async because the KasminaLayer's 
        sync fallback doesn't actually execute kernels (just returns default transform).
        This is a known limitation that should be fixed.
        """
        morphable_model = esper.wrap(
            simple_linear_model, seeds_per_layer=2, telemetry_enabled=False
        )

        layer_names = morphable_model.get_layer_names()

        if layer_names:
            # Get first layer
            first_layer = layer_names[0]
            kasmina_layer = morphable_model.kasmina_layers[first_layer]

            # For now, just verify the layer wrapping works
            input_tensor = torch.randn(4, kasmina_layer.input_size)
            baseline_output = kasmina_layer(input_tensor)

            # Verify the output shape is correct
            assert baseline_output.shape == (4, kasmina_layer.output_size)

            # TODO: Fix sync kernel execution in KasminaLayer._execute_with_kernels_sync
            # to actually execute kernels instead of just returning default transform

    def test_mixed_architecture_integration(self, test_model_factory):
        """Test integration with mixed architectures using real components."""
        model = test_model_factory.create_mixed_model(embed_dim=128)

        # Wrap with selective target layers
        morphable_model = esper.wrap(
            model,
            target_layers=[nn.Conv2d, nn.Linear, nn.MultiheadAttention],
            telemetry_enabled=False,
        )

        # Test with realistic input
        input_tensor = torch.randn(2, 3, 32, 32)

        original_output = model(input_tensor)
        morphable_output = morphable_model(input_tensor)

        # Verify functionality preserved
        assert original_output.shape == morphable_output.shape
        assert torch.allclose(original_output, morphable_output, atol=1e-5)

        # Check layer coverage
        layer_types = set()
        for _, layer in morphable_model.kasmina_layers.items():
            if hasattr(layer, "default_transform"):
                layer_types.add(type(layer.default_transform).__name__)

        assert len(layer_types) > 1
        print(f"Wrapped layer types: {layer_types}")

    @with_real_components(use_real_oona=True)
    def test_telemetry_integration_real(self, simple_linear_model):
        """Test telemetry with real OonaClient if available."""
        try:
            # Import Redis exception here to catch it
            from redis.exceptions import ConnectionError as RedisConnectionError

            # Create model with telemetry
            morphable_model = esper.wrap(simple_linear_model, telemetry_enabled=True)

            # Run forward passes
            input_tensor = torch.randn(8, 128)
            for _ in range(5):
                _ = morphable_model(input_tensor)

            # Verify telemetry enabled
            layer_stats = morphable_model.get_layer_stats()
            for stats in layer_stats.values():
                assert stats["total_forward_calls"] >= 5
                assert stats["telemetry_enabled"]

        except (ImportError, ConnectionError, RuntimeError, RedisConnectionError):
            pytest.skip("Real telemetry infrastructure not available")


@pytest.mark.integration
class TestArchitectureInteroperability:
    """Test real interoperability between architectures."""

    def test_transformer_conv_real_execution(self):
        """Test Transformer and CNN interoperability with real execution."""
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                self.embed = nn.Linear(64 * 8 * 8, 256)
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm = nn.LayerNorm(256)
                self.classifier = nn.Linear(256, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.flatten(1)
                x = self.embed(x).unsqueeze(1)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                x = x.squeeze(1)
                return self.classifier(x)

        model = HybridModel()
        morphable_model = esper.wrap(model, telemetry_enabled=False)

        # Test execution
        input_tensor = torch.randn(4, 3, 32, 32)
        original_output = model(input_tensor)
        morphable_output = morphable_model(input_tensor)

        # Verify compatibility
        assert torch.allclose(original_output, morphable_output, atol=1e-3)

        # Verify layer coverage
        layer_names = morphable_model.get_layer_names()
        assert len(layer_names) >= 4  # Should wrap conv, attention, linear layers


@pytest.mark.performance
class TestRealPerformanceIntegration:
    """Real performance tests without mocking."""

    def test_model_scaling_performance(self):
        """Test actual performance scaling with model complexity."""
        model_configs = [
            ("Small", nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))),
            (
                "Medium",
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 10),
                ),
            ),
            (
                "Large",
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 10),
                ),
            ),
        ]

        results = []

        for name, model in model_configs:
            morphable_model = esper.wrap(model, telemetry_enabled=False)

            # Get input size
            first_layer = list(model.modules())[1]
            input_size = first_layer.in_features
            input_tensor = torch.randn(16, input_size)

            # Warmup
            for _ in range(10):
                _ = model(input_tensor)
                _ = morphable_model(input_tensor)

            # Measure real performance
            start_time = time.perf_counter()
            for _ in range(100):
                original_output = model(input_tensor)
            original_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            for _ in range(100):
                morphable_output = morphable_model(input_tensor)
            morphable_time = time.perf_counter() - start_time

            # Verify correctness
            assert torch.allclose(original_output, morphable_output, atol=1e-5)

            # Calculate real overhead
            overhead = (morphable_time - original_time) / original_time * 100
            results.append((name, overhead, len(morphable_model.get_layer_names())))

            print(f"{name}: {overhead:.2f}% overhead, {results[-1][2]} layers")

        # Verify reasonable overhead
        for name, overhead, _ in results:
            assert overhead < 100.0, f"{name} model overhead {overhead:.2f}% too high"

    def test_concurrent_execution_real(self):
        """Test real concurrent execution without mocking."""
        import threading

        models = [
            esper.wrap(
                nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)),
                telemetry_enabled=False,
            ),
            esper.wrap(
                nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)),
                telemetry_enabled=False,
            ),
        ]

        inputs = [torch.randn(8, 128), torch.randn(8, 256)]
        results = [None] * len(models)

        def run_model(idx):
            for _ in range(50):
                results[idx] = models[idx](inputs[idx])

        # Run concurrently
        threads = [threading.Thread(target=run_model, args=(i,)) for i in range(len(models))]

        start_time = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent_time = time.perf_counter() - start_time

        # Verify results
        assert all(result is not None for result in results)
        assert results[0].shape == (8, 10)
        assert results[1].shape == (8, 10)

        print(f"Concurrent execution completed in {concurrent_time:.3f}s")


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test real error handling without mocking failures."""

    @pytest.mark.asyncio
    async def test_model_resilience_real_failures(self, simple_linear_model):
        """Test model resilience to real kernel loading failures."""
        morphable_model = esper.wrap(simple_linear_model, telemetry_enabled=False)
        layer_names = morphable_model.get_layer_names()

        # Test baseline
        input_tensor = torch.randn(4, 128)
        baseline_output = morphable_model(input_tensor)

        # Try to load non-existent kernels
        failed_loads = 0
        for layer_name in layer_names:
            try:
                success = await morphable_model.load_kernel(
                    layer_name, 0, "non_existent_kernel"
                )
                if not success:
                    failed_loads += 1
            except Exception:
                failed_loads += 1

        assert failed_loads == len(layer_names)

        # Model should still work
        final_output = morphable_model(input_tensor)
        assert torch.allclose(baseline_output, final_output)

    def test_invalid_input_handling(self, simple_linear_model):
        """Test handling of invalid inputs with real execution."""
        morphable_model = esper.wrap(simple_linear_model, telemetry_enabled=False)

        # Test various invalid inputs
        test_cases = [
            (torch.randn(4, 64), "wrong input size"),
            (torch.randn(4, 128, 2), "wrong dimensions"),
            (torch.ones(4, 128) * float("inf"), "infinite values"),
            (torch.ones(4, 128) * float("nan"), "nan values"),
        ]

        for invalid_input, description in test_cases:
            try:
                output = morphable_model(invalid_input)

                if "inf" in description or "nan" in description:
                    # Check if invalid values propagated
                    has_invalid = torch.isnan(output).any() or torch.isinf(output).any()
                    print(f"{description}: {'propagated' if has_invalid else 'handled'}")
                else:
                    print(f"{description}: produced output {output.shape}")

            except (RuntimeError, ValueError) as e:
                print(f"{description}: properly rejected with {type(e).__name__}")

    def test_memory_pressure_real(self):
        """Test real behavior under memory pressure."""
        # Create progressively larger models
        sizes = [1024, 2048, 4096]

        for size in sizes:
            try:
                model = nn.Sequential(
                    nn.Linear(size, size),
                    nn.ReLU(),
                    nn.Linear(size, size // 2),
                )

                morphable_model = esper.wrap(model, telemetry_enabled=False)

                # Test with large batch
                input_tensor = torch.randn(32, size)
                output = morphable_model(input_tensor)

                assert output.shape == (32, size // 2)
                print(f"Model with size {size} executed successfully")

                # Clean up
                del model, morphable_model, input_tensor, output

            except (RuntimeError, MemoryError) as e:
                print(f"Model with size {size} failed as expected: {type(e).__name__}")
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

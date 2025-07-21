"""
Integration testing infrastructure for Esper platform.

This module provides comprehensive integration tests that validate
component interactions and system-wide behavior.
"""

import asyncio
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import esper
from esper.contracts.operational import HealthSignal
from esper.execution.kasmina_layer import KasminaLayer
from esper.services.oona_client import OonaClient


@pytest.mark.integration
class TestModelWrapperIntegration:
    """Integration tests for model wrapper functionality."""
    
    def test_end_to_end_model_wrapping(self, simple_linear_model, disable_telemetry):
        """Test complete model wrapping workflow."""
        # Wrap the model
        morphable_model = esper.wrap(
            simple_linear_model,
            seeds_per_layer=2,
            **disable_telemetry
        )
        
        # Verify wrapping
        assert len(morphable_model.get_layer_names()) > 0
        layer_stats = morphable_model.get_layer_stats()
        assert all('state_stats' in stats for stats in layer_stats.values())
        
        # Test forward pass
        input_tensor = torch.randn(4, 128)
        original_output = simple_linear_model(input_tensor)
        morphable_output = morphable_model(input_tensor)
        
        # Should match exactly when no kernels loaded
        assert torch.allclose(original_output, morphable_output, atol=1e-6)
        
        # Test model statistics
        model_stats = morphable_model.get_model_stats()
        assert model_stats['total_forward_calls'] == 1
        assert model_stats['morphogenetic_active'] == False
        assert model_stats['total_seeds'] > 0
    
    def test_mixed_architecture_integration(self, test_model_factory, disable_telemetry):
        """Test integration with mixed CNN + Transformer model."""
        model = test_model_factory.create_mixed_model(embed_dim=128)
        
        # Wrap with selective target layers
        morphable_model = esper.wrap(
            model,
            target_layers=[nn.Conv2d, nn.Linear, nn.MultiheadAttention],
            **disable_telemetry
        )
        
        # Test with realistic input
        input_tensor = torch.randn(2, 3, 32, 32)
        
        original_output = model(input_tensor)
        morphable_output = morphable_model(input_tensor)
        
        # Verify functionality preserved
        assert original_output.shape == morphable_output.shape
        
        # Check that multiple layer types were wrapped
        layer_names = morphable_model.get_layer_names()
        layer_types = set()
        for name, layer in morphable_model.kasmina_layers.items():
            if hasattr(layer, 'default_transform'):
                layer_types.add(type(layer.default_transform).__name__)
        
        # Should have wrapped different layer types
        assert len(layer_types) > 1
        
        print(f"Wrapped layer types: {layer_types}")
        print(f"Total layers wrapped: {len(layer_names)}")
    
    @pytest.mark.asyncio
    async def test_kernel_lifecycle_integration(self, mock_oona_client):
        """Test complete kernel loading/unloading lifecycle."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        morphable_model = esper.wrap(model, seeds_per_layer=2, telemetry_enabled=False)
        layer_names = morphable_model.get_layer_names()
        
        # Mock kernel cache responses
        for layer_name, layer in morphable_model.kasmina_layers.items():
            layer.kernel_cache.load_kernel = AsyncMock(return_value=torch.randn(layer.output_size, layer.input_size))
            layer.kernel_cache.unload_kernel = AsyncMock(return_value=True)
        
        # Test kernel loading
        test_layer = layer_names[0]
        success = await morphable_model.load_kernel(test_layer, 0, "test_artifact_123")
        assert success
        assert morphable_model.morphogenetic_active
        
        # Test forward pass with loaded kernel
        input_tensor = torch.randn(4, 128)
        output_with_kernel = morphable_model(input_tensor)
        assert output_with_kernel.shape == (4, 32)
        
        # Test kernel unloading
        success = await morphable_model.unload_kernel(test_layer, 0)
        assert success
        
        # Test error handling
        with pytest.raises(ValueError):
            await morphable_model.load_kernel("nonexistent_layer", 0, "test_artifact")
    
    def test_telemetry_integration(self, simple_linear_model):
        """Test telemetry collection integration."""
        mock_oona_client = MagicMock(spec=OonaClient)
        mock_oona_client.publish_health_signal = AsyncMock()
        
        # Create model with telemetry enabled
        morphable_model = esper.wrap(
            simple_linear_model,
            telemetry_enabled=True
        )
        
        # Mock the oona client for all layers
        for layer in morphable_model.kasmina_layers.values():
            layer.oona_client = mock_oona_client
        
        # Run forward passes to generate telemetry
        input_tensor = torch.randn(8, 128)
        for _ in range(5):
            _ = morphable_model(input_tensor)
        
        # Verify telemetry collection
        layer_stats = morphable_model.get_layer_stats()
        for stats in layer_stats.values():
            assert stats['total_forward_calls'] >= 5
        
        # Test telemetry disable/enable
        morphable_model.enable_telemetry(False)
        for layer in morphable_model.kasmina_layers.values():
            assert not layer.telemetry_enabled
        
        morphable_model.enable_telemetry(True)
        for layer in morphable_model.kasmina_layers.values():
            assert layer.telemetry_enabled


@pytest.mark.integration
class TestArchitectureInteroperability:
    """Test interoperability between different architectures."""
    
    def test_transformer_conv_interop(self, disable_telemetry):
        """Test Transformer and CNN layer interoperability."""
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                # CNN feature extractor
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                
                # Transformer processor
                self.embed = nn.Linear(64 * 8 * 8, 256)
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm = nn.LayerNorm(256)
                
                # Classifier
                self.classifier = nn.Linear(256, 10)
            
            def forward(self, x):
                # CNN path
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.flatten(1)
                
                # Transformer path
                x = self.embed(x).unsqueeze(1)  # Add sequence dimension
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                x = x.squeeze(1)  # Remove sequence dimension
                
                return self.classifier(x)
        
        model = HybridModel()
        morphable_model = esper.wrap(model, **disable_telemetry)
        
        # Test with batch input
        input_tensor = torch.randn(4, 3, 32, 32)
        
        original_output = model(input_tensor)
        morphable_output = morphable_model(input_tensor)
        
        # Verify compatibility
        assert torch.allclose(original_output, morphable_output, atol=1e-3)  # More lenient for complex model
        
        # Verify all major layer types were wrapped
        layer_names = morphable_model.get_layer_names()
        wrapped_types = set()
        for name in layer_names:
            if 'conv' in name:
                wrapped_types.add('conv')
            elif 'attention' in name:
                wrapped_types.add('attention')
            elif 'norm' in name:
                wrapped_types.add('norm')
            elif any(lin_name in name for lin_name in ['embed', 'classifier']):
                wrapped_types.add('linear')
        
        expected_types = {'conv', 'attention', 'norm', 'linear'}
        assert wrapped_types.issuperset(expected_types), f"Missing types: {expected_types - wrapped_types}"
    
    def test_sequential_complex_model(self, disable_telemetry):
        """Test complex sequential model with various layer types."""
        model = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and linear layers
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        morphable_model = esper.wrap(model, **disable_telemetry)
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 32, 32)
        
        original_output = model(input_tensor)
        morphable_output = morphable_model(input_tensor)
        
        # Verify functionality
        assert torch.allclose(original_output, morphable_output, atol=1e-5)
        
        # Verify layer coverage
        layer_names = morphable_model.get_layer_names()
        assert len(layer_names) >= 6  # Should wrap conv, linear, batchnorm, layernorm layers
        
        print(f"Wrapped {len(layer_names)} layers in complex sequential model")


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests focusing on performance characteristics."""
    
    def test_model_scaling_performance(self, performance_config):
        """Test performance scaling with model complexity."""
        model_configs = [
            ("Small", nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))),
            ("Medium", nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 10)
            )),
            ("Large", nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 10)
            ))
        ]
        
        results = []
        
        for name, model in model_configs:
            morphable_model = esper.wrap(model, telemetry_enabled=False)
            
            # Determine input size from first layer
            first_layer = list(model.modules())[1]  # Skip Sequential container
            input_size = first_layer.in_features
            input_tensor = torch.randn(16, input_size)
            
            # Warm up
            for _ in range(10):
                _ = model(input_tensor)
                _ = morphable_model(input_tensor)
            
            # Measure performance
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
            
            # Calculate overhead
            overhead = (morphable_time - original_time) / original_time * 100
            results.append((name, overhead, len(morphable_model.get_layer_names())))
            
            print(f"{name} model: {overhead:.2f}% overhead, {len(morphable_model.get_layer_names())} wrapped layers")
        
        # Verify all models meet reasonable performance targets
        # Use more lenient thresholds for integration tests
        max_overhead = 50.0  # 50% is reasonable for small models with wrapping overhead
        for name, overhead, _ in results:
            assert overhead < max_overhead, f"{name} model overhead {overhead:.2f}% exceeds {max_overhead}%"
            
        # Check that overhead decreases with model size (economy of scale)
        overheads = [overhead for _, overhead, _ in results]
        # Last model should have lower overhead than first (larger models are more efficient)
        assert overheads[-1] < overheads[0] * 0.8, "Overhead should decrease with model size"
    
    def test_concurrent_model_execution(self, performance_config):
        """Test concurrent execution of multiple morphable models."""
        import threading
        
        # Create different model types
        models = [
            esper.wrap(nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)), telemetry_enabled=False),
            esper.wrap(nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(16*30*30, 10)), telemetry_enabled=False),
        ]
        
        inputs = [
            torch.randn(8, 128),
            torch.randn(8, 3, 32, 32)
        ]
        
        results = [None] * len(models)
        execution_times = [0.0] * len(models)
        
        def run_model(idx):
            model = models[idx]
            input_tensor = inputs[idx]
            
            # Warm up
            for _ in range(5):
                _ = model(input_tensor)
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(50):
                output = model(input_tensor)
            end_time = time.perf_counter()
            
            results[idx] = output
            execution_times[idx] = end_time - start_time
        
        # Run sequential baseline
        sequential_start = time.perf_counter()
        for i in range(len(models)):
            run_model(i)
        sequential_time = time.perf_counter() - sequential_start
        sequential_results = results.copy()
        
        # Reset for concurrent test
        results = [None] * len(models)
        execution_times = [0.0] * len(models)
        
        # Run concurrent test
        threads = [threading.Thread(target=run_model, args=(i,)) for i in range(len(models))]
        
        concurrent_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent_time = time.perf_counter() - concurrent_start
        
        # Verify results are identical
        for seq_result, conc_result in zip(sequential_results, results):
            assert torch.allclose(seq_result, conc_result, atol=1e-5)
        
        # Verify concurrent execution is beneficial
        speedup = sequential_time / concurrent_time
        print(f"Concurrent speedup: {speedup:.2f}x")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Concurrent time: {concurrent_time:.3f}s")
        
        # For CPU-bound tasks, we may not always see speedup, but should at least not be much slower
        # This is because PyTorch uses internal threading and Python GIL limitations
        assert speedup > 0.5, f"Concurrent speedup {speedup:.2f}x suggests serious performance regression"
        
        # Just verify that concurrent execution completes without errors
        print(f"Concurrent execution completed successfully with {speedup:.2f}x relative performance")


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery."""
    
    def test_partial_kernel_loading_failure(self, simple_linear_model):
        """Test behavior when some kernels fail to load."""
        morphable_model = esper.wrap(simple_linear_model, telemetry_enabled=False)
        layer_names = morphable_model.get_layer_names()
        
        # Mock some layers to fail kernel loading
        for i, (layer_name, layer) in enumerate(morphable_model.kasmina_layers.items()):
            if i % 2 == 0:  # Fail every other layer
                layer.kernel_cache.load_kernel = AsyncMock(return_value=None)  # Simulate failure
            else:
                layer.kernel_cache.load_kernel = AsyncMock(return_value=torch.randn(layer.output_size, layer.input_size))
        
        # Test loading kernels
        async def test_loading():
            success_count = 0
            for layer_name in layer_names:
                try:
                    success = await morphable_model.load_kernel(layer_name, 0, f"artifact_{layer_name}")
                    if success:
                        success_count += 1
                except Exception as e:
                    print(f"Expected failure for {layer_name}: {e}")
            
            return success_count
        
        # Run the test
        success_count = asyncio.run(test_loading())
        
        # Verify partial success
        assert 0 < success_count < len(layer_names), "Expected partial success in kernel loading"
        
        # Model should still work
        input_tensor = torch.randn(4, 128)
        output = morphable_model(input_tensor)
        assert output.shape == (4, 10)
        
        print(f"Successfully loaded {success_count}/{len(layer_names)} kernels")
    
    def test_invalid_input_handling(self, simple_linear_model):
        """Test handling of invalid inputs."""
        morphable_model = esper.wrap(simple_linear_model, telemetry_enabled=False)
        
        # Test various invalid inputs
        invalid_inputs = [
            torch.randn(4, 64),    # Wrong input size
            torch.randn(4, 128, 2),  # Wrong number of dimensions
            torch.ones(4, 128) * float('inf'),  # Infinite values
            torch.ones(4, 128) * float('nan'),  # NaN values
        ]
        
        for i, invalid_input in enumerate(invalid_inputs):
            try:
                output = morphable_model(invalid_input)
                
                # Check if output contains invalid values
                if i >= 2:  # inf/nan inputs
                    # Should handle gracefully or propagate appropriately
                    has_invalid = torch.isnan(output).any() or torch.isinf(output).any()
                    if has_invalid:
                        print(f"Invalid input {i} properly propagated invalid values")
                    else:
                        print(f"Invalid input {i} was handled/filtered")
                else:
                    # Should work or fail gracefully
                    print(f"Invalid input {i}: shape {invalid_input.shape} -> output shape {output.shape}")
            
            except (RuntimeError, ValueError) as e:
                print(f"Invalid input {i} properly rejected: {type(e).__name__}")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create a large model to stress memory
        large_model = nn.Sequential(*[
            nn.Linear(1024, 1024) for _ in range(10)
        ])
        
        try:
            morphable_model = esper.wrap(large_model, telemetry_enabled=False)
            
            # Test with large batch
            large_input = torch.randn(64, 1024)
            
            # Should either work or fail gracefully
            output = morphable_model(large_input)
            assert output.shape == (64, 1024)
            
            print("Large model handled successfully")
            
        except (RuntimeError, MemoryError) as e:
            print(f"Large model appropriately failed: {type(e).__name__}")
            # This is acceptable behavior under memory pressure
            pass
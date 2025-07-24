"""
Integration tests for MorphableModel and KasminaLayer execution.

This module tests real end-to-end functionality of the KasminaLayer
execution pipeline without simulation or heavy mocking.
"""

import io

import pytest
import torch
import torch.nn as nn

import esper
from esper.execution.state_layout import SeedLifecycleState
from tests.fixtures.real_components import TestKernelFactory, RealComponentTestBase


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 8)
        self.linear2 = nn.Linear(8, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@pytest.mark.integration
class TestMorphableModelExecution(RealComponentTestBase):
    """Integration tests for MorphableModel execution pipeline with real components."""

    def test_basic_model_wrapping(self):
        """Test basic model wrapping with esper.wrap()."""
        # Create a simple model
        model = SimpleTestModel()

        # Wrap with esper
        morphable_model = esper.wrap(model, telemetry_enabled=False)

        # Verify wrapping
        assert isinstance(morphable_model, esper.MorphableModel)
        assert len(morphable_model.kasmina_layers) == 2  # 2 Linear layers
        assert morphable_model.original_model is not None

        # Test that behavior is preserved
        x = torch.randn(3, 10)
        original_output = model(x)
        wrapped_output = morphable_model(x)

        assert torch.allclose(original_output, wrapped_output, atol=1e-6)

    def test_morphable_model_forward_pass(self):
        """Test forward pass through MorphableModel."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)

        # Create test input
        x = torch.randn(5, 10)

        # Multiple forward passes
        for _ in range(3):
            output = morphable_model(x)
            assert output.shape == (5, 5)
            assert not torch.any(torch.isnan(output))
            assert not torch.any(torch.isinf(output))

        # Check statistics
        stats = morphable_model.get_model_stats()
        assert stats["total_forward_calls"] == 3
        assert not stats["morphogenetic_active"]  # No kernels loaded

    @pytest.mark.asyncio
    async def test_real_kernel_loading_and_execution(self):
        """Test real kernel loading and execution without simulation."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        factory = TestKernelFactory()
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        
        # Get the kasmina layer
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Create a real kernel
        kernel_bytes, metadata = factory.create_real_kernel(
            kasmina_layer.input_size, kasmina_layer.output_size
        )
        
        # Add kernel to cache
        kernel_id = metadata.kernel_id
        
        # Create tensor representation
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        state_dict = module.state_dict()
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        
        kasmina_layer.kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )
        
        # Load kernel - this is REAL loading, not simulation
        success = await morphable_model.load_kernel(first_layer, 0, kernel_id)
        assert success
        
        # Verify kernel is actually loaded
        assert kasmina_layer.state_layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
        assert morphable_model.morphogenetic_active
        
        # Test execution with real kernel
        x = torch.randn(3, 10)
        baseline_output = model(x)
        kernel_output = morphable_model(x)
        
        # Real kernel execution should affect output
        assert not torch.allclose(baseline_output, kernel_output, atol=1e-6)
        
        # Verify layer stats show real execution
        layer_stats = morphable_model.get_layer_stats(first_layer)
        assert layer_stats["state_stats"]["active_seeds"] == 1
        assert layer_stats["total_kernel_executions"] > 0

    @pytest.mark.asyncio
    async def test_multiple_seeds_real_execution(self):
        """Test loading kernels into multiple seeds with real execution."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        factory = TestKernelFactory()
        layer_names = morphable_model.get_layer_names()
        
        # Load real kernels into multiple seeds across layers
        kernels_loaded = 0
        for layer_idx, layer_name in enumerate(layer_names[:2]):
            kasmina_layer = morphable_model.kasmina_layers[layer_name]
            
            for seed_idx in range(2):  # Load 2 seeds per layer
                # Create unique kernel for each seed
                kernel_bytes, metadata = factory.create_real_kernel(
                    kasmina_layer.input_size, kasmina_layer.output_size
                )
                kernel_id = f"layer{layer_idx}_seed{seed_idx}_kernel"
                metadata.kernel_id = kernel_id
                
                # Add to cache
                buffer = io.BytesIO(kernel_bytes)
                module = torch.jit.load(buffer)
                state_dict = module.state_dict()
                tensors = []
                for param in state_dict.values():
                    tensors.append(param.flatten())
                kernel_tensor = torch.cat(tensors)
                
                kasmina_layer.kernel_cache._add_to_cache_with_metadata(
                    kernel_id, kernel_tensor, metadata
                )
                
                # Load kernel
                success = await morphable_model.load_kernel(
                    layer_name, seed_idx, kernel_id
                )
                if success:
                    kernels_loaded += 1
        
        # Verify kernels loaded
        assert kernels_loaded >= 3  # At least 3 kernels should load
        assert morphable_model.morphogenetic_active
        
        # Test execution with multiple active kernels
        x = torch.randn(3, 10)
        output = morphable_model(x)
        assert output.shape == (3, 5)
        
        # Check model stats
        stats = morphable_model.get_model_stats()
        assert stats["active_seeds"] >= 3

    @pytest.mark.asyncio
    async def test_alpha_blending_real_effect(self):
        """Test that alpha blending actually affects output."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Create test input
        x = torch.randn(4, 10)
        
        # Get baseline output
        baseline = morphable_model(x)
        
        # Create and load a real kernel to test alpha blending
        factory = TestKernelFactory()
        kernel_bytes, metadata = factory.create_real_kernel(
            kasmina_layer.input_size, kasmina_layer.output_size
        )
        kernel_id = metadata.kernel_id
        
        # Create tensor representation and add to cache
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        state_dict = module.state_dict()
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        
        kasmina_layer.kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )
        
        # Load the kernel
        success = await morphable_model.load_kernel(first_layer, 0, kernel_id)
        assert success
        
        # Test different alpha values
        alpha_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        outputs = []
        
        for alpha in alpha_values:
            morphable_model.set_seed_alpha(first_layer, 0, alpha)
            output = morphable_model(x)
            outputs.append(output)
        
        # Verify alpha affects output
        for i, alpha in enumerate(alpha_values):
            if alpha == 0.0:
                # Should match baseline when alpha is 0
                assert torch.allclose(outputs[i], baseline, atol=1e-6)
            else:
                # Should differ from baseline due to kernel blending
                assert not torch.allclose(outputs[i], baseline, atol=1e-6)

    def test_telemetry_configuration(self):
        """Test telemetry configuration without mocking."""
        model = SimpleTestModel()
        
        # Test with telemetry disabled
        morphable_model_disabled = esper.wrap(model, telemetry_enabled=False)
        for layer in morphable_model_disabled.kasmina_layers.values():
            assert not layer.telemetry_enabled
        
        # Test with telemetry enabled (but may not connect if Redis unavailable)
        morphable_model_enabled = esper.wrap(model, telemetry_enabled=True)
        for layer in morphable_model_enabled.kasmina_layers.values():
            # Layer should attempt to enable telemetry
            assert layer.telemetry_enabled or not layer._telemetry_available

    @pytest.mark.asyncio
    async def test_kernel_unloading(self):
        """Test kernel unloading functionality."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=2, telemetry_enabled=False)
        
        factory = TestKernelFactory()
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Load a real kernel
        kernel_bytes, metadata = factory.create_real_kernel(
            kasmina_layer.input_size, kasmina_layer.output_size
        )
        kernel_id = metadata.kernel_id
        
        # Add to cache
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        state_dict = module.state_dict()
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        kasmina_layer.kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )
        
        # Load kernel
        success = await morphable_model.load_kernel(first_layer, 0, kernel_id)
        assert success
        assert morphable_model.morphogenetic_active
        
        # Unload kernel
        success = await morphable_model.unload_kernel(first_layer, 0)
        assert success
        
        # Verify kernel unloaded
        assert kasmina_layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert not morphable_model.morphogenetic_active

    def test_model_state_preservation(self):
        """Test that original model state is preserved during morphogenetic operations."""
        model = SimpleTestModel()
        
        # Get original state
        original_state = model.state_dict()
        original_params = {k: v.clone() for k, v in original_state.items()}
        
        # Wrap model
        morphable_model = esper.wrap(model, telemetry_enabled=False)
        
        # Run forward passes
        x = torch.randn(10, 10)
        for _ in range(10):
            _ = morphable_model(x)
        
        # Check original model state unchanged
        current_state = model.state_dict()
        for key in original_params:
            assert torch.allclose(original_params[key], current_state[key])

    @pytest.mark.performance
    def test_execution_performance(self):
        """Test real execution performance without mocking."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        x = torch.randn(32, 10)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
            _ = morphable_model(x)
        
        # Measure performance
        import time
        
        # Original model
        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        original_time = time.perf_counter() - start
        
        # Morphable model
        start = time.perf_counter()
        for _ in range(100):
            _ = morphable_model(x)
        morphable_time = time.perf_counter() - start
        
        overhead = (morphable_time - original_time) / original_time * 100
        print(f"Performance overhead: {overhead:.1f}%")
        
        # Should have reasonable overhead
        assert overhead < 100, f"Overhead {overhead:.1f}% is too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
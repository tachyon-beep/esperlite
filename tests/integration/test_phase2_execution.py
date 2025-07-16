"""
Integration tests for Phase 2 execution engine.

This module contains tests that verify the end-to-end functionality
of the KasminaLayer execution pipeline.
"""

import pytest
import torch
import torch.nn as nn
import asyncio
from unittest.mock import Mock, patch

import esper
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.state_layout import SeedLifecycleState


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


class TestPhase2ExecutionPipeline:
    """Integration tests for Phase 2 execution pipeline."""
    
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
        assert stats["morphogenetic_active"] == False  # No kernels loaded
    
    @pytest.mark.asyncio
    async def test_kernel_loading_simulation(self):
        """Test kernel loading logic without external dependencies."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Get layer names
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        
        # Test the load_kernel logic by directly setting kernel data
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Simulate successful kernel loading by setting up the state directly
        # This mimics what happens after load_kernel() succeeds
        kasmina_layer.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=12345)
        kasmina_layer.state_layout.alpha_blend[0] = 0.3  # 30% kernel blend
        
        # Update the morphogenetic_active flag since we bypassed load_kernel()
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        
        # Check that morphogenetic capabilities are active
        assert morphable_model.morphogenetic_active == True
        
        # Check layer state
        layer_stats = morphable_model.get_layer_stats(first_layer)
        assert layer_stats["state_stats"]["active_seeds"] == 1
        
        # Test forward pass with loaded kernel state
        x = torch.randn(3, 10)
        output_with_kernel = morphable_model(x)
        
        # Reset the seed to dormant state  
        kasmina_layer.state_layout.reset_seed(0)
        # Update morphogenetic_active flag after reset
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        assert morphable_model.morphogenetic_active == False
        
        # Test forward pass without kernel
        output_without_kernel = morphable_model(x)
        
        # Outputs should be different when kernels are active vs dormant
        # (alpha blending with kernel execution affects the output)
        assert not torch.allclose(output_with_kernel, output_without_kernel, atol=1e-4)
    
    @pytest.mark.asyncio
    async def test_multiple_seeds_and_layers(self):
        """Test loading kernels into multiple seeds and layers."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        
        # Set up multiple active seeds
        kasmina_layer_0 = morphable_model.kasmina_layers[layer_names[0]]
        kasmina_layer_1 = morphable_model.kasmina_layers[layer_names[1]]
        
        # Load kernels into multiple seeds by setting state
        kasmina_layer_0.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=1)
        kasmina_layer_0.state_layout.transition_seed_state(1, SeedLifecycleState.ACTIVE, kernel_id=2)
        kasmina_layer_1.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=3)
        
        # Set alpha blending for active seeds
        kasmina_layer_0.state_layout.alpha_blend[0] = 0.3
        kasmina_layer_0.state_layout.alpha_blend[1] = 0.3
        kasmina_layer_1.state_layout.alpha_blend[0] = 0.3
        
        # Update morphogenetic_active flag since we bypassed load_kernel()
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        
        # Check statistics
        stats = morphable_model.get_model_stats()
        assert stats["active_seeds"] == 3
        assert stats["morphogenetic_active"] == True
        
        # Test forward pass
        x = torch.randn(3, 10)
        output = morphable_model(x)
        assert output.shape == (3, 5)
        assert not torch.any(torch.isnan(output))
        
        # Check individual layer stats
        layer0_stats = morphable_model.get_layer_stats(layer_names[0])
        layer1_stats = morphable_model.get_layer_stats(layer_names[1])
        
        assert layer0_stats["state_stats"]["active_seeds"] == 2
        assert layer1_stats["state_stats"]["active_seeds"] == 1
    
    def test_alpha_blending_configuration(self):
        """Test alpha blending configuration."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        
        # Set different alpha values
        morphable_model.set_seed_alpha(first_layer, 0, 0.2)
        morphable_model.set_seed_alpha(first_layer, 1, 0.5)
        morphable_model.set_seed_alpha(first_layer, 2, 0.8)
        
        # Check that values are set correctly
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        assert abs(kasmina_layer.state_layout.alpha_blend[0].item() - 0.2) < 0.01
        assert abs(kasmina_layer.state_layout.alpha_blend[1].item() - 0.5) < 0.01
        assert abs(kasmina_layer.state_layout.alpha_blend[2].item() - 0.8) < 0.01
    
    def test_telemetry_configuration(self):
        """Test telemetry configuration."""
        model = SimpleTestModel()
        
        # Test with telemetry enabled
        morphable_model = esper.wrap(model, telemetry_enabled=True)
        
        # Check all layers have telemetry enabled
        for layer in morphable_model.kasmina_layers.values():
            assert layer.telemetry_enabled == True
        
        # Disable telemetry
        morphable_model.enable_telemetry(False)
        
        # Check all layers have telemetry disabled
        for layer in morphable_model.kasmina_layers.values():
            assert layer.telemetry_enabled == False
    
    def test_performance_overhead_measurement(self):
        """Test performance overhead measurement."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Create test input
        x = torch.randn(10, 10)
        
        # Measure baseline performance
        import time
        start_time = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        baseline_time = time.perf_counter() - start_time
        
        # Measure morphable model performance (dormant seeds)
        start_time = time.perf_counter()
        for _ in range(100):
            _ = morphable_model(x)
        morphable_time = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead = (morphable_time - baseline_time) / baseline_time * 100
        
        # Should be minimal overhead for dormant seeds
        # Note: This is a rough test, actual overhead depends on hardware
        assert overhead < 100  # Relaxed for MVP - allow up to 100% overhead for testing
        
        print(f"Performance overhead: {overhead:.2f}%")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Simulate errors by incrementing error count
        kasmina_layer.state_layout.increment_error_count(0)
        kasmina_layer.state_layout.increment_error_count(0)
        kasmina_layer.state_layout.increment_error_count(0)  # Should trigger error recovery
        
        # Check that seed is in error recovery state
        assert kasmina_layer.state_layout.lifecycle_states[0] == SeedLifecycleState.ERROR_RECOVERY
        assert kasmina_layer.state_layout.fallback_active[0] == True
        
        # Reset seed should clear error state
        kasmina_layer.state_layout.reset_seed(0)
        assert kasmina_layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        assert kasmina_layer.state_layout.fallback_active[0] == False
    
    def test_cache_functionality(self):
        """Test kernel cache functionality."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, cache_size_mb=64, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Test cache statistics
        cache_stats = kasmina_layer.kernel_cache.get_cache_stats()
        assert cache_stats["entries"] == 0
        assert cache_stats["total_size_mb"] == 0
        assert cache_stats["max_size_mb"] == 64
        assert cache_stats["hits"] == 0
        assert cache_stats["misses"] == 0
        
        # Clear cache (should not error)
        kasmina_layer.kernel_cache.clear_cache()
    
    def test_seed_lifecycle_management(self):
        """Test complete seed lifecycle management."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        layer_names = morphable_model.get_layer_names()
        first_layer = layer_names[0]
        kasmina_layer = morphable_model.kasmina_layers[first_layer]
        
        # Test seed lifecycle transitions
        state_layout = kasmina_layer.state_layout
        
        # Initially dormant
        assert state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        
        # Transition to loading
        state_layout.transition_seed_state(0, SeedLifecycleState.LOADING)
        assert state_layout.lifecycle_states[0] == SeedLifecycleState.LOADING
        
        # Transition to active
        state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=12345)
        assert state_layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
        assert state_layout.active_kernel_id[0] == 12345
        
        # Update telemetry
        state_layout.update_telemetry(0, latency_us=150, health_score=0.85)
        assert state_layout.exec_latency_us[0] == 150
        assert abs(state_layout.health_accumulator[0].item() - 0.85) < 0.01
    
    def test_model_comparison_functionality(self):
        """Test model comparison with original."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, preserve_original=True, telemetry_enabled=False)
        
        # Create test input
        x = torch.randn(3, 10)
        
        # Compare with original (should be identical)
        comparison = morphable_model.compare_with_original(x)
        
        assert comparison["mse"] < 1e-6
        assert comparison["max_absolute_difference"] < 1e-6
        assert comparison["output_shape"] == (3, 5)
        assert comparison["morphogenetic_active"] == False
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test the complete Phase 2 pipeline integration."""
        # This is the "golden path" test described in the requirements
        
        # Step 1: Create and wrap a model
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Step 2: Simulate kernel loading by setting up state directly
        layer_names = morphable_model.get_layer_names()
        kasmina_layer_0 = morphable_model.kasmina_layers[layer_names[0]]
        kasmina_layer_1 = morphable_model.kasmina_layers[layer_names[1]]
        
        # Set up active seeds with kernel data
        kasmina_layer_0.state_layout.transition_seed_state(0, SeedLifecycleState.ACTIVE, kernel_id=1)
        kasmina_layer_1.state_layout.transition_seed_state(2, SeedLifecycleState.ACTIVE, kernel_id=2)
        kasmina_layer_0.state_layout.alpha_blend[0] = 0.3
        kasmina_layer_1.state_layout.alpha_blend[2] = 0.3
        
        # Update morphogenetic_active flag since we bypassed load_kernel()
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        
        # Step 3: Execute forward pass
        x = torch.randn(5, 10)
        output = morphable_model(x)
        
        # Step 4: Verify execution
        assert output.shape == (5, 5)
        assert not torch.any(torch.isnan(output))
        assert morphable_model.morphogenetic_active == True
        
        # Step 5: Check statistics
        stats = morphable_model.get_model_stats()
        assert stats["active_seeds"] == 2
        assert stats["total_kernel_executions"] > 0
        
        # Step 6: Verify layer-specific stats
        layer0_stats = morphable_model.get_layer_stats(layer_names[0])
        layer1_stats = morphable_model.get_layer_stats(layer_names[1])
        
        assert layer0_stats["state_stats"]["active_seeds"] == 1
        assert layer1_stats["state_stats"]["active_seeds"] == 1
        
        # Step 7: Test kernel unloading
        kasmina_layer_0.state_layout.reset_seed(0)
        # Update morphogenetic_active flag after reset
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        assert morphable_model.get_model_stats()["active_seeds"] == 1
        
        # Step 8: Test complete cleanup
        kasmina_layer_1.state_layout.reset_seed(2)
        # Update morphogenetic_active flag after final reset
        morphable_model.morphogenetic_active = morphable_model._check_morphogenetic_active()
        assert morphable_model.morphogenetic_active == False
        
        print("✅ Full Phase 2 pipeline integration test completed successfully!")


class TestPhase2EdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model_wrapping(self):
        """Test wrapping a model with no target layers."""
        
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self.relu(x)
        
        model = EmptyModel()
        morphable_model = esper.wrap(model, telemetry_enabled=False)
        
        assert len(morphable_model.kasmina_layers) == 0
        assert morphable_model.morphogenetic_active == False
        
        # Should still work normally
        x = torch.randn(3, 10)
        output = morphable_model(x)
        assert torch.allclose(output, model(x))
    
    @pytest.mark.asyncio
    async def test_invalid_operations(self):
        """Test invalid operations and error handling."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Test invalid layer name for stats (expects ValueError, not KeyError)
        with pytest.raises(ValueError, match="not found"):
            morphable_model.get_layer_stats("nonexistent")
        
        # Test invalid seed index for alpha setting
        layer_names = morphable_model.get_layer_names()
        with pytest.raises((IndexError, ValueError)):
            morphable_model.set_seed_alpha(layer_names[0], 10, 0.5)  # seed index too high
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        model = SimpleTestModel().cuda()
        morphable_model = esper.wrap(model, telemetry_enabled=False)
        
        # Move to GPU
        morphable_model.to(torch.device("cuda"))
        
        # Test forward pass on GPU
        x = torch.randn(3, 10).cuda()
        output = morphable_model(x)
        
        assert output.is_cuda
        assert output.shape == (3, 5)
    
    def test_large_batch_processing(self):
        """Test processing large batches."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Test with large batch
        x = torch.randn(100, 10)
        output = morphable_model(x)
        
        assert output.shape == (100, 5)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access."""
        model = SimpleTestModel()
        morphable_model = esper.wrap(model, seeds_per_layer=4, telemetry_enabled=False)
        
        # Test multiple concurrent forward passes
        import threading
        results = []
        
        def forward_pass():
            x = torch.randn(3, 10)
            output = morphable_model(x)
            results.append(output)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=forward_pass)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All results should be valid
        assert len(results) == 5
        for output in results:
            assert output.shape == (3, 5)
            assert not torch.any(torch.isnan(output))

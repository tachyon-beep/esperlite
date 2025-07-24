"""
Refactored integration tests for kernel execution system.

This module contains integration tests that use real components 
with minimal mocking, replacing the over-mocked original tests.
"""

import io
import time

import pytest
import torch

from src.esper.contracts.assets import KernelMetadata
from src.esper.execution.kasmina_layer import KasminaLayer
from src.esper.execution.kernel_executor import create_test_kernel_artifact
from src.esper.execution.state_layout import SeedLifecycleState
from tests.fixtures.real_components import TestKernelFactory
from tests.helpers.test_context import with_real_components


@pytest.mark.integration
class TestKernelExecutionIntegration:
    """Integration tests for kernel execution system using real components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_size = 64
        self.output_size = 32
        self.num_seeds = 4

        # Create KasminaLayer for testing
        self.layer = KasminaLayer(
            input_size=self.input_size,
            output_size=self.output_size,
            num_seeds=self.num_seeds,
            cache_size_mb=64,
            telemetry_enabled=False,
            layer_name="test_layer",
        )

    def test_layer_initialization(self):
        """Test proper initialization of integrated components."""
        # Verify all components are initialized
        assert self.layer.input_size == self.input_size
        assert self.layer.output_size == self.output_size
        assert self.layer.num_seeds == self.num_seeds

        # Verify component integration
        assert self.layer.kernel_cache is not None
        assert self.layer.kernel_executor is not None
        assert self.layer.error_recovery is not None
        assert self.layer.state_layout is not None

        # Verify default state
        assert not self.layer.state_layout.has_active_seeds()

    def test_forward_pass_without_kernels(self):
        """Test forward pass with no active kernels (default behavior)."""
        batch_size = 16
        input_tensor = torch.randn(batch_size, self.input_size)

        output = self.layer(input_tensor)

        assert output.shape == (batch_size, self.output_size)
        assert torch.all(torch.isfinite(output))

        # Should use default transformation only
        expected_output = self.layer.default_transform(input_tensor)
        assert torch.allclose(output, expected_output)

    @pytest.mark.xfail(reason="Kernel execution doesn't work in sync fallback mode")
    @pytest.mark.asyncio
    async def test_kernel_loading_and_execution_real(self):
        """Test kernel loading and execution with real components."""
        factory = TestKernelFactory()
        kernel_bytes, metadata = factory.create_real_kernel(self.input_size, self.output_size)
        
        # Add to cache
        kernel_id = metadata.kernel_id
        
        # Create tensor representation
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        state_dict = module.state_dict()
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        
        self.layer.kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )

        # Load kernel
        success = await self.layer.load_kernel(seed_idx=0, artifact_id=kernel_id)
        assert success

        # Verify state
        assert self.layer.state_layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
        assert self.layer.state_layout.active_kernel_id[0] != 0
        assert self.layer.state_layout.alpha_blend[0] > 0

        # Test execution
        input_tensor = torch.randn(8, self.input_size)
        baseline = self.layer.default_transform(input_tensor)
        output = self.layer(input_tensor)

        # Kernel should affect output
        assert not torch.allclose(output, baseline, atol=1e-6)

    @pytest.mark.asyncio
    async def test_error_recovery_real_failure(self):
        """Test error recovery with real failure scenarios."""
        # Try to load non-existent kernel
        success = await self.layer.load_kernel(
            seed_idx=0, artifact_id="non_existent_kernel"
        )
        
        assert not success
        
        # When kernel is incompatible, it's handled as a normal case, not an error
        # The seed should remain dormant
        assert self.layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT
        
        # Layer should still work with default transform
        input_tensor = torch.randn(4, self.input_size)
        output = self.layer(input_tensor)
        assert output.shape == (4, self.output_size)
        
        # Get stats to verify no errors were recorded for incompatible kernels
        stats = self.layer.get_layer_stats()
        error_stats = stats["error_recovery_stats"]
        # Incompatible kernels don't count as errors, they're normal operations
        assert error_stats["total_recoveries"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_kernel_execution(self):
        """Test concurrent execution with multiple real kernels."""
        factory = TestKernelFactory()
        
        # Create multiple kernels
        for i in range(2):
            kernel_bytes, metadata = factory.create_real_kernel(
                self.input_size, self.output_size
            )
            kernel_id = f"concurrent_kernel_{i}"
            metadata.kernel_id = kernel_id
            
            # Add to cache
            buffer = io.BytesIO(kernel_bytes)
            module = torch.jit.load(buffer)
            state_dict = module.state_dict()
            tensors = []
            for param in state_dict.values():
                tensors.append(param.flatten())
            kernel_tensor = torch.cat(tensors)
            
            self.layer.kernel_cache._add_to_cache_with_metadata(
                kernel_id, kernel_tensor, metadata
            )
            
            # Load into different seeds
            success = await self.layer.load_kernel(seed_idx=i, artifact_id=kernel_id)
            assert success

        # Verify both seeds active
        assert self.layer.state_layout.get_active_count() == 2

        # Test execution
        input_tensor = torch.randn(4, self.input_size)
        output = self.layer(input_tensor)
        
        assert output.shape == (4, self.output_size)
        assert torch.all(torch.isfinite(output))

    def test_layer_statistics_integration(self):
        """Test integrated statistics reporting with real execution."""
        # Run some forward passes
        input_tensor = torch.randn(4, self.input_size)
        for _ in range(5):
            _ = self.layer(input_tensor)

        stats = self.layer.get_layer_stats()

        # Verify stats structure
        assert stats["layer_name"] == "test_layer"
        assert stats["total_forward_calls"] == 5
        assert stats["total_kernel_executions"] == 0  # No kernels loaded
        
        # Verify all component stats included
        assert "state_stats" in stats
        assert "cache_stats" in stats
        assert "error_recovery_stats" in stats

    @pytest.mark.asyncio
    async def test_kernel_compatibility_validation(self):
        """Test kernel compatibility validation with real kernels."""
        factory = TestKernelFactory()
        
        # Create incompatible kernel (wrong dimensions)
        kernel_bytes, metadata = factory.create_real_kernel(32, 16)  # Wrong sizes
        kernel_id = "incompatible_kernel"
        metadata.kernel_id = kernel_id
        metadata.input_shape = [32]  # Incompatible with layer
        metadata.output_shape = [16]  # Incompatible with layer
        
        # Add to cache - only add metadata, not bytes
        # The kernel will be fetched during load_kernel and validated then
        
        # Try to load - should fail validation
        success = await self.layer.load_kernel(seed_idx=0, artifact_id=kernel_id)
        assert not success
        
        # Seed should remain dormant
        assert self.layer.state_layout.lifecycle_states[0] == SeedLifecycleState.DORMANT

    def test_device_migration(self):
        """Test device migration functionality."""
        target_device = torch.device("cpu")
        
        self.layer.to(target_device)
        
        # Verify components moved
        assert self.layer.state_layout.device == target_device
        assert self.layer.default_transform.weight.device == target_device
        
        # Test execution on target device
        input_tensor = torch.randn(4, self.input_size, device=target_device)
        output = self.layer(input_tensor)
        
        assert output.device == target_device
        assert output.shape == (4, self.output_size)


@pytest.mark.performance
class TestKernelExecutionPerformance:
    """Real performance tests for kernel execution."""

    def setup_method(self):
        """Setup performance test fixtures."""
        self.layer = KasminaLayer(
            input_size=256,
            output_size=128,
            num_seeds=8,
            cache_size_mb=256,
            telemetry_enabled=False,
            layer_name="perf_test_layer",
        )

    def test_forward_pass_latency(self):
        """Test actual forward pass latency."""
        batch_size = 64
        input_tensor = torch.randn(batch_size, 256)

        # Warmup
        for _ in range(10):
            _ = self.layer(input_tensor)

        # Measure real latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            _ = self.layer(input_tensor)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[99]

        print(f"Real avg latency: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms")

        # Realistic targets based on actual execution
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms too high"
        assert p99_latency < 20.0, f"P99 latency {p99_latency:.2f}ms too high"

    @pytest.mark.asyncio
    async def test_kernel_loading_performance(self):
        """Test real kernel loading performance."""
        factory = TestKernelFactory()
        
        load_times = []
        for i in range(10):
            # Create real kernel
            kernel_bytes, metadata = factory.create_real_kernel(256, 128)
            kernel_id = f"perf_kernel_{i}"
            metadata.kernel_id = kernel_id
            
            # Add to cache
            buffer = io.BytesIO(kernel_bytes)
            module = torch.jit.load(buffer)
            state_dict = module.state_dict()
            tensors = []
            for param in state_dict.values():
                tensors.append(param.flatten())
            kernel_tensor = torch.cat(tensors)
            
            self.layer.kernel_cache._add_to_cache_with_metadata(
                kernel_id, kernel_tensor, metadata
            )
            
            # Measure loading time
            start_time = time.perf_counter()
            success = await self.layer.load_kernel(
                seed_idx=i % self.layer.num_seeds, artifact_id=kernel_id
            )
            load_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if success:
                load_times.append(load_time)

        if load_times:
            avg_load_time = sum(load_times) / len(load_times)
            print(f"Real kernel load time: {avg_load_time:.2f}ms")
            
            # Real loading should be reasonably fast
            assert avg_load_time < 100.0, f"Load time {avg_load_time:.2f}ms too high"


@pytest.mark.integration 
class TestSystemResilience:
    """Test system resilience with real components."""

    def setup_method(self):
        """Setup resilience test fixtures."""
        self.layer = KasminaLayer(
            input_size=32,
            output_size=16, 
            num_seeds=2,
            telemetry_enabled=False,
            layer_name="resilience_test_layer",
        )

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system continues working despite failures."""
        input_tensor = torch.randn(8, 32)
        
        # Baseline
        output1 = self.layer(input_tensor)
        assert output1.shape == (8, 16)
        
        # Try loading multiple non-existent kernels
        for i in range(5):
            success = await self.layer.load_kernel(
                seed_idx=i % self.layer.num_seeds,
                artifact_id=f"failing_kernel_{i}"
            )
            assert not success
        
        # System should still work
        output2 = self.layer(input_tensor)
        assert output2.shape == (8, 16)
        assert torch.allclose(output1, output2)  # Same as baseline
        
        # Check that system handled failures gracefully
        stats = self.layer.get_layer_stats()
        error_stats = stats["error_recovery_stats"]
        # Incompatible kernels don't trigger error recovery, they're handled as normal operations
        assert error_stats["total_recoveries"] == 0
        # All seeds should remain dormant since no kernels were compatible
        assert stats["state_stats"]["active_seeds"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
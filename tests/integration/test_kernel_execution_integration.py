"""
Integration tests for the complete kernel execution system.

This module contains end-to-end tests that verify the integration between
KasminaLayer, RealKernelExecutor, EnhancedKernelCache, and error recovery.
"""

import time
from unittest.mock import patch

import pytest
import torch

from src.esper.contracts.assets import KernelMetadata
from src.esper.execution.kasmina_layer import KasminaLayer
from src.esper.execution.kernel_executor import create_test_kernel_artifact
from src.esper.execution.state_layout import SeedLifecycleState
from src.esper.utils.config import ServiceConfig


class MockServiceConfig(ServiceConfig):
    """Mock service configuration for testing."""

    def __init__(self):
        self.cache_size_mb = 128
        self.max_cache_entries = 64
        self.http_timeout = 30.0
        self.retry_attempts = 3

    def get_urza_api_url(self):
        return "http://localhost:8080/api"


@pytest.mark.integration
class TestKernelExecutionIntegration:
    """Integration tests for kernel execution system."""

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
            telemetry_enabled=False,  # Disable for testing
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

    @pytest.mark.asyncio
    async def test_kernel_loading_and_execution(self):
        """Test complete kernel loading and execution workflow."""
        # Mock kernel data
        test_metadata = KernelMetadata(
            kernel_id="test_kernel_123",
            blueprint_id="test_blueprint",
            name="Test Integration Kernel",
            input_shape=[self.input_size],
            output_shape=[self.output_size],
            parameter_count=self.input_size * self.output_size + self.output_size,
            device_requirements=["cpu"],
            memory_footprint_mb=2.0,
        )

        test_tensor = torch.randn(1000)  # Mock kernel tensor
        test_artifact = create_test_kernel_artifact(self.input_size, self.output_size)

        # Mock the cache methods
        with (
            patch.object(
                self.layer.kernel_cache, "load_kernel_with_validation"
            ) as mock_load,
            patch.object(self.layer.kernel_cache, "get_kernel_bytes") as mock_get_bytes,
        ):

            mock_load.return_value = (test_tensor, test_metadata)
            mock_get_bytes.return_value = test_artifact

            # Load kernel into first seed
            success = await self.layer.load_kernel(
                seed_idx=0, artifact_id="test_kernel_123"
            )

            assert success

            # Verify seed state changed
            assert (
                self.layer.state_layout.lifecycle_states[0] == SeedLifecycleState.ACTIVE
            )
            assert self.layer.state_layout.active_kernel_id[0] != 0
            assert self.layer.state_layout.alpha_blend[0] > 0

    @pytest.mark.asyncio
    async def test_forward_pass_with_kernels(self):
        """Test forward pass with active kernels."""
        # Setup kernel
        test_metadata = KernelMetadata(
            kernel_id="execution_test_kernel",
            blueprint_id="test_blueprint",
            name="Execution Test Kernel",
            input_shape=[self.input_size],
            output_shape=[self.output_size],
            parameter_count=100,
            device_requirements=["cpu"],
            memory_footprint_mb=1.0,
        )

        test_tensor = torch.randn(500)
        test_artifact = create_test_kernel_artifact(self.input_size, self.output_size)

        # Mock cache and executor
        with (
            patch.object(
                self.layer.kernel_cache, "load_kernel_with_validation"
            ) as mock_load,
            patch.object(self.layer.kernel_cache, "get_kernel_bytes") as mock_get_bytes,
        ):

            mock_load.return_value = (test_tensor, test_metadata)
            mock_get_bytes.return_value = test_artifact

            # Load kernel
            await self.layer.load_kernel(
                seed_idx=0, artifact_id="execution_test_kernel"
            )

            # Test forward pass
            batch_size = 8
            input_tensor = torch.randn(batch_size, self.input_size)

            output = self.layer(input_tensor)

            assert output.shape == (batch_size, self.output_size)
            assert torch.all(torch.isfinite(output))

            # Output should differ from pure default transformation due to blending
            default_output = self.layer.default_transform(input_tensor)
            assert not torch.allclose(output, default_output)

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across the integrated system."""
        # Mock a failing kernel load
        with patch.object(
            self.layer.kernel_cache, "load_kernel_with_validation"
        ) as mock_load:
            mock_load.side_effect = Exception("Mock loading error")

            # Attempt to load kernel
            success = await self.layer.load_kernel(
                seed_idx=0, artifact_id="failing_kernel"
            )

            assert not success

            # Verify error was recorded in recovery system
            assert len(self.layer.error_recovery.error_tracker.error_history) > 0

            # Seed should remain dormant
            assert (
                self.layer.state_layout.lifecycle_states[0]
                == SeedLifecycleState.DORMANT
            )

    @pytest.mark.asyncio
    async def test_multiple_seeds_execution(self):
        """Test execution with multiple active seeds."""
        # Setup multiple kernels
        test_artifacts = []
        for i in range(2):
            metadata = KernelMetadata(
                kernel_id=f"multi_kernel_{i}",
                blueprint_id=f"blueprint_{i}",
                name=f"Multi Test Kernel {i}",
                input_shape=[self.input_size],
                output_shape=[self.output_size],
                parameter_count=100,
                device_requirements=["cpu"],
                memory_footprint_mb=1.0,
            )

            tensor = torch.randn(200)
            artifact = create_test_kernel_artifact(self.input_size, self.output_size)
            test_artifacts.append((tensor, metadata, artifact))

        # Mock cache methods to return different data for each kernel
        def mock_load_side_effect(artifact_id, **_kwargs):
            if "multi_kernel_0" in artifact_id:
                return test_artifacts[0][:2]  # tensor, metadata
            elif "multi_kernel_1" in artifact_id:
                return test_artifacts[1][:2]
            return None

        def mock_get_bytes_side_effect(artifact_id):
            if "multi_kernel_0" in artifact_id:
                return test_artifacts[0][2]  # artifact
            elif "multi_kernel_1" in artifact_id:
                return test_artifacts[1][2]
            return None

        with (
            patch.object(
                self.layer.kernel_cache,
                "load_kernel_with_validation",
                side_effect=mock_load_side_effect,
            ),
            patch.object(
                self.layer.kernel_cache,
                "get_kernel_bytes",
                side_effect=mock_get_bytes_side_effect,
            ),
        ):

            # Load kernels into multiple seeds
            success1 = await self.layer.load_kernel(
                seed_idx=0, artifact_id="multi_kernel_0"
            )
            success2 = await self.layer.load_kernel(
                seed_idx=1, artifact_id="multi_kernel_1"
            )

            assert success1 and success2

            # Verify both seeds are active
            assert self.layer.state_layout.has_active_seeds()
            assert self.layer.state_layout.get_active_count() == 2

            # Test forward pass with multiple kernels
            input_tensor = torch.randn(4, self.input_size)
            output = self.layer(input_tensor)

            assert output.shape == (4, self.output_size)
            assert torch.all(torch.isfinite(output))

    def test_layer_statistics_integration(self):
        """Test integrated statistics reporting."""
        stats = self.layer.get_layer_stats()

        # Verify all component stats are included
        assert "layer_name" in stats
        assert "total_forward_calls" in stats
        assert "state_stats" in stats
        assert "cache_stats" in stats
        assert "error_recovery_stats" in stats

        # Verify nested stats structure
        assert "metadata_cache_size" in stats["cache_stats"]
        assert "compatibility_rate" in stats["cache_stats"]
        assert "total_recoveries" in stats["error_recovery_stats"]
        assert "error_stats" in stats["error_recovery_stats"]

    @pytest.mark.asyncio
    async def test_kernel_compatibility_filtering(self):
        """Test kernel compatibility filtering in integrated system."""
        # Create incompatible kernel metadata
        _ = KernelMetadata(
            kernel_id="incompatible_kernel",
            blueprint_id="test_blueprint",
            name="Incompatible Kernel",
            input_shape=[32],  # Wrong input size
            output_shape=[16],  # Wrong output size
            parameter_count=100,
            device_requirements=["cpu"],
            memory_footprint_mb=1.0,
        )

        _ = torch.randn(100)

        with patch.object(
            self.layer.kernel_cache, "load_kernel_with_validation"
        ) as mock_load:
            # Return None for incompatible kernel
            mock_load.return_value = None

            success = await self.layer.load_kernel(
                seed_idx=0, artifact_id="incompatible_kernel"
            )

            assert not success

            # Verify compatibility check was performed
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert (
                call_args[1]["target_shape"][1] == self.input_size
            )  # Batch dim excluded

    def test_compatible_kernels_discovery(self):
        """Test discovery of compatible kernels."""
        # Add some test metadata to cache
        compatible_metadata = KernelMetadata(
            kernel_id="compatible_kernel",
            blueprint_id="test_blueprint",
            name="Compatible Kernel",
            input_shape=[self.input_size],
            output_shape=[self.output_size],
            parameter_count=100,
            device_requirements=["cpu"],
            memory_footprint_mb=1.0,
        )

        test_tensor = torch.randn(100)

        # Manually add to cache for testing
        self.layer.kernel_cache._add_to_cache_with_metadata(
            "compatible_kernel", test_tensor, compatible_metadata
        )

        # Find compatible kernels
        compatible = self.layer.find_compatible_kernels(max_memory_mb=50.0)

        assert len(compatible) == 1
        assert compatible[0][0] == "compatible_kernel"
        assert compatible[0][1].kernel_id == "compatible_kernel"

    @pytest.mark.asyncio
    async def test_device_migration(self):
        """Test device migration with integrated system."""
        # Test moving to different device (CPU in this case)
        target_device = torch.device("cpu")

        self.layer.to(target_device)

        # Verify all components moved to target device
        assert self.layer.state_layout.device == target_device
        assert self.layer.default_transform.weight.device == target_device

        # Verify forward pass still works
        input_tensor = torch.randn(4, self.input_size, device=target_device)
        output = self.layer(input_tensor)

        assert output.device == target_device
        assert output.shape == (4, self.output_size)


@pytest.mark.performance
class TestKernelExecutionPerformance:
    """Performance tests for integrated kernel execution."""

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
        """Test forward pass latency targets."""
        batch_size = 64
        input_tensor = torch.randn(batch_size, 256)

        # Warmup
        for _ in range(10):
            _ = self.layer(input_tensor)

        # Measure latency
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            _ = self.layer(input_tensor)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[99]

        # Performance targets (these may need adjustment based on hardware)
        assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds target"
        assert p99_latency < 10.0, f"P99 latency {p99_latency:.2f}ms exceeds target"

    @pytest.mark.asyncio
    async def test_kernel_loading_performance(self):
        """Test kernel loading performance."""
        # Mock fast kernel loading
        test_metadata = KernelMetadata(
            kernel_id="perf_kernel",
            blueprint_id="perf_blueprint",
            name="Performance Test Kernel",
            input_shape=[256],
            output_shape=[128],
            parameter_count=1000,
            device_requirements=["cpu"],
            memory_footprint_mb=5.0,
        )

        test_tensor = torch.randn(1000)
        test_artifact = create_test_kernel_artifact(256, 128)

        with (
            patch.object(
                self.layer.kernel_cache, "load_kernel_with_validation"
            ) as mock_load,
            patch.object(self.layer.kernel_cache, "get_kernel_bytes") as mock_get_bytes,
        ):

            mock_load.return_value = (test_tensor, test_metadata)
            mock_get_bytes.return_value = test_artifact

            # Measure kernel loading time
            load_times = []
            for i in range(10):
                start_time = time.perf_counter()
                success = await self.layer.load_kernel(
                    seed_idx=i % self.layer.num_seeds, artifact_id=f"perf_kernel_{i}"
                )
                load_time = (time.perf_counter() - start_time) * 1000  # ms
                load_times.append(load_time)
                assert success

            avg_load_time = sum(load_times) / len(load_times)

            # Kernel loading should be fast (target < 100ms)
            assert (
                avg_load_time < 100.0
            ), f"Average load time {avg_load_time:.2f}ms exceeds target"

    def test_memory_usage(self):
        """Test memory usage characteristics."""
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Create multiple layers to test memory scaling
        layers = []
        for i in range(5):
            layer = KasminaLayer(
                input_size=128,
                output_size=64,
                num_seeds=4,
                cache_size_mb=32,
                telemetry_enabled=False,
                layer_name=f"memory_test_layer_{i}",
            )
            layers.append(layer)

        # Run forward passes
        input_tensor = torch.randn(32, 128)
        for layer in layers:
            _ = layer(input_tensor)

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (target < 500MB for 5 layers)
            memory_increase_mb = memory_increase / (1024 * 1024)
            assert (
                memory_increase_mb < 500.0
            ), f"Memory increase {memory_increase_mb:.1f}MB exceeds target"


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error recovery."""

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
        """Test graceful degradation under failures."""
        # Test forward pass continues working even after kernel failures
        input_tensor = torch.randn(8, 32)

        # Normal operation
        output1 = self.layer(input_tensor)
        assert output1.shape == (8, 16)

        # Simulate kernel loading failures
        with patch.object(
            self.layer.kernel_cache, "load_kernel_with_validation"
        ) as mock_load:
            mock_load.side_effect = Exception("Simulated failure")

            # Attempt to load failing kernel
            success = await self.layer.load_kernel(
                seed_idx=0, artifact_id="failing_kernel"
            )
            assert not success

            # Forward pass should still work (graceful degradation)
            output2 = self.layer(input_tensor)
            assert output2.shape == (8, 16)
            assert torch.allclose(output1, output2)  # Should be same as default

    @pytest.mark.asyncio
    async def test_error_recovery_effectiveness(self):
        """Test effectiveness of error recovery mechanisms."""
        # Generate multiple errors and verify recovery
        errors_generated = 0

        async def generate_error():
            nonlocal errors_generated
            try:
                # Simulate various failure scenarios
                if errors_generated % 3 == 0:
                    raise ConnectionError("Network failure")
                elif errors_generated % 3 == 1:
                    raise TimeoutError("Operation timeout")
                else:
                    raise RuntimeError("General failure")
            finally:
                errors_generated += 1

        # Generate and handle multiple errors
        for _ in range(9):
            try:
                await generate_error()
            except Exception as e:
                # Errors should be classified and handled
                from src.esper.execution.error_recovery import classify_kernel_error
                from src.esper.execution.error_recovery import create_error_context

                error_type = classify_kernel_error(e)
                context = create_error_context(
                    error_type=error_type,
                    component="test_component",
                    layer_name="resilience_test",
                    exception=e,
                )

                await self.layer.error_recovery.handle_error(context)

        # Verify error recovery system tracked all errors
        assert len(self.layer.error_recovery.error_tracker.error_history) == 9
        assert len(self.layer.error_recovery.recovery_history) == 9

        # Check recovery success rate
        stats = self.layer.error_recovery.get_recovery_stats()
        assert stats["total_recoveries"] == 9
        assert stats["recovery_success_rate"] > 0.0  # Some recoveries should succeed


if __name__ == "__main__":
    pytest.main([__file__])

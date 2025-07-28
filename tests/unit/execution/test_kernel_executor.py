"""
Unit tests for the RealKernelExecutor.

This module contains comprehensive tests for kernel execution, validation,
error handling, and performance characteristics.
"""

import asyncio
import time

import pytest
import torch
import torch.nn as nn

from src.esper.execution.kernel_executor import ExecutionStats
from src.esper.execution.kernel_executor import KernelExecutionError
from src.esper.execution.kernel_executor import KernelValidator
from src.esper.execution.kernel_executor import RealKernelExecutor
from src.esper.execution.kernel_executor import create_test_kernel_artifact
from src.esper.execution.kernel_executor import validate_kernel_artifact


class TestExecutionStats:
    """Test ExecutionStats functionality."""

    def test_stats_initialization(self):
        """Test ExecutionStats initialization."""
        stats = ExecutionStats()

        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.total_execution_time == 0.0
        assert stats.success_rate == 1.0
        assert stats.average_execution_time == 0.0

    def test_record_success(self):
        """Test recording successful executions."""
        stats = ExecutionStats()

        stats.record_success(0.1)
        stats.record_success(0.2)

        assert stats.total_executions == 2
        assert stats.successful_executions == 2
        assert stats.failed_executions == 0
        assert stats.success_rate == 1.0
        assert (
            abs(stats.average_execution_time - 150.0) < 0.001
        )  # (0.1 + 0.2) / 2 * 1000

    def test_record_error(self):
        """Test recording execution errors."""
        stats = ExecutionStats()

        stats.record_error("deserialize")
        stats.record_error("shape")
        stats.record_error("runtime")

        assert stats.total_executions == 3
        assert stats.successful_executions == 0
        assert stats.failed_executions == 3
        assert stats.success_rate == 0.0
        assert stats.deserialize_errors == 1
        assert stats.shape_errors == 1
        assert stats.runtime_errors == 1

    def test_mixed_results(self):
        """Test mixed success and error recording."""
        stats = ExecutionStats()

        stats.record_success(0.1)
        stats.record_error("runtime")
        stats.record_success(0.2)

        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert stats.failed_executions == 1
        assert stats.success_rate == 2 / 3
        assert (
            abs(stats.average_execution_time - 150.0) < 0.001
        )  # (0.1 + 0.2) / 2 * 1000


class TestKernelValidator:
    """Test KernelValidator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = KernelValidator()

    def test_validate_simple_module(self):
        """Test validation of simple allowed module."""
        module = nn.Linear(10, 5)
        is_valid, error_msg = self.validator.validate_module(module)

        assert is_valid
        assert error_msg == ""

    def test_validate_complex_module(self):
        """Test validation of complex allowed module."""
        module = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5), nn.Sigmoid()
        )
        is_valid, error_msg = self.validator.validate_module(module)

        assert is_valid
        assert error_msg == ""

    def test_validate_too_large_module(self):
        """Test validation failure for oversized module."""
        # Create module with too many parameters
        module = nn.Linear(5000, 5000)  # 25M parameters
        is_valid, error_msg = self.validator.validate_module(module)

        assert not is_valid
        assert "too many parameters" in error_msg.lower()

    def test_validate_shape_compatibility(self):
        """Test shape compatibility validation."""
        input_shape = torch.Size([32, 10])
        expected_input_shape = torch.Size([32, 10])
        output_shape = torch.Size([32, 5])
        expected_output_shape = torch.Size([32, 5])

        is_valid, error_msg = self.validator.validate_shapes(
            input_shape, expected_input_shape, output_shape, expected_output_shape
        )

        assert is_valid
        assert error_msg == ""

    def test_validate_shape_mismatch(self):
        """Test shape mismatch detection."""
        input_shape = torch.Size([32, 10])
        expected_input_shape = torch.Size([32, 8])  # Mismatch
        output_shape = torch.Size([32, 5])
        expected_output_shape = torch.Size([32, 5])

        is_valid, error_msg = self.validator.validate_shapes(
            input_shape, expected_input_shape, output_shape, expected_output_shape
        )

        assert not is_valid
        assert "mismatch" in error_msg.lower()


class TestRealKernelExecutor:
    """Test RealKernelExecutor functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device("cpu")
        self.executor = RealKernelExecutor(
            device=self.device,
            max_kernel_cache_size=10,
            enable_validation=True,
            execution_timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_basic_kernel_execution(self):
        """Test basic kernel execution functionality."""
        # Create simple kernel
        input_size, output_size = 10, 5
        kernel_artifact = create_test_kernel_artifact(input_size, output_size)

        # Test execution
        input_tensor = torch.randn(32, input_size)
        result = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=0.5,
            kernel_id="test_kernel",
        )

        assert result.shape == (32, output_size)
        assert torch.all(torch.isfinite(result))
        assert self.executor.stats.successful_executions == 1

    @pytest.mark.asyncio
    async def test_kernel_caching(self):
        """Test kernel deserialization caching."""
        kernel_artifact = create_test_kernel_artifact(10, 5)
        input_tensor = torch.randn(32, 10)

        # First execution - should cache the kernel
        result1 = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=1.0,
            kernel_id="cached_kernel",
        )

        assert len(self.executor.kernel_cache) == 1
        assert "cached_kernel" in self.executor.kernel_cache

        # Second execution - should use cached kernel
        result2 = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=1.0,
            kernel_id="cached_kernel",
        )

        assert len(self.executor.kernel_cache) == 1
        assert torch.allclose(result1, result2)

    @pytest.mark.asyncio
    async def test_alpha_blending(self):
        """Test alpha blending functionality."""
        kernel_artifact = create_test_kernel_artifact(10, 10)  # Same input/output size
        input_tensor = torch.randn(32, 10)

        # Test with alpha = 0 (should return input)
        result_alpha_0 = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=0.0,
        )

        assert torch.allclose(result_alpha_0, input_tensor)

        # Test with alpha = 1 (should return kernel output)
        result_alpha_1 = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=1.0,
        )

        assert not torch.allclose(result_alpha_1, input_tensor)

        # Test with alpha = 0.5 (should blend)
        result_alpha_05 = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=0.5,
        )

        # Result should be between input and kernel output
        expected_blend = 0.5 * input_tensor + 0.5 * result_alpha_1
        assert torch.allclose(result_alpha_05, expected_blend, atol=1e-6)

    @pytest.mark.asyncio
    async def test_corrupted_kernel_handling(self):
        """Test handling of corrupted kernel artifacts."""
        corrupted_artifact = b"invalid_kernel_data"
        input_tensor = torch.randn(32, 10)

        with pytest.raises(KernelExecutionError):
            await self.executor.execute_kernel(
                kernel_artifact=corrupted_artifact,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=0.5,
            )

        assert self.executor.stats.failed_executions == 1
        assert self.executor.stats.deserialize_errors == 1

    @pytest.mark.asyncio
    async def test_shape_mismatch_handling(self):
        """Test handling of shape mismatches."""
        # Create kernel expecting different input size
        kernel_artifact = create_test_kernel_artifact(8, 5)
        input_tensor = torch.randn(32, 10)  # Wrong input size

        with pytest.raises(KernelExecutionError):
            await self.executor.execute_kernel(
                kernel_artifact=kernel_artifact,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=0.5,
            )

        assert self.executor.stats.failed_executions == 1

    @pytest.mark.asyncio
    async def test_zero_alpha_blend(self):
        """Test that zero alpha blend returns input unchanged."""
        kernel_artifact = create_test_kernel_artifact(10, 5)
        input_tensor = torch.randn(32, 10)

        # With alpha=0.0, should return input tensor unchanged
        result = await self.executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=0.0,
            kernel_id="test_kernel",
        )

        # Should return exactly the same tensor
        assert torch.equal(
            result, input_tensor
        ), "Zero alpha should return input unchanged"

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU cache eviction."""
        # Create executor with small cache
        small_cache_executor = RealKernelExecutor(
            device=self.device, max_kernel_cache_size=2
        )

        # Add kernels to exceed cache size
        for i in range(3):
            kernel_artifact = create_test_kernel_artifact(10, 5)
            input_tensor = torch.randn(1, 10)

            await small_cache_executor.execute_kernel(
                kernel_artifact=kernel_artifact,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=1.0,
                kernel_id=f"kernel_{i}",
            )

        # Cache should contain only 2 kernels (LRU eviction)
        assert len(small_cache_executor.kernel_cache) == 2
        # First kernel should be evicted
        assert "kernel_0" not in small_cache_executor.kernel_cache
        assert "kernel_1" in small_cache_executor.kernel_cache
        assert "kernel_2" in small_cache_executor.kernel_cache

    def test_get_stats(self):
        """Test statistics reporting."""
        stats = self.executor.get_stats()

        assert isinstance(stats, dict)
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats
        assert "success_rate" in stats
        assert "cached_kernels" in stats
        assert "error_recovery_stats" in stats

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some kernels to cache
        create_test_kernel_artifact(10, 5)

        # Manually add to cache for testing
        test_module = nn.Linear(10, 5)
        self.executor.kernel_cache["test"] = test_module
        self.executor.cache_access_times["test"] = time.time()

        assert len(self.executor.kernel_cache) == 1

        self.executor.clear_cache()

        assert len(self.executor.kernel_cache) == 0
        assert len(self.executor.cache_access_times) == 0


class TestKernelArtifacts:
    """Test kernel artifact creation and validation."""

    def test_create_test_kernel_artifact(self):
        """Test test kernel artifact creation."""
        artifact = create_test_kernel_artifact(10, 5)

        assert isinstance(artifact, bytes)
        assert len(artifact) > 0

    def test_validate_kernel_artifact(self):
        """Test kernel artifact validation."""
        # Valid artifact
        valid_artifact = create_test_kernel_artifact(10, 5)
        is_valid, error_msg = validate_kernel_artifact(valid_artifact)

        assert is_valid
        assert error_msg == ""

        # Invalid artifact
        invalid_artifact = b"not_a_kernel"
        is_valid, error_msg = validate_kernel_artifact(invalid_artifact)

        assert not is_valid
        assert len(error_msg) > 0


class TestIntegration:
    """Integration tests for kernel execution system."""

    @pytest.mark.asyncio
    async def test_end_to_end_execution(self):
        """Test complete end-to-end kernel execution."""
        executor = RealKernelExecutor(torch.device("cpu"))

        # Create and execute multiple kernels
        kernels = []
        for i in range(3):
            artifact = create_test_kernel_artifact(10, 5)
            kernels.append(artifact)

        input_tensor = torch.randn(16, 10)
        results = []

        for i, kernel_artifact in enumerate(kernels):
            result = await executor.execute_kernel(
                kernel_artifact=kernel_artifact,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=0.3,
                kernel_id=f"kernel_{i}",
            )
            results.append(result)

        # Verify all executions succeeded
        assert len(results) == 3
        assert all(r.shape == (16, 5) for r in results)
        assert executor.stats.successful_executions == 3
        assert executor.stats.failed_executions == 0

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent kernel execution."""
        executor = RealKernelExecutor(torch.device("cpu"))

        async def execute_kernel_task(kernel_id: str):
            artifact = create_test_kernel_artifact(10, 5)
            input_tensor = torch.randn(8, 10)

            return await executor.execute_kernel(
                kernel_artifact=artifact,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=0.5,
                kernel_id=kernel_id,
            )

        # Execute multiple kernels concurrently
        tasks = [execute_kernel_task(f"concurrent_kernel_{i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.shape == (8, 5) for r in results)
        assert executor.stats.successful_executions == 5


@pytest.mark.performance
class TestPerformance:
    """Performance tests for kernel execution."""

    @pytest.mark.asyncio
    async def test_execution_latency(self):
        """Test kernel execution latency targets."""
        executor = RealKernelExecutor(torch.device("cpu"))

        # Test small kernel latency
        small_kernel = create_test_kernel_artifact(32, 32)
        input_tensor = torch.randn(32, 32)

        # Warmup
        for _ in range(5):
            await executor.execute_kernel(
                kernel_artifact=small_kernel,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=1.0,
            )

        # Measure latency
        latencies = []
        for _ in range(10):
            start_time = time.perf_counter()
            await executor.execute_kernel(
                kernel_artifact=small_kernel,
                input_tensor=input_tensor,
                original_shape=input_tensor.shape,
                blend_alpha=1.0,
            )
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        # Performance targets for small kernels
        assert (
            avg_latency < 10.0
        ), f"Average latency {avg_latency:.2f}ms exceeds 10ms target"
        assert (
            p99_latency < 20.0
        ), f"P99 latency {p99_latency:.2f}ms exceeds 20ms target"

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache hit/miss performance."""
        executor = RealKernelExecutor(torch.device("cpu"))

        kernel_artifact = create_test_kernel_artifact(64, 64)
        input_tensor = torch.randn(32, 64)

        # First execution (cache miss)
        start_time = time.perf_counter()
        await executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=1.0,
            kernel_id="cached_perf_kernel",
        )
        cache_miss_time = time.perf_counter() - start_time

        # Second execution (cache hit)
        start_time = time.perf_counter()
        await executor.execute_kernel(
            kernel_artifact=kernel_artifact,
            input_tensor=input_tensor,
            original_shape=input_tensor.shape,
            blend_alpha=1.0,
            kernel_id="cached_perf_kernel",
        )
        cache_hit_time = time.perf_counter() - start_time

        # Cache hit should be significantly faster
        speedup_ratio = cache_miss_time / cache_hit_time
        assert speedup_ratio > 1.5, f"Cache speedup {speedup_ratio:.2f}x is too low"


if __name__ == "__main__":
    pytest.main([__file__])

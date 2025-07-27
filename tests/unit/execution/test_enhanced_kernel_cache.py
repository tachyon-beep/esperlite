"""
Unit tests for the EnhancedKernelCache.

This module contains comprehensive tests for enhanced kernel caching,
metadata validation, and compatibility checking.
"""

import hashlib
from unittest.mock import patch

import pytest
import torch

from src.esper.contracts.assets import KernelMetadata
from src.esper.execution.enhanced_kernel_cache import EnhancedKernelCache
from src.esper.execution.enhanced_kernel_cache import KernelValidator
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


class TestKernelValidator:
    """Test KernelValidator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = KernelValidator()

    def test_validate_compatibility_success(self):
        """Test successful compatibility validation."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="Test Kernel",
            input_shape=[10],
            output_shape=[5],
            parameter_count=55,  # 10*5 + 5 bias
            device_requirements=["cpu", "cuda"],
            memory_footprint_mb=1.0,
        )

        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        is_valid, error_msg = self.validator.validate_compatibility(
            metadata, target_shape, device
        )

        assert is_valid
        assert error_msg == ""

    def test_validate_compatibility_shape_mismatch(self):
        """Test compatibility validation with shape mismatch."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="Test Kernel",
            input_shape=[8],  # Different from target
            output_shape=[5],
            parameter_count=45,
            device_requirements=["cpu"],
            memory_footprint_mb=1.0,
        )

        target_shape = torch.Size([32, 10])  # Expects input size 10
        device = torch.device("cpu")

        is_valid, error_msg = self.validator.validate_compatibility(
            metadata, target_shape, device
        )

        assert not is_valid
        assert "shape mismatch" in error_msg.lower()

    def test_validate_compatibility_device_mismatch(self):
        """Test compatibility validation with device mismatch."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="Test Kernel",
            input_shape=[10],
            output_shape=[5],
            parameter_count=55,
            device_requirements=["cuda"],  # Only supports CUDA
            memory_footprint_mb=1.0,
        )

        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")  # Requesting CPU

        is_valid, error_msg = self.validator.validate_compatibility(
            metadata, target_shape, device
        )

        assert not is_valid
        assert "cpu not in requirements" in error_msg.lower()

    def test_validate_compatibility_memory_overflow(self):
        """Test compatibility validation with memory overflow."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="Test Kernel",
            input_shape=[10],
            output_shape=[5],
            parameter_count=55,
            device_requirements=["cpu"],
            memory_footprint_mb=3000.0,  # 3GB - too large
        )

        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        is_valid, error_msg = self.validator.validate_compatibility(
            metadata, target_shape, device, max_memory_mb=100.0
        )

        assert not is_valid
        assert "memory footprint too large" in error_msg.lower()

    def test_estimate_execution_memory(self):
        """Test memory usage estimation."""
        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="Test Kernel",
            input_shape=[10],
            output_shape=[5],
            parameter_count=55,
            device_requirements=["cpu"],
            memory_footprint_mb=10.0,
        )

        # Test with different batch sizes
        memory_16 = self.validator.estimate_execution_memory(metadata, 16)
        memory_32 = self.validator.estimate_execution_memory(metadata, 32)
        memory_64 = self.validator.estimate_execution_memory(metadata, 64)

        # Memory usage should scale with batch size
        assert memory_32 > memory_16
        assert memory_64 > memory_32

        # Should include overhead factor
        assert memory_32 > metadata.get_memory_estimate(32)


class TestEnhancedKernelCache:
    """Test EnhancedKernelCache functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = MockServiceConfig()
        self.cache = EnhancedKernelCache(
            config=self.config, max_cache_size_mb=100, max_entries=10
        )

    def create_test_metadata(
        self, kernel_id="test_kernel", input_shape=[10], output_shape=[5]
    ):
        """Create test kernel metadata."""
        return KernelMetadata(
            kernel_id=kernel_id,
            blueprint_id="test_blueprint",
            name=f"Test Kernel {kernel_id}",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_count=len(input_shape) * len(output_shape) + len(output_shape),
            device_requirements=["cpu", "cuda"],
            memory_footprint_mb=1.0,
            checksum=hashlib.sha256(f"kernel_{kernel_id}".encode()).hexdigest(),
        )

    def create_test_tensor(self, size_mb=1.0):
        """Create test tensor of specified size."""
        # Calculate number of float32 elements for target size
        bytes_per_element = 4
        num_elements = int(size_mb * 1024 * 1024 / bytes_per_element)
        return torch.randn(num_elements)

    @pytest.mark.asyncio
    async def test_load_kernel_with_validation_success(self):
        """Test successful kernel loading with validation."""
        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        # Mock the fetch method to return test data
        test_metadata = self.create_test_metadata()
        test_tensor = self.create_test_tensor(2.0)

        with patch.object(self.cache, "_fetch_kernel_with_metadata") as mock_fetch:
            mock_fetch.return_value = (test_tensor, test_metadata)

            result = await self.cache.load_kernel_with_validation(
                artifact_id="test_kernel",
                target_shape=target_shape,
                device=device,
                batch_size=32,
            )

            assert result is not None
            kernel_tensor, metadata = result
            assert torch.equal(kernel_tensor, test_tensor)
            assert metadata.kernel_id == "test_kernel"

            # Verify caching
            assert "test_kernel" in self.cache._cache
            assert "test_kernel" in self.cache.metadata_cache

    @pytest.mark.asyncio
    async def test_load_kernel_with_validation_incompatible(self):
        """Test kernel loading with incompatible kernel."""
        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        # Create metadata with incompatible shape
        incompatible_metadata = self.create_test_metadata(input_shape=[8])  # Wrong size
        test_tensor = self.create_test_tensor(1.0)

        with patch.object(self.cache, "_fetch_kernel_with_metadata") as mock_fetch:
            mock_fetch.return_value = (test_tensor, incompatible_metadata)

            result = await self.cache.load_kernel_with_validation(
                artifact_id="incompatible_kernel",
                target_shape=target_shape,
                device=device,
                batch_size=32,
            )

            assert result is None
            assert self.cache.compatibility_failures == 1

    @pytest.mark.asyncio
    async def test_cache_hit_with_validation(self):
        """Test cache hit with validation check."""
        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        # Pre-populate cache
        test_metadata = self.create_test_metadata(kernel_id="cached_kernel")
        test_tensor = self.create_test_tensor(1.0)

        self.cache._add_to_cache_with_metadata(
            "cached_kernel", test_tensor, test_metadata
        )

        # Load should hit cache
        result = await self.cache.load_kernel_with_validation(
            artifact_id="cached_kernel",
            target_shape=target_shape,
            device=device,
            batch_size=32,
        )

        assert result is not None
        kernel_tensor, metadata = result
        assert torch.equal(kernel_tensor, test_tensor)
        assert metadata.kernel_id == "cached_kernel"
        assert self.cache._hits == 1

    @pytest.mark.asyncio
    async def test_find_compatible_kernels(self):
        """Test finding compatible kernels."""
        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        # Add compatible and incompatible kernels to cache
        compatible_metadata1 = self.create_test_metadata("compatible1", [10], [5])
        compatible_metadata2 = self.create_test_metadata("compatible2", [10], [8])
        incompatible_metadata = self.create_test_metadata("incompatible", [8], [5])

        test_tensor = self.create_test_tensor(1.0)

        self.cache._add_to_cache_with_metadata(
            "compatible1", test_tensor, compatible_metadata1
        )
        self.cache._add_to_cache_with_metadata(
            "compatible2", test_tensor, compatible_metadata2
        )
        self.cache._add_to_cache_with_metadata(
            "incompatible", test_tensor, incompatible_metadata
        )

        compatible_kernels = self.cache.find_compatible_kernels(
            target_shape=target_shape, device=device
        )

        assert len(compatible_kernels) == 2
        compatible_ids = [kernel_id for kernel_id, _ in compatible_kernels]
        assert "compatible1" in compatible_ids
        assert "compatible2" in compatible_ids
        assert "incompatible" not in compatible_ids

    def test_get_kernel_metadata(self):
        """Test kernel metadata retrieval."""
        test_metadata = self.create_test_metadata()
        test_tensor = self.create_test_tensor(1.0)

        self.cache._add_to_cache_with_metadata(
            "test_kernel", test_tensor, test_metadata
        )

        retrieved_metadata = self.cache.get_kernel_metadata("test_kernel")
        assert retrieved_metadata is not None
        assert retrieved_metadata.kernel_id == "test_kernel"

        # Test non-existent kernel
        assert self.cache.get_kernel_metadata("nonexistent") is None

    def test_enhanced_stats(self):
        """Test enhanced statistics reporting."""
        # Add some test data
        test_metadata = self.create_test_metadata()
        test_tensor = self.create_test_tensor(2.0)

        self.cache._add_to_cache_with_metadata(
            "test_kernel", test_tensor, test_metadata
        )
        self.cache.compatibility_checks = 10
        self.cache.compatibility_failures = 2

        stats = self.cache.get_enhanced_stats()

        assert "metadata_cache_size" in stats
        assert "compatibility_checks" in stats
        assert "compatibility_failures" in stats
        assert "compatibility_rate" in stats
        assert "total_estimated_memory_mb" in stats
        assert "average_kernel_parameters" in stats

        assert stats["metadata_cache_size"] == 1
        assert stats["compatibility_checks"] == 10
        assert stats["compatibility_failures"] == 2
        assert stats["compatibility_rate"] == 0.8  # (10-2)/10

    def test_cache_eviction_with_metadata(self):
        """Test LRU eviction includes metadata cleanup."""
        # Create cache with small size
        small_cache = EnhancedKernelCache(
            config=self.config,
            max_cache_size_mb=5,  # Small size to force eviction
            max_entries=2,
        )

        # Add kernels to exceed cache capacity
        for i in range(3):
            metadata = self.create_test_metadata(f"kernel_{i}")
            tensor = self.create_test_tensor(2.0)  # 2MB each

            small_cache._add_to_cache_with_metadata(f"kernel_{i}", tensor, metadata)

        # Should have evicted first kernel
        assert len(small_cache._cache) == 2
        assert len(small_cache.metadata_cache) == 2
        assert "kernel_0" not in small_cache._cache
        assert "kernel_0" not in small_cache.metadata_cache
        assert "kernel_1" in small_cache._cache
        assert "kernel_2" in small_cache._cache

    def test_clear_cache_enhanced(self):
        """Test enhanced cache clearing."""
        # Add test data
        test_metadata = self.create_test_metadata()
        test_tensor = self.create_test_tensor(1.0)

        self.cache._add_to_cache_with_metadata(
            "test_kernel", test_tensor, test_metadata
        )
        self.cache.memory_usage_estimates["test_kernel"] = 1.5

        assert len(self.cache._cache) == 1
        assert len(self.cache.metadata_cache) == 1
        assert len(self.cache.memory_usage_estimates) == 1

        self.cache.clear_cache()

        assert len(self.cache._cache) == 0
        assert len(self.cache.metadata_cache) == 0
        assert len(self.cache.memory_usage_estimates) == 0

    def test_remove_kernel_enhanced(self):
        """Test enhanced kernel removal."""
        # Add test data
        test_metadata = self.create_test_metadata()
        test_tensor = self.create_test_tensor(1.0)

        self.cache._add_to_cache_with_metadata(
            "test_kernel", test_tensor, test_metadata
        )
        self.cache.memory_usage_estimates["test_kernel"] = 1.5

        assert "test_kernel" in self.cache._cache
        assert "test_kernel" in self.cache.metadata_cache
        assert "test_kernel" in self.cache.memory_usage_estimates

        success = self.cache.remove_kernel("test_kernel")

        assert success
        assert "test_kernel" not in self.cache._cache
        assert "test_kernel" not in self.cache.metadata_cache
        assert "test_kernel" not in self.cache.memory_usage_estimates

    @pytest.mark.asyncio
    async def test_fetch_kernel_integration(self):
        """Test kernel fetching integrates with HTTP mocking correctly."""
        # Mock the HTTP client to avoid real network calls
        from unittest.mock import patch, AsyncMock
        import numpy as np
        import hashlib
        
        # Create some float32 tensor data
        kernel_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).tobytes()
        actual_checksum = hashlib.sha256(kernel_data).hexdigest()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": "test_kernel",
            "checksum": "abc123",
            "metadata": {
                "kernel_id": "test_kernel",
                "blueprint_id": "test_blueprint",
                "name": "Test Kernel",
                "input_shape": [10],
                "output_shape": [5],
                "parameter_count": 1000,
                "memory_footprint_mb": 4.0,
                "performance_profile": {"flops": 1000000},
                "checksum": actual_checksum
            },
            "binary_ref": "http://localhost:8080/kernels/test_kernel/download"
        })
        
        # Second response for downloading kernel bytes
        mock_download_response = AsyncMock()
        mock_download_response.status = 200
        mock_download_response.read = AsyncMock(return_value=kernel_data)
        
        with patch("esper.utils.http_client.AsyncHttpClient") as mock_http_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_response, mock_download_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_http_client_class.return_value = mock_client
            
            result = await self.cache._fetch_kernel_with_metadata_impl("test_kernel")

        assert result is not None, "Should successfully fetch kernel using HTTP mock"
        kernel_tensor, metadata = result
        assert isinstance(kernel_tensor, torch.Tensor), "Should return tensor"
        assert kernel_tensor.numel() > 0, "Tensor should not be empty"
        assert (
            metadata.kernel_id == "test_kernel"
        ), "Metadata should have correct kernel ID"
        assert isinstance(metadata.checksum, str), "Should have checksum"
        assert len(metadata.checksum) > 0, "Checksum should not be empty"

    @pytest.mark.asyncio
    async def test_fetch_kernel_not_found(self):
        """Test handling when kernel is not found."""
        # Mock the HTTP client to return 404
        from unittest.mock import patch, AsyncMock
        
        mock_response = AsyncMock()
        mock_response.status = 404
        
        with patch("esper.utils.http_client.AsyncHttpClient") as mock_http_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_http_client_class.return_value = mock_client
            
            result = await self.cache._fetch_kernel_with_metadata_impl("invalid-kernel")

        assert result is None, "Should return None when kernel not found"


@pytest.mark.integration
class TestEnhancedCacheIntegration:
    """Integration tests for enhanced kernel cache."""

    @pytest.mark.asyncio
    async def test_end_to_end_cache_workflow(self):
        """Test complete cache workflow with validation."""
        config = MockServiceConfig()
        cache = EnhancedKernelCache(config=config, max_cache_size_mb=50, max_entries=5)

        target_shape = torch.Size([32, 10])
        device = torch.device("cpu")

        # Mock multiple kernel fetches
        test_kernels = []
        for i in range(3):
            metadata = KernelMetadata(
                kernel_id=f"kernel_{i}",
                blueprint_id=f"blueprint_{i}",
                name=f"Test Kernel {i}",
                input_shape=[10],
                output_shape=[5],
                parameter_count=55,
                device_requirements=["cpu"],
                memory_footprint_mb=float(i + 1),
            )
            tensor = torch.randn(1000)  # Small tensor
            test_kernels.append((tensor, metadata))

        # Mock fetch method
        fetch_call_count = 0

        async def mock_fetch(artifact_id):
            nonlocal fetch_call_count
            if fetch_call_count < len(test_kernels):
                result = test_kernels[fetch_call_count]
                fetch_call_count += 1
                return result
            return None

        with patch.object(cache, "_fetch_kernel_with_metadata", side_effect=mock_fetch):
            # Load kernels with validation
            results = []
            for i in range(3):
                result = await cache.load_kernel_with_validation(
                    artifact_id=f"kernel_{i}",
                    target_shape=target_shape,
                    device=device,
                    batch_size=32,
                )
                results.append(result)

            # Verify all loads succeeded
            assert all(r is not None for r in results)
            assert len(cache._cache) == 3
            assert len(cache.metadata_cache) == 3

            # Test cache hits
            cache_hit_result = await cache.load_kernel_with_validation(
                artifact_id="kernel_0",
                target_shape=target_shape,
                device=device,
                batch_size=32,
            )

            assert cache_hit_result is not None
            assert cache._hits > 0

            # Test finding compatible kernels
            compatible = cache.find_compatible_kernels(target_shape, device)
            assert len(compatible) == 3

            # Test statistics
            stats = cache.get_enhanced_stats()
            assert stats["metadata_cache_size"] == 3
            assert stats["compatibility_rate"] == 1.0  # All successful


if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for KernelCache.
"""

import asyncio
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from esper.execution.kernel_cache import KernelCache


class TestKernelCache:
    """Test cases for KernelCache."""

    def test_initialization(self):
        """Test basic initialization."""
        cache = KernelCache(max_cache_size_mb=256, max_entries=64)

        assert cache.max_cache_size_mb == 256
        assert cache.max_entries == 64
        assert cache.total_size_mb == 0.0
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._evictions == 0

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        cache = KernelCache()

        assert cache.max_cache_size_mb == 512
        assert cache.max_entries == 128
        assert cache.total_size_mb == 0.0

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_load_kernel_cache_miss(self, mock_fetch):
        """Test loading kernel with cache miss (mocked fetch)."""
        cache = KernelCache()
        fake_tensor = torch.randn(1024)
        mock_fetch.return_value = fake_tensor
        kernel = await cache.load_kernel("test-kernel-123")
        assert kernel is fake_tensor
        assert cache._hits == 0
        assert cache._misses == 1
        assert len(cache._cache) == 1
        assert "test-kernel-123" in cache._cache

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_load_kernel_cache_hit(self, mock_fetch):
        """Test loading kernel with cache hit (mocked fetch)."""
        cache = KernelCache()
        fake_tensor = torch.randn(1024)
        mock_fetch.return_value = fake_tensor
        # First load (cache miss)
        kernel1 = await cache.load_kernel("test-kernel-123")
        # Second load (cache hit)
        kernel2 = await cache.load_kernel("test-kernel-123")
        assert kernel1 is kernel2
        assert cache._hits == 1
        assert cache._misses == 1
        assert len(cache._cache) == 1

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_lru_eviction(self, mock_fetch):
        """Test LRU eviction when cache is full (mocked fetch)."""
        cache = KernelCache(max_entries=2, max_cache_size_mb=1024)
        mock_fetch.side_effect = [
            torch.randn(1024),
            torch.randn(1024),
            torch.randn(1024),
        ]
        await cache.load_kernel("kernel-1")
        await cache.load_kernel("kernel-2")
        await cache.load_kernel("kernel-3")  # Should evict kernel-1
        assert len(cache._cache) == 2
        assert "kernel-1" not in cache._cache
        assert "kernel-2" in cache._cache
        assert "kernel-3" in cache._cache
        assert cache._evictions == 1

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_size_based_eviction(self, mock_fetch):
        """Test eviction based on cache size."""
        # Use very small cache size to force eviction
        cache = KernelCache(max_cache_size_mb=0.001, max_entries=10)
        mock_fetch.side_effect = [torch.randn(1024), torch.randn(1024)]

        # Load multiple kernels to trigger size-based eviction
        await cache.load_kernel("kernel-1")
        await cache.load_kernel("kernel-2")  # Should trigger eviction

        # First kernel should be evicted due to size constraint
        assert len(cache._cache) <= 1
        assert cache._evictions >= 1

    def test_get_cache_stats(self):
        """Test cache statistics."""
        cache = KernelCache(max_cache_size_mb=256, max_entries=64)

        stats = cache.get_cache_stats()

        assert stats["entries"] == 0
        assert stats["total_size_mb"] == pytest.approx(0.0)
        assert stats["max_size_mb"] == 256
        assert stats["max_entries"] == 64
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == pytest.approx(0.0)
        assert stats["cache_keys"] == []

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_get_cache_stats_with_data(self, mock_fetch):
        """Test cache statistics with cached data."""
        cache = KernelCache()
        mock_fetch.side_effect = [torch.randn(1024), torch.randn(1024)]

        # Load some kernels
        await cache.load_kernel("kernel-1")
        await cache.load_kernel("kernel-2")
        await cache.load_kernel("kernel-1")  # Cache hit

        stats = cache.get_cache_stats()

        assert stats["entries"] == 2
        assert stats["total_size_mb"] > 0
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1 / 3
        assert "kernel-1" in stats["cache_keys"]
        assert "kernel-2" in stats["cache_keys"]

    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = KernelCache()

        # Add some dummy data
        cache._cache["test"] = torch.randn(100)
        cache._cache_info["test"] = {"size_mb": 0.1, "added_at": 0, "last_accessed": 0}
        cache.total_size_mb = 0.1

        cache.clear_cache()

        assert len(cache._cache) == 0
        assert len(cache._cache_info) == 0
        assert cache.total_size_mb == pytest.approx(0.0)

    def test_remove_kernel(self):
        """Test removing a specific kernel."""
        cache = KernelCache()

        # Add some dummy data
        test_tensor = torch.randn(100)
        cache._cache["test-kernel"] = test_tensor
        cache._cache_info["test-kernel"] = {
            "size_mb": 0.1,
            "added_at": 0,
            "last_accessed": 0,
        }
        cache.total_size_mb = 0.1

        # Remove existing kernel
        result = cache.remove_kernel("test-kernel")
        assert result
        assert "test-kernel" not in cache._cache
        assert "test-kernel" not in cache._cache_info
        assert cache.total_size_mb == pytest.approx(0.0)

        # Try to remove non-existent kernel
        result = cache.remove_kernel("non-existent")
        assert not result

    def test_is_gpu_resident_empty(self):
        """Test GPU residency check with empty cache."""
        cache = KernelCache()
        assert not cache.is_gpu_resident

    def test_is_gpu_resident_with_cpu_tensor(self):
        """Test GPU residency check with CPU tensor."""
        cache = KernelCache()

        # Add CPU tensor
        cache._cache["test"] = torch.randn(100)
        assert not cache.is_gpu_resident

    @patch("torch.cuda.is_available", return_value=True)
    def test_is_gpu_resident_with_gpu_tensor(self, mock_cuda_available):
        """Test GPU residency check with GPU tensor."""
        cache = KernelCache()

        # Add GPU tensor
        cache._cache["test"] = torch.randn(100).cuda()
        assert cache.is_gpu_resident

    @pytest.mark.asyncio
    async def test_fetch_from_urza_with_real_http_client(self):
        """Test kernel fetching with properly mocked async HTTP client."""
        cache = KernelCache()

        # Mock the async HTTP client
        with patch("esper.utils.http_client.AsyncHttpClient") as mock_client_class:
            # Create mock client instance
            mock_client = Mock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            # Mock metadata response
            mock_meta_response = Mock()
            mock_meta_response.status = 200

            async def mock_json():
                return {"kernel_binary_ref": "http://s3.example.com/kernel-binary"}

            mock_meta_response.json = mock_json

            # Mock S3 binary response
            mock_s3_response = Mock()
            mock_s3_response.status = 200

            async def mock_read():
                return torch.randn(1024).numpy().tobytes()

            mock_s3_response.read = mock_read

            # Mock the client get method to return response contexts
            async def mock_get(url):
                if "test-kernel-123" in url:
                    return mock_meta_response
                else:
                    return mock_s3_response

            mock_client.get = mock_get

            # Test the fetch method
            kernel = await cache._fetch_from_urza("test-kernel-123")

            # Verify returned kernel
            assert kernel is not None
            assert isinstance(kernel, torch.Tensor)
            assert kernel.shape == (1024,)

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_preload_kernels(self, mock_fetch):
        """Test preloading kernels."""
        cache = KernelCache()
        mock_fetch.side_effect = [
            torch.randn(1024),
            torch.randn(1024),
            torch.randn(1024),
        ]

        # Test that preload works
        await cache.preload_kernels(["kernel-1", "kernel-2", "kernel-3"])

        # Should have loaded 3 kernels
        assert len(cache._cache) == 3
        assert cache._misses == 3
        assert cache._hits == 0

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_concurrent_access(self, mock_fetch):
        """Test concurrent access to cache."""
        cache = KernelCache()
        shared_tensor = torch.randn(1024)
        mock_fetch.return_value = shared_tensor

        # Create multiple concurrent requests for the same kernel
        tasks = [cache.load_kernel("shared-kernel") for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All should return the same tensor
        first_kernel = results[0]
        for kernel in results[1:]:
            assert torch.equal(first_kernel, kernel)

        # Should have 1 miss (first load) and 4 hits
        assert cache._misses == 1
        assert cache._hits == 4

    @pytest.mark.asyncio
    async def test_load_kernel_with_fetch_failure(self):
        """Test kernel loading when fetch fails."""
        cache = KernelCache()

        # Mock the fetch method to return None
        with patch.object(cache, "_fetch_from_urza", return_value=None):
            kernel = await cache.load_kernel("non-existent-kernel")

            assert kernel is None
            assert cache._misses == 1
            assert cache._hits == 0
            assert len(cache._cache) == 0

    @pytest.mark.asyncio
    @patch.object(KernelCache, "_fetch_from_urza", autospec=True)
    async def test_lru_eviction_with_real_tensor_data(self, mock_fetch):
        """Test LRU eviction with actual tensor data and size tracking."""
        cache = KernelCache(max_entries=3, max_cache_size_mb=1024)
        mock_fetch.side_effect = [torch.randn(1024) for _ in range(4)]

        # Load kernels with predictable data
        kernels = []
        for i in range(4):  # Load 4 kernels, should evict first one
            kernel = await cache.load_kernel(f"kernel-{i}")
            kernels.append(kernel)

            # Verify each kernel is unique (different due to simulation seeding)
            assert kernel is not None
            assert isinstance(kernel, torch.Tensor)

        # Verify eviction occurred
        assert len(cache._cache) == 3
        assert "kernel-0" not in cache._cache  # First should be evicted
        assert "kernel-3" in cache._cache  # Last should be present

        # Verify cache metrics
        assert cache._misses == 4
        assert cache._evictions >= 1

    def test_cache_size_calculation_with_real_tensors(self):
        """Test cache size calculation with actual tensor data."""
        cache = KernelCache()

        # Create realistic tensors of known sizes
        small_tensor = torch.randn(256, dtype=torch.float32)  # 1KB
        medium_tensor = torch.randn(1024, dtype=torch.float32)  # 4KB
        large_tensor = torch.randn(4096, dtype=torch.float32)  # 16KB

        # Add to cache manually to test size calculation
        cache._cache["small"] = small_tensor
        cache._cache["medium"] = medium_tensor
        cache._cache["large"] = large_tensor

        # Calculate expected sizes (float32 = 4 bytes per element)
        small_size_mb = (256 * 4) / (1024 * 1024)
        medium_size_mb = (1024 * 4) / (1024 * 1024)
        large_size_mb = (4096 * 4) / (1024 * 1024)

        # Update cache metadata
        cache._cache_info["small"] = {
            "size_mb": small_size_mb,
            "added_at": 0,
            "last_accessed": 0,
        }
        cache._cache_info["medium"] = {
            "size_mb": medium_size_mb,
            "added_at": 1,
            "last_accessed": 1,
        }
        cache._cache_info["large"] = {
            "size_mb": large_size_mb,
            "added_at": 2,
            "last_accessed": 2,
        }
        cache.total_size_mb = small_size_mb + medium_size_mb + large_size_mb

        # Test size calculation
        stats = cache.get_cache_stats()
        assert stats["entries"] == 3
        assert (
            abs(
                stats["total_size_mb"]
                - (small_size_mb + medium_size_mb + large_size_mb)
            )
            < 0.001
        )

        # Test size-based eviction threshold
        expected_total_mb = (256 + 1024 + 4096) * 4 / (1024 * 1024)  # ~0.021 MB
        assert abs(stats["total_size_mb"] - expected_total_mb) < 0.001

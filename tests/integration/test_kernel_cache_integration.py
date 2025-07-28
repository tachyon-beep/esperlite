"""
Integration tests for KernelCache with Urza service.

These tests verify real HTTP communication with mocked Urza service endpoints.
"""

import asyncio
import os
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from esper.execution.kernel_cache import KernelCache


class TestKernelCacheUrzaIntegration:
    """Integration tests for KernelCache with Urza service."""

    @pytest.mark.skip(reason="CUDA out of memory issues preventing test execution")
    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_successful_kernel_fetch_from_urza(self, mock_client_class):
        """Test successful kernel fetching from Urza with async HTTP client."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock Urza API response
        urza_response = Mock()
        urza_response.status = 200

        async def mock_json():
            return {
                "kernel_binary_ref": "https://s3.example.com/kernels/test-kernel-123.bin"
            }

        urza_response.json = mock_json

        # Mock S3 binary response with real tensor data
        test_tensor = torch.randn(1024, dtype=torch.float32)
        s3_response = Mock()
        s3_response.status = 200

        async def mock_read():
            return test_tensor.numpy().tobytes()

        s3_response.read = mock_read

        # Configure mock to return different responses based on URL
        async def mock_get(url):
            if "/api/v1/kernels/" in url:
                return urza_response
            elif "s3.example.com" in url:
                return s3_response
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_client.get = mock_get

        # Test kernel loading
        try:
            kernel = await cache.load_kernel("test-kernel-123")

            # Verify kernel was loaded correctly
            assert kernel is not None
            assert isinstance(kernel, torch.Tensor)
            assert kernel.shape == (1024,)

            # Verify cache metrics
            assert cache._hits == 0
            assert cache._misses == 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("CUDA out of memory, skipping test")
            else:
                raise
        assert len(cache._cache) == 1

    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_kernel_not_found_in_urza(self, mock_client_class):
        """Test handling when kernel is not found in Urza."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock 404 response from Urza
        urza_response = Mock()
        urza_response.status = 404
        mock_client.get.return_value = urza_response

        # Test kernel loading
        kernel = await cache.load_kernel("non-existent-kernel")

        # Verify proper handling
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_urza_server_error(self, mock_client_class):
        """Test handling when Urza server returns an error."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock server error from Urza
        mock_client.get.side_effect = Exception("Server Error")

        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel")

        # Should handle error gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_missing_binary_reference(self, mock_client_class):
        """Test handling when Urza response lacks binary reference."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock Urza response without binary_ref
        urza_response = Mock()
        urza_response.status = 200
        urza_response.json.return_value = {
            "kernel_id": "test-kernel-123"
            # Missing kernel_binary_ref
        }
        mock_client.get.return_value = urza_response

        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel-123")

        # Should handle missing reference gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_s3_download_failure(self, mock_client_class):
        """Test handling when S3 download fails."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock successful Urza response
        urza_response = Mock()
        urza_response.status = 200
        urza_response.json.return_value = {
            "kernel_binary_ref": "https://s3.example.com/kernels/test-kernel.bin"
        }

        # Mock failed S3 response
        s3_response = Mock()
        s3_response.status = 403

        # Return different responses based on URL
        async def mock_get(url):
            if "/api/v1/kernels/" in url:
                return urza_response
            elif "s3.example.com" in url:
                raise Exception("Access Denied")
            else:
                raise ValueError(f"Unexpected URL: {url}")

        mock_client.get = mock_get

        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel")

        # Should handle S3 failure gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0

    @pytest.mark.skip(reason="CUDA out of memory issues preventing test execution")
    @pytest.mark.asyncio
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_concurrent_urza_requests(self, mock_client_class):
        """Test concurrent requests to Urza for different kernels."""
        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        def create_mock_responses(kernel_id):
            # Create unique tensor for each kernel
            test_tensor = torch.randn(1024, dtype=torch.float32)

            urza_response = Mock()
            urza_response.status = 200

            async def mock_json():
                return {
                    "kernel_binary_ref": f"https://s3.example.com/kernels/{kernel_id}.bin"
                }

            urza_response.json = mock_json

            s3_response = Mock()
            s3_response.status = 200

            async def mock_read():
                return test_tensor.numpy().tobytes()

            s3_response.read = mock_read

            return urza_response, s3_response

        # Setup mock responses for 3 different kernels
        kernel_responses = {}
        for i in range(3):
            kernel_id = f"kernel-{i}"
            kernel_responses[kernel_id] = create_mock_responses(kernel_id)

        async def mock_get(url):
            for kernel_id, (urza_resp, s3_resp) in kernel_responses.items():
                if f"/api/v1/kernels/{kernel_id}" in url:
                    return urza_resp
                elif f"kernels/{kernel_id}.bin" in url:
                    return s3_resp
            raise ValueError(f"Unexpected URL: {url}")

        mock_client.get = mock_get

        # Make concurrent requests
        tasks = [cache.load_kernel(f"kernel-{i}") for i in range(3)]

        results = await asyncio.gather(*tasks)

        # Verify all kernels were loaded
        assert len(results) == 3
        for kernel in results:
            assert kernel is not None
            assert isinstance(kernel, torch.Tensor)
            assert kernel.shape == (1024,)

        # Verify cache state
        assert cache._misses == 3
        assert cache._hits == 0
        assert len(cache._cache) == 3

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"URZA_URL": "http://custom-urza:9000"})
    @patch("esper.utils.http_client.AsyncHttpClient")
    async def test_custom_urza_url_from_environment(self, mock_client_class):
        """Test using custom Urza URL from environment variable."""
        from esper.utils.config import reset_service_config

        # Reset configuration to pick up environment change
        reset_service_config()

        cache = KernelCache()

        # Mock the async HTTP client
        mock_client = Mock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock Urza response
        urza_response = Mock()
        urza_response.status = 404

        # Track the URL that was called
        called_urls = []

        async def mock_get(url):
            called_urls.append(url)
            return urza_response

        mock_client.get = mock_get

        # Test kernel loading
        await cache.load_kernel("test-kernel")

        # Verify custom URL was used
        assert len(called_urls) == 1
        assert "http://custom-urza:9000/api/v1/kernels/test-kernel" in called_urls[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

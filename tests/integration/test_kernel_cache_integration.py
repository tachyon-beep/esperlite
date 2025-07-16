"""
Integration tests for KernelCache with Urza service.

These tests verify real HTTP communication with mocked Urza service endpoints.
"""

import pytest
import torch
import asyncio
from unittest.mock import patch, Mock
from esper.execution.kernel_cache import KernelCache


class TestKernelCacheUrzaIntegration:
    """Integration tests for KernelCache with Urza service."""
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_successful_kernel_fetch_from_urza(self, mock_get):
        """Test successful kernel fetching from Urza with real HTTP calls."""
        cache = KernelCache()
        
        # Mock Urza API response
        urza_response = Mock()
        urza_response.status_code = 200
        urza_response.json.return_value = {
            "kernel_binary_ref": "https://s3.example.com/kernels/test-kernel-123.bin"
        }
        
        # Mock S3 binary response with real tensor data
        test_tensor = torch.randn(1024, dtype=torch.float32)
        s3_response = Mock()
        s3_response.status_code = 200
        s3_response.content = test_tensor.numpy().tobytes()
        
        # Configure mock to return different responses based on URL
        def mock_request_handler(url, **kwargs):
            if "/api/v1/kernels/" in url:
                return urza_response
            elif "s3.example.com" in url:
                return s3_response
            else:
                raise ValueError(f"Unexpected URL: {url}")
        
        mock_get.side_effect = mock_request_handler
        
        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel-123")
        
        # Verify HTTP calls were made correctly
        assert mock_get.call_count == 2
        calls = mock_get.call_args_list
        
        # First call should be to Urza API
        urza_call = calls[0]
        assert "/api/v1/kernels/test-kernel-123" in urza_call[0][0]
        assert urza_call[1]['timeout'] == 5
        
        # Second call should be to S3
        s3_call = calls[1] 
        assert "s3.example.com" in s3_call[0][0]
        assert s3_call[1]['timeout'] == 10
        
        # Verify kernel was loaded correctly
        assert kernel is not None
        assert isinstance(kernel, torch.Tensor)
        assert kernel.shape == (1024,)
        
        # Verify cache metrics
        assert cache._hits == 0
        assert cache._misses == 1
        assert len(cache._cache) == 1
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_kernel_not_found_in_urza(self, mock_get):
        """Test handling when kernel is not found in Urza."""
        cache = KernelCache()
        
        # Mock 404 response from Urza
        urza_response = Mock()
        urza_response.status_code = 404
        mock_get.return_value = urza_response
        
        # Test kernel loading
        kernel = await cache.load_kernel("non-existent-kernel")
        
        # Verify proper handling
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0
        
        # Verify API call was made
        mock_get.assert_called_once()
        assert "/api/v1/kernels/non-existent-kernel" in mock_get.call_args[0][0]
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_urza_server_error(self, mock_get):
        """Test handling when Urza server returns an error."""
        cache = KernelCache()
        
        # Mock server error from Urza
        urza_response = Mock()
        urza_response.status_code = 500
        urza_response.raise_for_status.side_effect = Exception("Server Error")
        mock_get.return_value = urza_response
        
        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel")
        
        # Should handle error gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_missing_binary_reference(self, mock_get):
        """Test handling when Urza response lacks binary reference."""
        cache = KernelCache()
        
        # Mock Urza response without binary_ref
        urza_response = Mock()
        urza_response.status_code = 200
        urza_response.json.return_value = {
            "kernel_id": "test-kernel-123"
            # Missing kernel_binary_ref
        }
        mock_get.return_value = urza_response
        
        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel-123")
        
        # Should handle missing reference gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_s3_download_failure(self, mock_get):
        """Test handling when S3 download fails."""
        cache = KernelCache()
        
        # Mock successful Urza response
        urza_response = Mock()
        urza_response.status_code = 200
        urza_response.json.return_value = {
            "kernel_binary_ref": "https://s3.example.com/kernels/test-kernel.bin"
        }
        
        # Mock failed S3 response
        s3_response = Mock()
        s3_response.status_code = 403
        s3_response.raise_for_status.side_effect = Exception("Access Denied")
        
        def mock_request_handler(url, **kwargs):
            if "/api/v1/kernels/" in url:
                return urza_response
            elif "s3.example.com" in url:
                return s3_response
            else:
                raise ValueError(f"Unexpected URL: {url}")
        
        mock_get.side_effect = mock_request_handler
        
        # Test kernel loading
        kernel = await cache.load_kernel("test-kernel")
        
        # Should handle S3 failure gracefully
        assert kernel is None
        assert cache._misses == 1
        assert len(cache._cache) == 0
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_concurrent_urza_requests(self, mock_get):
        """Test concurrent requests to Urza for different kernels."""
        cache = KernelCache()
        
        def create_mock_responses(kernel_id):
            # Create unique tensor for each kernel
            test_tensor = torch.randn(1024, dtype=torch.float32)
            
            urza_response = Mock()
            urza_response.status_code = 200
            urza_response.json.return_value = {
                "kernel_binary_ref": f"https://s3.example.com/kernels/{kernel_id}.bin"
            }
            
            s3_response = Mock()
            s3_response.status_code = 200
            s3_response.content = test_tensor.numpy().tobytes()
            
            return urza_response, s3_response
        
        # Setup mock responses for 3 different kernels
        kernel_responses = {}
        for i in range(3):
            kernel_id = f"kernel-{i}"
            kernel_responses[kernel_id] = create_mock_responses(kernel_id)
        
        def mock_request_handler(url, **kwargs):
            for kernel_id, (urza_resp, s3_resp) in kernel_responses.items():
                if f"/api/v1/kernels/{kernel_id}" in url:
                    return urza_resp
                elif f"kernels/{kernel_id}.bin" in url:
                    return s3_resp
            raise ValueError(f"Unexpected URL: {url}")
        
        mock_get.side_effect = mock_request_handler
        
        # Make concurrent requests
        tasks = [
            cache.load_kernel(f"kernel-{i}")
            for i in range(3)
        ]
        
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
    @patch('os.getenv')
    @patch('requests.get')
    async def test_custom_urza_url_from_environment(self, mock_get, mock_getenv):
        """Test using custom Urza URL from environment variable."""
        cache = KernelCache()
        
        # Mock environment variable
        mock_getenv.return_value = "http://custom-urza:9000"
        
        # Mock Urza response
        urza_response = Mock()
        urza_response.status_code = 404
        mock_get.return_value = urza_response
        
        # Test kernel loading
        await cache.load_kernel("test-kernel")
        
        # Verify custom URL was used
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "http://custom-urza:9000/api/v1/kernels/test-kernel" == call_url
        
        # Verify environment variable was checked
        mock_getenv.assert_called_once_with("URZA_API_URL", "http://localhost:8000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

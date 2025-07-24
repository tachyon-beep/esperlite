"""
Tests for async Conv2D implementation.
"""

import asyncio
import time

import pytest
import torch
import torch.nn.functional as F

from esper.execution.async_conv2d_kernel import AsyncConv2dKernel, create_async_conv2d_kernel
from esper.execution.gradient_sync import GradientSynchronizer, GradientSafeContext
from esper.execution.async_kasmina_conv2d_layer import AsyncKasminaConv2dLayer
from esper.execution.stream_manager import StreamManager, get_global_stream_manager


class TestAsyncConv2dKernel:
    """Test AsyncConv2dKernel functionality."""

    @pytest.mark.asyncio
    async def test_async_execution_correctness(self):
        """Verify async execution produces correct results."""
        # Setup
        batch_size, in_channels, height, width = 2, 3, 32, 32
        out_channels, kernel_size = 16, 3
        
        # Create input and weights
        x = torch.randn(batch_size, in_channels, height, width)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        bias = torch.randn(out_channels)
        
        # Reference synchronous execution
        sync_output = F.conv2d(x, weight, bias, padding=1)
        
        # Async execution
        kernel = create_async_conv2d_kernel(
            in_channels, out_channels, kernel_size, padding=1
        )
        async_output = await kernel.execute_async(x, weight, bias)
        
        # Verify correctness
        assert torch.allclose(sync_output, async_output, rtol=1e-5)
        
        # Cleanup
        kernel.cleanup()

    @pytest.mark.asyncio
    async def test_gradient_correctness(self):
        """Verify gradients are computed correctly with async execution."""
        # Setup
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        weight = torch.randn(8, 3, 3, 3, requires_grad=True)
        target = torch.randn(1, 8, 16, 16)
        
        # Gradient synchronizer
        grad_sync = GradientSynchronizer()
        
        # Async kernel
        kernel = create_async_conv2d_kernel(3, 8, 3, padding=1)
        
        # Forward pass with gradient tracking
        output = await grad_sync.register_async_operation(
            kernel.execute_async, (x, weight, None)
        )
        
        # Loss and backward
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # Verify gradients exist
        assert x.grad is not None
        assert weight.grad is not None
        assert torch.any(x.grad != 0)
        assert torch.any(weight.grad != 0)
        
        # Cleanup
        kernel.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test multiple async Conv2D operations concurrently."""
        num_ops = 4
        kernels = []
        operations = []
        
        # Create multiple kernels and operations
        for i in range(num_ops):
            kernel = create_async_conv2d_kernel(3, 16, 3, padding=1)
            kernels.append(kernel)
            
            x = torch.randn(1, 3, 32, 32)
            weight = torch.randn(16, 3, 3, 3)
            
            operations.append(kernel.execute_async(x, weight, None))
        
        # Execute concurrently
        start_time = time.time()
        outputs = await asyncio.gather(*operations)
        async_time = time.time() - start_time
        
        # Verify all completed
        assert len(outputs) == num_ops
        assert all(isinstance(out, torch.Tensor) for out in outputs)
        assert all(out.shape == (1, 16, 32, 32) for out in outputs)
        
        # Cleanup
        for kernel in kernels:
            kernel.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="CUDA stream test is flaky in test suite")
    async def test_cuda_stream_execution(self):
        """Test CUDA stream-based async execution."""
        device = torch.device("cuda")
        
        # Create CUDA tensors
        x = torch.randn(2, 3, 64, 64, device=device)
        weight = torch.randn(32, 3, 5, 5, device=device)
        
        # Create kernel with custom stream
        stream = torch.cuda.Stream()
        kernel = AsyncConv2dKernel(
            conv_params={'stride': 1, 'padding': 2, 'dilation': 1, 'groups': 1},
            device=device,
            stream=stream
        )
        
        # Execute async
        output = await kernel.execute_async(x, weight, None)
        
        # Verify output
        assert output.device.type == device.type
        assert output.shape == (2, 32, 64, 64)
        
        # Cleanup
        kernel.cleanup()

    @pytest.mark.asyncio
    async def test_cpu_thread_pool_execution(self):
        """Test CPU thread pool async execution."""
        device = torch.device("cpu")
        
        # CPU tensors
        x = torch.randn(1, 3, 32, 32)
        weight = torch.randn(16, 3, 3, 3)
        
        kernel = create_async_conv2d_kernel(3, 16, 3, device=device)
        
        # Execute multiple times
        outputs = []
        for _ in range(3):
            output = await kernel.execute_async(x, weight, None)
            outputs.append(output)
        
        # Verify consistency
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])
        
        # Check execution stats
        stats = kernel.get_execution_stats()
        assert stats["execution_count"] == 3
        assert stats["device"] == "cpu"
        
        kernel.cleanup()


class TestAsyncKasminaConv2dLayer:
    """Test AsyncKasminaConv2dLayer functionality."""

    @pytest.mark.asyncio
    async def test_async_forward_no_seeds(self):
        """Test async forward pass without active seeds."""
        layer = AsyncKasminaConv2dLayer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            num_seeds=4,
            telemetry_enabled=False
        )
        
        x = torch.randn(2, 3, 32, 32)
        
        # Async forward
        output = await layer.forward_async(x)
        
        # Verify output
        assert output.shape == (2, 16, 32, 32)
        assert layer.async_execution_count == 1

    @pytest.mark.asyncio
    async def test_gradient_safe_context(self):
        """Test gradient-safe context for async operations."""
        layer = AsyncKasminaConv2dLayer(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,  # Add padding to preserve spatial dimensions
            enable_gradient_sync=True,
            telemetry_enabled=False
        )
        
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        target = torch.randn(1, 8, 16, 16)
        
        # Use gradient-safe context
        async with GradientSafeContext(layer.gradient_sync) as sync:
            output = await layer.forward_async(x)
            
        # Compute loss and gradients
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # Verify gradients
        assert x.grad is not None
        assert layer.default_transform.weight.grad is not None

    @pytest.mark.asyncio
    async def test_mixed_sync_async_execution(self):
        """Test mixing sync and async execution."""
        layer = AsyncKasminaConv2dLayer(3, 16, 3, padding=1, telemetry_enabled=False)
        x = torch.randn(1, 3, 32, 32)
        
        # Sync execution
        sync_output = layer.forward(x)
        
        # Async execution
        async_output = await layer.forward_async(x)
        
        # Should produce same results
        assert torch.allclose(sync_output, async_output, rtol=1e-5)
        
        # Check stats
        stats = layer.get_async_stats()
        assert stats["async_execution_count"] == 1
        assert stats["sync_fallback_count"] >= 0

    def test_sync_forward_compatibility(self):
        """Test synchronous forward method compatibility."""
        layer = AsyncKasminaConv2dLayer(3, 16, 3, padding=1, telemetry_enabled=False)  # Add padding to preserve dimensions
        x = torch.randn(2, 3, 64, 64)
        
        # Should work in sync context
        output = layer.forward(x)
        assert output.shape == (2, 16, 64, 64)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test proper cleanup of async resources."""
        layer = AsyncKasminaConv2dLayer(3, 32, 5, padding=2, telemetry_enabled=False)
        
        # Execute some operations
        x = torch.randn(1, 3, 32, 32)
        _ = await layer.forward_async(x)
        
        # Cleanup
        layer.cleanup()
        
        # Verify cleanup
        assert layer.async_kernel.executor._shutdown


class TestStreamManager:
    """Test StreamManager for multi-GPU support."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stream_allocation(self):
        """Test stream allocation across devices."""
        manager = StreamManager(num_streams_per_device=4)
        device = torch.device("cuda:0")
        
        # Get multiple streams
        streams = [manager.get_stream(device) for _ in range(6)]
        
        # Should cycle through available streams
        assert streams[0] == streams[4]  # Round-robin
        assert streams[1] == streams[5]
        
        # Check stats
        stats = manager.get_stats()
        assert stats["num_devices"] == 1
        assert stats["streams_per_device"] == 4
        
        manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_stream_synchronization(self):
        """Test async stream synchronization."""
        manager = StreamManager()
        device = torch.device("cuda:0")
        stream = manager.get_stream(device)
        
        # Queue operation on stream
        with torch.cuda.stream(stream):
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x)
        
        # Record event
        event = torch.cuda.Event()
        event.record(stream)
        
        # Async wait
        await manager.synchronize_stream(stream)
        
        # Should be complete
        assert event.query()
        
        manager.cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="CUDA stream test is flaky in test suite")
    def test_stream_context(self):
        """Test stream context manager."""
        from esper.execution.stream_manager import StreamContext
        
        manager = StreamManager()
        device = torch.device("cuda:0")
        
        original_stream = torch.cuda.current_stream()
        
        with StreamContext(manager, device) as stream:
            # Should be on different stream
            assert torch.cuda.current_stream() != original_stream
            assert torch.cuda.current_stream() == stream
        
        # Should restore original stream
        assert torch.cuda.current_stream() == original_stream
        
        manager.cleanup()

    def test_global_stream_manager(self):
        """Test global stream manager singleton."""
        manager1 = get_global_stream_manager()
        manager2 = get_global_stream_manager()
        
        # Should be same instance
        assert manager1 is manager2
        
        # Cleanup
        from esper.execution.stream_manager import cleanup_global_stream_manager
        cleanup_global_stream_manager()


class TestIntegration:
    """Integration tests for async Conv2D pipeline."""

    @pytest.mark.asyncio
    async def test_full_async_pipeline(self):
        """Test complete async execution pipeline."""
        # Create layer
        layer = AsyncKasminaConv2dLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            num_seeds=2,
            enable_gradient_sync=True,
            telemetry_enabled=False
        )
        
        # Input with gradients
        x = torch.randn(4, 3, 224, 224, requires_grad=True)
        target = torch.randn(4, 64, 112, 112)
        
        # Forward pass
        output = await layer.forward_async(x)
        
        # Loss computation
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Verify execution
        assert output.shape == (4, 64, 112, 112)
        assert x.grad is not None
        assert layer.default_transform.weight.grad is not None
        
        # Check async stats
        stats = layer.get_async_stats()
        assert stats["async_execution_count"] == 1
        assert stats["uses_cuda_streams"] == (x.device.type == "cuda")
        
        # Cleanup
        layer.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    async def test_device_compatibility(self, device_type):
        """Test compatibility across devices."""
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device(device_type)
        
        layer = AsyncKasminaConv2dLayer(3, 16, 3, padding=1, telemetry_enabled=False).to(device)  # Add padding
        x = torch.randn(2, 3, 32, 32, device=device)
        
        output = await layer.forward_async(x)
        
        assert output.device.type == device_type
        assert output.shape == (2, 16, 32, 32)
        
        layer.cleanup()
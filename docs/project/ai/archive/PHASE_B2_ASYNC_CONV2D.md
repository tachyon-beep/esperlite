# Phase B2: Async Support for Conv2D Layers - Detailed Implementation

## Overview

Phase B2 addresses the synchronous execution fallback in Conv2D layers when operating in async contexts. This phase enables true asynchronous execution for convolutional operations, maintaining performance while ensuring gradient correctness.

## Current State Analysis

### Problem Statement
- Conv2D kernel execution falls back to synchronous execution in async contexts
- Current workaround uses `asyncio.run_until_complete()` which blocks the event loop
- This creates performance bottlenecks in async training pipelines
- Gradient computation must remain correct with async execution

### Existing Code Issues
```python
# Current problematic code in KasminaConv2dLayer
async def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.active_seed and self.execution_mode == "morpho":
        # This falls back to sync execution!
        return await self._execute_kernel_async(x, self.active_seed)
    return self._default_conv2d(x)
```

## Detailed Implementation Plan

### 1. Async Conv2D Kernel Wrapper

**File**: `src/esper/execution/async_conv2d_kernel.py`

```python
import torch
import asyncio
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

class AsyncConv2dKernel:
    """Enables true async execution for Conv2D operations."""
    
    def __init__(self, 
                 conv_params: Dict[str, Any],
                 device: torch.device,
                 stream: Optional[torch.cuda.Stream] = None):
        self.conv_params = conv_params
        self.device = device
        self.stream = stream or torch.cuda.default_stream()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    async def execute_async(self, 
                          x: torch.Tensor, 
                          weight: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute Conv2D operation asynchronously.
        
        Strategy:
        1. Use CUDA streams for GPU async
        2. Use thread pool for CPU ops
        3. Maintain gradient tape integrity
        """
        if x.is_cuda:
            return await self._execute_cuda_async(x, weight, bias)
        else:
            return await self._execute_cpu_async(x, weight, bias)
            
    async def _execute_cuda_async(self, x, weight, bias):
        """GPU async execution using CUDA streams."""
        # Record current stream state
        original_stream = torch.cuda.current_stream()
        
        # Switch to our async stream
        with torch.cuda.stream(self.stream):
            # Queue Conv2D operation
            output = torch.nn.functional.conv2d(
                x, weight, bias,
                stride=self.conv_params['stride'],
                padding=self.conv_params['padding'],
                dilation=self.conv_params['dilation'],
                groups=self.conv_params['groups']
            )
            
            # Create event for synchronization
            event = torch.cuda.Event()
            event.record(self.stream)
        
        # Return awaitable future
        future = asyncio.create_future()
        
        def check_completion():
            if event.query():
                future.set_result(output)
                return False
            return True  # Continue checking
            
        # Poll for completion without blocking
        async def wait_for_completion():
            while check_completion():
                await asyncio.sleep(0)  # Yield to event loop
            return future.result()
            
        return await wait_for_completion()
        
    async def _execute_cpu_async(self, x, weight, bias):
        """CPU async execution using thread pool."""
        loop = asyncio.get_event_loop()
        
        def conv_operation():
            return torch.nn.functional.conv2d(
                x, weight, bias,
                stride=self.conv_params['stride'],
                padding=self.conv_params['padding'],
                dilation=self.conv_params['dilation'],
                groups=self.conv_params['groups']
            )
            
        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(self.executor, conv_operation)
```

### 2. Gradient-Safe Async Execution

**File**: `src/esper/execution/gradient_sync.py`

```python
class GradientSynchronizer:
    """Ensures gradient correctness with async operations."""
    
    def __init__(self):
        self.pending_operations = []
        self.gradient_events = []
        
    async def register_async_operation(self, 
                                     operation: Callable,
                                     inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Register async operation for gradient tracking.
        
        Ensures:
        1. Gradient graph is maintained
        2. Backward pass waits for forward completion
        3. No race conditions in gradient accumulation
        """
        # Create gradient placeholder
        grad_placeholder = GradientPlaceholder(inputs)
        
        # Execute operation
        output = await operation(*inputs)
        
        # Register for backward sync
        if any(inp.requires_grad for inp in inputs):
            self._register_backward_hook(output, grad_placeholder)
            
        return output
        
    def _register_backward_hook(self, tensor: torch.Tensor, placeholder: GradientPlaceholder):
        """Register hook to sync gradients during backward pass."""
        def grad_sync_hook(grad):
            # Ensure all async ops complete before backward
            asyncio.create_task(self._sync_before_backward(placeholder))
            return grad
            
        tensor.register_hook(grad_sync_hook)
        
    async def _sync_before_backward(self, placeholder: GradientPlaceholder):
        """Synchronize all pending operations before gradient computation."""
        await asyncio.gather(*self.pending_operations)
        self.pending_operations.clear()
```

### 3. Updated KasminaConv2dLayer

**File**: `src/esper/execution/kasmina_conv2d_layer.py` (updates)

```python
class KasminaConv2dLayer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize async kernel wrapper
        self.async_kernel = AsyncConv2dKernel(
            conv_params={
                'stride': self.stride,
                'padding': self.padding,
                'dilation': self.dilation,
                'groups': self.groups
            },
            device=self.weight.device
        )
        self.gradient_sync = GradientSynchronizer()
        
    async def _execute_kernel_real(self, x: torch.Tensor, seed_data: SeedData) -> torch.Tensor:
        """
        Execute kernel asynchronously without blocking.
        """
        try:
            # Load kernel parameters
            kernel_weight = seed_data.parameters.get('weight', self.weight)
            kernel_bias = seed_data.parameters.get('bias', self.bias)
            
            # Execute asynchronously with gradient safety
            output = await self.gradient_sync.register_async_operation(
                lambda x, w, b: self.async_kernel.execute_async(x, w, b),
                (x, kernel_weight, kernel_bias)
            )
            
            # Record telemetry without blocking
            asyncio.create_task(self._record_async_telemetry(
                kernel_id=seed_data.kernel_id,
                execution_time=time.time()
            ))
            
            return output
            
        except Exception as e:
            logger.error(f"Async kernel execution failed: {e}")
            # Fallback to sync execution
            return self._default_conv2d(x)
            
    async def _record_async_telemetry(self, kernel_id: str, execution_time: float):
        """Record telemetry without blocking execution."""
        # Non-blocking telemetry recording
        await self.telemetry.record_async({
            'kernel_id': kernel_id,
            'execution_time': execution_time,
            'device': str(self.weight.device),
            'async': True
        })
```

### 4. Stream Management for Multi-GPU

**File**: `src/esper/execution/stream_manager.py`

```python
class StreamManager:
    """Manages CUDA streams for async execution across devices."""
    
    def __init__(self, num_streams_per_device: int = 4):
        self.streams = {}
        self.num_streams = num_streams_per_device
        self.stream_index = {}
        
    def get_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Get next available stream for device (round-robin)."""
        device_id = device.index or 0
        
        if device_id not in self.streams:
            self.streams[device_id] = [
                torch.cuda.Stream(device=device) 
                for _ in range(self.num_streams)
            ]
            self.stream_index[device_id] = 0
            
        # Round-robin stream selection
        idx = self.stream_index[device_id]
        self.stream_index[device_id] = (idx + 1) % self.num_streams
        
        return self.streams[device_id][idx]
        
    async def synchronize_all(self):
        """Synchronize all streams across all devices."""
        sync_tasks = []
        for device_streams in self.streams.values():
            for stream in device_streams:
                sync_tasks.append(self._sync_stream(stream))
        await asyncio.gather(*sync_tasks)
        
    async def _sync_stream(self, stream: torch.cuda.Stream):
        """Asynchronously wait for stream completion."""
        event = torch.cuda.Event()
        event.record(stream)
        
        while not event.query():
            await asyncio.sleep(0)
```

### 5. Testing Framework for Async Correctness

**File**: `tests/execution/test_async_conv2d.py`

```python
import pytest
import torch
import asyncio

class TestAsyncConv2d:
    @pytest.mark.asyncio
    async def test_async_execution_correctness(self):
        """Verify async execution produces correct results."""
        # Setup
        layer = KasminaConv2dLayer(3, 64, 3, padding=1)
        x = torch.randn(1, 3, 224, 224)
        
        # Sync reference
        sync_output = layer._default_conv2d(x)
        
        # Async execution
        async_output = await layer.forward(x)
        
        # Verify correctness
        assert torch.allclose(sync_output, async_output, rtol=1e-5)
        
    @pytest.mark.asyncio
    async def test_gradient_correctness(self):
        """Verify gradients are computed correctly with async execution."""
        # Setup
        layer = KasminaConv2dLayer(3, 64, 3, padding=1)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        target = torch.randn(1, 64, 32, 32)
        
        # Forward pass
        output = await layer.forward(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert x.grad is not None
        assert torch.any(x.grad != 0)
        assert layer.weight.grad is not None
        
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test multiple async Conv2D operations concurrently."""
        layers = [KasminaConv2dLayer(3, 64, 3) for _ in range(4)]
        inputs = [torch.randn(1, 3, 32, 32) for _ in range(4)]
        
        # Execute concurrently
        outputs = await asyncio.gather(*[
            layer.forward(x) for layer, x in zip(layers, inputs)
        ])
        
        # Verify all completed
        assert len(outputs) == 4
        assert all(isinstance(out, torch.Tensor) for out in outputs)
        
    @pytest.mark.asyncio
    async def test_stream_isolation(self):
        """Verify operations on different streams don't interfere."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        layer1 = KasminaConv2dLayer(3, 64, 3).cuda()
        layer2 = KasminaConv2dLayer(3, 64, 3).cuda()
        
        x1 = torch.randn(1, 3, 224, 224).cuda()
        x2 = torch.randn(1, 3, 224, 224).cuda()
        
        # Execute on different streams
        out1, out2 = await asyncio.gather(
            layer1.forward(x1),
            layer2.forward(x2)
        )
        
        # Verify both completed correctly
        assert out1.shape == (1, 64, 224, 224)
        assert out2.shape == (1, 64, 224, 224)
```

## Performance Benchmarks

**File**: `tests/performance/test_async_conv2d_performance.py`

```python
class AsyncConv2dBenchmark:
    async def benchmark_async_vs_sync(self):
        """Compare async vs sync performance."""
        results = {
            'sync_time': [],
            'async_time': [],
            'speedup': []
        }
        
        for batch_size in [1, 8, 32, 128]:
            # Sync execution
            sync_time = self._measure_sync_execution(batch_size)
            
            # Async execution
            async_time = await self._measure_async_execution(batch_size)
            
            results['sync_time'].append(sync_time)
            results['async_time'].append(async_time)
            results['speedup'].append(sync_time / async_time)
            
        return results
```

## Integration Guide

### 1. Feature Flag Configuration
```python
# config/features.py
ASYNC_CONV2D_CONFIG = {
    "enabled": False,  # Start disabled
    "fallback_on_error": True,
    "max_concurrent_ops": 4,
    "stream_pool_size": 4
}
```

### 2. Gradual Rollout Plan
1. Enable for 1% of Conv2D operations
2. Monitor error rates and performance
3. Increase to 10%, 50%, 100%
4. Remove sync fallback after stability

### 3. Monitoring Metrics
- Async execution success rate
- Average execution time comparison
- Gradient computation correctness
- Memory usage patterns
- Stream utilization

## Success Criteria

### Functional Requirements
- ✅ Conv2D executes asynchronously without blocking
- ✅ Gradient computation remains correct
- ✅ No race conditions in concurrent execution
- ✅ Proper error handling and fallback

### Performance Requirements
- Async overhead < 5% vs sync execution
- Concurrent execution scaling > 0.8x per stream
- No memory leaks in long-running training
- Stream utilization > 75%

### Reliability Requirements
- Async execution success rate > 99.9%
- Gradient correctness validation 100%
- Proper cleanup on errors
- No deadlocks or race conditions

## Risk Mitigation

### Risk 1: Gradient Computation Errors
**Mitigation**: Extensive gradient checking, numerical validation tests

### Risk 2: Memory Leaks
**Mitigation**: Proper stream cleanup, memory profiling in CI/CD

### Risk 3: Performance Regression
**Mitigation**: Continuous benchmarking, automatic rollback

### Risk 4: Device Synchronization Issues
**Mitigation**: Explicit synchronization points, thorough testing

## Documentation Requirements

1. **Developer Guide**: How to use async Conv2D layers
2. **Performance Guide**: Tuning async execution
3. **Debugging Guide**: Common issues and solutions
4. **Migration Guide**: Converting sync to async code
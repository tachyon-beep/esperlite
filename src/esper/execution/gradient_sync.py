"""
Gradient Synchronization Module.

This module ensures gradient correctness when executing operations
asynchronously, preventing race conditions and maintaining proper
backward pass synchronization.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from weakref import WeakKeyDictionary

import torch

logger = logging.getLogger(__name__)


@dataclass
class GradientPlaceholder:
    """Placeholder for gradient computation tracking."""
    inputs: Tuple[torch.Tensor, ...]
    operation_id: str
    requires_sync: bool = True
    completed: bool = False


class GradientSynchronizer:
    """
    Ensures gradient correctness with async operations.
    
    This class tracks async operations and ensures all forward passes
    complete before backward passes begin, maintaining gradient
    computation integrity.
    """

    def __init__(self):
        """Initialize the gradient synchronizer."""
        self.pending_operations: List[asyncio.Task] = []
        self.gradient_events: List[torch.cuda.Event] = []
        self.operation_registry: WeakKeyDictionary = WeakKeyDictionary()
        self._operation_counter = 0
        self._sync_enabled = True

    async def register_async_operation(
        self,
        operation: Callable,
        inputs: Tuple[torch.Tensor, ...],
        **kwargs
    ) -> torch.Tensor:
        """
        Register async operation for gradient tracking.
        
        Ensures:
        1. Gradient graph is maintained
        2. Backward pass waits for forward completion
        3. No race conditions in gradient accumulation
        
        Args:
            operation: Async operation to execute
            inputs: Input tensors
            **kwargs: Additional arguments for operation
            
        Returns:
            Output tensor with gradient tracking
        """
        self._operation_counter += 1
        op_id = f"op_{self._operation_counter}"

        # Create gradient placeholder
        grad_placeholder = GradientPlaceholder(inputs, op_id)

        # Execute operation
        if asyncio.iscoroutinefunction(operation):
            output = await operation(*inputs, **kwargs)
        else:
            # Wrap sync operation for consistency
            output = operation(*inputs, **kwargs)

        # Register for backward sync if gradients required
        if self._sync_enabled and any(inp.requires_grad for inp in inputs):
            self._register_backward_hook(output, grad_placeholder)
            self.operation_registry[output] = grad_placeholder

        return output

    def _register_backward_hook(
        self, tensor: torch.Tensor, placeholder: GradientPlaceholder
    ):
        """Register hook to sync gradients during backward pass."""
        def grad_sync_hook(grad):
            """Synchronize before computing gradients."""
            if not placeholder.completed:
                # Synchronously wait for async operations
                # This is called during backward pass which is synchronous
                import concurrent.futures
                import threading

                # Create a future to wait on
                future = concurrent.futures.Future()

                def run_sync():
                    try:
                        # Create new event loop in thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._sync_before_backward(placeholder))
                        loop.close()
                        future.set_result(None)
                    except Exception as e:
                        future.set_exception(e)

                # Run in a separate thread to avoid event loop conflicts
                thread = threading.Thread(target=run_sync)
                thread.start()
                thread.join()

                # Get result or raise exception
                future.result()
                placeholder.completed = True
            return grad

        if tensor.requires_grad:
            tensor.register_hook(grad_sync_hook)

    async def _sync_before_backward(self, placeholder: GradientPlaceholder):
        """Synchronize all pending operations before gradient computation."""
        if self.pending_operations:
            logger.debug(
                f"Synchronizing {len(self.pending_operations)} operations "
                f"before backward for {placeholder.operation_id}"
            )

            # Wait for all pending operations
            await asyncio.gather(*self.pending_operations, return_exceptions=True)
            self.pending_operations.clear()

        # Synchronize CUDA events if any
        if self.gradient_events:
            for event in self.gradient_events:
                if not event.query():
                    # Wait for CUDA operation to complete
                    while not event.query():
                        await asyncio.sleep(0)
            self.gradient_events.clear()

    def add_pending_operation(self, task: asyncio.Task):
        """Add an operation to track for synchronization."""
        self.pending_operations.append(task)

    def add_cuda_event(self, event: torch.cuda.Event):
        """Add a CUDA event to track for synchronization."""
        self.gradient_events.append(event)

    async def synchronize_all(self):
        """Manually synchronize all pending operations."""
        if self.pending_operations:
            await asyncio.gather(*self.pending_operations, return_exceptions=True)
            self.pending_operations.clear()

        # Clear CUDA events
        self.gradient_events.clear()

    def enable_sync(self):
        """Enable gradient synchronization."""
        self._sync_enabled = True

    def disable_sync(self):
        """Disable gradient synchronization (use with caution)."""
        self._sync_enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            "total_operations": self._operation_counter,
            "pending_operations": len(self.pending_operations),
            "pending_cuda_events": len(self.gradient_events),
            "sync_enabled": self._sync_enabled,
        }


class GradientSafeContext:
    """
    Context manager for gradient-safe async execution.
    
    Ensures all async operations within the context complete
    before gradients are computed.
    """

    def __init__(self, synchronizer: Optional[GradientSynchronizer] = None):
        """
        Initialize gradient-safe context.
        
        Args:
            synchronizer: Optional existing synchronizer to use
        """
        self.synchronizer = synchronizer or GradientSynchronizer()
        self._original_sync_state = None

    async def __aenter__(self) -> GradientSynchronizer:
        """Enter gradient-safe context."""
        self._original_sync_state = self.synchronizer._sync_enabled
        self.synchronizer.enable_sync()
        return self.synchronizer

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and ensure synchronization."""
        # Synchronize all pending operations
        await self.synchronizer.synchronize_all()

        # Restore original sync state
        if self._original_sync_state is not None:
            self.synchronizer._sync_enabled = self._original_sync_state


def create_gradient_checkpoint(
    tensors: List[torch.Tensor],
    name: str = "checkpoint"
) -> torch.cuda.Event:
    """
    Create a gradient checkpoint for a set of tensors.
    
    This ensures all operations on these tensors complete before
    gradient computation begins.
    
    Args:
        tensors: List of tensors to checkpoint
        name: Name for the checkpoint
        
    Returns:
        CUDA event for synchronization (if using CUDA)
    """
    if any(t.is_cuda for t in tensors):
        event = torch.cuda.Event()
        event.record()
        logger.debug(f"Created CUDA gradient checkpoint: {name}")
        return event
    else:
        logger.debug(f"Created CPU gradient checkpoint: {name}")
        return None


class AsyncGradientAccumulator:
    """
    Accumulates gradients from multiple async operations.
    
    Ensures thread-safe gradient accumulation when multiple
    async operations update the same parameters.
    """

    def __init__(self):
        """Initialize the gradient accumulator."""
        self._gradient_buffers: Dict[torch.Tensor, torch.Tensor] = {}
        self._lock = asyncio.Lock()

    async def accumulate_gradient(
        self,
        parameter: torch.Tensor,
        gradient: torch.Tensor
    ):
        """
        Accumulate gradient for a parameter.
        
        Args:
            parameter: Parameter tensor
            gradient: Gradient to accumulate
        """
        async with self._lock:
            if parameter in self._gradient_buffers:
                self._gradient_buffers[parameter] += gradient
            else:
                self._gradient_buffers[parameter] = gradient.clone()

    async def apply_gradients(self):
        """Apply accumulated gradients to parameters."""
        async with self._lock:
            for param, grad in self._gradient_buffers.items():
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad

            # Clear buffers
            self._gradient_buffers.clear()

    def clear(self):
        """Clear accumulated gradients."""
        self._gradient_buffers.clear()

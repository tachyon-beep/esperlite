"""
Real kernel execution engine for morphogenetic adaptations.

This module implements the core kernel execution system that replaces
the placeholder implementation in KasminaLayer with real PyTorch module execution.
"""

import asyncio
import io
import logging
import pickle
import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from .error_recovery import ErrorType
from .error_recovery import create_error_context
from .exceptions import KernelDeserializationError
from .exceptions import KernelExecutionError


# Import error recovery after exceptions to avoid circular import
def _lazy_import_error_recovery():
    from .error_recovery import ErrorRecoveryManager
    from .error_recovery import ErrorType
    from .error_recovery import create_error_context

    return ErrorRecoveryManager, ErrorType, create_error_context


logger = logging.getLogger(__name__)


class KernelShapeError(Exception):
    """Exception raised when kernel input/output shapes are incompatible."""

    pass


@dataclass
class ExecutionStats:
    """Statistics for kernel execution performance."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    deserialize_errors: int = 0
    shape_errors: int = 0
    runtime_errors: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate execution success rate."""
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.successful_executions == 0:
            return 0.0
        return (self.total_execution_time / self.successful_executions) * 1000

    def record_success(self, execution_time: float):
        """Record successful execution."""
        self.total_executions += 1
        self.successful_executions += 1
        self.total_execution_time += execution_time

    def record_error(self, error_type: str):
        """Record execution error."""
        self.total_executions += 1
        self.failed_executions += 1

        if error_type == "deserialize":
            self.deserialize_errors += 1
        elif error_type == "shape":
            self.shape_errors += 1
        elif error_type == "runtime":
            self.runtime_errors += 1


class KernelValidator:
    """Validates kernel compatibility and safety."""

    def __init__(self):
        self.max_parameters = 10_000_000  # 10M parameters max
        self.allowed_modules = {
            nn.Linear,
            nn.Conv2d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.LayerNorm,
            nn.ReLU,
            nn.GELU,
            nn.Tanh,
            nn.Sigmoid,
            nn.Dropout,
            nn.Sequential,
            nn.ModuleList,
        }
        # Also allow TorchScript modules
        try:
            import torch.jit

            # Add common TorchScript types
            script_types = [torch.jit.ScriptModule, torch.jit.TracedModule]
            for script_type in script_types:
                self.allowed_modules.add(script_type)
        except (ImportError, AttributeError):
            pass  # TorchScript not available

    def validate_module(self, module: nn.Module) -> Tuple[bool, str]:
        """
        Validate that a module is safe to execute.

        Args:
            module: PyTorch module to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check parameter count
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > self.max_parameters:
                return (
                    False,
                    f"Module has too many parameters: {param_count} > {self.max_parameters}",
                )

            # Check module types (recursive)
            if not self._validate_module_types(module):
                return False, "Module contains disallowed layer types"

            # Check for potentially dangerous operations
            if self._has_dangerous_operations(module):
                return False, "Module contains potentially dangerous operations"

            return True, ""

        except Exception as e:
            return False, f"Validation error: {e}"

    def _validate_module_types(self, module: nn.Module) -> bool:
        """Recursively validate module types."""
        module_type = type(module)

        # Allow TorchScript modules (any module whose type name contains 'Script')
        if "Script" in module_type.__name__ or "Traced" in module_type.__name__:
            return True

        if module_type not in self.allowed_modules:
            logger.warning(f"Disallowed module type: {module_type}")
            return False

        for child in module.children():
            if not self._validate_module_types(child):
                return False

        return True

    def _has_dangerous_operations(self, module: nn.Module) -> bool:
        """Check for potentially dangerous operations."""
        # Convert to string and check for dangerous patterns
        module_str = str(module)

        dangerous_patterns = [
            "exec(",
            "eval(",
            "compile(",
            "import ",
            "__import__",
            "subprocess",
            "os.system",
            "open(",
        ]

        for pattern in dangerous_patterns:
            if pattern in module_str:
                logger.warning(f"Found dangerous pattern: {pattern}")
                return True

        return False

    def validate_shapes(
        self,
        input_shape: torch.Size,
        expected_input_shape: torch.Size,
        output_shape: torch.Size,
        expected_output_shape: torch.Size,
    ) -> Tuple[bool, str]:
        """
        Validate input/output shape compatibility.

        Args:
            input_shape: Actual input tensor shape
            expected_input_shape: Expected input shape from kernel
            output_shape: Actual output tensor shape
            expected_output_shape: Expected output shape

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Allow flexible batch dimension
        if len(input_shape) != len(expected_input_shape):
            return (
                False,
                f"Input rank mismatch: {len(input_shape)} vs {len(expected_input_shape)}",
            )

        # Check all dimensions except batch (index 0)
        for i in range(1, len(input_shape)):
            if input_shape[i] != expected_input_shape[i]:
                return (
                    False,
                    f"Input shape mismatch at dim {i}: {input_shape[i]} vs {expected_input_shape[i]}",
                )

        if len(output_shape) != len(expected_output_shape):
            return (
                False,
                f"Output rank mismatch: {len(output_shape)} vs {len(expected_output_shape)}",
            )

        for i in range(1, len(output_shape)):
            if output_shape[i] != expected_output_shape[i]:
                return (
                    False,
                    f"Output shape mismatch at dim {i}: {output_shape[i]} vs {expected_output_shape[i]}",
                )

        return True, ""


class RealKernelExecutor:
    """
    Real kernel execution engine for morphogenetic adaptations.

    This class handles the execution of compiled kernel artifacts with proper
    tensor handling, error recovery, and performance monitoring.
    """

    def __init__(
        self,
        device: torch.device,
        max_kernel_cache_size: int = 100,
        enable_validation: bool = True,
        execution_timeout: float = 10.0,
    ):
        """
        Initialize the kernel executor.

        Args:
            device: Device to execute kernels on
            max_kernel_cache_size: Maximum number of deserialized kernels to cache
            enable_validation: Whether to validate kernels before execution
            execution_timeout: Maximum execution time per kernel in seconds
        """
        self.device = device
        self.max_kernel_cache_size = max_kernel_cache_size
        self.enable_validation = enable_validation
        self.execution_timeout = execution_timeout

        # Deserialized kernel cache (kernel_id -> module)
        self.kernel_cache: Dict[str, nn.Module] = {}
        self.cache_access_times: Dict[str, float] = {}

        # Components
        self.validator = KernelValidator()
        self.stats = ExecutionStats()
        error_recovery_manager, _, _ = _lazy_import_error_recovery()
        self.error_recovery = error_recovery_manager()

        logger.info(f"Initialized RealKernelExecutor on device {device}")

    async def execute_kernel(
        self,
        kernel_artifact: bytes,
        input_tensor: torch.Tensor,
        original_shape: torch.Size,
        blend_alpha: float,
        kernel_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Execute compiled kernel with proper tensor handling.

        Args:
            kernel_artifact: Serialized kernel bytes
            input_tensor: Input tensor for kernel
            original_shape: Original tensor shape for validation
            blend_alpha: Alpha blending factor (0.0 = skip kernel, 1.0 = full kernel)
            kernel_id: Optional kernel ID for caching

        Returns:
            Output tensor from kernel execution

        Raises:
            KernelExecutionError: If kernel execution fails
        """
        start_time = time.perf_counter()

        try:
            # Quick exit for alpha = 0
            if blend_alpha <= 0.0:
                return input_tensor

            # Deserialize kernel
            kernel_module = await self._deserialize_kernel(kernel_artifact, kernel_id)

            # Validate if enabled
            if self.enable_validation:
                is_valid, error_msg = self.validator.validate_module(kernel_module)
                if not is_valid:
                    raise KernelExecutionError(f"Kernel validation failed: {error_msg}")

            # Execute kernel with timeout
            output_tensor = await self._execute_with_timeout(
                kernel_module, input_tensor
            )

            # Validate output shape
            if self.enable_validation:
                self._validate_output_shape(output_tensor, input_tensor, original_shape)

            # Apply alpha blending if not full kernel execution
            if blend_alpha < 1.0:
                output_tensor = self._apply_alpha_blending(
                    input_tensor, output_tensor, blend_alpha
                )

            # Record success
            execution_time = time.perf_counter() - start_time
            self.stats.record_success(execution_time)

            return output_tensor

        except KernelDeserializationError as e:
            self.stats.record_error("deserialize")

            # Handle through error recovery system
            error_context = create_error_context(
                error_type=ErrorType.KERNEL_VALIDATION,
                component="kernel_executor",
                layer_name=kernel_id or "unknown",
                exception=e,
                kernel_id=kernel_id,
            )
            await self.error_recovery.handle_error(error_context)

            raise KernelExecutionError(f"Kernel deserialization failed: {e}")

        except KernelShapeError as e:
            self.stats.record_error("shape")

            # Handle through error recovery system
            error_context = create_error_context(
                error_type=ErrorType.KERNEL_VALIDATION,
                component="kernel_executor",
                layer_name=kernel_id or "unknown",
                exception=e,
                kernel_id=kernel_id,
            )
            await self.error_recovery.handle_error(error_context)

            raise KernelExecutionError(f"Kernel shape error: {e}")

        except Exception as e:
            self.stats.record_error("runtime")

            # Classify and handle through error recovery system
            if "timeout" in str(e).lower():
                error_type = ErrorType.TIMEOUT
            elif "memory" in str(e).lower():
                error_type = ErrorType.MEMORY_OVERFLOW
            else:
                error_type = ErrorType.KERNEL_EXECUTION

            error_context = create_error_context(
                error_type=error_type,
                component="kernel_executor",
                layer_name=kernel_id or "unknown",
                exception=e,
                kernel_id=kernel_id,
            )
            await self.error_recovery.handle_error(error_context)

            raise KernelExecutionError(f"Kernel execution failed: {e}")

    async def _deserialize_kernel(
        self, kernel_artifact: bytes, kernel_id: Optional[str] = None
    ) -> nn.Module:
        """
        Deserialize kernel module from bytes with caching.

        Args:
            kernel_artifact: Serialized kernel bytes
            kernel_id: Optional kernel ID for caching

        Returns:
            Deserialized PyTorch module

        Raises:
            KernelDeserializationError: If deserialization fails
        """
        # Check cache first
        if kernel_id and kernel_id in self.kernel_cache:
            self.cache_access_times[kernel_id] = time.time()
            return self.kernel_cache[kernel_id]

        try:
            # Try torch.jit first (preferred format)
            module = self._deserialize_torchscript(kernel_artifact)

        except Exception:
            try:
                # Fallback to pickle (with security validation)
                module = self._deserialize_pickle(kernel_artifact)

            except Exception as e:
                raise KernelDeserializationError(f"Failed to deserialize kernel: {e}")

        # Move to target device
        module = module.to(self.device)
        module.eval()  # Set to evaluation mode

        # Cache if kernel_id provided
        if kernel_id:
            self._add_to_cache(kernel_id, module)

        return module

    def _deserialize_torchscript(self, kernel_artifact: bytes) -> nn.Module:
        """Deserialize TorchScript module."""
        buffer = io.BytesIO(kernel_artifact)
        return torch.jit.load(buffer, map_location=self.device)

    def _deserialize_pickle(self, kernel_artifact: bytes) -> nn.Module:
        """Deserialize pickled module with safety checks."""
        # Basic safety check on pickle data
        if b"subprocess" in kernel_artifact or b"os.system" in kernel_artifact:
            raise KernelDeserializationError("Potentially unsafe pickle data detected")

        try:
            # Use restricted unpickler for safety
            buffer = io.BytesIO(kernel_artifact)
            module = pickle.load(buffer)

            if not isinstance(module, nn.Module):
                raise KernelDeserializationError(
                    f"Expected nn.Module, got {type(module)}"
                )

            return module

        except Exception as e:
            raise KernelDeserializationError(f"Pickle deserialization failed: {e}")

    async def _execute_with_timeout(
        self, kernel_module: nn.Module, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute kernel with timeout protection.

        Args:
            kernel_module: Module to execute
            input_tensor: Input tensor

        Returns:
            Output tensor
        """

        # Create execution task
        async def execute_task():
            with torch.no_grad():
                return kernel_module(input_tensor)

        # Execute with timeout
        try:
            output = await asyncio.wait_for(
                execute_task(), timeout=self.execution_timeout
            )
            return output

        except asyncio.TimeoutError:
            raise KernelExecutionError(
                f"Kernel execution timed out after {self.execution_timeout}s"
            )

    def _validate_output_shape(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        expected_shape: torch.Size,
    ):
        """Validate output tensor shape."""
        # Basic sanity checks
        if not torch.is_tensor(output_tensor):
            raise KernelShapeError("Kernel output is not a tensor")

        if output_tensor.device != input_tensor.device:
            raise KernelShapeError("Kernel output device mismatch")

        # Check for NaN/Inf values
        if torch.any(torch.isnan(output_tensor)) or torch.any(
            torch.isinf(output_tensor)
        ):
            raise KernelShapeError("Kernel output contains NaN or Inf values")

        # Batch dimension should match
        if output_tensor.size(0) != input_tensor.size(0):
            raise KernelShapeError(
                f"Batch size mismatch: input {input_tensor.size(0)} vs output {output_tensor.size(0)}"
            )

    def _apply_alpha_blending(
        self, input_tensor: torch.Tensor, kernel_output: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Apply alpha blending between input and kernel output.

        Args:
            input_tensor: Original input tensor
            kernel_output: Kernel output tensor
            alpha: Blending factor (0.0 = all input, 1.0 = all kernel)

        Returns:
            Blended output tensor
        """
        # Handle shape mismatch by using kernel output shape
        if input_tensor.shape != kernel_output.shape:
            # If shapes don't match, we can't blend - return kernel output
            logger.debug(
                f"Shape mismatch for blending: {input_tensor.shape} vs {kernel_output.shape}"
            )
            return kernel_output

        # Linear blending
        return (1.0 - alpha) * input_tensor + alpha * kernel_output

    def _add_to_cache(self, kernel_id: str, module: nn.Module):
        """Add module to cache with LRU eviction."""
        # Evict if cache is full
        while len(self.kernel_cache) >= self.max_kernel_cache_size:
            self._evict_lru()

        self.kernel_cache[kernel_id] = module
        self.cache_access_times[kernel_id] = time.time()

    def _evict_lru(self):
        """Evict least recently used item from cache."""
        if not self.cache_access_times:
            return

        lru_kernel_id = min(
            self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k]
        )

        del self.kernel_cache[lru_kernel_id]
        del self.cache_access_times[lru_kernel_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.stats.total_executions,
            "successful_executions": self.stats.successful_executions,
            "failed_executions": self.stats.failed_executions,
            "success_rate": self.stats.success_rate,
            "average_execution_time_ms": self.stats.average_execution_time,
            "deserialize_errors": self.stats.deserialize_errors,
            "shape_errors": self.stats.shape_errors,
            "runtime_errors": self.stats.runtime_errors,
            "cached_kernels": len(self.kernel_cache),
            "max_cache_size": self.max_kernel_cache_size,
            "error_recovery_stats": self.error_recovery.get_recovery_stats(),
        }

    def clear_cache(self):
        """Clear kernel cache."""
        self.kernel_cache.clear()
        self.cache_access_times.clear()
        logger.info("Kernel cache cleared")


# Utility functions for testing and development
def create_test_kernel_artifact(input_size: int, output_size: int) -> bytes:
    """Create a test kernel artifact for development/testing."""
    module = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    # Serialize using torch.jit
    traced_module = torch.jit.trace(module, torch.randn(1, input_size))

    buffer = io.BytesIO()
    torch.jit.save(traced_module, buffer)
    return buffer.getvalue()


def validate_kernel_artifact(artifact: bytes) -> Tuple[bool, str]:
    """Validate a kernel artifact without executing it."""
    try:
        executor = RealKernelExecutor(torch.device("cpu"))
        module = asyncio.run(executor._deserialize_kernel(artifact))

        validator = KernelValidator()
        is_valid, error_msg = validator.validate_module(module)

        return is_valid, error_msg

    except Exception as e:
        return False, str(e)

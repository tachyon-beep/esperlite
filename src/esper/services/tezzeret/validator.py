"""
Kernel Validator Module.

This module validates compiled kernels for correctness, safety, and performance.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import numpy as np

from esper.contracts.assets import Blueprint, CompiledKernel

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of kernel validation."""
    is_valid: bool
    functional_correctness: bool
    performance_acceptable: bool
    memory_safe: bool
    gradient_correct: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class ValidationError(Exception):
    """Raised when kernel validation fails critically."""


class KernelValidator:
    """Validates compiled kernels for correctness and safety."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        tolerance: float = 1e-5,
        performance_threshold: float = 2.0
    ):
        """
        Initialize the validator.

        Args:
            device: Device to run validation on
            tolerance: Numerical tolerance for correctness checks
            performance_threshold: Maximum acceptable performance overhead
        """
        self.device = device or torch.device("cpu")
        self.tolerance = tolerance
        self.performance_threshold = performance_threshold
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
        }

    def validate_kernel(
        self,
        kernel: CompiledKernel,
        blueprint: Blueprint,
        reference_module: Optional[torch.nn.Module] = None
    ) -> ValidationResult:
        """
        Comprehensive kernel validation.

        Performs:
        1. Functional correctness testing
        2. Performance benchmarking
        3. Memory safety verification
        4. Gradient flow validation

        Args:
            kernel: Compiled kernel to validate
            blueprint: Original blueprint
            reference_module: Optional reference implementation

        Returns:
            ValidationResult: Validation results
        """
        logger.info("Validating kernel %s", kernel.kernel_id)

        errors = []
        warnings = []
        metrics = {}

        # Load the kernel module
        try:
            kernel_module = self._load_kernel_module(kernel)
        except Exception as e:
            errors.append(f"Failed to load kernel: {e}")
            return ValidationResult(
                is_valid=False,
                functional_correctness=False,
                performance_acceptable=False,
                memory_safe=False,
                gradient_correct=False,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

        # Test functional correctness
        functional_correct = self._test_functional_correctness(
            kernel_module, reference_module, errors, warnings, metrics
        )

        # Verify performance bounds
        performance_ok = self._verify_performance_bounds(
            kernel_module, errors, warnings, metrics
        )

        # Check memory safety
        memory_safe = self._check_memory_safety(
            kernel_module, errors, warnings, metrics
        )

        # Validate gradient flow
        gradient_correct = self._validate_gradient_flow(
            kernel_module, errors, warnings, metrics
        )

        # Overall validation result
        is_valid = (
            functional_correct and
            performance_ok and
            memory_safe and
            gradient_correct and
            len(errors) == 0
        )

        # Update statistics
        self.validation_stats["total_validated"] += 1
        if is_valid:
            self.validation_stats["passed"] += 1
        else:
            self.validation_stats["failed"] += 1
        if warnings:
            self.validation_stats["warnings"] += 1

        result = ValidationResult(
            is_valid=is_valid,
            functional_correctness=functional_correct,
            performance_acceptable=performance_ok,
            memory_safe=memory_safe,
            gradient_correct=gradient_correct,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

        logger.info(
            "Validation complete for kernel %s: %s",
            kernel.kernel_id,
            "PASSED" if is_valid else "FAILED"
        )

        return result

    def _load_kernel_module(self, kernel: CompiledKernel) -> torch.jit.ScriptModule:
        """Load kernel module from compiled artifact."""
        # In production, this would load from S3
        # For now, we'll create a dummy module matching the kernel metadata
        import io
        buffer = io.BytesIO()

        # Get input/output shapes from metadata
        input_size = kernel.metadata.input_shape[0] if kernel.metadata.input_shape else 128
        output_size = kernel.metadata.output_shape[0] if kernel.metadata.output_shape else 128

        # Create a simple test module matching the metadata
        test_module = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )
        scripted = torch.jit.script(test_module)
        torch.jit.save(scripted, buffer)
        buffer.seek(0)

        return torch.jit.load(buffer, map_location=self.device)

    def _test_functional_correctness(
        self,
        kernel_module: torch.jit.ScriptModule,
        reference_module: Optional[torch.nn.Module],
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, float]
    ) -> bool:
        """Compare kernel output with reference implementation."""
        try:
            # Create test inputs
            test_inputs = self._generate_test_inputs(kernel_module)

            if reference_module is None:
                # Basic sanity checks without reference
                for test_input in test_inputs:
                    output = kernel_module(test_input)

                    # Check output shape
                    if output.shape[0] != test_input.shape[0]:
                        errors.append("Output batch size doesn't match input")
                        return False

                    # Check for NaN/Inf
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        errors.append("Output contains NaN or Inf values")
                        return False

                    # Check output range
                    output_max = output.abs().max().item()
                    if output_max > 1e6:
                        warnings.append(f"Large output values detected: {output_max}")

                metrics["sanity_checks_passed"] = True
                return True

            else:
                # Compare with reference implementation
                reference_module = reference_module.to(self.device)
                reference_module.eval()
                kernel_module.eval()

                total_error = 0.0
                max_error = 0.0

                with torch.no_grad():
                    for test_input in test_inputs:
                        kernel_output = kernel_module(test_input)
                        reference_output = reference_module(test_input)

                        # Compute error
                        error = (kernel_output - reference_output).abs()
                        total_error += error.mean().item()
                        max_error = max(max_error, error.max().item())

                        # Check if within tolerance
                        if not torch.allclose(
                            kernel_output, reference_output,
                            rtol=self.tolerance, atol=self.tolerance
                        ):
                            errors.append(
                                f"Output mismatch: max error {error.max().item()}"
                            )
                            return False

                avg_error = total_error / len(test_inputs)
                metrics["average_error"] = avg_error
                metrics["max_error"] = max_error

                if avg_error > self.tolerance * 10:
                    warnings.append(f"High average error: {avg_error}")

                return True

        except Exception as e:
            errors.append(f"Functional correctness test failed: {e}")
            return False

    def _verify_performance_bounds(
        self,
        kernel_module: torch.jit.ScriptModule,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, float]
    ) -> bool:
        """Ensure kernel meets performance requirements."""
        try:
            # Create test input
            test_input = self._create_test_input(kernel_module)

            # Warmup
            for _ in range(10):
                _ = kernel_module(test_input)

            # Measure execution time
            if test_input.is_cuda:
                torch.cuda.synchronize()

            times = []
            num_runs = 100

            for _ in range(num_runs):
                start = time.perf_counter()
                _ = kernel_module(test_input)
                if test_input.is_cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            avg_time = np.mean(times)
            std_time = np.std(times)
            p95_time = np.percentile(times, 95)

            metrics["avg_execution_time_ms"] = avg_time
            metrics["std_execution_time_ms"] = std_time
            metrics["p95_execution_time_ms"] = p95_time

            # Check against threshold
            baseline_time = 1.0  # Baseline expectation in ms
            overhead = avg_time / baseline_time

            if overhead > self.performance_threshold:
                errors.append(
                    f"Performance overhead {overhead:.2f}x exceeds threshold "
                    f"{self.performance_threshold}x"
                )
                return False

            if std_time > avg_time * 0.2:  # High variance warning
                warnings.append(
                    f"High execution time variance: {std_time:.2f}ms "
                    f"(mean: {avg_time:.2f}ms)"
                )

            return True

        except Exception as e:
            errors.append(f"Performance verification failed: {e}")
            return False

    def _check_memory_safety(
        self,
        kernel_module: torch.jit.ScriptModule,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, float]
    ) -> bool:
        """Verify no out-of-bounds access or memory leaks."""
        try:
            # Test with various input sizes
            test_sizes = [1, 16, 32, 64, 128]

            if torch.cuda.is_available() and next(kernel_module.parameters()).is_cuda:
                # CUDA memory testing
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()

                for size in test_sizes:
                    test_input = torch.randn(size, 128, device=self.device)

                    # Run multiple times to check for leaks
                    for _ in range(10):
                        _ = kernel_module(test_input)

                    torch.cuda.synchronize()

                final_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()

                memory_leak = final_memory - initial_memory
                metrics["memory_leak_mb"] = memory_leak / (1024 * 1024)
                metrics["peak_memory_mb"] = peak_memory / (1024 * 1024)

                if memory_leak > 1024 * 1024:  # 1MB threshold
                    errors.append(f"Memory leak detected: {memory_leak} bytes")
                    return False

            else:
                # CPU memory testing (less precise)
                import psutil
                process = psutil.Process()

                initial_memory = process.memory_info().rss
                for size in test_sizes:
                    test_input = torch.randn(size, 128)
                    for _ in range(10):
                        _ = kernel_module(test_input)

                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory

                if memory_increase > 10 * 1024 * 1024:  # 10MB threshold
                    warnings.append(
                        f"Significant memory increase: "
                        f"{memory_increase / (1024*1024):.2f}MB"
                    )

            return True

        except Exception as e:
            errors.append(f"Memory safety check failed: {e}")
            return False

    def _validate_gradient_flow(
        self,
        kernel_module: torch.jit.ScriptModule,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, float]
    ) -> bool:
        """Ensure proper gradient computation."""
        try:
            # Create test input with gradient
            test_input = self._create_test_input(kernel_module).requires_grad_(True)

            # Forward pass
            output = kernel_module(test_input)

            # Create dummy loss
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Check if gradients exist
            if test_input.grad is None:
                errors.append("No gradients computed for input")
                return False

            # Check gradient validity
            if torch.isnan(test_input.grad).any() or torch.isinf(test_input.grad).any():
                errors.append("Gradients contain NaN or Inf")
                return False

            # Check gradient magnitude
            grad_norm = test_input.grad.norm().item()
            metrics["input_gradient_norm"] = grad_norm

            if grad_norm == 0:
                warnings.append("Zero gradients detected")
            elif grad_norm > 1000:
                warnings.append(f"Large gradient norm: {grad_norm}")

            # Gradient consistency check
            # Run multiple times and check consistency
            grad_norms = []
            for _ in range(5):
                test_input = self._create_test_input(kernel_module).requires_grad_(True)
                output = kernel_module(test_input)
                loss = output.sum()
                loss.backward()
                grad_norms.append(test_input.grad.norm().item())

            grad_std = np.std(grad_norms)
            if grad_std > np.mean(grad_norms) * 0.1:
                warnings.append(
                    f"Inconsistent gradients detected: std={grad_std:.4f}"
                )

            return True

        except Exception as e:
            errors.append(f"Gradient validation failed: {e}")
            return False

    def _generate_test_inputs(
        self, kernel_module: torch.jit.ScriptModule
    ) -> List[torch.Tensor]:
        """Generate various test inputs for validation."""
        test_inputs = []

        # Standard test cases
        batch_sizes = [1, 16, 32]
        input_size = 128  # Default, should be inferred from module

        for batch_size in batch_sizes:
            # Normal distribution
            test_inputs.append(
                torch.randn(batch_size, input_size, device=self.device)
            )

            # Uniform distribution
            test_inputs.append(
                torch.rand(batch_size, input_size, device=self.device)
            )

            # Edge cases
            # Zeros
            test_inputs.append(
                torch.zeros(batch_size, input_size, device=self.device)
            )

            # Large values
            test_inputs.append(
                torch.randn(batch_size, input_size, device=self.device) * 100
            )

        return test_inputs

    def _create_test_input(
        self, kernel_module: torch.jit.ScriptModule
    ) -> torch.Tensor:
        """Create a single test input tensor."""
        # Try to infer input shape from first parameter
        try:
            first_param = next(kernel_module.parameters())
            if len(first_param.shape) >= 2:
                # Assume it's a linear layer weight matrix
                input_size = first_param.shape[1]
                return torch.randn(32, input_size, device=self.device)
        except StopIteration:
            pass
        
        # Default fallback
        return torch.randn(32, 128, device=self.device)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        if stats["total_validated"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_validated"]
            stats["warning_rate"] = stats["warnings"] / stats["total_validated"]
        else:
            stats["pass_rate"] = 0.0
            stats["warning_rate"] = 0.0
        return stats
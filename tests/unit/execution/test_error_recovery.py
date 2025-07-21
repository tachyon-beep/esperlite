"""
Unit tests for the error recovery system.

This module contains comprehensive tests for error handling, recovery strategies,
and system health monitoring.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from src.esper.execution.error_recovery import (
    ErrorType,
    RecoveryStrategy,
    ErrorContext,
    RecoveryAction,
    ErrorTracker,
    ErrorRecoveryManager,
    HealthMonitor,
    create_error_context,
    classify_kernel_error
)
from src.esper.execution.kernel_executor import KernelExecutionError
from src.esper.contracts.operational import HealthSignal


class TestErrorType:
    """Test ErrorType enumeration."""
    
    def test_error_types_exist(self):
        """Test that all expected error types exist."""
        expected_types = [
            "KERNEL_EXECUTION",
            "KERNEL_LOADING", 
            "KERNEL_VALIDATION",
            "MEMORY_OVERFLOW",
            "DEVICE_ERROR",
            "NETWORK_ERROR",
            "CIRCUIT_BREAKER",
            "TIMEOUT",
            "UNKNOWN"
        ]
        
        for error_type in expected_types:
            assert hasattr(ErrorType, error_type)


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation and serialization."""
        exception = ValueError("Test error")
        
        context = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer",
            seed_idx=1,
            kernel_id="test_kernel",
            exception=exception,
            metadata={"test_key": "test_value"}
        )
        
        assert context.error_type == ErrorType.KERNEL_EXECUTION
        assert context.component == "test_component"
        assert context.layer_name == "test_layer"
        assert context.seed_idx == 1
        assert context.kernel_id == "test_kernel"
        assert context.exception == exception
        assert context.metadata["test_key"] == "test_value"
    
    def test_error_context_to_dict(self):
        """Test ErrorContext serialization to dictionary."""
        exception = RuntimeError("Runtime error")
        
        context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            component="test_component",
            layer_name="test_layer",
            exception=exception
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["error_type"] == "timeout"
        assert context_dict["component"] == "test_component"
        assert context_dict["layer_name"] == "test_layer"
        assert context_dict["exception_type"] == "RuntimeError"
        assert context_dict["exception_message"] == "Runtime error"
        assert "timestamp" in context_dict


class TestErrorTracker:
    """Test ErrorTracker functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = ErrorTracker(window_size=5)
    
    def test_record_error(self):
        """Test error recording."""
        context = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer"
        )
        
        self.tracker.record_error(context)
        
        assert len(self.tracker.error_history) == 1
        assert self.tracker.error_counts[ErrorType.KERNEL_EXECUTION] == 1
        assert len(self.tracker.component_errors["test_component"]) == 1
    
    def test_sliding_window(self):
        """Test sliding window behavior."""
        # Add more errors than window size
        for i in range(7):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component=f"component_{i}",
                layer_name="test_layer"
            )
            self.tracker.record_error(context)
        
        # Should maintain window size
        assert len(self.tracker.error_history) == 5
        assert self.tracker.error_counts[ErrorType.KERNEL_EXECUTION] == 5
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        # Add errors over time
        base_time = time.time()
        
        for i in range(3):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component="test_component",
                layer_name="test_layer"
            )
            context.timestamp = base_time + i * 10  # 10 second intervals
            self.tracker.record_error(context)
        
        # Calculate error rate for 30 second window
        with patch('time.time', return_value=base_time + 30):
            error_rate = self.tracker.get_error_rate(time_window=30.0)
            assert error_rate == 3.0 / 30.0  # 3 errors in 30 seconds
    
    def test_problematic_kernel_detection(self):
        """Test problematic kernel detection."""
        kernel_id = "problematic_kernel"
        
        # Add multiple errors for same kernel
        for i in range(4):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component="test_component",
                layer_name="test_layer",
                kernel_id=kernel_id
            )
            self.tracker.record_error(context)
        
        assert self.tracker.is_problematic_kernel(kernel_id, threshold=3)
        assert not self.tracker.is_problematic_kernel("good_kernel", threshold=3)
    
    def test_problematic_component_detection(self):
        """Test problematic component detection."""
        component = "problematic_component"
        
        # Add multiple errors for same component
        for i in range(12):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component=component,
                layer_name="test_layer"
            )
            self.tracker.record_error(context)
        
        assert self.tracker.is_problematic_component(component, threshold=10)
        assert not self.tracker.is_problematic_component("good_component", threshold=10)
    
    def test_get_stats(self):
        """Test statistics generation."""
        # Add some test errors
        for error_type in [ErrorType.KERNEL_EXECUTION, ErrorType.TIMEOUT, ErrorType.TIMEOUT]:
            context = ErrorContext(
                error_type=error_type,
                component="test_component", 
                layer_name="test_layer"
            )
            self.tracker.record_error(context)
        
        stats = self.tracker.get_stats()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts"][ErrorType.KERNEL_EXECUTION.value] == 1
        assert stats["error_counts"][ErrorType.TIMEOUT.value] == 2
        assert "error_rate_5min" in stats
        assert "problematic_kernels" in stats
        assert "problematic_components" in stats


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = ErrorRecoveryManager()
    
    @pytest.mark.asyncio
    async def test_handle_error_basic(self):
        """Test basic error handling."""
        context = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer",
            exception=RuntimeError("Test error")
        )
        
        success = await self.manager.handle_error(context)
        
        # Should have recorded the error
        assert len(self.manager.error_tracker.error_history) == 1
        assert len(self.manager.recovery_history) == 1
        
        # Should have attempted recovery
        recovery_record = self.manager.recovery_history[0]
        assert recovery_record["error_context"]["error_type"] == "kernel_execution"
        assert recovery_record["strategy"] == "fallback"  # Default for kernel execution
    
    @pytest.mark.asyncio
    async def test_handle_error_with_fallback(self):
        """Test error handling with fallback action."""
        fallback_called = False
        
        def test_fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"
        
        context = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer"
        )
        
        success = await self.manager.handle_error(context, fallback_action=test_fallback)
        
        assert success
        assert fallback_called
    
    @pytest.mark.asyncio
    async def test_escalation_conditions(self):
        """Test error escalation conditions."""
        kernel_id = "problematic_kernel"
        
        # Add enough errors to trigger escalation
        for i in range(5):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component="test_component",
                layer_name="test_layer",
                kernel_id=kernel_id
            )
            self.manager.error_tracker.record_error(context)
        
        # Next error should escalate
        context = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer",
            kernel_id=kernel_id
        )
        
        # Mock the escalation check
        with patch.object(self.manager.error_tracker, 'is_problematic_kernel', return_value=True):
            success = await self.manager.handle_error(context)
            
            # Should have escalated
            recovery_record = self.manager.recovery_history[-1]
            assert recovery_record["strategy"] == "escalate"
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery_prevention(self):
        """Test prevention of concurrent recovery for same component."""
        context1 = ErrorContext(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer"
        )
        
        context2 = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            component="test_component", 
            layer_name="test_layer"
        )
        
        # Start first recovery
        task1 = asyncio.create_task(self.manager.handle_error(context1))
        
        # Try to start second recovery for same component while first is running
        # Add small delay to ensure first recovery starts
        await asyncio.sleep(0.01)
        success2 = await self.manager.handle_error(context2)
        
        # Wait for first to complete
        success1 = await task1
        
        # Second should be rejected (already in progress)
        assert not success2
        assert success1
    
    def test_recovery_strategy_assignment(self):
        """Test recovery strategy assignment based on error type."""
        test_cases = [
            (ErrorType.KERNEL_EXECUTION, RecoveryStrategy.FALLBACK),
            (ErrorType.KERNEL_LOADING, RecoveryStrategy.RETRY),
            (ErrorType.MEMORY_OVERFLOW, RecoveryStrategy.CIRCUIT_BREAKER),
            (ErrorType.NETWORK_ERROR, RecoveryStrategy.RETRY),
            (ErrorType.UNKNOWN, RecoveryStrategy.FALLBACK)
        ]
        
        for error_type, expected_strategy in test_cases:
            strategy = self.manager.recovery_strategies.get(error_type)
            assert strategy == expected_strategy
    
    def test_get_recovery_stats(self):
        """Test recovery statistics."""
        stats = self.manager.get_recovery_stats()
        
        assert "total_recoveries" in stats
        assert "recent_recoveries" in stats
        assert "recovery_success_rate" in stats
        assert "active_recoveries" in stats
        assert "error_stats" in stats


class TestHealthMonitor:
    """Test HealthMonitor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.recovery_manager = ErrorRecoveryManager()
        self.monitor = HealthMonitor(self.recovery_manager)
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping health monitoring."""
        assert not self.monitor.monitoring_active
        
        # Start monitoring
        monitoring_task = asyncio.create_task(self.monitor.start_monitoring())
        
        # Wait a bit to ensure monitoring starts
        await asyncio.sleep(0.1)
        assert self.monitor.monitoring_active
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        await asyncio.sleep(0.1)
        assert not self.monitor.monitoring_active
        
        # Cancel the task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
    
    def test_add_health_signal(self):
        """Test health signal addition and cleanup."""
        # Create test health signals
        signals = []
        base_time = time.time()
        
        for i in range(5):
            signal = HealthSignal(
                layer_id=1,
                seed_id=i,
                chunk_id=0,
                epoch=0,
                activation_variance=base_time - (i * 1800),  # Using as timestamp
                dead_neuron_ratio=0.1,
                avg_correlation=0.8,
                is_ready_for_transition=False
            )
            signals.append(signal)
            self.monitor.add_health_signal(signal)
        
        # Should keep recent signals
        assert len(self.monitor.health_signals) <= 5
    
    def test_get_health_summary(self):
        """Test health summary generation."""
        summary = self.monitor.get_health_summary()
        
        assert "monitoring_active" in summary
        assert "health_signals_count" in summary
        assert "recovery_stats" in summary
        assert "thresholds" in summary
        
        assert isinstance(summary["monitoring_active"], bool)
        assert isinstance(summary["health_signals_count"], int)
        assert isinstance(summary["thresholds"], dict)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_error_context(self):
        """Test error context creation utility."""
        exception = ValueError("Test error")
        
        context = create_error_context(
            error_type=ErrorType.KERNEL_EXECUTION,
            component="test_component",
            layer_name="test_layer",
            exception=exception,
            seed_idx=1,
            kernel_id="test_kernel"
        )
        
        assert context.error_type == ErrorType.KERNEL_EXECUTION
        assert context.component == "test_component"
        assert context.layer_name == "test_layer"
        assert context.exception == exception
        assert context.seed_idx == 1
        assert context.kernel_id == "test_kernel"
    
    def test_classify_kernel_error(self):
        """Test kernel error classification."""
        test_cases = [
            (KernelExecutionError("execution failed"), ErrorType.KERNEL_EXECUTION),
            (torch.cuda.OutOfMemoryError("out of memory"), ErrorType.MEMORY_OVERFLOW),
            (ConnectionError("connection failed"), ErrorType.NETWORK_ERROR),
            (TimeoutError("timeout"), ErrorType.NETWORK_ERROR),
            (RuntimeError("timeout occurred"), ErrorType.TIMEOUT),
            (ValueError("unknown error"), ErrorType.UNKNOWN)
        ]
        
        for exception, expected_type in test_cases:
            classified_type = classify_kernel_error(exception)
            assert classified_type == expected_type


@pytest.mark.integration 
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self):
        """Test complete error handling workflow."""
        manager = ErrorRecoveryManager()
        
        # Simulate a series of errors
        errors = [
            (ErrorType.KERNEL_EXECUTION, "component_a", "layer_1"),
            (ErrorType.TIMEOUT, "component_a", "layer_1"),
            (ErrorType.KERNEL_LOADING, "component_b", "layer_2"),
            (ErrorType.MEMORY_OVERFLOW, "component_c", "layer_3")
        ]
        
        recovery_results = []
        
        for error_type, component, layer in errors:
            context = ErrorContext(
                error_type=error_type,
                component=component,
                layer_name=layer,
                exception=RuntimeError(f"Test {error_type.value} error")
            )
            
            result = await manager.handle_error(context)
            recovery_results.append(result)
        
        # Verify all errors were handled
        assert len(manager.error_tracker.error_history) == 4
        assert len(manager.recovery_history) == 4
        
        # Check recovery strategies were applied
        strategies_used = [r["strategy"] for r in manager.recovery_history]
        assert "fallback" in strategies_used  # For kernel execution
        assert "retry" in strategies_used     # For kernel loading and timeout
        assert "circuit_breaker" in strategies_used  # For memory overflow
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test integration between health monitoring and error recovery."""
        recovery_manager = ErrorRecoveryManager()
        health_monitor = HealthMonitor(recovery_manager)
        
        # Add some errors to trigger health alerts
        for i in range(10):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component="stressed_component",
                layer_name="test_layer"
            )
            await recovery_manager.handle_error(context)
        
        # Check health monitoring detects the issue
        error_rate = recovery_manager.error_tracker.get_error_rate(time_window=60.0)
        assert error_rate > health_monitor.health_thresholds["error_rate"]
        
        # Verify health summary includes error information
        summary = health_monitor.get_health_summary()
        assert summary["recovery_stats"]["total_recoveries"] == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test handling of concurrent errors."""
        manager = ErrorRecoveryManager()
        
        async def generate_error(error_id: int):
            context = ErrorContext(
                error_type=ErrorType.KERNEL_EXECUTION,
                component=f"component_{error_id % 3}",  # 3 different components
                layer_name=f"layer_{error_id}",
                exception=RuntimeError(f"Concurrent error {error_id}")
            )
            return await manager.handle_error(context)
        
        # Generate 10 concurrent errors
        tasks = [generate_error(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Some recoveries may be rejected if they target the same component
        successful_recoveries = sum(1 for r in results if r)
        
        # Should have handled at least some errors
        assert successful_recoveries >= 3  # At least one per component
        assert len(manager.error_tracker.error_history) == 10
        assert len(manager.recovery_history) >= 3


if __name__ == "__main__":
    pytest.main([__file__])